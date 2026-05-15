"""Decode side of the disaggregated PD pipeline.

A `DecodeWorker`:

  1. Accepts a `KVHandoff` from the prefill worker.
  2. Adds a fresh request slot to its `PagedKVCache`.
  3. Materializes the handoff's per-stream KV into the cache (one
     `append_stream_packed` call per (layer, stream) pair). Block
     allocation happens implicitly on the first stream of layer 0.
  4. Yields the handoff's first sampled token as the first output token.
  5. Runs the decode loop one token at a time, sampling from each step's
     last logit, until EOS or `max_tokens` is reached.
  6. Releases the request slot.

The "first token comes from the handoff, not from this worker's model"
detail matters for parity: prefill saw the full prompt context when it
sampled, so it produced the same token target-alone would produce. The
decode worker continues from there. Token-for-token greedy parity vs
`ContinuousScheduler` falls out by construction.

Limitations match the prefill side:
  - `kv_quant` must be `None`.
  - Single-request: do not call `decode()` concurrently.
  - The handoff's tensors are moved to this worker's device on entry.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import sample
from mini_infer.workers.kv_handoff import KVHandoff

logger = logging.getLogger(__name__)


class DecodeWorker:
    """Owns a `ModelRunner`; ingests a `KVHandoff` and streams decoded tokens."""

    def __init__(self, runner: ModelRunner) -> None:
        if runner.block_pool.kv_quant is not None:
            raise NotImplementedError(
                f"DecodeWorker requires kv_quant=None (got "
                f"kv_quant={runner.block_pool.kv_quant!r}); KV-quant + PD is not wired."
            )
        self._runner = runner

    @property
    def runner(self) -> ModelRunner:
        return self._runner

    def decode(self, handoff: KVHandoff) -> Iterator[int]:
        """Yield decoded token ids one at a time, starting with the handoff's first token.

        Stops at the first of:
          - `len(emitted) == handoff.max_tokens`,
          - emitted token equals `handoff.eos_token_id`.

        The first yielded token is `handoff.first_sampled_token_id` (the
        token sampled from the last prefill logit, already determined by
        the prefill worker). Subsequent tokens come from this worker's
        decode steps.
        """
        cache = PagedKVCache(self._runner.block_pool)
        batch_idx = cache.add_request_slot()
        try:
            self._materialize_handoff_into_cache(cache, handoff)
            yield from self._decode_loop(cache, handoff)
        finally:
            cache.remove_request(batch_idx)

    def _materialize_handoff_into_cache(self, cache: PagedKVCache, handoff: KVHandoff) -> None:
        """Write the handoff's KV into `cache`'s freshly-allocated request slot.

        Calls `append_stream_packed` once per (layer, stream). The first
        call (layer 0, first stream of layer 0) advances per-slot token
        counts and allocates blocks; subsequent calls only write into the
        existing blocks.

        Stream order matters: `append_stream_packed` expects the first
        stream of layer 0 to be the allocation trigger, so we iterate
        in `pool.stream_names(layer_idx)` order.
        """
        pool = self._runner.block_pool
        prefill_len = handoff.prefill_len
        cu_seqlens_q_new = torch.tensor(
            [0, prefill_len],
            dtype=torch.int32,
            device=pool.storage_for_stream(0, pool.stream_names(0)[0]).device,
        )
        if pool.num_layers != handoff.num_layers:
            raise ValueError(
                f"handoff has {handoff.num_layers} layers but pool has {pool.num_layers}"
            )
        for layer_idx in range(pool.num_layers):
            expected_streams = pool.stream_names(layer_idx)
            handoff_streams = handoff.kv_streams_per_layer[layer_idx]
            if set(expected_streams) != set(handoff_streams.keys()):
                raise ValueError(
                    f"handoff layer {layer_idx} streams {sorted(handoff_streams.keys())} "
                    f"don't match pool streams {sorted(expected_streams)}"
                )
            for stream_name in expected_streams:
                stream_packed = handoff_streams[stream_name]
                # Move to the decode worker's device. When both workers
                # share a runner this is a no-op; otherwise the handoff
                # tensors land on the prefill device and need a hop.
                if stream_packed.device != cu_seqlens_q_new.device:
                    stream_packed = stream_packed.to(device=cu_seqlens_q_new.device)
                if stream_packed.shape[0] != prefill_len:
                    raise ValueError(
                        f"handoff layer {layer_idx} stream {stream_name!r} has "
                        f"{stream_packed.shape[0]} positions; expected {prefill_len}"
                    )
                cache.append_stream_packed(
                    stream_packed=stream_packed,
                    cu_seqlens_q_new=cu_seqlens_q_new,
                    layer_idx=layer_idx,
                    stream_name=stream_name,
                )

    def _decode_loop(self, cache: PagedKVCache, handoff: KVHandoff) -> Iterator[int]:
        """Yield decoded tokens starting with the handoff's first token.

        Decode-loop contract (must match what `ContinuousScheduler` does so
        the parity test holds):

          - Emit `first_sampled_token_id` immediately (it was sampled from
            the last prefill logit, already correct).
          - For each subsequent step, feed the most-recently-emitted token
            into `decode_batch`, sample from the returned logits, emit.

        Termination: stop after `max_tokens` total emitted, OR when the
        emitted token equals `eos_token_id`.
        """
        if handoff.max_tokens <= 0:
            return
        # The first token: already sampled by the prefill worker. Yield it
        # without running the decode model — `ContinuousScheduler` does the
        # same (its first emitted token is `sample(prefill_logits, params)`,
        # which is what's in the handoff).
        first = handoff.first_sampled_token_id
        yield first
        if handoff.eos_token_id is not None and first == handoff.eos_token_id:
            return

        emitted = 1
        last_token = first
        while emitted < handoff.max_tokens:
            cache, logits = self._runner.decode(cache, last_token)
            next_token = sample(logits, handoff.sampling_params)
            yield next_token
            emitted += 1
            if handoff.eos_token_id is not None and next_token == handoff.eos_token_id:
                return
            last_token = next_token
