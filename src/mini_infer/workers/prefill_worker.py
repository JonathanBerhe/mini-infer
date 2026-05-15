"""Prefill side of the disaggregated PD pipeline.

A `PrefillWorker`:

  1. Tokenizes the incoming prompt via its `ModelRunner`'s tokenizer.
  2. Adds a fresh request slot to its `PagedKVCache`.
  3. Runs one packed-varlen forward over the prompt tokens.
  4. Samples the first output token from the last prefill logit.
  5. Extracts every layer's per-stream KV into a `KVHandoff`.
  6. Releases the request slot (the KV data lives in the handoff now).
  7. Returns the handoff.

Single-request only: do not call `prefill()` concurrently.

Limitations today:
  - `kv_quant` must be `None`. The compressed-pool extraction path goes
    through a separate API and is not exercised here.
  - The handoff carries the prefill worker's device-resident tensors;
    the decode worker moves them to its own device if needed.
"""

from __future__ import annotations

import logging
import uuid

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams, sample
from mini_infer.scheduler.request_state import Request
from mini_infer.workers.kv_handoff import KVHandoff

logger = logging.getLogger(__name__)


class PrefillWorker:
    """Owns a `ModelRunner`; runs single-request prefill and emits a `KVHandoff`."""

    def __init__(self, runner: ModelRunner) -> None:
        if runner.block_pool.kv_quant is not None:
            raise NotImplementedError(
                f"PrefillWorker requires kv_quant=None (got "
                f"kv_quant={runner.block_pool.kv_quant!r}); KV-quant + PD is not wired."
            )
        self._runner = runner

    @property
    def runner(self) -> ModelRunner:
        return self._runner

    def prefill(self, request: Request) -> KVHandoff:
        """Run prefill on `request.prompt` and return the handoff for decode.

        Caller invariants:
          - This worker is single-request: do not call `prefill()` concurrently.
          - The request's `sampling_params` and `max_tokens` propagate to the
            handoff; the decode worker uses them.
        """
        prompt_token_ids = self._runner.tokenizer.encode(request.prompt)
        if not prompt_token_ids:
            raise ValueError("PrefillWorker received an empty prompt")

        # One-shot single-request prefill: build a fresh cache, run the forward,
        # sample the first token, extract per-stream KV, free the slot.
        cache = PagedKVCache(self._runner.block_pool)
        batch_idx = cache.add_request_slot()
        try:
            cache, last_logits = self._runner.prefill_chunk(
                cache, prompt_token_ids, position_offset=0
            )
            first_token_id = sample(last_logits, request.sampling_params)
            handoff = self._extract_handoff(
                cache=cache,
                request_id=str(uuid.uuid4()),
                prefill_len=len(prompt_token_ids),
                first_sampled_token_id=first_token_id,
                sampling_params=request.sampling_params,
                max_tokens=request.max_tokens,
                eos_token_id=self._runner.tokenizer.eos_token_id,
            )
        finally:
            # Free the slot whether or not extraction succeeded; otherwise a
            # raised exception leaks blocks back into the next request's pool.
            cache.remove_request(batch_idx)
        logger.debug(
            "PrefillWorker emitted handoff request_id=%s prefill_len=%d first_token=%d",
            handoff.request_id,
            handoff.prefill_len,
            handoff.first_sampled_token_id,
        )
        return handoff

    def prefill_batch(self, requests: list[Request]) -> list[KVHandoff]:
        """Run prefill for a batch of requests in ONE packed-varlen forward.

        Each request becomes its own slot in a fresh `PagedKVCache`; all
        prompts are tokenized, packed into a single varlen tensor with
        per-request `cu_seqlens_q`, and processed in one forward call. The
        per-request last logits are sampled to produce each request's
        first output token. Per-layer per-stream KV is then extracted
        from the cache via a single `materialize_packed_stream` call
        per (layer, stream), sliced per-request using `cu_seqlens_k`, and
        packaged into B handoffs.

        Equivalent in output to running `prefill(request)` once per
        request (same sampled tokens, same KV bytes) — but it uses one
        forward call instead of B, which is the throughput win
        production engines deliver on the prefill side. Parity is
        validated against the per-request loop in
        `tests/unit/test_workers_batch.py`.

        Limitations today:
          - All requests share the cache's block pool. If the combined
            prompt length exceeds pool capacity, this raises (the
            single-request path is what falls back).
          - No mid-batch admission. The batch is fixed at entry; a
            ContinuousScheduler-style admission loop is out of scope here.

        Caller invariants:
          - Single-call: don't share a `PrefillWorker` instance across
            concurrent batched calls.
        """
        if not requests:
            return []
        tokenizer = self._runner.tokenizer
        pool = self._runner.block_pool
        cache = PagedKVCache(pool)

        per_request_token_ids: list[list[int]] = []
        for request in requests:
            token_ids = tokenizer.encode(request.prompt)
            if not token_ids:
                raise ValueError(
                    f"PrefillWorker.prefill_batch received an empty prompt in batch slot "
                    f"{len(per_request_token_ids)}"
                )
            per_request_token_ids.append(token_ids)

        batch_idxs: list[int] = []
        try:
            packed_input_ids: list[int] = []
            cu_seqlens_q: list[int] = [0]
            for token_ids in per_request_token_ids:
                cache.add_request_slot()
                batch_idxs.append(cache.batch_size - 1)
                packed_input_ids.extend(token_ids)
                cu_seqlens_q.append(cu_seqlens_q[-1] + len(token_ids))
            # All slots start at position 0 (fresh prefill).
            position_offsets = [0] * len(requests)
            per_request_logits = self._runner.forward_step(
                cache, packed_input_ids, cu_seqlens_q, position_offsets
            )
            first_tokens = [
                sample(per_request_logits[i], requests[i].sampling_params)
                for i in range(len(requests))
            ]
            handoffs = self._extract_batch_handoff(
                cache=cache,
                requests=requests,
                per_request_token_ids=per_request_token_ids,
                first_sampled_token_ids=first_tokens,
            )
        finally:
            # Remove slots in reverse order so each index stays valid as we shrink.
            for batch_idx in reversed(batch_idxs):
                cache.remove_request(batch_idx)
        return handoffs

    def _extract_batch_handoff(
        self,
        *,
        cache: PagedKVCache,
        requests: list[Request],
        per_request_token_ids: list[list[int]],
        first_sampled_token_ids: list[int],
    ) -> list[KVHandoff]:
        """Slice a multi-slot cache's per-stream packed KV into per-request handoffs.

        `materialize_packed_stream` returns `(total_k, h, d)` plus a
        `cu_seqlens_k` tensor with the per-request K boundaries. We
        materialize each (layer, stream) once and slice it B ways using
        those boundaries. This is the multi-slot generalization of
        `_extract_handoff`.
        """
        pool = self._runner.block_pool
        eos_token_id = self._runner.tokenizer.eos_token_id
        batch_size = len(requests)

        # Pre-build the empty per-request layer lists so we can index into them.
        per_request_layer_streams: list[list[dict[str, torch.Tensor]]] = [
            [] for _ in range(batch_size)
        ]
        for layer_idx in range(pool.num_layers):
            for stream_name in pool.stream_names(layer_idx):
                stream_packed, cu_seqlens_k, _max_seq = cache.materialize_packed_stream(
                    layer_idx, stream_name
                )
                offsets = cu_seqlens_k.tolist()
                if len(offsets) != batch_size + 1:
                    raise RuntimeError(
                        f"materialize_packed_stream returned cu_seqlens_k of length "
                        f"{len(offsets)}, expected {batch_size + 1}"
                    )
                for r in range(batch_size):
                    start, end = offsets[r], offsets[r + 1]
                    expected = len(per_request_token_ids[r])
                    if end - start != expected:
                        raise RuntimeError(
                            f"layer {layer_idx} stream {stream_name!r} request {r}: "
                            f"got {end - start} positions, expected {expected}"
                        )
                    # Allocate a layer dict on demand for this request.
                    if len(per_request_layer_streams[r]) <= layer_idx:
                        per_request_layer_streams[r].append({})
                    per_request_layer_streams[r][layer_idx][stream_name] = stream_packed[
                        start:end
                    ].clone()

        handoffs: list[KVHandoff] = []
        for r in range(batch_size):
            handoffs.append(
                KVHandoff(
                    request_id=str(uuid.uuid4()),
                    kv_streams_per_layer=per_request_layer_streams[r],
                    prefill_len=len(per_request_token_ids[r]),
                    first_sampled_token_id=first_sampled_token_ids[r],
                    sampling_params=requests[r].sampling_params,
                    max_tokens=requests[r].max_tokens,
                    eos_token_id=eos_token_id,
                )
            )
        return handoffs

    def _extract_handoff(
        self,
        *,
        cache: PagedKVCache,
        request_id: str,
        prefill_len: int,
        first_sampled_token_id: int,
        sampling_params: SamplingParams,
        max_tokens: int,
        eos_token_id: int | None,
    ) -> KVHandoff:
        """Pull every layer's per-stream packed KV out of the prefill cache.

        `materialize_packed_stream` returns tensors of shape
        `(total_k, num_kv_heads_s, head_dim_s)` concatenated over all
        requests in the cache; since the prefill cache holds exactly one
        request here, `total_k == prefill_len` and no slicing is needed.
        """
        pool = self._runner.block_pool
        kv_streams_per_layer: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(pool.num_layers):
            layer_streams: dict[str, torch.Tensor] = {}
            for stream_name in pool.stream_names(layer_idx):
                stream_packed, _cu_seqlens_k, _max_seq = cache.materialize_packed_stream(
                    layer_idx, stream_name
                )
                if stream_packed.shape[0] != prefill_len:
                    # Sanity: we built the cache with exactly one request of
                    # prefill_len tokens. If the materialize disagrees, the
                    # cache state has drifted (a previous slot wasn't freed,
                    # or a bug in `add_request_slot`).
                    raise RuntimeError(
                        f"materialize_packed_stream returned total_k="
                        f"{stream_packed.shape[0]} but prefill_len={prefill_len} "
                        f"(layer={layer_idx} stream={stream_name!r})"
                    )
                # `.clone()` decouples the handoff tensors from the pool
                # storage so freeing the slot doesn't invalidate them.
                layer_streams[stream_name] = stream_packed.clone()
            kv_streams_per_layer.append(layer_streams)
        return KVHandoff(
            request_id=request_id,
            kv_streams_per_layer=kv_streams_per_layer,
            prefill_len=prefill_len,
            first_sampled_token_id=first_sampled_token_id,
            sampling_params=sampling_params,
            max_tokens=max_tokens,
            eos_token_id=eos_token_id,
        )
