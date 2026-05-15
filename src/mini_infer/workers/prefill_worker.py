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
