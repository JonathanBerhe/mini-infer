# ADR-006: Chunked prefill (head-of-line-blocking fix)

Date: 2026-04-27
Status: Accepted (with packed-forward optimization deferred to a follow-up ADR)

## Context

After continuous batching shipped (ADR-005), the scheduler processes prefill in one shot and decode in batched form. A long-prompt request that lands while short requests are decoding still blocks the engine for the entire duration of its prefill: `model.forward(...)` over a 4k-token prompt is one big GPU call, and during that call no decoders make progress. This is the classic "head-of-line blocking" pathology in inference engines, and it's what chunked prefill exists to fix.

The structural change is to advance prefill in fixed-size chunks (default 256 tokens) interleaved with decoder steps. With chunk_size=256, a 4k-prompt prefill is split into 16 engine steps, each step also running a batched decode forward over any in-flight decoders.

Two implementations of this fit the same scheduler structure:

1. **Two forwards per step** (this slice). One `model.forward` for the prefill chunk (q_len=256, paged-aware via existing materialization), one for the batched decode (q_len=1, paged kernel). Decode requests still wait the duration of the chunk forward, but the chunk is bounded — vs the unbounded full-prefill block.
2. **One packed forward per step** (deferred follow-up). One `model.forward` over a packed sequence: prefill-chunk q-tokens + decode q-tokens concatenated, varlen attention via FlashAttention. This is the vLLM/SGLang shape and is what gets the maximum throughput win.

This ADR covers (1). (2) is queued as the next slice and will get its own ADR when it lands.

## Decision

1. **State machine**: `RequestState` gains `CHUNKED_PREFILLING`. `RunningRequest` gains `tokens_prefilled: int` and `prefill_cache: PagedKVCache | None`. Lifecycle: `WAITING -> PREFILLING -> CHUNKED_PREFILLING -> DECODING -> DONE`. `PREFILLING` is a one-step transient (set on admission, immediately advanced to `CHUNKED_PREFILLING` on the next step).
2. **Per-request prefill cache**: each prefilling request owns a temp `PagedKVCache(batch_size=1)` allocated at admission time. Successive chunks `prefill_chunk(cache, chunk_tokens, position_offset)` accumulate K/V across multiple engine steps. After the final chunk, the cache merges into the scheduler's shared batched cache via `merge_request` (block ownership transfers cleanly; same pattern as ADR-005).
3. **Chunk size = 256 tokens by default**, configurable via `ContinuousScheduler(chunk_size=...)`. 256 is the vLLM/SGLang default and balances per-step overhead against decoder responsiveness. Benchmarks should sweep this.
4. **`PagedKVCache.append_kv_packed(packed_k, packed_v, cu_seqlens_q_new, layer_idx)`**: new primitive that writes ragged per-slot appends (zero is allowed for a slot, meaning "no append this step"). The existing uniform-length `append_kv` becomes a wrapper that builds uniform `cu_seqlens` and delegates. Same physical write pattern; new entry point opens up the packed-forward path for the follow-up slice.
5. **`PagedKVCache.get_mask_sizes(query_length, layer_idx)`** is overridden. Without this, HF's `create_causal_mask` infers `kv_length` from `DynamicCache.layers` (which our paging-backed cache does not populate) and returns `(query_length, 0)` instead of `(existing_seq_len + query_length, 0)`. The mask gets sized too small, attention truncates to only the new chunk's K/V, and outputs silently diverge from the un-chunked path. Fix: return `(max(self._num_tokens) + query_length, 0)`. Caught by `test_chunked_prefill_matches_unchunked` before it ever shipped.
6. **No changes to the patched Qwen2 attention**: the paged-decode fast path (q_len=1, batch_size>0) is unchanged; prefill chunks (q_len=chunk_size > 1) flow through HF's stock attention via `cache.update`, which materializes the full K/V history for the layer's SDPA call. Same code path as the un-chunked single-shot prefill we shipped in ADR-005.

## Alternatives considered

- **One packed forward per step now (Approach 2)**. Higher throughput because one forward amortizes the matmul cost across all in-flight q-tokens (chunk + decoders). Requires varlen attention with paged K/V — `flash_attn_varlen_func` doesn't support paged, `flash_attn_with_kvcache` accepts paged K/V but not naturally varlen-q across requests. Either we materialize K/V from blocks for varlen calls (negates paging at runtime), or we write a custom Triton varlen-paged kernel. Either way is a separate slice with its own ADR. Shipping the two-forward variant first lets us validate the chunked prefill state machine, the per-request cache transfer, and the get_mask_sizes fix in isolation; the kernel work doesn't risk breaking the structural correctness.
- **Skip chunked prefill, go straight to packed forward**. Possible but riskier — kernel + state machine in one slice with no intermediate validation point. The bug we found (`get_mask_sizes`) was structural and would have been masked by simultaneous changes to the attention path.
- **Keep `RequestState.PREFILLING` only, drop `CHUNKED_PREFILLING`**. Functionally equivalent (the scheduler treats both states the same in `_step`). Kept both for state-machine clarity in the docstring; cost is zero (one extra enum value, no runtime branching).
- **Per-request configurable chunk sizes**. Real engines do this for tuning. Out of scope here; one default suffices for correctness validation, and the benchmark sweep can drive a default-tuning decision later.

## Consequences

- **Positive**:
  - Head-of-line blocking is gone for prompts up to ~chunk_size × max_step_time / decode_time. A 4k-prompt prefill no longer freezes the engine; decoders advance roughly every chunk.
  - The scheduler's structure now cleanly supports the packed-forward upgrade — `_advance_chunked_prefill` produces the per-request K/V needed by the packed path.
  - `append_kv_packed` is the right primitive for varlen K/V appends and is now in place.
  - The HF mask construction bug (`get_mask_sizes` discrepancy) is found and documented; any future cache subclass faces a clear invariant.
- **Negative**:
  - Throughput at mixed-length workloads is bounded by two-forwards-per-step. We pay two model.forward calls per step (prefill chunk + decode batch) when both are present. The full Approach 2 win is deferred.
  - `prefill_cache` is a field on `RunningRequest` that's only meaningful while the request is in `PREFILLING` / `CHUNKED_PREFILLING`. After merge it goes back to None. Slight type-shape impurity; the alternative (a `dict[RequestID, PagedKVCache]` in the scheduler) is more code.
- **Reversibility**:
  - Reverting chunked prefill means restoring the old `_prefill_and_merge` (full prefill in one shot) and dropping `prefill_chunk` / `_advance_chunked_prefill`. About 80 lines.
  - The `append_kv_packed` and `get_mask_sizes` additions are independent improvements and stay regardless.

## Validation

- **Local (M1, CPU PyTorch path)**:
  - `test_chunked_prefill_matches_unchunked`: a long prompt processed with `chunk_size = prompt_len // 3` produces token-for-token identical greedy output to the same prompt processed in a single chunk. **Catches the kind of bug `get_mask_sizes` produced before the fix.**
  - 4 new `test_paged_kv_cache.py` tests for `append_kv_packed`: uniform-length matches `append_kv`, ragged writes go to correct slots, zero-length skips don't disturb other slots, malformed `cu_seqlens` raises.
  - 2 stress tests in `tests/stress/`: 8 concurrent requests of mixed prompt lengths (short / medium / long, 5–144 tokens), chunk_size=32 forcing multi-chunk prefills — outputs match serial reference token-for-token. Memory pressure with a 32-block pool; all 6 submitted requests complete and the pool releases all blocks.
- **CI**: 73 tests pass (including the 4 cache-level + 1 scheduler-level chunked prefill tests). Stress tests are excluded via `@pytest.mark.slow`.
- **Modal**: deferred to the follow-up packed-forward slice — the meaningful CUDA validation is the kernel-path benchmark, and it makes more sense to validate the final design than the intermediate two-forward shape.

## Pointers

- State machine: `src/mini_infer/scheduler/request_state.py` (`RequestState.CHUNKED_PREFILLING`, `RunningRequest.tokens_prefilled`, `RunningRequest.prefill_cache`)
- Cache primitive: `src/mini_infer/cache/paged_kv_cache.py` (`append_kv_packed`, `_write_packed_kv`, `get_mask_sizes`)
- Runner entry point: `src/mini_infer/engine/model_runner.py` (`prefill_chunk`)
- Scheduler step: `src/mini_infer/scheduler/continuous_scheduler.py` (`_advance_chunked_prefill`, `chunk_size` config)
- Unit tests: `tests/unit/test_paged_kv_cache.py`, `tests/unit/test_scheduler.py::test_chunked_prefill_matches_unchunked`
- Stress tests: `tests/stress/test_chunked_prefill_load.py`

## Follow-up

Packed-forward implementation: a separate slice will replace the two-forwards-per-step pattern with a single `model.forward` call per step over a packed sequence of all in-flight q-tokens (prefill chunks + decode tokens), using `flash_attn_varlen_func` for varlen attention with materialization from the paged cache (and an optimization path to paged-aware varlen if benchmarks demand). New ADR (ADR-007) will document that work.
