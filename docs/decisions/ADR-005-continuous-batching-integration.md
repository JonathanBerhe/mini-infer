# ADR-005: Continuous batching integration — batch-aware PagedKVCache

Date: 2026-04-27
Status: Accepted

## Context

Slice 2.3a (commit `650dee4`) introduced `ContinuousScheduler` with a dedicated engine thread, FIFO admission, and per-request handles, but each step still issued ONE forward per request. Slice 2.3b (commit `06d98be`) added the batched Triton kernel and PyTorch reference (`paged_attention_decode_batched`), validated by 5 unit tests. The plumbing was in place but inert — the scheduler wasn't dispatching batched.

This slice (2.3c) wires it together: the scheduler now runs ONE forward over all currently-decoding requests per step. The blocking constraint surfaced during exploration was that HF's Qwen2 forward expects ONE `Cache` object that handles batch dim B internally, while `PagedKVCache` was hardcoded for batch=1 (`_num_tokens: int`, `_block_ids: list[int]`).

## Decision

1. **Refactor `PagedKVCache` to be natively batch-aware.** `_block_ids: list[list[int]]` and `_num_tokens: list[int]`, indexed by `batch_idx`. New methods `add_request_slot()`, `remove_request(batch_idx)`, `merge_request(other)`, batched `append_kv(K, V, layer_idx)` for `(B, num_kv_heads, new_seq_len, head_dim)` input, `block_tables_per_request_tensor(device)` and `seq_lens_list()` for the kernel.
2. **One long-lived shared cache per scheduler.** `ContinuousScheduler._batched_cache` is created lazily on first admission. It grows when a prefill completes (`merge_request`) and shrinks when a request finishes (`remove_request`). Block ownership transfers from temp prefill cache to the shared cache via `merge_request`; the temp's `free()` becomes a no-op after merge.
3. **Prefill stays single-request.** Variable-length prompts make batched prefill its own design problem (chunked prefill, separately queued on the ROADMAP). Each new admit gets a temp `PagedKVCache(batch_size=1)`, runs single-request prefill, then merges into the shared cache.
4. **Decode is fully batched.** `runner.decode_batch(cache, last_tokens)` builds `(B, 1)` `input_ids`, per-request `position_ids`, and a ragged `attention_mask` shape `(B, max_seq+1)` and runs ONE forward. The patched Qwen2 attention dispatches `paged_attention_decode_batched(...)` at every layer.
5. **Per-request `batch_idx` tracking on `RunningRequest`.** When a request finishes, `_reap_done` walks finished requests in descending `batch_idx` order, removes each from the shared cache, and decrements `batch_idx` for every survivor at a higher index in the same critical section. The engine thread is the sole mutator, so this is single-threaded by construction.
6. **`runner.decode()` kept as a single-request convenience wrapper** around `decode_batch`. Avoids churn in golden tests and the decode-step latency benchmark, both of which iterate prefill → decode in a loop with `cache.batch_size == 1`. Three lines.
7. **`update()` still satisfies the HF `Cache` contract** for the prefill path (called by HF's stock attention when q_len > 1, B always 1 in our flow). The batch-aware `_materialize` zero-pads to `max_seq_len` for safety against future code paths that hit it with B > 1; today only the prefill path does, with B=1.

## Alternatives considered

- **Fused wrapper (Option A from the plan).** Keep `PagedKVCache` per-request; add a `BatchedPagedKVCache` wrapper that holds a list of single-request caches and presents the HF `Cache` interface for the model's forward. Smaller blast radius (PagedKVCache untouched), but creates two cache types where one will do, and the wrapper still has to route K/V appends per-request internally. The user explicitly chose Option B for the cleaner long-term shape, matching vLLM's design.
- **Refactor PagedKVCache and run prefills batched too.** Would require ragged batching by prompt length and chunked prefill to interleave long prompts with running decodes. Both are real performance features but each their own slice. Deferred; chunked prefill is a Phase 2 deliverable on the ROADMAP.
- **Stable per-request slot IDs (no `batch_idx` drift).** A request's slot stays the same throughout its lifetime; removing a finished request leaves a "hole" the scheduler tracks. Avoids the shift-on-remove logic but complicates the kernel (sparse batch dim) and the cache (memory not contiguous along the batch axis). Drift is simple and the engine thread does it serially; the test `test_short_request_finishing_first_does_not_corrupt_others` proves it.
- **Use one tensor for `_num_tokens` instead of a list.** A `torch.Tensor` of shape `(batch_size,)` would be slightly faster to index from the kernel-side path, but the kernel takes `seq_lens: list[int]` as Python ints anyway (the Triton launcher converts to a tensor). A Python list keeps the cache code free of device handling for a dimension that's tiny.

## Consequences

- **Positive**:
  - The scheduler's hot path is now a single batched forward per step. End-to-end throughput at C=8 hits 4.1× C=1 on A10 (`docs/benchmarks/2026-04-27-continuous-batching.md`).
  - The kernel call site is identical for B=1 and B>=2: the patched Qwen2 forward calls `paged_attention_decode_batched(...)` unconditionally, with batch size determined by the cache. No special-case path for single-request.
  - `merge_request` makes the prefill-then-decode handoff explicit. The temp cache's blocks transfer to the shared cache without ever being freed and re-allocated.
  - Removing the per-request cache field from `RunningRequest` eliminates a class of bugs where `RunningRequest.cache` and the scheduler's view of the world could diverge.
- **Negative**:
  - More code in `PagedKVCache` (~80 lines added, mostly the batch-iteration in `append_kv` / `_materialize`). The prior single-request implementation was simpler.
  - `batch_idx` drift on remove is a subtle invariant. The drift test catches the obvious failure mode but a future change to admission/finish ordering could violate it; mitigation is to keep all `_batched_cache` mutations inside `_reap_done` / `_admit_waiting` / `_prefill_and_merge`.
  - `attention_mask` is now constructed per-step in `decode_batch` (small Python loop sized by batch). Negligible at our batch sizes; a tensor-builder would be faster at C=128+.
- **Reversibility**:
  - Reverting to per-request decode means: drop `decode_batch` and the multi-batch code paths in PagedKVCache, restore `runner.decode` as the primary API, and revert the scheduler's `_batched_decode_step` to the per-request loop. The kernel and tests stay; they're useful regardless. About 200 lines.
  - The batch-aware cache shape is the foundational change: every Phase 3 feature (speculative decoding, P/D disaggregation) assumes a multi-request KV cache, so reverting this would undo the foundation, not just a tactical choice.

## Validation summary

- **Local (61 unit tests + 3 golden, all green on M1 / CPU PyTorch path):**
  - 14 tests for the batch-aware `PagedKVCache` (add/remove/merge/append/materialize, including the "remove shifts indices" invariant).
  - `test_batched_decode_matches_serial`: 3 concurrent requests through the batched scheduler match a serial reference token-for-token.
  - `test_short_request_finishing_first_does_not_corrupt_others`: a request finishing mid-batch doesn't corrupt the survivors' outputs (the `batch_idx` drift test).
  - All 3 golden tests still token-for-token vs HF reference.
- **Modal smoke (A10):** 4 concurrent requests via `scripts/modal_concurrent_smoke.py`; all 4 outputs match a serial reference run on the same hardware.
- **Modal benchmark (A10):** throughput at C ∈ {1, 2, 4, 8} in `docs/benchmarks/2026-04-27-continuous-batching.md`. C=8 → 4.1× C=1.

## Pointers

- Cache: `src/mini_infer/cache/paged_kv_cache.py`
- Kernel call site (unchanged contract): `src/mini_infer/engine/attention_patches/qwen2.py`
- Runner batched entry point: `src/mini_infer/engine/model_runner.py` (`decode_batch` + back-compat `decode`)
- Scheduler: `src/mini_infer/scheduler/continuous_scheduler.py` (`_step`, `_admit_waiting`, `_prefill_and_merge`, `_batched_decode_step`, `_reap_done`)
- Per-request state: `src/mini_infer/scheduler/request_state.py` (`RunningRequest.batch_idx`)
- Tests: `tests/unit/test_paged_kv_cache.py`, `tests/unit/test_scheduler.py`
- Modal entrypoints: `scripts/modal_concurrent_smoke.py`, `scripts/modal_concurrent_bench.py`
- Benchmark report: `docs/benchmarks/2026-04-27-continuous-batching.md`
