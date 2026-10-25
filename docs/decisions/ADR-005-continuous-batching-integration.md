# ADR-005: Continuous batching integration — batch-aware PagedKVCache

Date: 2026-04-27
Status: Accepted

## Context

Slice 2.3a (commit `650dee4`) added `ContinuousScheduler`, but it still ran
one forward per request per step. Slice 2.3b (commit `06d98be`) added the
batched Triton kernel (`paged_attention_decode_batched`), but the scheduler
did not call it yet. This slice (2.3c) runs one forward over all decoding
requests per step. The blocker was the cache. The Qwen2 forward of HF
expects one `Cache` object for the whole batch, but `PagedKVCache`
supported only batch 1.

## Decision

1. Make `PagedKVCache` batch-aware. `_block_ids: list[list[int]]` and
   `_num_tokens: list[int]` are indexed by `batch_idx`. The new methods
   include `add_request_slot()`, `remove_request(batch_idx)`,
   `merge_request(other)` and a batched `append_kv(K, V, layer_idx)`.
2. Keep one long-lived shared cache per scheduler,
   `ContinuousScheduler._batched_cache`. It grows when a prefill completes
   (`merge_request`) and shrinks when a request finishes
   (`remove_request`).
3. Keep prefill single-request. Each new request prefills into a temporary
   `PagedKVCache(batch_size=1)`. `merge_request` then moves its blocks into
   the shared cache, so a later `free()` of the temporary cache does
   nothing.
4. Batch all decode. `runner.decode_batch(cache, last_tokens)` runs one
   forward with a ragged `attention_mask`. The patched Qwen2 attention calls
   `paged_attention_decode_batched(...)` at every layer, for every batch
   size.
5. Track `batch_idx` on `RunningRequest`. `_reap_done` removes finished
   requests in descending `batch_idx` order. It decrements the `batch_idx`
   of each survivor at a higher index. Only the engine thread changes the
   cache.
6. Keep `runner.decode()` as a three-line single-request wrapper around
   `decode_batch`, for the golden tests and the latency benchmark.
7. `update()` still satisfies the HF `Cache` contract for prefill (q_len > 1,
   B = 1). `_materialize` zero-pads to `max_seq_len` in case a later code
   path calls it with B > 1.

## Alternatives considered

- A `BatchedPagedKVCache` wrapper around single-request caches (Option A of
  the plan). It leaves `PagedKVCache` unchanged, but gives two cache types
  where one is enough. The owner chose Option B, the cleaner long-term shape
  that vLLM also uses.
- Batched prefill. It needs ragged batching by prompt length and chunked
  prefill. Each is its own slice.
- Stable slot IDs with no `batch_idx` drift. Finished requests leave holes,
  so the kernel gets a sparse batch dimension. Drift is simpler.
- A tensor for `_num_tokens`. The kernel takes `seq_lens: list[int]`
  anyway.

## Consequences

- On an A10, throughput at C=8 is 4.1× the throughput at C=1
  (`docs/benchmarks/2026-04-27-continuous-batching.md`).
- `merge_request` moves blocks from prefill to decode with no free and no
  new allocation. `RunningRequest` no longer holds its own cache, so two
  views of the cache cannot diverge.
- `PagedKVCache` grew by about 80 lines.
- `batch_idx` drift is a subtle invariant. A change to the order of
  admission and finish can break it. So all `_batched_cache` changes stay
  inside `_reap_done`, `_admit_waiting` and `_prefill_and_merge`.
- `decode_batch` builds `attention_mask` in a Python loop each step. The
  cost is negligible at our batch sizes, but a tensor builder is faster at
  C=128+.
- Speculative decoding and P/D disaggregation assume this multi-request
  cache. A revert to per-request decode is about 200 lines.
- 61 unit tests and 3 golden tests pass on M1 (CPU). 14 tests cover the
  batch-aware cache. `test_batched_decode_matches_serial` runs 3 concurrent
  requests and matches a serial reference token-for-token.
  `test_short_request_finishing_first_does_not_corrupt_others` covers the
  drift. On a Modal A10, `scripts/modal_concurrent_smoke.py` runs 4
  concurrent requests that match a serial run.
