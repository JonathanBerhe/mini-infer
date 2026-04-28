# ADR-009: Prefix caching via chained-hash, block-granular, refcounted LRU

Date: 2026-04-28
Status: Accepted

## Context

After ADR-005 through ADR-008 the engine is structured around a shared
`PagedKVCache` with one packed-varlen forward per scheduler step. Every
prefill rebuilds K/V from scratch, even when the same system prompt has been
served minutes earlier and its blocks are still sitting in the pool's
free-list waiting to be reused.

Real chat workloads are dominated by a small set of repeated prefixes (system
prompts, few-shot exemplars, tool descriptions). vLLM and SGLang treat this as
the highest-leverage optimization in the prefill path: vLLM ships hash-based
block sharing; SGLang ships RadixAttention with sub-block matching. Both report
multi-x TTFT improvements on chat-template workloads.

This slice adds prefix caching to mini-infer.

## Decision

1. **Hash-based, block-granular sharing.** Each fully-filled `block_size`-token
   block gets a chained hash
   `h_i = blake2b-128(h_{i-1} || token_ids[i*B : (i+1)*B])`. Two prompts share
   a cache entry iff they share both the local block tokens AND every prior
   block in the chain.
2. **Standalone `PrefixCache` data structure** (`cache/prefix_cache.py`). Owns
   the hash → block_id table, refcounts, and an `OrderedDict` for LRU. No
   knowledge of `BlockPool` storage; integration is at the `BlockPool` /
   `PagedKVCache` boundary.
3. **`BlockPool` is cache-aware.** `allocate()` pops from the free list; on
   exhaustion, asks `PrefixCache.evict_lru()` for an unreferenced cached block.
   `free()` of a cached block calls `PrefixCache.decref` (block stays in cache,
   becomes evictable at refcount=0); free of an uncached block returns it to
   the free list as before.
4. **`PagedKVCache.add_request_slot(prompt_token_ids=...)`** looks up the
   prompt's chained hashes and pre-populates the slot's `_block_ids` with
   matched blocks (incref'd to pin them). The slot's `_num_tokens` reflects
   how many prompt tokens are already cached; the chunked-prefill loop in the
   scheduler picks up from there.
5. **Last-token rule.** If the entire prompt would be cache-resident, the
   slot drops the last cached block. The scheduler must run forward over at
   least one prompt token to obtain logits for the next sample; without this,
   a fully-cached prompt has no logits to sample from.
6. **Publish-on-last-layer.** During chunked prefill, blocks fill across many
   step / many chunks. They are only published to the cache after the last
   layer of a step writes the K/V (`layer_idx == num_layers - 1`). Publishing
   earlier would cache K/V for layers 0..N-1 that haven't been written yet,
   so a future hit would read stale K/V at the upper layers.
7. **Coalesce on duplicate publish.** Two slots with identical prompts that
   fill the same block in the same step both call `publish`; the second one
   sees the canonical entry, returns its just-allocated block to the free
   list, and rewrites its `_block_ids` to point at the canonical block. The
   K/V values are bit-equal in either block (same tokens + same model =
   same K/V), so attention reads after the swap are correct.
8. **Opt-in via `ModelRunner.from_pretrained(prefix_cache=True)`**. Default
   stays off so existing tests and the golden suite are unaffected. The
   continuous scheduler counts `PrefixCache.num_evictable` toward admission
   capacity (cached blocks are reclaimable on demand), so a "full" pool
   doesn't artificially block admission.

## Why hash-based and not a radix tree

Two paths considered:

- **Hash-based, block-granular** (vLLM-style). Simple to implement and test.
  Sharing only at `block_size` granularity; sub-block prefix overlap is not
  exploited.
- **Radix tree, sub-block-granular** (SGLang's RadixAttention). Sharing at
  any token boundary, including mid-block. More implementation surface (node
  splitting on partial match, suffix sharing).

At our defaults (`block_size=16`), the additional sharing a radix tree buys is
small: chat workloads are dominated by long shared system prompts that happen
to be much longer than `block_size`, so the prefix match boundary almost
always falls at a block edge anyway. Hash-based gets >80% of the wins with a
fraction of the implementation surface and is easier to reason about for
correctness.

The radix-tree path remains a defensible follow-up if benchmarks ever show
that mid-block sharing matters at scale.

## Numerical correctness

The K/V values returned by a cache hit are bit-equal to the values that would
be produced by re-prefilling the same prompt. The cached block's K/V was
written by an earlier prefill of the same prompt against the same model
weights; the model's prefill is deterministic on a fixed device + dtype, so
re-prefilling produces the same K/V. After a hit, attention reads the cached
K/V at positions covered by the hit and reads freshly-computed K/V at later
positions, exactly as it would in a cold-cache run.

This is verified end-to-end by `test_prefix_cache_matches_no_cache` in the
scheduler test file: a long shared-prefix prompt run with prefix caching
enabled produces token-for-token identical output to the same prompt run with
caching disabled.

## Alternatives considered

- **Radix tree** (SGLang's RadixAttention). Rejected for V1; documented as a
  follow-up (above).
- **Persisting the cache across server restarts.** Rejected; orthogonal and
  not the hot path.
- **Cache-aware scheduling** (admit the request with the largest expected
  cache hit first). Rejected; admission is FIFO today and prefix-cache hits
  reduce the per-request prefill work either way. A reasonable follow-up
  once benchmarks are in.
- **Caching decoded continuations** (cache K/V past the prompt boundary).
  Rejected for V1 because hit rates are low under sampling and the bookkeeping
  cost is non-trivial. Greedy-deterministic workloads might benefit; revisit
  when a real workload demands it.

## Consequences

- **Positive**:
  - Repeat prompts and shared system-prompt workloads skip prefill on the
    cached prefix entirely. Modal benchmark TBD; expected TTFT reduction is
    proportional to `prompt_length / new_unique_tokens`.
  - The `PrefixCache` is a clean, single-purpose data structure (~150 LOC)
    that's easy to test in isolation and would serve as the foundation for a
    radix-tree variant if we go that route later.
  - `BlockPool` is now refcount-aware end-to-end; useful for future features
    that share block storage (e.g. CoW for speculative decoding state).
- **Negative**:
  - More moving parts: per-slot prompt token tracking (`_slot_prompt_tokens`,
    `_slot_num_published`, `_slot_parent_hash`) lives alongside `_block_ids`
    and `_num_tokens`. Concentrated in `PagedKVCache.add_request_slot` and
    `_publish_filled_blocks`; the rest of the cache API is unchanged.
  - Last-token rule means a fully-cached prompt still pays for one prefill
    forward to get the next-token logits. Negligible.
- **Reversibility**: removing the prefix cache is a clean revert; the
  `PrefixCache` parameter on `BlockPool` and `PagedKVCache` is optional,
  defaults off, and is only constructed when explicitly requested.

## Validation

- **M1 (CPU, fp32)**: 21 + 12 = 33 prefix-cache unit/integration tests pass.
  Existing 75-test suite stays green; the cache-disabled path is byte-for-byte
  unchanged.
- **Real-model parity (Qwen2.5-0.5B, M1)**: prefix-cached output equals the
  no-cache reference token-for-token for a multi-prompt shared-system-prompt
  workload (`tests/unit/test_scheduler.py::test_prefix_cache_matches_no_cache`,
  `tests/stress/test_prefix_cache_load.py::test_shared_system_prompt_parity`).
- **Eviction under pressure**: 20 unique long prompts on a 64-block pool
  complete without OOM, exercising the BlockPool fallback path
  (`tests/stress/test_prefix_cache_load.py::test_eviction_under_unique_prompts`).
- **CUDA (A10, Qwen2.5-0.5B, bf16)**: 15.9k-token shared system prompt + 8
  unique short user questions. **Warm-TTFT 158x speedup** (74ms vs 11.7s);
  **concurrent throughput 13–32x** across C ∈ {1, 4, 8}. Numbers in
  `docs/benchmarks/2026-04-28-prefix-caching.md`.

## Pointers

- Data structure: `src/mini_infer/cache/prefix_cache.py`.
- Pool integration: `src/mini_infer/cache/block_pool.py` (`allocate`/`free`
  fallbacks).
- Slot integration: `src/mini_infer/cache/paged_kv_cache.py`
  (`add_request_slot`, `_publish_filled_blocks`).
- Scheduler integration:
  `src/mini_infer/scheduler/continuous_scheduler.py::_admit_waiting`.
- Engine flag: `ModelRunner.from_pretrained(..., prefix_cache=True)`.
- Earlier ADRs: ADR-005 (continuous batching), ADR-006 (chunked prefill),
  ADR-007 (packed varlen forward), ADR-008 (paged FA varlen).

## Follow-ups

- **Radix tree** (sub-block-granular sharing). Worth doing only if a real
  workload shows the cost of block-edge alignment matters.
- **Cache-aware admission**: prioritize the request with the largest expected
  cache hit. Improves TTFT under contention.
- **H100 sweep** for the prefix workload, to characterise how the speedup
  scales with hardware. The bottleneck on cache-OFF is prefill compute, so
  faster GPUs should narrow the speedup proportionally.
