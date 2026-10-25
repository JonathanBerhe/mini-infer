# ADR-009: Prefix caching via chained-hash, block-granular, refcounted LRU

Date: 2026-04-28
Status: Accepted

## Context

After ADR-005 through ADR-008, the engine runs one packed-varlen forward per
scheduler step over a shared `PagedKVCache`. Each prefill computes K/V again,
even when the pool still holds the blocks of the same system prompt.

Chat workloads repeat a small set of prefixes: system prompts, few-shot
examples and tool descriptions. vLLM shares blocks by hash, and SGLang uses
RadixAttention. Both report a TTFT that is several times lower on
chat-template workloads.

## Decision

1. Share full blocks by a chained hash. Each full block of `block_size`
   tokens gets `h_i = blake2b-128(h_{i-1} || token_ids[i*B : (i+1)*B])`. Two
   prompts share an entry only if that block and all earlier blocks match.
2. A standalone `PrefixCache` (`cache/prefix_cache.py`) holds the hash to
   block ID table, the refcounts and an `OrderedDict` for LRU.
3. `BlockPool` knows the cache. When the free list is empty, `allocate()`
   asks `PrefixCache.evict_lru()` for an unreferenced cached block. `free()`
   of a cached block calls `PrefixCache.decref`, so the block stays cached
   and becomes evictable at refcount 0.
4. `PagedKVCache.add_request_slot(prompt_token_ids=...)` looks up the hashes
   of the prompt and puts the matched blocks into `_block_ids`, with an
   incref to pin them. `_num_tokens` counts the cached tokens, and chunked
   prefill starts after them.
5. Last-token rule. If the full prompt is cached, the slot drops the last
   cached block. The forward must run on at least one prompt token to get
   the logits for the next sample.
6. Publish on the last layer. A block goes into the cache only after the
   last layer writes its K/V (`layer_idx == num_layers - 1`). An earlier
   publish lets a hit read stale K/V in the upper layers.
7. Coalesce a duplicate publish. Two slots with the same prompt can fill the
   same block in one step. Then the second slot frees its own block and
   points `_block_ids` at the canonical block, which holds bit-equal K/V.
8. Opt in with `ModelRunner.from_pretrained(prefix_cache=True)`. The default
   is off, so the golden suite does not change. At admission, the continuous
   scheduler counts `PrefixCache.num_evictable` as free capacity.

Prefill is deterministic on a fixed device and dtype. So a hit gives the
same K/V bits as a new prefill of the same prompt.

## Alternatives considered

- A radix tree (SGLang's RadixAttention), which shares at any token. At
  `block_size=16` it adds little: long system prompts make the match end at a
  block edge almost every time. Hashing gets >80% of the gain with much less
  code. It stays a follow-up if mid-block sharing matters for a workload.
- Keep the cache across server restarts. Rejected, because it is not on the
  hot path.
- Cache-aware admission (largest expected hit first). Admission stays FIFO.
  This follow-up can lower TTFT under contention.
- Cache the decoded tokens after the prompt. Hit rates are low under
  sampling, and the bookkeeping is large.

## Consequences

- A repeated prefix skips prefill. On an A10 (Qwen2.5-0.5B, bf16), the test
  used a 15.9k-token shared system prompt and 8 short user questions. Warm
  TTFT is 74ms instead of 11.7s (158x), and concurrent throughput is 13–32x
  higher for C ∈ {1, 4, 8}. The numbers are in
  `docs/benchmarks/2026-04-28-prefix-caching.md`.
- Each slot also tracks `_slot_prompt_tokens`, `_slot_num_published` and
  `_slot_parent_hash`. This code stays in `add_request_slot` and
  `_publish_filled_blocks`.
- A fully cached prompt still runs one forward for the next-token logits.
- On M1 (CPU, fp32), 21 + 12 = 33 prefix-cache tests pass, and the 75
  existing tests stay green. On Qwen2.5-0.5B, the cached output equals the
  uncached output token for token
  (`tests/unit/test_scheduler.py::test_prefix_cache_matches_no_cache`,
  `tests/stress/test_prefix_cache_load.py::test_shared_system_prompt_parity`).
  20 unique long prompts on a 64-block pool complete without OOM
  (`tests/stress/test_prefix_cache_load.py::test_eviction_under_unique_prompts`).
