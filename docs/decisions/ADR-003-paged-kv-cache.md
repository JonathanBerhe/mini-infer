# ADR-003: Block-based paged KV cache

Date: 2026-04-25
Status: Accepted

## Context

Phase 1 used a single contiguous KV tensor per request (`KVCache(DynamicCache)`). That works but:

- Wastes memory: every request reserves the full max-sequence-length tensor up front, or grows reallocating; both are unfriendly to multi-request serving.
- Fragments memory across requests: when one finishes mid-pool, the remaining holes can't be reused.
- Doesn't reflect how production engines (vLLM, TensorRT-LLM, SGLang) actually structure KV memory.

Phase 2 is the right place to fix this, because everything else Phase 2 wants (continuous batching, prefix caching, copy-on-write) presupposes a block-based memory manager.

## Decision

Adopt vLLM's PagedAttention pattern for KV cache management:

- A single `BlockPool` is pre-allocated at engine startup, sized as one contiguous `torch.Tensor` of shape `(num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)`. The "2" is (key, value).
- Per-request `PagedKVCache` instances hold a small `block_table: list[int]` mapping logical positions to physical block IDs in the pool.
- A new block is allocated only when the current one fills up (`block_size` defaults to 16 tokens).
- When the request completes, `cache.free()` returns its blocks to the pool's free list. Wrapped in `try/finally` in the scheduler so a crash mid-stream still releases.
- `PagedKVCache` subclasses `transformers.DynamicCache` so HF's per-layer plumbing (`get_seq_length`, layer indexing) is inherited; `update()` is overridden to read/write through blocks.

In Slice 2.1 (this ADR), `update()` materializes the full block list back into a contiguous K/V tensor before returning it to HF's attention layer. This keeps the existing model.forward() path unchanged. The materialization is the expected stepping-stone; **Slice 2.2 replaces it with a paged attention kernel** that reads non-contiguous blocks directly. That's where the performance benefit lands.

## Alternatives Considered

- **Stay on `DynamicCache`**: simplest, but blocks the rest of Phase 2 (continuous batching, prefix sharing) which fundamentally need block-level addressing. We'd have to rewrite the cache later anyway.
- **Reimplement attention layers ourselves to read paged K/V directly**: more invasive than needed for Slice 2.1; reserves architectural choices for the kernel slice.
- **Use a numpy array for the pool storage**: rejected. K/V passes through the pool as `torch.Tensor` on the model's device (MPS / CUDA). Numpy is CPU-only, doesn't support `bfloat16` / `float16`, and would force a torch↔numpy + CPU↔GPU round-trip on every layer of every step. Plain Python `list[int]` is fine for the bookkeeping (block table, free list) where the data is small integers; torch tensors for the actual K/V storage.
- **Variable-size blocks**: matches request length better but kills the simple allocator and the future kernel's vectorization assumptions. Fixed-size is the standard PagedAttention design.

## Consequences

- **Positive**:
  - Block bookkeeping is in place; Slices 2.2 (kernel), 2.3 (continuous batching), 2.4 (prefix caching) build on it without architectural change.
  - Memory accounting is now first-class: `pool.num_free_blocks` is observable and testable.
  - Pre-allocated pool eliminates fragmentation entirely (fixed-size blocks; allocate/free is O(1)).
  - Hardware-agnostic: the block manager runs on MPS / CUDA / CPU. No Modal cost incurred for this slice.
- **Negative**:
  - Materialization in `update()` is wasteful: a fresh contiguous tensor is built every layer per step, gathering from blocks. O(seq_len) per call. Slice 2.2 fixes this; until then performance is similar to or marginally worse than the old `DynamicCache` path. Acceptable for correctness-first Phase 2.1.
  - Pool size is fixed at startup (~190 MB for the 0.5B model defaults: 1024 blocks × 24 layers × 16 slots × 2 KV heads × 64 head_dim × fp16). If a request exceeds capacity, we raise `OutOfMemoryError` rather than growing the pool. This matches production-engine behavior; the right answer is admission control (Slice 2.3), not unbounded growth.
- **Reversibility**:
  - The cache type is internal. Swapping back to `DynamicCache` would mean undoing the `model_runner` and `scheduler` edits; the public API of `Scheduler.run / stream / GenerationResult / GenerationStep` is unchanged.
  - The next slice (kernel) will replace `_materialize()` and possibly `_write_new_kv()` with kernel-driven equivalents. The block pool layout was chosen with that future kernel in mind (contiguous-by-layer storage, `(num_blocks, block_size, ...)` axes).

## Verification

- Golden tests still pass token-for-token at the full 16-token reference (`tests/golden/`). This is the critical correctness gate; any regression here would mean materialization is broken.
- Unit tests (`tests/unit/test_block_pool.py`, `tests/unit/test_paged_kv_cache.py`) cover the allocator and the per-request cache without needing a model. **Run in CI.**
- Manual smoke confirmed: pool starts at `num_blocks` free, decreases during a request, returns to `num_blocks` free after. `cache.free()` invoked via `try/finally`.
