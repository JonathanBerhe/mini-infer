# ADR-003: Block-based paged KV cache

Date: 2026-04-25
Status: Accepted

## Context

Phase 1 kept one contiguous KV tensor per request (`KVCache(DynamicCache)`).
Each request reserved memory for the maximum sequence length, and a finished
request left a hole that stayed unused. Continuous batching,
prefix caching and copy-on-write all need a block-based memory manager first.

## Decision

Use the PagedAttention pattern of vLLM:

- At startup the engine allocates one `BlockPool` tensor of shape
  `(num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)`. A block
  holds 16 tokens by default.
- Each request has a `PagedKVCache`, a `DynamicCache` subclass whose
  `block_table: list[int]` maps positions to block IDs. It takes a new block
  only when the current block is full.
- `cache.free()` returns the blocks to the free list. The scheduler calls it
  in `try/finally`, so a crash also releases them.

In Slice 2.1, `update()` copies the blocks into one contiguous K/V tensor for
the HF attention layer, so the model code does not change. Slice 2.2 replaces
this copy with a paged attention kernel, which gives the speedup.

## Alternatives Considered

- Keep `DynamicCache`. Rejected, because continuous batching and prefix
  sharing need block addresses.
- Write our own attention layers now. This is too invasive for this slice.
- Store the pool in numpy. It is CPU-only, has no `bfloat16`, and forces a
  CPU-GPU copy on each layer.
- Use blocks of variable size. The allocator gets complex and the kernel
  loses its vectorization assumptions.

## Consequences

- Slices 2.2 to 2.4 (kernel, continuous batching, prefix caching) build on
  the blocks. `pool.num_free_blocks` makes memory use testable. Fixed-size
  blocks remove fragmentation, and allocate and free are O(1).
- Until Slice 2.2, the copy in `update()` costs O(seq_len) per layer per step.
  Speed is the same as `DynamicCache` or a little worse.
- The pool size is fixed, about 190 MB for the 0.5B model. A request that
  needs more raises `OutOfMemoryError`. Admission control (Slice 2.3) handles
  this, not a pool that grows.
- The golden tests pass token-for-token. `tests/unit/cache/test_block_pool.py` and
  `tests/unit/cache/test_paged_kv_cache.py` test the allocator and the cache in CI.
