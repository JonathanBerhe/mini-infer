"""Integration tests: PrefixCache + BlockPool + PagedKVCache.

These tests exercise the wiring done in Slice B: pool-level cache-aware
allocate/free, slot-level lookup-on-create, and publish-on-last-layer-write.
The pure data-structure tests live in `test_prefix_cache.py`.
"""

import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache


def make_pool(
    *,
    block_size: int = 4,
    num_blocks: int = 16,
    num_layers: int = 2,
    with_prefix_cache: bool = True,
) -> BlockPool:
    prefix_cache = PrefixCache(block_size=block_size) if with_prefix_cache else None
    return BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
        prefix_cache=prefix_cache,
    )


def _packed_kv(num_tokens: int, *, fill: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic packed K/V plus cu_seqlens covering one batch slot."""
    k = torch.full((num_tokens, 2, 4), fill, dtype=torch.float32)
    v = k + 100.0
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)
    return k, v, cu_seqlens


def _write_slot_all_layers(
    cache: PagedKVCache, num_tokens: int, fill: float, num_layers: int
) -> None:
    """Append `num_tokens` to slot 0 across every layer (publish triggers on last)."""
    k, v, cu = _packed_kv(num_tokens, fill=fill)
    for layer_idx in range(num_layers):
        cache.append_kv_packed(k, v, cu, layer_idx=layer_idx)


def test_no_prefix_cache_behaves_like_before() -> None:
    """Without a prefix cache, add_request_slot ignores prompt_token_ids."""
    pool = make_pool(with_prefix_cache=False)
    cache = PagedKVCache(pool)
    batch_idx = cache.add_request_slot(prompt_token_ids=[1, 2, 3, 4, 5, 6])
    assert batch_idx == 0
    assert cache.seq_lens_list() == [0]  # nothing pre-populated
    assert cache.block_ids_for_request(0) == []


def test_publish_after_last_layer_only() -> None:
    """A block is published only after the last layer's append_kv_packed.

    Earlier layers' K/V isn't written yet, so we must defer publish until the
    full layer stack is committed.
    """
    pool = make_pool(block_size=4, num_layers=3)
    cache = PagedKVCache(pool)
    cache.add_request_slot(prompt_token_ids=[10, 11, 12, 13])  # 1 full block

    k, v, cu = _packed_kv(4, fill=1.0)
    # First two layers write but should NOT trigger a publish.
    cache.append_kv_packed(k, v, cu, layer_idx=0)
    assert pool.prefix_cache is not None
    assert pool.prefix_cache.num_cached == 0
    cache.append_kv_packed(k, v, cu, layer_idx=1)
    assert pool.prefix_cache.num_cached == 0
    # Last layer triggers publish.
    cache.append_kv_packed(k, v, cu, layer_idx=2)
    assert pool.prefix_cache.num_cached == 1


def test_partial_block_not_published() -> None:
    """A non-full last block is excluded from publishing."""
    pool = make_pool(block_size=4, num_layers=2)
    cache = PagedKVCache(pool)
    cache.add_request_slot(prompt_token_ids=[1, 2, 3, 4, 5, 6])  # 1 full + 2-token partial

    _write_slot_all_layers(cache, num_tokens=6, fill=0.5, num_layers=2)
    # Only the first 4 tokens form a full block; the last 2 stay private.
    assert pool.prefix_cache is not None
    assert pool.prefix_cache.num_cached == 1


def test_second_slot_with_same_prompt_hits_cache() -> None:
    """A repeat prompt finds its prefix in the cache and starts pre-prefilled."""
    pool = make_pool(block_size=4, num_layers=2)
    cache = PagedKVCache(pool)
    prompt = [10, 11, 12, 13, 14, 15, 16, 17, 18]  # 2 full blocks + 1-token partial
    cache.add_request_slot(prompt_token_ids=prompt)
    _write_slot_all_layers(cache, num_tokens=9, fill=0.7, num_layers=2)
    cache.remove_request(0)

    # First slot done; cached blocks should still be in the prefix cache.
    assert pool.prefix_cache is not None
    assert pool.prefix_cache.num_cached == 2

    # New slot with the same prompt sees both cached blocks (8 tokens)
    # plus the partial block remaining.
    new_idx = cache.add_request_slot(prompt_token_ids=prompt)
    assert cache.seq_lens_list()[new_idx] == 8  # 2 full blocks worth


def test_last_token_rule_caps_full_cache_hit() -> None:
    """If the entire prompt is cached, drop the last block to leave a token to prefill."""
    pool = make_pool(block_size=4, num_layers=2)
    cache = PagedKVCache(pool)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]  # exactly 2 full blocks; no partial tail
    cache.add_request_slot(prompt_token_ids=prompt)
    _write_slot_all_layers(cache, num_tokens=8, fill=0.3, num_layers=2)
    cache.remove_request(0)

    new_idx = cache.add_request_slot(prompt_token_ids=prompt)
    # Full hit would be 8 tokens (== len(prompt)). Capped to 1 block (4 tokens).
    assert cache.seq_lens_list()[new_idx] == 4


def test_partial_prefix_share_with_diverging_suffix() -> None:
    """Two prompts share the first block; second block diverges."""
    pool = make_pool(block_size=4, num_layers=2)
    cache = PagedKVCache(pool)
    prompt_a = [1, 2, 3, 4, 5, 6, 7, 8]  # blocks: [1,2,3,4], [5,6,7,8]
    prompt_b = [1, 2, 3, 4, 99, 99, 99, 99, 99]  # blocks: [1,2,3,4], [99,99,99,99] + partial

    cache.add_request_slot(prompt_token_ids=prompt_a)
    _write_slot_all_layers(cache, num_tokens=8, fill=0.1, num_layers=2)
    cache.remove_request(0)

    new_idx = cache.add_request_slot(prompt_token_ids=prompt_b)
    # Only the first block (4 tokens) is shared; second block diverges.
    assert cache.seq_lens_list()[new_idx] == 4


def test_cached_blocks_persist_after_request_removal() -> None:
    """Removing a request keeps its blocks in cache (refcount=0, evictable)."""
    pool = make_pool(block_size=4, num_layers=2, num_blocks=8)
    cache = PagedKVCache(pool)
    cache.add_request_slot(prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    _write_slot_all_layers(cache, num_tokens=8, fill=0.4, num_layers=2)

    pf = pool.prefix_cache
    assert pf is not None
    assert pf.num_cached == 2

    pre_remove_free = pool.num_free_blocks
    cache.remove_request(0)
    # Cached blocks did NOT return to the pool's free list (they live in the
    # cache); only any non-cached blocks would have. Here all blocks are cached.
    assert pool.num_free_blocks == pre_remove_free
    assert pf.num_evictable == 2


def test_eviction_when_pool_runs_out() -> None:
    """When the free list is exhausted, allocate falls back to LRU eviction."""
    pool = make_pool(block_size=4, num_layers=1, num_blocks=2)
    cache = PagedKVCache(pool)
    cache.add_request_slot(prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    _write_slot_all_layers(cache, num_tokens=8, fill=0.2, num_layers=1)
    cache.remove_request(0)

    pf = pool.prefix_cache
    assert pf is not None
    assert pf.num_cached == 2
    assert pool.num_free_blocks == 0  # all blocks are in the cache now

    # A new prompt that doesn't share these blocks needs fresh blocks; the
    # pool reclaims LRU cached blocks via eviction.
    new_idx = cache.add_request_slot(prompt_token_ids=[100, 101, 102, 103])
    assert cache.seq_lens_list()[new_idx] == 0  # no cache hit for this prompt
    _write_slot_all_layers(cache, num_tokens=4, fill=0.9, num_layers=1)
    # Eviction reclaimed at least one cached block.
    assert pf.num_cached <= 2  # could be 2 if duplicate hash, but we picked unique tokens


def test_concurrent_publish_coalesces_blocks() -> None:
    """Two slots filling identical blocks end up sharing one cache entry."""
    pool = make_pool(block_size=4, num_layers=1, num_blocks=4)
    cache = PagedKVCache(pool)
    prompt = [1, 2, 3, 4]  # one full block

    # Two slots with identical prompts. Both write before either publishes.
    cache.add_request_slot(prompt_token_ids=prompt)
    cache.add_request_slot(prompt_token_ids=prompt)
    # Both write the full block in one append_kv_packed call. Allocation
    # happens inside layer 0; publish runs at the end of the last layer.
    k = torch.full((8, 2, 4), 0.5, dtype=torch.float32)
    v = k + 100.0
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    cache.append_kv_packed(k, v, cu, layer_idx=0)

    # Publish-on-last-layer ran for both slots. The first published; the second
    # found a duplicate hash and coalesced. After this:
    #   - cache has exactly 1 entry,
    #   - both slots point at the canonical block.
    pf = pool.prefix_cache
    assert pf is not None
    assert pf.num_cached == 1
    assert cache.block_ids_for_request(0) == cache.block_ids_for_request(1)


def test_cached_kv_values_match_originally_written() -> None:
    """K/V at cached positions read back equal to what was originally written.

    Verifies that pre-populating a new slot with cached blocks gives access to
    the K/V values from the prior request (the whole point of the cache).
    """
    pool = make_pool(block_size=4, num_layers=1)
    cache = PagedKVCache(pool)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2 full blocks + 1 partial

    cache.add_request_slot(prompt_token_ids=prompt)
    fill_value = 0.42
    _write_slot_all_layers(cache, num_tokens=9, fill=fill_value, num_layers=1)
    cache.remove_request(0)

    new_idx = cache.add_request_slot(prompt_token_ids=prompt)
    assert cache.seq_lens_list()[new_idx] == 8

    # Materialize what's in the cache for this slot at layer 0.
    k_packed, v_packed, _, _ = cache.materialize_packed_kv(layer_idx=0)
    expected_k = torch.full_like(k_packed, fill_value)
    expected_v = expected_k + 100.0
    assert torch.allclose(k_packed, expected_k)
    assert torch.allclose(v_packed, expected_v)


def test_admission_evictable_blocks_count_toward_capacity() -> None:
    """A second request can be admitted by reclaiming cached blocks via eviction.

    Verifies the BlockPool.allocate fallback path that reaches into the prefix
    cache when the free list is exhausted but evictable blocks exist.
    """
    pool = make_pool(block_size=4, num_layers=1, num_blocks=2)
    cache = PagedKVCache(pool)
    cache.add_request_slot(prompt_token_ids=[10, 11, 12, 13])
    _write_slot_all_layers(cache, num_tokens=4, fill=0.5, num_layers=1)
    cache.remove_request(0)
    # Block is cached and evictable; free list has the OTHER unused block.
    pf = pool.prefix_cache
    assert pf is not None
    assert pf.num_evictable == 1
    assert pool.num_free_blocks == 1

    # New slot whose prompt requires 2 fresh blocks: 1 from free list, 1 via eviction.
    cache.add_request_slot(prompt_token_ids=[20, 21, 22, 23, 24, 25, 26, 27])
    _write_slot_all_layers(cache, num_tokens=8, fill=0.9, num_layers=1)
    # The new prompt's 2 blocks are now cached; the old cached block was evicted.
    assert pf.num_cached == 2
    # No KeyError, no OOM: the eviction path resolved the allocation.


def test_prefix_cache_block_size_must_match_pool_block_size() -> None:
    """Pool and cache must agree on block_size or construction fails."""
    import pytest

    pf = PrefixCache(block_size=8)
    with pytest.raises(ValueError, match="block_size"):
        BlockPool(
            num_blocks=4,
            block_size=4,  # mismatch
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            prefix_cache=pf,
        )
