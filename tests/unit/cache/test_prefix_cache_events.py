"""KV-cache events from the prefix cache (BlockStored, BlockRemoved).

A KV-cache-aware router keys each stored block by its tokens and finds the
prefix through `parent_block_hash`. It drops a block whose parent it has not
seen, so a parent must be stored before its child and must name the right hash.
"""

import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.kv_events import BlockRemoved, BlockStored, KVEvent
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache

BLOCK_SIZE = 4
NUM_LAYERS = 2


def _recording_cache() -> tuple[PrefixCache, list[KVEvent]]:
    cache = PrefixCache(block_size=BLOCK_SIZE)
    events: list[KVEvent] = []
    cache.set_event_listener(events.append)
    return cache, events


def _pool_with_events(num_blocks: int = 16) -> tuple[BlockPool, list[KVEvent]]:
    prefix_cache, events = _recording_cache()
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
        prefix_cache=prefix_cache,
    )
    return pool, events


def _prefill_slot0(cache: PagedKVCache, num_tokens: int) -> None:
    """Write `num_tokens` new K/V rows for slot 0 in every layer."""
    k = torch.ones((num_tokens, 2, 4), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)
    for layer_idx in range(NUM_LAYERS):
        cache.append_kv_packed(k, k + 1.0, cu_seqlens, layer_idx=layer_idx)


def test_publish_emits_block_stored_with_parent() -> None:
    cache, events = _recording_cache()
    first, second = (1, 2, 3, 4), (5, 6, 7, 8)
    h1 = PrefixCache.hash_block(None, first)
    h2 = PrefixCache.hash_block(h1, second)

    cache.publish(h1, first, 0)
    cache.publish(h2, second, 1, parent_hash=h1)

    assert events == [
        BlockStored(block_hashes=(h1,), parent_block_hash=None, token_ids=first, block_size=4),
        BlockStored(block_hashes=(h2,), parent_block_hash=h1, token_ids=second, block_size=4),
    ]


def test_duplicate_publish_emits_no_event() -> None:
    cache, events = _recording_cache()
    tokens = (1, 2, 3, 4)
    block_hash = PrefixCache.hash_block(None, tokens)
    cache.publish(block_hash, tokens, 0)
    events.clear()

    canonical, was_duplicate = cache.publish(block_hash, tokens, 1)

    assert (canonical, was_duplicate) == (0, True)
    assert events == []


def test_eviction_emits_block_removed() -> None:
    cache, events = _recording_cache()
    tokens = (1, 2, 3, 4)
    block_hash = PrefixCache.hash_block(None, tokens)
    cache.publish(block_hash, tokens, 7)
    cache.decref(7)
    events.clear()

    assert cache.evict_lru() == 7
    assert events == [BlockRemoved(block_hashes=(block_hash,))]


def test_removing_the_listener_stops_events() -> None:
    cache, events = _recording_cache()
    cache.set_event_listener(None)
    tokens = (1, 2, 3, 4)
    cache.publish(PrefixCache.hash_block(None, tokens), tokens, 0)
    assert events == []


def test_paged_cache_stores_the_chain_parent_first() -> None:
    """A 10-token prompt fills two blocks; their events chain in prompt order."""
    pool, events = _pool_with_events()
    cache = PagedKVCache(pool)
    prompt = list(range(100, 110))
    cache.add_request_slot(prompt_token_ids=prompt)

    _prefill_slot0(cache, num_tokens=len(prompt))

    (h1, t1), (h2, t2) = PrefixCache.compute_block_hashes(prompt, BLOCK_SIZE)
    assert events == [
        BlockStored(block_hashes=(h1,), parent_block_hash=None, token_ids=t1, block_size=4),
        BlockStored(block_hashes=(h2,), parent_block_hash=h1, token_ids=t2, block_size=4),
    ]


def test_prefix_hit_continues_the_chain_from_the_cached_block() -> None:
    """After a hit on two cached blocks, the next new block names the second as parent."""
    pool, events = _pool_with_events()
    cache = PagedKVCache(pool)
    shared = [1, 2, 3, 4, 5, 6, 7, 8]
    cache.add_request_slot(prompt_token_ids=shared)
    _prefill_slot0(cache, num_tokens=len(shared))
    cache.remove_request(0)
    events.clear()

    prompt = [*shared, 50, 51, 52, 53, 54]
    cache.add_request_slot(prompt_token_ids=prompt)
    assert cache.seq_lens_list() == [8]
    _prefill_slot0(cache, num_tokens=len(prompt) - 8)

    chain = PrefixCache.compute_block_hashes(prompt, BLOCK_SIZE)
    assert events == [
        BlockStored(
            block_hashes=(chain[2][0],),
            parent_block_hash=chain[1][0],
            token_ids=(50, 51, 52, 53),
            block_size=4,
        )
    ]


def test_pool_eviction_reports_the_removed_block() -> None:
    """When the pool takes back a cached block, the router hears that it is gone."""
    pool, events = _pool_with_events(num_blocks=2)
    cache = PagedKVCache(pool)
    prompt = [1, 2, 3, 4]
    cache.add_request_slot(prompt_token_ids=prompt)
    _prefill_slot0(cache, num_tokens=len(prompt))
    cache.remove_request(0)
    events.clear()

    # Two fresh blocks: the free one, then the cached one through eviction.
    pool.allocate()
    pool.allocate()

    [(cached_hash, _)] = PrefixCache.compute_block_hashes(prompt, BLOCK_SIZE)
    assert events == [BlockRemoved(block_hashes=(cached_hash,))]
