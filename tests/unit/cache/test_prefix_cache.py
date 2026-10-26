import pytest

from mini_infer.cache.prefix_cache import PrefixCache


def test_hash_block_is_deterministic() -> None:
    """Same parent + same tokens always produces the same hash."""
    h1 = PrefixCache.hash_block(None, (1, 2, 3))
    h2 = PrefixCache.hash_block(None, (1, 2, 3))
    assert h1 == h2


def test_hash_block_chains_distinguish_same_tokens_different_prefix() -> None:
    """Identical tokens after different parents must hash differently.

    This is the correctness property the chain exists for: K/V at a position
    depends on every prior token, so two blocks with identical token contents
    but different histories must NOT share a cache key.
    """
    parent_a = PrefixCache.hash_block(None, (1, 2))
    parent_b = PrefixCache.hash_block(None, (3, 4))
    assert PrefixCache.hash_block(parent_a, (5, 6)) != PrefixCache.hash_block(parent_b, (5, 6))


def test_hash_block_changes_with_tokens() -> None:
    """Different token contents under the same parent produce different hashes."""
    h1 = PrefixCache.hash_block(None, (1, 2, 3))
    h2 = PrefixCache.hash_block(None, (1, 2, 4))
    assert h1 != h2


def test_compute_block_hashes_full_blocks_only() -> None:
    """Trailing partial block is excluded; only fully-filled blocks are hashed."""
    block_size = 4
    tokens = list(range(10))  # 2 full blocks (8 tokens) + 1 partial (2 tokens)
    chain = PrefixCache.compute_block_hashes(tokens, block_size)
    assert len(chain) == 2
    assert chain[0][1] == (0, 1, 2, 3)
    assert chain[1][1] == (4, 5, 6, 7)


def test_compute_block_hashes_chains_correctly() -> None:
    """Block N's hash depends on block N-1's hash (chain property end-to-end)."""
    block_size = 4
    chain_a = PrefixCache.compute_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size)
    # Same first block, different second-block tokens — second hash must differ.
    chain_b = PrefixCache.compute_block_hashes([1, 2, 3, 4, 99, 6, 7, 8], block_size)
    assert chain_a[0][0] == chain_b[0][0]
    assert chain_a[1][0] != chain_b[1][0]


def test_compute_block_hashes_returns_empty_for_too_few_tokens() -> None:
    """Prompt shorter than block_size produces no cacheable blocks."""
    assert PrefixCache.compute_block_hashes([1, 2, 3], block_size=4) == []


def test_publish_then_lookup_returns_block_id() -> None:
    """Single-block round-trip: publish, look up the same hash, get the block_id."""
    cache = PrefixCache(block_size=4)
    chain = PrefixCache.compute_block_hashes(list(range(4)), block_size=4)
    block_hash, tokens = chain[0]
    cache.publish(block_hash, tokens, block_id=42)
    assert cache.lookup(chain) == [42]


def test_lookup_misses_at_first_unknown_hash() -> None:
    """Chain [a, b, c] with only [a, b] cached returns 2 block_ids."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2, 3, 4, 5, 6], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=10)
    cache.publish(chain[1][0], chain[1][1], block_id=11)
    # chain[2] never published.
    matches = cache.lookup(chain)
    assert matches == [10, 11]


def test_lookup_does_not_change_refcount() -> None:
    """lookup is read-only; matched blocks must remain LRU-eligible."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=7)
    cache.decref(7)  # refcount 1 -> 0; block enters _evictable
    assert cache.num_evictable == 1
    cache.lookup(chain)  # read-only
    assert cache.num_evictable == 1


def test_publish_duplicate_returns_canonical_and_increments_refcount() -> None:
    """Two slots filling identical blocks coalesce to the cache's canonical entry."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    h, tokens = chain[0]

    canonical, dup_first = cache.publish(h, tokens, block_id=100)
    assert canonical == 100
    assert dup_first is False

    canonical2, dup_second = cache.publish(h, tokens, block_id=200)
    assert canonical2 == 100
    assert dup_second is True

    # Two refs now: original publish + duplicate. Decref twice to evict.
    cache.decref(100)
    assert cache.num_evictable == 0  # still has 1 ref
    cache.decref(100)
    assert cache.num_evictable == 1  # now evictable


def test_publish_with_hash_collision_raises() -> None:
    """Same hash but different tokens is treated as an unrecoverable collision.

    With blake2b-128 this is statistically impossible, but we still defend
    against the case to avoid silently returning wrong K/V.
    """
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    h, tokens = chain[0]
    cache.publish(h, tokens, block_id=1)
    with pytest.raises(RuntimeError, match="hash collision"):
        cache.publish(h, (9, 9), block_id=2)


def test_decref_to_zero_makes_evictable() -> None:
    """A block becomes LRU-eligible exactly when refcount hits 0."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=5)
    assert cache.num_evictable == 0  # refcount=1
    cache.decref(5)
    assert cache.num_evictable == 1  # refcount=0


def test_incref_pins_against_eviction() -> None:
    """A re-incref'd block leaves the evictable set."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=5)
    cache.decref(5)
    assert cache.num_evictable == 1
    cache.incref(5)
    assert cache.num_evictable == 0


def test_evict_lru_returns_oldest_evictable() -> None:
    """Insertion order = decref order = LRU order."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2, 3, 4, 5, 6], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=10)
    cache.publish(chain[1][0], chain[1][1], block_id=11)
    cache.publish(chain[2][0], chain[2][1], block_id=12)
    cache.decref(11)
    cache.decref(12)
    cache.decref(10)
    # Decref order: 11, 12, 10. evict_lru must return them in that order.
    assert cache.evict_lru() == 11
    assert cache.evict_lru() == 12
    assert cache.evict_lru() == 10
    assert cache.evict_lru() is None


def test_evict_lru_returns_none_when_nothing_evictable() -> None:
    """A pinned-only cache has no eviction candidates."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=10)
    # refcount=1; not evictable.
    assert cache.evict_lru() is None


def test_evict_lru_skips_pinned_blocks() -> None:
    """Mixed pinned + unpinned cache evicts only unpinned, in LRU order."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2, 3, 4, 5, 6], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=10)
    cache.publish(chain[1][0], chain[1][1], block_id=11)
    cache.publish(chain[2][0], chain[2][1], block_id=12)
    # Only 11 is decref'd; 10 and 12 stay pinned.
    cache.decref(11)
    assert cache.evict_lru() == 11
    assert cache.evict_lru() is None  # 10 and 12 still pinned


def test_evicted_block_no_longer_in_cache() -> None:
    """After evict, both lookup and is_cached should report the block as gone."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=10)
    cache.decref(10)
    assert cache.evict_lru() == 10
    assert cache.lookup(chain) == []
    assert cache.is_cached(10) is False


def test_decref_on_uncached_block_raises_keyerror() -> None:
    """Decref'ing a block_id we never published is a programming error."""
    cache = PrefixCache(block_size=2)
    with pytest.raises(KeyError, match="not in the prefix cache"):
        cache.decref(99)


def test_decref_below_zero_raises() -> None:
    """Double-free is a programming error and must be loud."""
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    cache.publish(chain[0][0], chain[0][1], block_id=10)
    cache.decref(10)
    with pytest.raises(RuntimeError, match="double-free or refcount leak"):
        cache.decref(10)


def test_block_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        PrefixCache(block_size=0)
    with pytest.raises(ValueError, match="block_size must be positive"):
        PrefixCache(block_size=-1)


def test_lookup_misses_on_token_collision_with_matching_hash() -> None:
    """If we somehow get a hash hit but token tuples disagree, treat as miss.

    Hand-construct the corruption by reaching into the cache (the legitimate
    path can never trigger this, so we have to fabricate the state).
    """
    cache = PrefixCache(block_size=2)
    chain = PrefixCache.compute_block_hashes([1, 2], block_size=2)
    h, _ = chain[0]
    cache.publish(h, (1, 2), block_id=10)
    # Lookup with mismatched tokens under the same hash; must miss.
    fake_chain = [(h, (9, 9))]
    assert cache.lookup(fake_chain) == []
