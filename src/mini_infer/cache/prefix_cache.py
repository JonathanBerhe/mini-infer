"""Block-level prefix cache for paged K/V.

When two requests share a prompt prefix, the second one can reuse the K/V
blocks the first computed instead of redoing the prefill work. This module
implements a hash-table-based prefix cache with refcounts and LRU eviction.

The cache stores fully-filled blocks (block_size tokens) keyed by a chained
hash of the block's token contents and its parent block's hash. The chain
ensures that identical tokens after different prefixes get different cache
keys, which is critical for correctness: the K/V at position N depends on
the entire context up to N, not just the local tokens.

Refcount lifecycle:
  - publish(): block enters the cache with refcount=1 (writer holds it).
  - incref(): caller is using the block; pinned against eviction.
  - decref(): caller is no longer using the block; if refcount hits zero,
    the block becomes LRU-eligible but stays cached until evicted.
  - evict_lru(): physically removes the oldest unreferenced block from the
    cache and returns its block_id so BlockPool can reuse it.

The cache is a pure data structure: it does not own block storage and does
not call into BlockPool. The integration layer (BlockPool / PagedKVCache)
wires `evict_lru` into the allocate path and `decref` into the free path.
"""

import dataclasses
import hashlib
from collections import OrderedDict

# 128-bit blake2b digest. Collision probability among N entries is ~N^2 / 2^128;
# at N=2^32 that is 2^-64, which is far below any realistic concern. We still
# verify the underlying token tuple on every hit (defense-in-depth and a useful
# debugging aid).
_HASH_BYTES = 16


@dataclasses.dataclass
class _CacheEntry:
    block_id: int
    block_hash: bytes
    token_ids: tuple[int, ...]
    refcount: int = 0


class PrefixCache:
    """Hash-keyed prefix cache with refcounts and LRU eviction.

    Operations are O(1) amortized: dict lookups for hash → entry and block_id →
    hash, and an OrderedDict for the evictable set keeps `evict_lru` O(1).
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._block_size = block_size
        self._entries: dict[bytes, _CacheEntry] = {}
        self._block_to_hash: dict[int, bytes] = {}
        # Insertion order tracks decref-time order; popitem(last=False) gives
        # the oldest evictable block.
        self._evictable: OrderedDict[int, None] = OrderedDict()

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_cached(self) -> int:
        return len(self._entries)

    @property
    def num_evictable(self) -> int:
        return len(self._evictable)

    @staticmethod
    def hash_block(parent_hash: bytes | None, token_ids: tuple[int, ...]) -> bytes:
        """Chained block hash; identical tokens after different prefixes hash differently.

        The chain mirrors the K/V dependency: position N's K/V depends on every
        prior token, so a block's cache key must depend on the full prefix that
        produced its K/V, not just the local tokens.
        """
        h = hashlib.blake2b(digest_size=_HASH_BYTES)
        if parent_hash is not None:
            h.update(parent_hash)
        # Encode each token id as 4 little-endian bytes; safe for any vocab
        # under 2^32, far above any real tokenizer.
        for token_id in token_ids:
            h.update(token_id.to_bytes(4, "little", signed=False))
        return h.digest()

    @classmethod
    def compute_block_hashes(
        cls, token_ids: list[int], block_size: int
    ) -> list[tuple[bytes, tuple[int, ...]]]:
        """Return (chained_hash, token_tuple) for each FULL block in `token_ids`.

        The trailing partial block (if any) is excluded; only fully-filled
        blocks are eligible for caching. With block_size=16 and a 20-token
        prompt, this returns one entry covering tokens [0:16]; tokens [16:20]
        live in the slot's private last block.
        """
        result: list[tuple[bytes, tuple[int, ...]]] = []
        parent: bytes | None = None
        num_full_blocks = len(token_ids) // block_size
        for block_idx in range(num_full_blocks):
            start = block_idx * block_size
            end = start + block_size
            tokens = tuple(token_ids[start:end])
            block_hash = cls.hash_block(parent, tokens)
            result.append((block_hash, tokens))
            parent = block_hash
        return result

    def lookup(self, block_hashes: list[tuple[bytes, tuple[int, ...]]]) -> list[int]:
        """Walk the hash chain; return cached block_ids for the longest matching prefix.

        Stops at the first miss or first hash collision (hash matches but token
        tuple disagrees, which is vanishingly rare with blake2b-128). Does not
        change refcounts; the caller calls `incref` on each returned block_id
        if it intends to hold the blocks.
        """
        result: list[int] = []
        for block_hash, expected_tokens in block_hashes:
            entry = self._entries.get(block_hash)
            if entry is None:
                break
            if entry.token_ids != expected_tokens:
                # Hash collision; treat as a miss so we don't return wrong K/V.
                break
            result.append(entry.block_id)
        return result

    def publish(
        self, block_hash: bytes, token_ids: tuple[int, ...], block_id: int
    ) -> tuple[int, bool]:
        """Register a fresh block under (block_hash, token_ids).

        Returns `(canonical_block_id, was_duplicate)`:
          - If `block_hash` is new: the block enters the cache with refcount=1
            (the writer holds the reference). Returns `(block_id, False)`.
          - If `block_hash` was already cached (concurrent fill of identical
            content): the caller's `block_id` is rejected, the canonical
            block_id is returned with incremented refcount. The caller must
            return their unused `block_id` to BlockPool. Returns
            `(canonical, True)`.

        Raises if a stored hash maps to different tokens than the publish call,
        which would indicate an unrecoverable hash collision.
        """
        existing = self._entries.get(block_hash)
        if existing is not None:
            if existing.token_ids != token_ids:
                raise RuntimeError(
                    f"hash collision on publish: stored tokens {existing.token_ids} "
                    f"vs new tokens {token_ids}"
                )
            self._incref(existing)
            return existing.block_id, True

        entry = _CacheEntry(
            block_id=block_id,
            block_hash=block_hash,
            token_ids=token_ids,
            refcount=1,
        )
        self._entries[block_hash] = entry
        self._block_to_hash[block_id] = block_hash
        # refcount=1 means not evictable yet; nothing to add to _evictable.
        return block_id, False

    def incref(self, block_id: int) -> None:
        """Pin a cached block against eviction."""
        block_hash = self._block_to_hash.get(block_id)
        if block_hash is None:
            raise KeyError(f"block_id={block_id} is not in the prefix cache")
        self._incref(self._entries[block_hash])

    def decref(self, block_id: int) -> None:
        """Release a cached block; becomes LRU-eligible at refcount=0."""
        block_hash = self._block_to_hash.get(block_id)
        if block_hash is None:
            raise KeyError(f"block_id={block_id} is not in the prefix cache")
        entry = self._entries[block_hash]
        if entry.refcount <= 0:
            raise RuntimeError(
                f"decref on block_id={block_id} with refcount={entry.refcount} "
                "(double-free or refcount leak)"
            )
        entry.refcount -= 1
        if entry.refcount == 0:
            # Newly evictable; insertion order = LRU order, so append to end.
            self._evictable[block_id] = None

    def evict_lru(self) -> int | None:
        """Remove the oldest refcount=0 block; return its block_id (or None).

        Caller (BlockPool) returns the freed block_id to its free list.
        """
        if not self._evictable:
            return None
        block_id, _ = self._evictable.popitem(last=False)
        block_hash = self._block_to_hash.pop(block_id)
        del self._entries[block_hash]
        return block_id

    def is_cached(self, block_id: int) -> bool:
        return block_id in self._block_to_hash

    def _incref(self, entry: _CacheEntry) -> None:
        if entry.refcount == 0:
            self._evictable.pop(entry.block_id, None)
        entry.refcount += 1
