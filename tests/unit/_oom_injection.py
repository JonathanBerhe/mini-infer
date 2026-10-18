"""Inject one `OutOfMemoryError` inside the REAL `append_kv_packed`.

Shared by the scheduler OOM tests. A fake runner that raises before it touches
the cache cannot reach the state that matters. On layer 0,
`append_kv_packed` advances each slot's token count and allocates its blocks
slot by slot, and only afterwards writes any K/V. A pool that runs dry at
slot j therefore leaves slots 0..j-1 claiming positions that hold no data.

`MidAppendOom` wraps one pool's `allocate` and produces that partial state with
the production append, in one of two shapes:

- By default it raises at the first allocation made after slot 0's count
  advanced. Slot 0 then claims positions with nothing written, and the slot
  that raised holds no new block.
- With `arm(slot=j)` it raises inside slot j's own allocation loop, after slot
  j already received at least one new block but before its count advanced.
  That slot then keeps blocks its unchanged length does not need.
"""

from __future__ import annotations

from collections.abc import Callable

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.exceptions import OutOfMemoryError


class MidAppendOom:
    """One-shot failure injection at the two-phase append's partial state.

    `cache` is a getter because the scheduler may replace its cache between
    steps; the injection reads whichever cache is live at allocation time.
    """

    def __init__(self, pool: BlockPool, cache: Callable[[], PagedKVCache | None]) -> None:
        self._real_allocate = pool.allocate
        self._cache = cache
        self._lens: list[int] | None = None
        self._blocks: list[int] = []
        self._identity: list[tuple[int, ...]] = []
        self._slot: int | None = None
        self.fired = False
        # An instance attribute shadows the bound method, so the production
        # `append_kv_packed` calls this wrapper through `self._pool.allocate()`.
        pool.allocate = self._allocate  # type: ignore[method-assign]

    def arm(self, *, slot: int | None = None) -> list[int]:
        """Arm for the next forward. Returns the slot lengths it starts from.

        `slot=None` raises after slot 0 advanced; `slot=j` raises inside slot
        j's loop once it holds a new block (the slot must need two or more).

        Arm right before a step in which no request finishes or arrives. The
        baseline is per slot INDEX, so a reap or an admission in the armed step
        shifts it onto the wrong requests. `_allocate` refuses to fire in that
        case: it compares the first block of every slot, not only the count.
        """
        cache = self._cache()
        assert cache is not None, "arm only once some request holds a cache slot"
        self._lens = cache.seq_lens_list()
        self._blocks = [len(ids) for ids in cache._block_ids]
        self._identity = [tuple(ids[:1]) for ids in cache._block_ids]
        self._slot = slot
        return list(self._lens)

    def _allocate(self) -> int:
        if self._lens is not None:
            cache = self._cache()
            assert cache is not None
            # A slot's first block never changes while it lives, so it names
            # the slot. Slots that held no block at arm time get their first
            # one during this very append, so only held blocks are compared.
            identity = [tuple(ids[:1]) for ids in cache._block_ids]
            assert len(identity) == len(self._identity) and all(
                now == then for now, then in zip(identity, self._identity, strict=True) if then
            ), (
                "a slot was added, reaped, or shifted between arm() and the forward; "
                "arm right before a step in which no request finishes or arrives"
            )
            lens = cache.seq_lens_list()
            if self._slot is None:
                due = lens[0] > self._lens[0]
            else:
                j = self._slot
                due = len(cache._block_ids[j]) > self._blocks[j] and lens[j] == self._lens[j]
            if due:
                self._lens = None
                self.fired = True
                raise OutOfMemoryError("forced mid-append OOM for test")
        return self._real_allocate()
