"""Inject one `OutOfMemoryError` inside the real `append_kv_packed`.

A fake runner that raises before it touches the cache cannot reach the state
that matters. On layer 0, `append_kv_packed` takes the slots in order. For each, it
allocates blocks, then advances the token count. It writes K/V only after this
loop. So an OOM at slot j leaves slots 0..j-1 claiming positions with no data.

`MidAppendOom` makes this state in one of two shapes. By default, it raises at
the first allocation after slot 0 advanced, so the slot that raised holds no new
block. With `arm(slot=j)`, it raises after slot j got a new block and before its
count advanced, so slot j keeps blocks that its unchanged length does not need.
"""

from __future__ import annotations

from collections.abc import Callable

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.exceptions import OutOfMemoryError


class MidAppendOom:
    """Raise one OOM in the middle of an append, at the partial state above.

    `cache` is a getter because the scheduler can replace its cache between steps.
    """

    def __init__(self, pool: BlockPool, cache: Callable[[], PagedKVCache | None]) -> None:
        self._real_allocate = pool.allocate
        self._cache = cache
        self._lens: list[int] | None = None
        self._blocks: list[int] = []
        self._identity: list[tuple[int, ...]] = []
        self._slot: int | None = None
        self.fired = False
        # The instance attribute shadows the method, so the production append calls this.
        pool.allocate = self._allocate  # type: ignore[method-assign]

    def arm(self, *, slot: int | None = None) -> list[int]:
        """Arm for the next forward. Return the slot lengths at arm time.

        With `slot=j`, slot j must need two or more new blocks. Arm right before a
        step in which no request finishes or arrives. The baseline is per slot
        index, so a reap or an admission moves it onto the wrong requests, and
        `_allocate` then raises an `AssertionError` instead of the OOM.
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
            # The first block of a slot never changes while the slot lives, so it
            # identifies the slot. A slot with no block at arm time gets one now, so skip it.
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
