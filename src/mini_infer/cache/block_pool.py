import torch

from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.exceptions import OutOfMemoryError


class BlockPool:
    """Pre-allocated pool of fixed-size K/V blocks shared across requests."""

    # Storage shape: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim).
    # The "2" is (key, value). All blocks live in one contiguous tensor for cache-friendly
    # access; allocate/free shuffle integer block IDs in a Python free list.

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if prefix_cache is not None and prefix_cache.block_size != block_size:
            raise ValueError(
                f"prefix_cache.block_size={prefix_cache.block_size} disagrees with "
                f"pool block_size={block_size}"
            )

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self._storage = torch.zeros(
            num_layers,
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._free_list: list[int] = list(range(num_blocks))
        self._prefix_cache = prefix_cache

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_list)

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    @property
    def prefix_cache(self) -> PrefixCache | None:
        return self._prefix_cache

    def allocate(self) -> int:
        if self._free_list:
            return self._free_list.pop()
        # Free pool empty: ask the prefix cache to surrender its oldest unreferenced
        # block. Returns None if the cache is empty or every cached block is pinned
        # by a running slot, in which case we genuinely have no memory.
        if self._prefix_cache is not None:
            evicted = self._prefix_cache.evict_lru()
            if evicted is not None:
                return evicted
        raise OutOfMemoryError("BlockPool: no free blocks available")

    def free(self, block_id: int) -> None:
        """Release a block. Cached blocks defer to PrefixCache (decref); else go to free list.

        With a prefix cache configured, a "freed" block that's cached stays in
        the cache (refcount-1) until LRU eviction reclaims it. Uncached blocks
        (e.g., the partial last block of a prompt) go straight to the free list.
        """
        if self._prefix_cache is not None and self._prefix_cache.is_cached(block_id):
            self._prefix_cache.decref(block_id)
        else:
            self._free_list.append(block_id)

    def view(self, block_id: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (key_block, value_block) views into storage for one block at one layer."""
        # Each view is shape (block_size, num_kv_heads, head_dim); writes go through.
        return self._storage[layer_idx, 0, block_id], self._storage[layer_idx, 1, block_id]
