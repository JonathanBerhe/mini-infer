import torch

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
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

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

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_list)

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    def allocate(self) -> int:
        if not self._free_list:
            raise OutOfMemoryError("BlockPool: no free blocks available")
        return self._free_list.pop()

    def free(self, block_id: int) -> None:
        # Caller's responsibility to free only blocks they allocated; we don't double-check
        # to keep the hot path cheap. Phase 2.3 (continuous batching) will add accounting.
        self._free_list.append(block_id)

    def view(self, block_id: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (key_block, value_block) views into storage for one block at one layer."""
        # Each view is shape (block_size, num_kv_heads, head_dim); writes go through.
        return self._storage[layer_idx, 0, block_id], self._storage[layer_idx, 1, block_id]
