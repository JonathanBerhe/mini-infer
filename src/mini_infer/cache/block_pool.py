import torch

from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.cache.turbo_quant import (
    dequantize_kv_block,
    generate_rotation_matrices,
    inverse_rotate,
    quantize_kv_block,
    rotate,
)
from mini_infer.exceptions import OutOfMemoryError

# Supported KV-cache compression modes. None = legacy bf16/fp16 storage.
# "turbo4" = TurboQuant V1: per-layer random rotation + per-block 4-bit quant.
_SUPPORTED_KV_QUANT = (None, "turbo4")


class BlockPool:
    """Pre-allocated pool of fixed-size K/V blocks shared across requests.

    Two storage modes:

    - **Uncompressed** (default, ``kv_quant=None``). Single bf16/fp16 tensor
      of shape ``(num_layers, 2, num_blocks, block_size, num_kv_heads,
      head_dim)``. The "2" is (key, value). Per-block read/write is a
      direct slice into ``_storage``.
    - **TurboQuant 4-bit** (``kv_quant="turbo4"``). Each block is stored
      as int8 packed bytes (two 4-bit values per byte) plus a bf16
      ``(num_kv_heads, head_dim, 2)`` scales tensor (low + scale per
      channel). A per-layer random orthogonal rotation is applied
      before quantization on write and inverted after dequantization on
      read. Math is identity vs the uncompressed path within
      quantization noise; storage is ~4x smaller.

    Compressed-mode blocks are read/written via ``read_compressed_block``
    and ``write_compressed_block`` rather than ``view`` / ``storage``,
    which only make sense for the uncompressed layout.
    """

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
        kv_quant: str | None = None,
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
        if kv_quant not in _SUPPORTED_KV_QUANT:
            raise ValueError(
                f"unsupported kv_quant={kv_quant!r}; expected one of {_SUPPORTED_KV_QUANT}"
            )
        if kv_quant == "turbo4" and (block_size * num_kv_heads * head_dim) % 2 != 0:
            raise ValueError(
                "turbo4 packs two 4-bit values per byte; "
                "block_size * num_kv_heads * head_dim must be even"
            )

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._kv_quant = kv_quant

        if kv_quant is None:
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
            self._compressed_storage: torch.Tensor | None = None
            self._scales_storage: torch.Tensor | None = None
            self._rotation: torch.Tensor | None = None
        else:
            # turbo4: int8 packed bytes per block + bf16 scales + rotation.
            packed_bytes_per_block = (block_size * num_kv_heads * head_dim) // 2
            self._compressed_storage = torch.zeros(
                num_layers,
                2,
                num_blocks,
                packed_bytes_per_block,
                dtype=torch.int8,
                device=device,
            )
            # Per-block, per-channel (low, scale) — last dim of size 2.
            self._scales_storage = torch.zeros(
                num_layers,
                2,
                num_blocks,
                num_kv_heads,
                head_dim,
                2,
                dtype=dtype,
                device=device,
            )
            self._rotation = generate_rotation_matrices(
                num_layers,
                head_dim,
                dtype=dtype,
                device=device,
                seed=0,
            )
            # No bf16 _storage in compressed mode; using `.storage` raises.
            self._storage = torch.empty(0, dtype=dtype, device=device)

        self._free_list: list[int] = list(range(num_blocks))
        self._prefix_cache = prefix_cache

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_list)

    @property
    def storage(self) -> torch.Tensor:
        if self._kv_quant is not None:
            raise RuntimeError(
                f"`storage` only valid for uncompressed pool; got kv_quant={self._kv_quant!r}. "
                "Use read_compressed_block / write_compressed_block instead."
            )
        return self._storage

    @property
    def kv_quant(self) -> str | None:
        return self._kv_quant

    @property
    def rotation(self) -> torch.Tensor | None:
        return self._rotation

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
        if self._kv_quant is not None:
            raise RuntimeError(
                f"`view` only valid for uncompressed pool; got kv_quant={self._kv_quant!r}"
            )
        # Each view is shape (block_size, num_kv_heads, head_dim); writes go through.
        return self._storage[layer_idx, 0, block_id], self._storage[layer_idx, 1, block_id]

    def read_compressed_block(self, layer_idx: int, kv_idx: int, block_id: int) -> torch.Tensor:
        """Read one compressed block as a `(block_size, num_kv_heads, head_dim)` tensor.

        Internally: load packed int8 + scales for this block, dequantize,
        apply inverse rotation. Returns a fresh bf16/fp16 tensor in the
        original (un-rotated) representation. Caller is responsible for
        only reading positions ``< self._num_tokens[batch_idx]``; tail
        slots beyond seq_len contain whatever the last write quantized
        (zero on freshly-allocated blocks).

        Compressed-mode only; raises on uncompressed pools.
        """
        if self._kv_quant != "turbo4":
            raise RuntimeError(
                f"read_compressed_block requires kv_quant='turbo4'; got {self._kv_quant!r}"
            )
        assert self._compressed_storage is not None
        assert self._scales_storage is not None
        assert self._rotation is not None
        packed = self._compressed_storage[layer_idx, kv_idx, block_id]
        # scales_storage[..., 0] is `low`, [..., 1] is `scale`.
        low = self._scales_storage[layer_idx, kv_idx, block_id, :, :, 0]
        scale = self._scales_storage[layer_idx, kv_idx, block_id, :, :, 1]
        rotated = dequantize_kv_block(
            packed,
            low,
            scale,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
        )
        return inverse_rotate(rotated, self._rotation[layer_idx])

    def write_compressed_block(
        self,
        layer_idx: int,
        kv_idx: int,
        block_id: int,
        block: torch.Tensor,
    ) -> None:
        """Write a full ``(block_size, num_kv_heads, head_dim)`` block to compressed storage.

        Internally: rotate, per-channel 4-bit quantize, write packed bytes
        + scales. The caller has already collected a complete block's
        worth of data (rotate + quant don't make sense on a sub-block).

        Compressed-mode only; raises on uncompressed pools.
        """
        if self._kv_quant != "turbo4":
            raise RuntimeError(
                f"write_compressed_block requires kv_quant='turbo4'; got {self._kv_quant!r}"
            )
        if block.shape != (self.block_size, self.num_kv_heads, self.head_dim):
            raise ValueError(
                f"expected block shape ({self.block_size}, {self.num_kv_heads}, "
                f"{self.head_dim}); got {tuple(block.shape)}"
            )
        assert self._compressed_storage is not None
        assert self._scales_storage is not None
        assert self._rotation is not None

        rotated = rotate(block, self._rotation[layer_idx])
        packed, low, scale = quantize_kv_block(rotated)
        self._compressed_storage[layer_idx, kv_idx, block_id] = packed
        self._scales_storage[layer_idx, kv_idx, block_id, :, :, 0] = low
        self._scales_storage[layer_idx, kv_idx, block_id, :, :, 1] = scale
