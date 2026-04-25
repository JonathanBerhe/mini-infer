from typing import Any

import torch
from transformers import DynamicCache

from mini_infer.cache.block_pool import BlockPool


class PagedKVCache(DynamicCache):  # type: ignore[misc]
    """Per-request paged KV cache; subclasses DynamicCache so HF's per-layer plumbing works.

    Slice 2.1 materializes blocks into contiguous K/V at every update() so HF's stock
    attention path keeps working. Slice 2.2 will replace the materialization with a
    paged attention kernel. update() and get_seq_length() are overridden to use blocks;
    DynamicCache's internal layer storage is unused by our path.
    """

    def __init__(self, pool: BlockPool) -> None:
        super().__init__()
        self._pool = pool
        self._block_ids: list[int] = []
        self._num_tokens = 0

    @property
    def block_ids(self) -> list[int]:
        return list(self._block_ids)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # key_states / value_states shape: (1, num_kv_heads, new_seq_len, head_dim).
        new_seq_len = key_states.shape[2]

        # Allocate any new blocks needed; advance the running token count once per step
        # (on layer 0 only, since update() is called per layer per step).
        if layer_idx == 0:
            new_total = self._num_tokens + new_seq_len
            block_size = self._pool.block_size
            required_blocks = (new_total + block_size - 1) // block_size
            while len(self._block_ids) < required_blocks:
                self._block_ids.append(self._pool.allocate())
            self._num_tokens = new_total

        self._write_new_kv(layer_idx, key_states, value_states, new_seq_len)
        return self._materialize(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._num_tokens

    def free(self) -> None:
        for block_id in self._block_ids:
            self._pool.free(block_id)
        self._block_ids.clear()
        self._num_tokens = 0

    def _write_new_kv(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        new_seq_len: int,
    ) -> None:
        block_size = self._pool.block_size
        # num_tokens is consistent across layers because it advances only on layer 0.
        start = self._num_tokens - new_seq_len
        for i in range(new_seq_len):
            pos = start + i
            block_id = self._block_ids[pos // block_size]
            slot = pos % block_size
            self._pool.storage[layer_idx, 0, block_id, slot] = key_states[0, :, i, :]
            self._pool.storage[layer_idx, 1, block_id, slot] = value_states[0, :, i, :]

    def _materialize(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._num_tokens == 0:
            shape = (1, self._pool.num_kv_heads, 0, self._pool.head_dim)
            empty = torch.empty(
                shape, dtype=self._pool.storage.dtype, device=self._pool.storage.device
            )
            return empty, empty

        block_size = self._pool.block_size
        device = self._pool.storage.device

        positions = torch.arange(self._num_tokens, device=device)
        block_ids_lookup = torch.tensor(self._block_ids, device=device)
        block_ids_per_pos = block_ids_lookup[positions // block_size]
        slots_per_pos = positions % block_size

        # Advanced indexing: storage[layer, 0, block_ids_per_pos, slots_per_pos] returns
        # shape (num_tokens, num_kv_heads, head_dim).
        key_full = self._pool.storage[layer_idx, 0, block_ids_per_pos, slots_per_pos]
        value_full = self._pool.storage[layer_idx, 1, block_ids_per_pos, slots_per_pos]

        # Reshape to HF's (batch=1, num_kv_heads, num_tokens, head_dim).
        key_full = key_full.permute(1, 0, 2).unsqueeze(0)
        value_full = value_full.permute(1, 0, 2).unsqueeze(0)
        return key_full, value_full
