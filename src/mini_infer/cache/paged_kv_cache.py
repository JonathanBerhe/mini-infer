from typing import Any

import torch
from transformers import DynamicCache

from mini_infer.cache.block_pool import BlockPool


class PagedKVCache(DynamicCache):  # type: ignore[misc]
    """Paged KV cache that natively holds state for B requests.

    A fresh cache has `batch_size=0`. Each `add_request_slot()` appends a new
    request with its own block list and seq_len; `remove_request(batch_idx)`
    frees its blocks and shifts later indices down by one. The patched Qwen2
    attention reads per-request `block_tables_per_request_tensor()` and
    `seq_lens_list()` and dispatches `paged_attention_decode_batched(...)`,
    so a single forward pass over `(B, 1)` decode tokens works without any
    further plumbing.

    `update()` still satisfies HF's `Cache` contract for the prefill path
    (called by HF's stock attention when q_len > 1, B always 1 in our flow);
    it materializes per-request K/V padded to `max_seq_len` when B > 1 so
    HF's attention can mask via `attention_mask`. The hot decode path bypasses
    `update()` entirely and uses `append_kv()`.
    """

    def __init__(self, pool: BlockPool) -> None:
        super().__init__()
        self._pool = pool
        self._block_ids: list[list[int]] = []
        self._num_tokens: list[int] = []

    @property
    def batch_size(self) -> int:
        return len(self._num_tokens)

    def add_request_slot(self) -> int:
        """Append an empty request slot; returns its batch_idx."""
        self._num_tokens.append(0)
        self._block_ids.append([])
        return len(self._num_tokens) - 1

    def remove_request(self, batch_idx: int) -> None:
        """Free this request's blocks and remove its slot. Shifts later indices down by 1."""
        if not 0 <= batch_idx < self.batch_size:
            raise IndexError(f"batch_idx={batch_idx} out of range for batch_size={self.batch_size}")
        for block_id in self._block_ids[batch_idx]:
            self._pool.free(block_id)
        self._block_ids.pop(batch_idx)
        self._num_tokens.pop(batch_idx)

    def merge_request(self, other: "PagedKVCache") -> int:
        """Absorb a single-request cache from prefill. Returns the new batch_idx in self.

        Transfers ownership of `other`'s blocks; `other` is cleared so its `free()`
        becomes a no-op. The blocks themselves stay allocated (now owned by `self`).
        """
        if other.batch_size != 1:
            raise ValueError(f"merge_request expects batch_size=1, got {other.batch_size}")
        if other._pool is not self._pool:
            raise ValueError("cannot merge caches backed by different pools")
        self._block_ids.append(other._block_ids[0])
        self._num_tokens.append(other._num_tokens[0])
        other._block_ids = []
        other._num_tokens = []
        return len(self._num_tokens) - 1

    def block_ids_for_request(self, batch_idx: int) -> list[int]:
        """Return a copy of the block-id list for one request."""
        return list(self._block_ids[batch_idx])

    def block_tables_per_request_tensor(
        self, device: torch.device | str, dtype: torch.dtype = torch.int32
    ) -> list[torch.Tensor]:
        """Per-request 1D block-id tensors on `device` for the batched paged kernel."""
        return [torch.tensor(ids, device=device, dtype=dtype) for ids in self._block_ids]

    def seq_lens_list(self) -> list[int]:
        return list(self._num_tokens)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # key_states / value_states shape: (B, num_kv_heads, new_seq_len, head_dim).
        # HF Cache contract: write the new K/V and return full materialized history.
        # In our flow, this path is hit only for prefill (q_len > 1, always B=1) since
        # the patched decode bypasses `update()`. The B > 1 branch is here only for
        # safety against future code paths that hit it.
        self.append_kv(key_states, value_states, layer_idx)
        return self._materialize(layer_idx)

    def append_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Write new K/V into block storage without materializing.

        Uniform-length wrapper around `append_kv_packed`. Input shape
        `(B, num_kv_heads, new_seq_len, head_dim)` with B == cache.batch_size and
        the same `new_seq_len` for every request. For prefill, B=1 and
        new_seq_len=prompt_len; for batched decode, B=batch_size and new_seq_len=1.

        Use `append_kv_packed` directly when slots receive different numbers of
        new tokens in the same step (chunked prefill mixed with decoders).
        """
        b_in = key_states.shape[0]
        new_seq_len = key_states.shape[2]
        if b_in != self.batch_size:
            raise ValueError(
                f"append_kv: input batch={b_in} but cache batch_size={self.batch_size}"
            )
        # Convert (B, num_kv_heads, new_seq_len, head_dim) to packed
        # (B * new_seq_len, num_kv_heads, head_dim) with batch-major ordering, so
        # request b's tokens occupy packed indices [b*N, (b+1)*N).
        num_kv_heads = key_states.shape[1]
        head_dim = key_states.shape[3]
        packed_k = key_states.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        packed_v = value_states.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        cu_seqlens_q_new = torch.arange(
            0,
            (self.batch_size + 1) * new_seq_len,
            new_seq_len,
            dtype=torch.int32,
            device=key_states.device,
        )
        self.append_kv_packed(packed_k, packed_v, cu_seqlens_q_new, layer_idx)

    def append_kv_packed(
        self,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Write new K/V to per-slot positions in packed (varlen) form.

        Each batch slot receives `cu_seqlens_q_new[batch_idx + 1] -
        cu_seqlens_q_new[batch_idx]` new tokens this step (zero is allowed and
        means "no append for that slot"). The packed K/V tensors have all new
        tokens concatenated along the leading dim; `cu_seqlens_q_new` slices
        them per-slot.

        Shapes:
          packed_k, packed_v : (total_new_tokens, num_kv_heads, head_dim)
          cu_seqlens_q_new   : (batch_size + 1,) int, monotonically non-decreasing

        Block allocation only happens on layer 0 (counts advance once per step).
        """
        if cu_seqlens_q_new.shape[0] != self.batch_size + 1:
            raise ValueError(
                f"cu_seqlens_q_new has {cu_seqlens_q_new.shape[0]} entries; "
                f"expected batch_size+1 = {self.batch_size + 1}"
            )
        if layer_idx == 0:
            block_size = self._pool.block_size
            for batch_idx in range(self.batch_size):
                new_tokens = int(cu_seqlens_q_new[batch_idx + 1] - cu_seqlens_q_new[batch_idx])
                if new_tokens == 0:
                    continue
                new_total = self._num_tokens[batch_idx] + new_tokens
                required_blocks = (new_total + block_size - 1) // block_size
                while len(self._block_ids[batch_idx]) < required_blocks:
                    self._block_ids[batch_idx].append(self._pool.allocate())
                self._num_tokens[batch_idx] = new_total
        self._write_packed_kv(layer_idx, packed_k, packed_v, cu_seqlens_q_new)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns max seq_len across the batch (HF assumes a single value).

        HF uses this to size attention masks / position embeddings. For batched
        decode with ragged seq_lens we return the max so HF allocates enough; the
        per-request seq_lens_list() drives our paged kernel directly.
        """
        if not self._num_tokens:
            return 0
        return max(self._num_tokens)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for HF's causal-mask construction.

        HF calls this BEFORE the attention forward to size the 4D causal mask.
        `kv_length` must be the size of the K/V that attention will see AFTER
        this step's append (existing cached tokens + the new query_length tokens).
        `kv_offset` is the absolute starting position of the K/V tensor (always 0
        for us — we materialize from position 0 every call).

        We override DynamicCache's default because DynamicCache infers the sizes
        from `self.layers`, which is empty for us (we store K/V in `BlockPool`,
        not in DynamicCache's per-layer tensor list). Without this override HF
        builds a mask with `kv_length == query_length`, which truncates attention
        to only the new chunk's K/V and produces wrong outputs for chunked prefill.
        """
        existing = max(self._num_tokens) if self._num_tokens else 0
        return existing + query_length, 0

    def free(self) -> None:
        for ids in self._block_ids:
            for block_id in ids:
                self._pool.free(block_id)
        self._block_ids.clear()
        self._num_tokens.clear()

    def _write_packed_kv(
        self,
        layer_idx: int,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
    ) -> None:
        block_size = self._pool.block_size
        for batch_idx in range(self.batch_size):
            packed_start = int(cu_seqlens_q_new[batch_idx])
            packed_end = int(cu_seqlens_q_new[batch_idx + 1])
            new_tokens = packed_end - packed_start
            if new_tokens == 0:
                continue
            # _num_tokens[batch_idx] reflects the post-append count; subtract
            # new_tokens to find where this step's tokens start in the slot.
            start_token = self._num_tokens[batch_idx] - new_tokens
            for token_offset in range(new_tokens):
                position = start_token + token_offset
                block_id = self._block_ids[batch_idx][position // block_size]
                slot_in_block = position % block_size
                self._pool.storage[layer_idx, 0, block_id, slot_in_block] = packed_k[
                    packed_start + token_offset
                ]
                self._pool.storage[layer_idx, 1, block_id, slot_in_block] = packed_v[
                    packed_start + token_offset
                ]

    def _materialize(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B, num_kv_heads, max_seq_len, head_dim), zero-padded for shorter requests."""
        if self.batch_size == 0:
            shape = (0, self._pool.num_kv_heads, 0, self._pool.head_dim)
            empty = torch.empty(
                shape, dtype=self._pool.storage.dtype, device=self._pool.storage.device
            )
            return empty, empty

        max_seq = max(self._num_tokens)
        if max_seq == 0:
            shape = (self.batch_size, self._pool.num_kv_heads, 0, self._pool.head_dim)
            empty = torch.empty(
                shape, dtype=self._pool.storage.dtype, device=self._pool.storage.device
            )
            return empty, empty

        block_size = self._pool.block_size
        device = self._pool.storage.device
        num_kv_heads = self._pool.num_kv_heads
        head_dim = self._pool.head_dim
        dtype = self._pool.storage.dtype

        key_full = torch.zeros(
            (self.batch_size, num_kv_heads, max_seq, head_dim), dtype=dtype, device=device
        )
        value_full = torch.zeros_like(key_full)

        for batch_idx in range(self.batch_size):
            seq_len = self._num_tokens[batch_idx]
            if seq_len == 0:
                continue
            # Gather K/V for this request from its blocks: shape
            # (seq_len, num_kv_heads, head_dim) after advanced indexing.
            positions = torch.arange(seq_len, device=device)
            block_ids_lookup = torch.tensor(self._block_ids[batch_idx], device=device)
            block_ids_per_pos = block_ids_lookup[positions // block_size]
            slots_per_pos = positions % block_size
            keys_gathered = self._pool.storage[layer_idx, 0, block_ids_per_pos, slots_per_pos]
            values_gathered = self._pool.storage[layer_idx, 1, block_ids_per_pos, slots_per_pos]
            # Reorder to HF's (num_kv_heads, seq_len, head_dim) and copy into the
            # padded (num_kv_heads, max_seq, head_dim) slot for this request.
            keys_in_hf_layout = keys_gathered.permute(1, 0, 2)
            values_in_hf_layout = values_gathered.permute(1, 0, 2)
            key_full[batch_idx, :, :seq_len, :] = keys_in_hf_layout
            value_full[batch_idx, :, :seq_len, :] = values_in_hf_layout

        return key_full, value_full
