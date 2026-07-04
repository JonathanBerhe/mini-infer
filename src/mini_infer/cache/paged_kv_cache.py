from typing import Any

import torch
from transformers import DynamicCache

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.cache.turbo_kernel import (
    fused_materialize_packed_kv,
    supports_fused_kernel,
)


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

    Prefix caching: if the underlying `BlockPool` was constructed with a
    `PrefixCache`, `add_request_slot(prompt_token_ids=...)` looks up cached
    block-hash chains and pre-populates the slot's blocks (with refcounts
    held in the prefix cache). On the last layer of each step's
    `append_kv_packed`, blocks that just filled with prompt tokens are
    published into the prefix cache so future requests can hit them.
    """

    def __init__(self, pool: BlockPool) -> None:
        super().__init__()
        self._pool = pool
        self._prefix_cache = pool.prefix_cache
        self._block_ids: list[list[int]] = []
        self._num_tokens: list[int] = []
        # Per-slot tracking for prefix-cache publish. Empty/0/None when prefix
        # caching is disabled OR when the slot was created without a prompt
        # (e.g., the legacy single-request prefill cache).
        self._slot_prompt_tokens: list[list[int]] = []
        self._slot_num_published: list[int] = []
        self._slot_parent_hash: list[bytes | None] = []
        # NVFP4 mode keeps a per-slot bf16 shadow of the slot's currently-
        # active (last) block across all (layer, side). The shadow is the
        # only place we hold bf16 K/V long enough to re-quantize the page
        # whenever a new token lands in it. Allocated lazily on first
        # write per slot; reset when the slot's active block changes.
        self._nvfp4_shadow: list[torch.Tensor | None] = []
        self._nvfp4_shadow_block_id: list[int | None] = []

    @property
    def batch_size(self) -> int:
        return len(self._num_tokens)

    def add_request_slot(self, prompt_token_ids: list[int] | None = None) -> int:
        """Append a request slot; returns its batch_idx.

        If `prompt_token_ids` is provided AND the pool has a prefix cache, the
        slot is pre-populated with cached blocks for any matching prompt prefix.
        On return, `self._num_tokens[batch_idx]` reflects how many tokens are
        already in the slot's K/V cache (zero if no hit, or no prefix cache).

        The "last-token rule": if the entire prompt is cached, we drop the last
        cached block so that at least one token of the prompt remains
        unprocessed. The scheduler relies on running forward over at least one
        token to obtain logits for the next sample; if everything is cached,
        there is nothing to run.
        """
        self._num_tokens.append(0)
        self._block_ids.append([])
        self._slot_prompt_tokens.append(list(prompt_token_ids) if prompt_token_ids else [])
        self._slot_num_published.append(0)
        self._slot_parent_hash.append(None)
        self._nvfp4_shadow.append(None)
        self._nvfp4_shadow_block_id.append(None)
        batch_idx = self.batch_size - 1

        if not prompt_token_ids or self._prefix_cache is None:
            return batch_idx

        chain = PrefixCache.compute_block_hashes(prompt_token_ids, self._pool.block_size)
        if not chain:
            return batch_idx

        matched = self._prefix_cache.lookup(chain)
        num_cached_blocks = len(matched)
        if num_cached_blocks == 0:
            return batch_idx
        # Last-token rule: leave at least one token un-prefilled so the
        # scheduler's first forward over this slot produces logits.
        if num_cached_blocks * self._pool.block_size >= len(prompt_token_ids):
            num_cached_blocks -= 1
        if num_cached_blocks <= 0:
            return batch_idx

        for block_id in matched[:num_cached_blocks]:
            self._prefix_cache.incref(block_id)
        self._block_ids[batch_idx] = list(matched[:num_cached_blocks])
        self._num_tokens[batch_idx] = num_cached_blocks * self._pool.block_size
        self._slot_num_published[batch_idx] = num_cached_blocks
        self._slot_parent_hash[batch_idx] = chain[num_cached_blocks - 1][0]
        return batch_idx

    def remove_request(self, batch_idx: int) -> None:
        """Free this request's blocks and remove its slot. Shifts later indices down by 1."""
        if not 0 <= batch_idx < self.batch_size:
            raise IndexError(f"batch_idx={batch_idx} out of range for batch_size={self.batch_size}")
        for block_id in self._block_ids[batch_idx]:
            self._pool.free(block_id)
        self._block_ids.pop(batch_idx)
        self._num_tokens.pop(batch_idx)
        self._slot_prompt_tokens.pop(batch_idx)
        self._slot_num_published.pop(batch_idx)
        self._slot_parent_hash.pop(batch_idx)
        self._nvfp4_shadow.pop(batch_idx)
        self._nvfp4_shadow_block_id.pop(batch_idx)

    def truncate_to(self, batch_idx: int, new_seq_len: int) -> None:
        """Roll back this slot to `new_seq_len`; free blocks beyond the new boundary.

        Frees blocks whose entire token range lies past `new_seq_len`. The
        block that contains `new_seq_len` is kept (its tail K/V values become
        stale and will be overwritten by the next append).

        Idempotent at the current length. Raises `ValueError` if asked to
        grow, or if the truncation would land inside a published prompt block
        (the block's K/V is shared with other slots via the prefix cache; we
        can't rewrite part of it on a later append without corrupting the
        cached entry).

        Common use: speculative decoding rolls back after rejected draft
        candidates. Spec-decode only truncates within decode-time blocks
        (never published), so the published-block guard never trips in that
        flow; it's a safety net for general use.
        """
        if not 0 <= batch_idx < self.batch_size:
            raise IndexError(f"batch_idx={batch_idx} out of range for batch_size={self.batch_size}")
        current = self._num_tokens[batch_idx]
        if new_seq_len < 0:
            raise ValueError(f"new_seq_len={new_seq_len} must be non-negative")
        if new_seq_len > current:
            raise ValueError(
                f"truncate_to(new_seq_len={new_seq_len}) > current {current}; "
                "truncation only shrinks"
            )
        if new_seq_len == current:
            return

        block_size = self._pool.block_size
        published_threshold = self._slot_num_published[batch_idx] * block_size
        if new_seq_len < published_threshold:
            raise ValueError(
                f"truncate_to(new_seq_len={new_seq_len}) lands inside a published "
                f"prompt block (published_threshold={published_threshold}); refusing "
                "to avoid corrupting the cache entry's K/V"
            )

        required_blocks = (new_seq_len + block_size - 1) // block_size
        blocks_to_free = self._block_ids[batch_idx][required_blocks:]
        self._block_ids[batch_idx] = self._block_ids[batch_idx][:required_blocks]
        for block_id in blocks_to_free:
            self._pool.free(block_id)
        self._num_tokens[batch_idx] = new_seq_len

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
        self._slot_prompt_tokens.append(other._slot_prompt_tokens[0])
        self._slot_num_published.append(other._slot_num_published[0])
        self._slot_parent_hash.append(other._slot_parent_hash[0])
        self._nvfp4_shadow.append(other._nvfp4_shadow[0])
        self._nvfp4_shadow_block_id.append(other._nvfp4_shadow_block_id[0])
        other._block_ids = []
        other._num_tokens = []
        other._slot_prompt_tokens = []
        other._slot_num_published = []
        other._slot_parent_hash = []
        other._nvfp4_shadow = []
        other._nvfp4_shadow_block_id = []
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

    def block_table_padded(
        self, device: torch.device | str, dtype: torch.dtype = torch.int32
    ) -> torch.Tensor:
        """Padded `(B, max_blocks)` block-id tensor for FlashAttention's paged varlen API.

        Each row is one request's block IDs, padded with zeros to `max_blocks`.
        FA's varlen + `block_table` path uses `cache_seqlens` (from
        `seq_lens_tensor`) to know how many of each row's slots are real, so
        the zero padding is never read.
        """
        if not self._block_ids:
            return torch.zeros((0, 0), device=device, dtype=dtype)
        max_blocks = max(len(ids) for ids in self._block_ids)
        if max_blocks == 0:
            return torch.zeros((self.batch_size, 0), device=device, dtype=dtype)
        table = torch.zeros((self.batch_size, max_blocks), device=device, dtype=dtype)
        for batch_idx, ids in enumerate(self._block_ids):
            if ids:
                table[batch_idx, : len(ids)] = torch.tensor(ids, device=device, dtype=dtype)
        return table

    def seq_lens_tensor(
        self, device: torch.device | str, dtype: torch.dtype = torch.int32
    ) -> torch.Tensor:
        """Per-request seq_lens as a `(B,)` tensor on `device`."""
        return torch.tensor(self._num_tokens, device=device, dtype=dtype)

    def pool_storage_for_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(K_pool, V_pool)` slices for one layer, shape
        `(num_blocks, block_size, num_kv_heads_l, head_dim_l)` each.

        Used by FA's paged varlen path to read K/V directly from blocks.
        Compressed-mode pools don't expose direct bf16 storage; the FA
        paged path is unsupported there and the dispatcher should fall
        back to the materialized path.
        """
        if self._pool.kv_quant is not None:
            raise RuntimeError(
                "pool_storage_for_layer is uncompressed-only; "
                f"got kv_quant={self._pool.kv_quant!r}. The FA paged path "
                "doesn't support compressed K/V; use the materialized path."
            )
        return self._pool.storage_for_layer(layer_idx)

    def pool_storage_for_stream(self, layer_idx: int, stream_name: str) -> torch.Tensor:
        """Return one named stream's pool storage for a layer, shape
        `(num_blocks, block_size, num_heads_s, head_dim_s)`.

        The named-stream counterpart of `pool_storage_for_layer`, for kernels
        that read a non-legacy layout (e.g. MiniMax-M3's `k`/`v`/`index_k`
        streams) directly from blocks via the block table.
        """
        if self._pool.kv_quant is not None:
            raise RuntimeError(
                "pool_storage_for_stream is uncompressed-only; "
                f"got kv_quant={self._pool.kv_quant!r}. Kernel paged paths "
                "don't support compressed K/V; use the materialized path."
            )
        return self._pool.storage_for_stream(layer_idx, stream_name)

    def materialize_packed_kv(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Gather per-request K/V from blocks into packed (varlen) form.

        Returns `(K_packed, V_packed, cu_seqlens_k, max_seqlen_k)`:
          K_packed, V_packed : `(total_k, num_kv_heads, head_dim)` — every request's
              full K/V history concatenated along dim 0, no padding.
          cu_seqlens_k       : `(batch_size + 1,)` int32 cumulative K boundaries on
              the same device as the pool storage.
          max_seqlen_k       : longest per-request seq_len (used by FlashAttention).

        This is the packed counterpart of `_materialize`, which pads to a 4D
        rectangular tensor for HF's stock attention. The packed form is what
        `flash_attn_varlen_func` and our PyTorch reference both consume.

        Compressed pool (`kv_quant="turbo4"`): per-block read goes through
        `pool.read_compressed_block`, which decompresses + inverse-rotates
        each block back to bf16 before slicing the relevant prefix.
        """
        num_kv_heads = self._pool.num_kv_heads_for_layer(layer_idx)
        head_dim = self._pool.head_dim_for_layer(layer_idx)
        block_size = self._pool.block_size
        dtype = self._pool.dtype
        if self._pool.kv_quant is None:
            device = self._pool.storage_for_layer(layer_idx)[0].device
        elif self._pool.kv_quant == "fp8":
            assert self._pool._fp8_storage is not None
            device = self._pool._fp8_storage.device
        elif self._pool.kv_quant == "nvfp4":
            assert self._pool._nvfp4_storage is not None
            device = self._pool._nvfp4_storage.device
        else:
            assert self._pool._compressed_storage is not None
            device = self._pool._compressed_storage.device

        seq_lens = self._num_tokens
        if not seq_lens or all(seq_len == 0 for seq_len in seq_lens):
            empty_kv = torch.empty((0, num_kv_heads, head_dim), dtype=dtype, device=device)
            cu_seqlens_k = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
            return empty_kv, empty_kv, cu_seqlens_k, 0

        cu_seqlens_k_list = [0]
        running = 0
        for seq_len in seq_lens:
            running += seq_len
            cu_seqlens_k_list.append(running)
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=device)
        total_k = cu_seqlens_k_list[-1]

        key_packed = torch.empty((total_k, num_kv_heads, head_dim), dtype=dtype, device=device)
        value_packed = torch.empty_like(key_packed)

        if self._pool.kv_quant is None:
            # Uncompressed fast path: gather directly from pool storage.
            k_pool, v_pool = self._pool.storage_for_layer(layer_idx)
            for batch_idx in range(self.batch_size):
                seq_len = seq_lens[batch_idx]
                if seq_len == 0:
                    continue
                offset = cu_seqlens_k_list[batch_idx]
                positions = torch.arange(seq_len, device=device)
                block_ids_lookup = torch.tensor(self._block_ids[batch_idx], device=device)
                block_ids_per_pos = block_ids_lookup[positions // block_size]
                slots_per_pos = positions % block_size
                key_packed[offset : offset + seq_len] = k_pool[block_ids_per_pos, slots_per_pos]
                value_packed[offset : offset + seq_len] = v_pool[block_ids_per_pos, slots_per_pos]
        elif self._pool.kv_quant == "fp8":
            # FP8 path used by the CPU/MPS reference fallback (the CUDA
            # FlashInfer path skips materialize entirely). Dequant per
            # request: gather fp8 K/V, cast to fp32, multiply by per-head
            # scale, cast to the pool's bf16/fp16 dtype.
            assert self._pool._fp8_storage is not None
            assert self._pool._fp8_scales is not None
            for batch_idx in range(self.batch_size):
                seq_len = seq_lens[batch_idx]
                if seq_len == 0:
                    continue
                offset = cu_seqlens_k_list[batch_idx]
                positions = torch.arange(seq_len, device=device)
                block_ids_lookup = torch.tensor(self._block_ids[batch_idx], device=device)
                block_ids_per_pos = block_ids_lookup[positions // block_size]
                slots_per_pos = positions % block_size
                k_fp8 = self._pool._fp8_storage[layer_idx, 0, block_ids_per_pos, slots_per_pos]
                v_fp8 = self._pool._fp8_storage[layer_idx, 1, block_ids_per_pos, slots_per_pos]
                k_scale = self._pool._fp8_scales[layer_idx, 0]  # (num_kv_heads,)
                v_scale = self._pool._fp8_scales[layer_idx, 1]
                key_packed[offset : offset + seq_len] = (
                    k_fp8.to(torch.float32) * k_scale.view(1, -1, 1)
                ).to(dtype)
                value_packed[offset : offset + seq_len] = (
                    v_fp8.to(torch.float32) * v_scale.view(1, -1, 1)
                ).to(dtype)
        elif self._pool.kv_quant == "nvfp4":
            # NVFP4 path: paged storage is read directly by FlashInfer's
            # attention kernel; the bf16 materialize path is only used by
            # CPU/MPS reference fallbacks that the FlashInfer-required
            # nvfp4 mode never reaches. We raise here rather than silently
            # return zeros so a misrouted call surfaces fast.
            raise NotImplementedError(
                "materialize_packed_kv is not supported for kv_quant='nvfp4': "
                "FlashInfer reads paged FP4 storage directly via the attention "
                "kernel, and FlashInfer does not currently expose a working "
                "paged-FP4 dequant helper for the bf16 fallback path."
            )
        elif self._pool.kv_quant == "turbo3" and supports_fused_kernel(device):
            # Fused fast path: a single Triton kernel per K/V replaces the
            # per-block Python loop below. Same numerical contract within
            # bf16 round-trip noise; falls through to the Python loop on
            # non-CUDA or with `_FUSED_DISABLED_FOR_BENCH`.
            task_block_ids: list[int] = []
            task_offsets: list[int] = []
            task_valid: list[int] = []
            for batch_idx in range(self.batch_size):
                seq_len = seq_lens[batch_idx]
                if seq_len == 0:
                    continue
                offset = cu_seqlens_k_list[batch_idx]
                pos_in_packed = 0
                num_blocks_used = (seq_len + block_size - 1) // block_size
                for block_idx_in_slot in range(num_blocks_used):
                    valid_in_block = min(block_size, seq_len - block_idx_in_slot * block_size)
                    task_block_ids.append(self._block_ids[batch_idx][block_idx_in_slot])
                    task_offsets.append(offset + pos_in_packed)
                    task_valid.append(valid_in_block)
                    pos_in_packed += valid_in_block

            task_block_ids_t = torch.tensor(task_block_ids, dtype=torch.int32, device=device)
            task_offsets_t = torch.tensor(task_offsets, dtype=torch.int32, device=device)
            task_valid_t = torch.tensor(task_valid, dtype=torch.int32, device=device)

            fused_materialize_packed_kv(
                self._pool,
                layer_idx,
                task_block_ids=task_block_ids_t,
                task_offsets=task_offsets_t,
                task_valid=task_valid_t,
                key_out=key_packed,
                value_out=value_packed,
            )
        else:
            # Compressed slow path: read each block (dequant + inverse-rotate),
            # then slice the seq_len-bounded prefix into the packed output.
            # Hit on non-CUDA, missing Triton, or kv_quant=="turbo4".
            for batch_idx in range(self.batch_size):
                seq_len = seq_lens[batch_idx]
                if seq_len == 0:
                    continue
                offset = cu_seqlens_k_list[batch_idx]
                pos_in_packed = 0
                num_blocks_used = (seq_len + block_size - 1) // block_size
                for block_idx_in_slot in range(num_blocks_used):
                    block_id = self._block_ids[batch_idx][block_idx_in_slot]
                    full_block_k = self._pool.read_compressed_block(layer_idx, 0, block_id)
                    full_block_v = self._pool.read_compressed_block(layer_idx, 1, block_id)
                    # How many slots of this block contain valid data.
                    block_start = block_idx_in_slot * block_size
                    valid_in_block = min(block_size, seq_len - block_start)
                    key_packed[offset + pos_in_packed : offset + pos_in_packed + valid_in_block] = (
                        full_block_k[:valid_in_block]
                    )
                    value_packed[
                        offset + pos_in_packed : offset + pos_in_packed + valid_in_block
                    ] = full_block_v[:valid_in_block]
                    pos_in_packed += valid_in_block

        return key_packed, value_packed, cu_seqlens_k, max(seq_lens)

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
        # Publish prompt blocks that just filled, but only on the LAST layer:
        # earlier layers' K/V isn't written yet, so the cached entry would be
        # incomplete. By the last layer's append, every layer has been written.
        if self._prefix_cache is not None and layer_idx == self._pool.num_layers - 1:
            for batch_idx in range(self.batch_size):
                self._publish_filled_blocks(batch_idx)

    def append_stream_packed(
        self,
        stream_packed: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
        layer_idx: int,
        stream_name: str,
    ) -> None:
        """Generalized varlen append for one named stream of one layer.

        Counterpart to `append_kv_packed` for the stream-descriptor API.
        Writes `stream_packed` (`(total_new_tokens, num_kv_heads_s,
        head_dim_s)`) into the per-stream pool storage; on the first
        stream of layer 0, also advances per-slot token counts and
        allocates fresh blocks.

        Used by MLA models that write per-layer streams of differing
        shape (`compressed_kv` 1xkv_lora_rank, `k_rope` 1xqk_rope_head_dim)
        rather than the standard K/V pair.
        """
        if cu_seqlens_q_new.shape[0] != self.batch_size + 1:
            raise ValueError(
                f"cu_seqlens_q_new has {cu_seqlens_q_new.shape[0]} entries; "
                f"expected batch_size+1 = {self.batch_size + 1}"
            )
        if self._pool.kv_quant is not None:
            raise RuntimeError(
                f"`append_stream_packed` is uncompressed-only; got "
                f"kv_quant={self._pool.kv_quant!r}."
            )
        # Advance counters + allocate blocks once per step. Callers append
        # streams in the order returned by `pool.stream_names(0)`; the
        # first stream of layer 0 is the trigger.
        first_stream_of_layer_0 = self._pool.stream_names(0)[0]
        if layer_idx == 0 and stream_name == first_stream_of_layer_0:
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
        # Write data into per-stream storage. For legacy K/V layers this
        # aliases `_layer_storage[layer_idx][0/1]` (no extra memory).
        stream_pool = self._pool.storage_for_stream(layer_idx, stream_name)
        block_size = self._pool.block_size
        for batch_idx in range(self.batch_size):
            packed_start = int(cu_seqlens_q_new[batch_idx])
            packed_end = int(cu_seqlens_q_new[batch_idx + 1])
            new_tokens = packed_end - packed_start
            if new_tokens == 0:
                continue
            seq_len_after = self._num_tokens[batch_idx]
            seq_len_before = seq_len_after - new_tokens
            packed_slice = stream_packed[packed_start:packed_end]
            for tok_idx in range(new_tokens):
                abs_pos = seq_len_before + tok_idx
                block_id = self._block_ids[batch_idx][abs_pos // block_size]
                slot = abs_pos % block_size
                stream_pool[block_id, slot] = packed_slice[tok_idx]
        # Publish prompt blocks once the LAST stream of the LAST layer
        # has been written this step (mirrors legacy `append_kv_packed`'s
        # publish logic).
        last_layer = self._pool.num_layers - 1
        last_stream = self._pool.stream_names(last_layer)[-1]
        if (
            self._prefix_cache is not None
            and layer_idx == last_layer
            and stream_name == last_stream
        ):
            for batch_idx in range(self.batch_size):
                self._publish_filled_blocks(batch_idx)

    def materialize_packed_stream(
        self, layer_idx: int, stream_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Gather one named stream's per-slot history into varlen-packed form.

        Returns `(stream_packed, cu_seqlens_k, max_seqlen_k)` where
        `stream_packed` is `(total_k, num_kv_heads_s, head_dim_s)` —
        every request's full history of this stream concatenated along
        dim 0, no padding.

        Generalizes `materialize_packed_kv` to a single named stream.
        Uncompressed pool only; compressed paths still go through the
        legacy 2-stream `materialize_packed_kv`.
        """
        if self._pool.kv_quant is not None:
            raise RuntimeError(
                f"`materialize_packed_stream` is uncompressed-only; got "
                f"kv_quant={self._pool.kv_quant!r}."
            )
        spec = self._pool.stream_spec(layer_idx, stream_name)
        block_size = self._pool.block_size
        dtype = self._pool.dtype
        stream_pool = self._pool.storage_for_stream(layer_idx, stream_name)
        device = stream_pool.device

        seq_lens = self._num_tokens
        if not seq_lens or all(seq_len == 0 for seq_len in seq_lens):
            empty = torch.empty((0, spec.num_kv_heads, spec.head_dim), dtype=dtype, device=device)
            cu_seqlens_k = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
            return empty, cu_seqlens_k, 0

        cu_seqlens_k_list = [0]
        running = 0
        for seq_len in seq_lens:
            running += seq_len
            cu_seqlens_k_list.append(running)
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=device)
        total_k = cu_seqlens_k_list[-1]
        max_seqlen_k = max(seq_lens) if seq_lens else 0

        stream_packed = torch.empty(
            (total_k, spec.num_kv_heads, spec.head_dim), dtype=dtype, device=device
        )
        for batch_idx in range(self.batch_size):
            seq_len = seq_lens[batch_idx]
            if seq_len == 0:
                continue
            offset = cu_seqlens_k_list[batch_idx]
            positions = torch.arange(seq_len, device=device)
            block_ids_lookup = torch.tensor(self._block_ids[batch_idx], device=device)
            block_ids_per_pos = block_ids_lookup[positions // block_size]
            slots_per_pos = positions % block_size
            stream_packed[offset : offset + seq_len] = stream_pool[block_ids_per_pos, slots_per_pos]
        return stream_packed, cu_seqlens_k, max_seqlen_k

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
        self._slot_prompt_tokens.clear()
        self._slot_num_published.clear()
        self._slot_parent_hash.clear()

    def _publish_filled_blocks(self, batch_idx: int) -> None:
        """Publish any blocks of this slot's prompt that just became fully filled.

        A block is publishable when (a) it lies entirely within the slot's
        prompt and (b) its block_size'th token has been written. We track
        which blocks are already published in `_slot_num_published`.

        On a duplicate publish (another slot's prompt produced the identical
        chained hash), the prefix cache returns its canonical block id; we
        return our just-allocated block to the pool and rewrite the slot's
        block list to point at the canonical block. The K/V values in the
        canonical block are bit-equal to ours (same tokens, same model), so
        attention reads after the swap are correct.
        """
        assert self._prefix_cache is not None
        prompt = self._slot_prompt_tokens[batch_idx]
        if not prompt:
            return
        block_size = self._pool.block_size
        num_full_prompt_blocks = len(prompt) // block_size
        target_count = min(self._num_tokens[batch_idx] // block_size, num_full_prompt_blocks)
        already = self._slot_num_published[batch_idx]
        if target_count <= already:
            return

        parent_hash = self._slot_parent_hash[batch_idx]
        for block_idx_in_slot in range(already, target_count):
            token_start = block_idx_in_slot * block_size
            token_end = token_start + block_size
            tokens = tuple(prompt[token_start:token_end])
            block_hash = PrefixCache.hash_block(parent_hash, tokens)
            block_id = self._block_ids[batch_idx][block_idx_in_slot]
            canonical_id, was_dup = self._prefix_cache.publish(block_hash, tokens, block_id)
            if was_dup:
                self._pool.free(block_id)
                self._block_ids[batch_idx][block_idx_in_slot] = canonical_id
            parent_hash = block_hash
        self._slot_num_published[batch_idx] = target_count
        self._slot_parent_hash[batch_idx] = parent_hash

    def _write_packed_kv(
        self,
        layer_idx: int,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
    ) -> None:
        if self._pool.kv_quant is None:
            self._write_packed_kv_uncompressed(layer_idx, packed_k, packed_v, cu_seqlens_q_new)
        elif self._pool.kv_quant == "fp8":
            self._write_packed_kv_fp8(layer_idx, packed_k, packed_v, cu_seqlens_q_new)
        elif self._pool.kv_quant == "nvfp4":
            self._write_packed_kv_nvfp4(layer_idx, packed_k, packed_v, cu_seqlens_q_new)
        else:
            self._write_packed_kv_compressed(layer_idx, packed_k, packed_v, cu_seqlens_q_new)

    def _write_packed_kv_uncompressed(
        self,
        layer_idx: int,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
    ) -> None:
        block_size = self._pool.block_size
        k_pool, v_pool = self._pool.storage_for_layer(layer_idx)
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
                k_pool[block_id, slot_in_block] = packed_k[packed_start + token_offset]
                v_pool[block_id, slot_in_block] = packed_v[packed_start + token_offset]

    def _write_packed_kv_fp8(
        self,
        layer_idx: int,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
    ) -> None:
        """FP8 e4m3fn write path: per-tensor scale set on first append,
        then used to quantize new K/V tokens before storing in the paged
        cache.

        FlashInfer's `run()` takes `k_scale` / `v_scale` as Python
        floats (per-tensor), not per-head tensors, so the quant side
        also has to use a single scalar per layer per side — otherwise
        the dequant during attention won't round-trip. We compute the
        scalar from the abs-max of the first batch's K (or V) and fill
        every head slot with it (the per-head storage layout is kept
        for forward compatibility with a future per-head FlashInfer
        path).
        """
        pool = self._pool
        assert pool._fp8_storage is not None
        assert pool._fp8_scales is not None
        assert pool._fp8_scales_initialized is not None

        block_size = pool.block_size
        fp8_max = 448.0  # max representable value for float8_e4m3fn

        for kv_idx, packed in ((0, packed_k), (1, packed_v)):
            if not bool(pool._fp8_scales_initialized[layer_idx, kv_idx]):
                abs_max = float(packed.abs().max().to(torch.float32).item())
                scalar = max(abs_max, 1e-6) / fp8_max
                pool._fp8_scales[layer_idx, kv_idx].fill_(scalar)
                pool._fp8_scales_initialized[layer_idx, kv_idx] = True

            scale = float(pool._fp8_scales[layer_idx, kv_idx, 0].item())
            quantized = (
                (packed.to(torch.float32) / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            )

            # Same per-token write as the uncompressed path; the only
            # difference is the dtype of the storage tensor.
            for batch_idx in range(self.batch_size):
                packed_start = int(cu_seqlens_q_new[batch_idx])
                packed_end = int(cu_seqlens_q_new[batch_idx + 1])
                new_tokens = packed_end - packed_start
                if new_tokens == 0:
                    continue
                start_token = self._num_tokens[batch_idx] - new_tokens
                for token_offset in range(new_tokens):
                    position = start_token + token_offset
                    block_id = self._block_ids[batch_idx][position // block_size]
                    slot_in_block = position % block_size
                    pool._fp8_storage[layer_idx, kv_idx, block_id, slot_in_block] = quantized[
                        packed_start + token_offset
                    ]

    def _write_packed_kv_nvfp4(
        self,
        layer_idx: int,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
    ) -> None:
        """NVFP4 write path via FlashInfer's paged quantizer.

        Mirrors the FP8 path's structure but with a per-slot bf16
        "shadow" of the active tail page. FlashInfer's
        `nvfp4_quantize_paged_kv_cache` quantizes a whole
        `(num_pages, block_size, num_kv_heads, head_dim)` bf16 input
        and produces FP4-packed `uint8` + per-page FP8 scales the
        wrapper consumes via `kv_cache_sf`. Because the API only
        operates on whole pages, we re-quantize each affected page
        on every step; the shadow holds the bf16 K/V of the
        currently-being-filled page so prior tokens of an in-flight
        page are available alongside this step's new tokens.

        First call per (layer, side) auto-computes the global scale
        from the batch's amax; we cache it on the pool and reuse on
        every subsequent call so re-quantizing a partially-filled
        page over multiple decode steps stays self-consistent.
        """
        from flashinfer.fp4_quantization import (  # type: ignore[import-not-found]
            nvfp4_quantize_paged_kv_cache,
        )

        pool = self._pool
        assert pool._nvfp4_storage is not None
        assert pool._nvfp4_block_scales is not None
        assert pool._nvfp4_global_sf is not None
        assert pool._nvfp4_initialized is not None

        block_size = pool.block_size
        num_kv_heads = pool.num_kv_heads
        head_dim = pool.head_dim
        device = pool._nvfp4_storage.device
        bf16_dtype = pool.dtype

        # Group writes by (batch_idx, block_id) so each affected page
        # is touched exactly once.
        writes: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for batch_idx in range(self.batch_size):
            packed_start = int(cu_seqlens_q_new[batch_idx])
            packed_end = int(cu_seqlens_q_new[batch_idx + 1])
            new_tokens = packed_end - packed_start
            if new_tokens == 0:
                continue
            start_token = self._num_tokens[batch_idx] - new_tokens
            for token_offset in range(new_tokens):
                position = start_token + token_offset
                block_id = self._block_ids[batch_idx][position // block_size]
                slot_in_block = position % block_size
                packed_idx = packed_start + token_offset
                writes.setdefault((batch_idx, block_id), []).append((slot_in_block, packed_idx))

        if not writes:
            return

        # Lazy-allocate the bf16 shadow for each slot that has writes.
        for batch_idx in {b for (b, _) in writes}:
            if self._nvfp4_shadow[batch_idx] is None:
                self._nvfp4_shadow[batch_idx] = torch.zeros(
                    pool.num_layers,
                    2,
                    block_size,
                    num_kv_heads,
                    head_dim,
                    dtype=bf16_dtype,
                    device=device,
                )

        affected = list(writes.keys())
        n_pages = len(affected)
        k_pages = torch.zeros(
            n_pages, block_size, num_kv_heads, head_dim, dtype=bf16_dtype, device=device
        )
        v_pages = torch.zeros_like(k_pages)
        for i, (batch_idx, block_id) in enumerate(affected):
            shadow = self._nvfp4_shadow[batch_idx]
            assert shadow is not None
            if self._nvfp4_shadow_block_id[batch_idx] == block_id:
                k_pages[i] = shadow[layer_idx, 0]
                v_pages[i] = shadow[layer_idx, 1]
            for slot_in_block, packed_idx in writes[(batch_idx, block_id)]:
                k_pages[i, slot_in_block] = packed_k[packed_idx]
                v_pages[i, slot_in_block] = packed_v[packed_idx]

        # Explicit global_sf: on first call per (layer, side), set from
        # this batch's amax using the textbook NVFP4 formula
        # `global_sf = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax = (448 * 6) /
        # amax`. This matches FlashInfer's reference helper
        # (`ref_fp4_quant` in `tests/test_helpers/utils_fp4.py`) and gives
        # full per-block-scale dynamic range (the largest block scale
        # lands exactly at the FP8 e4m3 saturation point of 448). Reusing
        # the same value on subsequent calls keeps decode-step
        # quantization consistent with the prefill.
        fp8_max_x_fp4_max = 448.0 * 6.0
        if not bool(pool._nvfp4_initialized[layer_idx, 0]):
            amax_k = max(float(k_pages.float().abs().max().item()), 1e-6)
            pool._nvfp4_global_sf[layer_idx, 0] = fp8_max_x_fp4_max / amax_k
            pool._nvfp4_initialized[layer_idx, 0] = True
        if not bool(pool._nvfp4_initialized[layer_idx, 1]):
            amax_v = max(float(v_pages.float().abs().max().item()), 1e-6)
            pool._nvfp4_global_sf[layer_idx, 1] = fp8_max_x_fp4_max / amax_v
            pool._nvfp4_initialized[layer_idx, 1] = True

        k_global_sf_arg = pool._nvfp4_global_sf[layer_idx, 0:1]
        v_global_sf_arg = pool._nvfp4_global_sf[layer_idx, 1:2]

        (k_packed, v_packed), (k_sf, v_sf), _k_scale_f, _v_scale_f = nvfp4_quantize_paged_kv_cache(
            k_pages,
            v_pages,
            kv_layout="NHD",
            k_global_sf=k_global_sf_arg,
            v_global_sf=v_global_sf_arg,
        )

        # Scatter per-page output into paged storage. FlashInfer
        # returns block scales as uint8 bytes; reinterpret as the
        # FP8 e4m3 storage dtype before assignment (byte width
        # matches; the wrapper reads them as FP8 on the dequant path).
        k_sf_e4m3 = k_sf.view(torch.float8_e4m3fn)
        v_sf_e4m3 = v_sf.view(torch.float8_e4m3fn)
        for i, (_, block_id) in enumerate(affected):
            pool._nvfp4_storage[layer_idx, 0, block_id] = k_packed[i]
            pool._nvfp4_storage[layer_idx, 1, block_id] = v_packed[i]
            pool._nvfp4_block_scales[layer_idx, 0, block_id] = k_sf_e4m3[i]
            pool._nvfp4_block_scales[layer_idx, 1, block_id] = v_sf_e4m3[i]

        # Update the shadow so it tracks each slot's new active block.
        for batch_idx in {b for (b, _) in writes}:
            if not self._block_ids[batch_idx]:
                continue
            active_block_id = self._block_ids[batch_idx][-1]
            shadow = self._nvfp4_shadow[batch_idx]
            assert shadow is not None
            if self._nvfp4_shadow_block_id[batch_idx] != active_block_id:
                shadow.zero_()
                self._nvfp4_shadow_block_id[batch_idx] = active_block_id
            key = (batch_idx, active_block_id)
            if key in writes:
                for slot_in_block, packed_idx in writes[key]:
                    shadow[layer_idx, 0, slot_in_block] = packed_k[packed_idx]
                    shadow[layer_idx, 1, slot_in_block] = packed_v[packed_idx]

    def _write_packed_kv_compressed(
        self,
        layer_idx: int,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        cu_seqlens_q_new: torch.Tensor,
    ) -> None:
        """Compressed-mode write: per-block dequant -> patch new tokens -> requantize.

        Quantization is per-block (block_size tokens share scales), so we
        can't write a single token without re-quantizing the affected
        block. V1 ships this slow-but-correct path; a Triton kernel that
        fuses partial-block updates is a follow-up.

        For each affected block we:
          1. Dequantize + inverse-rotate the existing block to bf16. Slots
             past the slot's seq_len are zeros (block was zero-initialized
             at allocation), which keeps the per-channel range tight.
          2. Patch in the new K/V tokens at their slot positions.
          3. Re-rotate + re-quantize + write back.
        """
        block_size = self._pool.block_size
        for batch_idx in range(self.batch_size):
            packed_start = int(cu_seqlens_q_new[batch_idx])
            packed_end = int(cu_seqlens_q_new[batch_idx + 1])
            new_tokens = packed_end - packed_start
            if new_tokens == 0:
                continue
            start_token = self._num_tokens[batch_idx] - new_tokens
            end_token = start_token + new_tokens

            first_block_idx = start_token // block_size
            last_block_idx = (end_token - 1) // block_size

            for block_idx_in_slot in range(first_block_idx, last_block_idx + 1):
                block_id = self._block_ids[batch_idx][block_idx_in_slot]
                block_token_start = block_idx_in_slot * block_size
                block_token_end = block_token_start + block_size

                full_block_k = self._pool.read_compressed_block(layer_idx, 0, block_id)
                full_block_v = self._pool.read_compressed_block(layer_idx, 1, block_id)

                write_token_start = max(start_token, block_token_start)
                write_token_end = min(end_token, block_token_end)
                for pos in range(write_token_start, write_token_end):
                    slot = pos - block_token_start
                    packed_idx = packed_start + (pos - start_token)
                    full_block_k[slot] = packed_k[packed_idx]
                    full_block_v[slot] = packed_v[packed_idx]

                self._pool.write_compressed_block(layer_idx, 0, block_id, full_block_k)
                self._pool.write_compressed_block(layer_idx, 1, block_id, full_block_v)

    def _materialize(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B, num_kv_heads, max_seq_len, head_dim), zero-padded for shorter requests.

        HF `Cache.update()` compatibility path, used only by the legacy
        single-request prefill flow. Compressed-mode pools must reach
        attention via the packed-varlen path (`materialize_packed_kv`);
        this method raises rather than allocate the bf16 fallback.
        """
        if self._pool.kv_quant is not None:
            raise RuntimeError(
                "_materialize (HF Cache.update path) is uncompressed-only; "
                f"got kv_quant={self._pool.kv_quant!r}. Compressed pools should "
                "reach attention via the packed-varlen forward path."
            )
        k_pool, v_pool = self._pool.storage_for_layer(layer_idx)
        num_kv_heads = self._pool.num_kv_heads_for_layer(layer_idx)
        head_dim = self._pool.head_dim_for_layer(layer_idx)
        device = k_pool.device
        dtype = k_pool.dtype

        if self.batch_size == 0:
            shape = (0, num_kv_heads, 0, head_dim)
            empty = torch.empty(shape, dtype=dtype, device=device)
            return empty, empty

        max_seq = max(self._num_tokens)
        if max_seq == 0:
            shape = (self.batch_size, num_kv_heads, 0, head_dim)
            empty = torch.empty(shape, dtype=dtype, device=device)
            return empty, empty

        block_size = self._pool.block_size

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
            keys_gathered = k_pool[block_ids_per_pos, slots_per_pos]
            values_gathered = v_pool[block_ids_per_pos, slots_per_pos]
            # Reorder to HF's (num_kv_heads, seq_len, head_dim) and copy into the
            # padded (num_kv_heads, max_seq, head_dim) slot for this request.
            keys_in_hf_layout = keys_gathered.permute(1, 0, 2)
            values_in_hf_layout = values_gathered.permute(1, 0, 2)
            key_full[batch_idx, :, :seq_len, :] = keys_in_hf_layout
            value_full[batch_idx, :, :seq_len, :] = values_in_hf_layout

        return key_full, value_full
