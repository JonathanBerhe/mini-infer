"""MiniMax Sparse Attention (MSA) block indexer (MiniMax-M3, arXiv 2606.13392).

The block-level cousin of V4's token-level `LightningIndexer`. A small scoring
branch picks which 128-token KV *blocks* each query attends to; the main GQA
attention then runs over only the selected blocks (top-16 + the local block).

Mechanism (faithful to HF `MiniMaxM3VLIndexer`, transformers `minimax_m3_vl`).
All comments are the load-bearing details:

    # per-head Gemma RMSNorm pre-RoPE; k is one shared head:
    idx_q = q_norm(q_proj(x).view(B, S, n_idx_heads, d)).transpose(1, 2)
    idx_k = k_norm(k_proj(x).view(B, S, 1, d)).transpose(1, 2)
    idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos, sin)  # full RoPE over head_dim
    scores = idx_q.float() @ idx_k.float().mT               # [B, heads, S, k_len]; NO /sqrt(d)
    scores = scores.masked_fill(k_pos > q_pos, -inf)        # causal, TOKEN granularity, BEFORE pool
    scores = pad(scores, last block up to block_size, -inf)
    # pool over the block's tokens (-1) AND the index heads (1) -> one score/query/block:
    block_scores = scores.view(B, n_idx_heads, S, n_blocks, block_size).amax(-1).amax(1)
    block_scores.scatter_(-1, local_blocks, +inf)           # force-include the query's own block(s)
    top_scores, top_idx = block_scores.topk(min(topk_blocks, n_blocks), -1)
    block_indices = top_idx.masked_fill(top_scores == -inf, -1)   # [B, S, topk]; -1 = padding

Two properties that took reading the source to pin down (the paper and an earlier
spec draft had these wrong):
  - The index scores have NO `1/sqrt(d)` scale (raw fp32 dot).
  - `block_scores` maxes over the block tokens AND the index heads, so the
    selection is a SINGLE set per query, shared across every main attention head
    (not one selection per GQA group). The mask is therefore `[B, 1, S, k_len]`.

`build_block_mask` turns the selection into the additive attention mask the main
attention adds before softmax; it folds the causal mask in (it REPLACES, not
augments, the causal mask). Reimplemented against HF's eager/sdpa path, which
builds this dense mask rather than gathering the selected blocks, so a
from-scratch port matches it bit-for-bit.

This module is the prefill / self-attention form (K is the current sequence).
The cached-decode form (score the new query against the full cached index-K
history) and tensor-parallel index-head sharding are wired in the model /
serving phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.models.blocks.gemma_rmsnorm import GemmaRMSNorm
from mini_infer.models.blocks.rope import apply_rotary_pos_emb

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class MiniMaxM3Indexer(nn.Module):
    """Block-level top-k selector for MSA. Returns per-query selected block ids."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        topk_blocks: int,
        local_blocks: int = 1,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if topk_blocks <= 0:
            raise ValueError(f"topk_blocks must be positive, got {topk_blocks}")
        if local_blocks < 0:
            raise ValueError(f"local_blocks must be non-negative, got {local_blocks}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.local_blocks = local_blocks
        # index_q_proj: (hidden -> num_heads * head_dim); index_k_proj: one shared
        # key head (hidden -> head_dim). Names match HF `index_{q,k}_proj`.
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        # Per-head Gemma (1+w) RMSNorm on q/k, applied on the (..., head_dim) view
        # BEFORE transpose/RoPE.
        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pick top-k block ids per query. Prefill form (K = current sequence).

        Args:
            hidden_states: `(B, S, hidden_size)`.
            cos, sin: RoPE tables, full width `head_dim`, shape `(B, S, head_dim)`.
            position_ids: `(B, S)` absolute positions.

        Returns:
            `(B, S, topk)` int64 block ids; `-1` marks a padding slot (fewer valid
            blocks than `topk`, i.e. an all-future / masked block).
        """
        bsz, seqlen, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim

        idx_q = self.q_norm(self.q_proj(hidden_states).view(bsz, seqlen, h, d)).transpose(1, 2)
        idx_k = self.k_norm(self.k_proj(hidden_states).view(bsz, seqlen, 1, d)).transpose(1, 2)
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos, sin)

        # [B, h, S, k_len], raw fp32 dot (NO 1/sqrt(d)). idx_k's single head
        # broadcasts across the h index heads.
        scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
        return self._select_blocks(scores, position_ids, idx_k.shape[2])

    def _select_blocks(
        self, scores: torch.Tensor, position_ids: torch.Tensor, k_len: int
    ) -> torch.Tensor:
        """Raw `(B, h, q, k_len)` scores -> `(B, q, topk)` block ids (-1 padded).

        Applies token-granularity causal masking, pads the last block, max-pools
        over the block's tokens AND the index heads (one selection per query,
        shared across all main heads), force-includes the local block(s), and
        takes the top-k. Shared by `forward` (batched prefill) and
        `forward_cached` (packed-varlen decode/prefill over the cache).
        """
        bsz, h, q, _ = scores.shape
        device = scores.device
        num_blocks = -(-k_len // self.block_size)  # ceil
        pad = num_blocks * self.block_size - k_len

        k_positions = torch.arange(k_len, device=device)
        token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
        scores = scores.masked_fill(token_future, float("-inf"))
        if pad:
            scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
        block_scores = scores.view(bsz, h, q, num_blocks, self.block_size).amax(-1).amax(1)

        if self.local_blocks > 0:
            q_block = position_ids // self.block_size
            local = torch.arange(self.local_blocks, device=device)
            local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
            block_scores.scatter_(-1, local_idx, float("inf"))

        topk = min(self.topk_blocks, num_blocks)
        top_scores, top_idx = block_scores.topk(topk, dim=-1)
        return top_idx.masked_fill(top_scores == float("-inf"), -1)

    def build_block_mask(
        self,
        block_indices: torch.Tensor,
        key_length: int,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Selected blocks -> additive attention bias `(B, 1, S, key_length)`.

        A key is kept (bias 0) iff its block was selected for the query AND it is
        causally visible (`key_pos <= query_pos`); every other key gets
        `finfo(dtype).min`. Folds the causal mask in (this REPLACES the causal
        mask in the main attention). Head dim is 1 (broadcast over all heads,
        since the selection is global per query).
        """
        device = block_indices.device
        bsz, seqlen, _ = block_indices.shape
        num_key_blocks = -(-key_length // self.block_size)  # ceil

        # -1 slots -> a throwaway column at index num_key_blocks; scatter 0 at the
        # real selected blocks, then drop the throwaway column.
        safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)
        bias = block_indices.new_full((bsz, seqlen, num_key_blocks + 1), float("-inf"), dtype=dtype)
        bias.scatter_(-1, safe, 0.0)
        bias = bias[..., :num_key_blocks]

        # Expand each kept block to its block_size key slots; add the head axis.
        block_keep = (
            (bias == 0.0).repeat_interleave(self.block_size, dim=-1)[..., :key_length].unsqueeze(1)
        )  # (B, 1, S, key_length) bool
        k_positions = torch.arange(key_length, device=device)
        token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
        keep = block_keep & ~token_future
        min_dtype = torch.finfo(dtype).min
        return torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(~keep, min_dtype)

    def select_cached(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        layer_idx: int,
        *,
        stream_name: str = "index_k",
    ) -> list[tuple[torch.Tensor, int]]:
        """Packed-varlen, cache-aware selection -> per-request selected block ids.

        Appends this step's index-K to `stream_name`, reads back each request's
        full index-K history, scores its queries against that history, and selects
        the top-k blocks. Returns one `((q_len_r, topk) int64 ids, k_len_r)` pair
        per request; `-1` marks a padding slot. Prefill (`q_len == k_len`) and
        decode (`q_len == 1`, `k_len == context`) run the same code; they differ
        only in the materialized per-request `k_len`. Absolute query positions are
        derived from the cache lengths (`[k_len - q_len, k_len)`), matching
        `GlmDsaIndexer.forward_cached` and the packed-varlen convention.

        Single source of truth for cache-aware selection: `forward_cached` builds
        dense masks from these ids (the torch oracle path) and the block-sparse
        decode kernel consumes them directly, so both paths share one selection.
        """
        _, total_q, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim
        device = hidden_states.device

        idx_q = self.q_norm(self.q_proj(hidden_states).view(1, total_q, h, d)).transpose(1, 2)
        idx_k = self.k_norm(self.k_proj(hidden_states).view(1, total_q, 1, d)).transpose(1, 2)
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos, sin)

        # Append this step's index-K (packed (total_q, 1, d)); read back full history.
        idx_k_packed = idx_k[0].transpose(0, 1).contiguous()  # (total_q, 1, d)
        past_key_values.append_stream_packed(idx_k_packed, cu_seqlens_q, layer_idx, stream_name)
        idx_k_full, cu_seqlens_k, _ = past_key_values.materialize_packed_stream(
            layer_idx, stream_name
        )  # (total_k, 1, d)

        # One host sync each (vs per-request int() casts in the loop; the
        # cu tensors live on the pool device during decode).
        cu_q = cu_seqlens_q.tolist()
        cu_k = cu_seqlens_k.tolist()
        selections: list[tuple[torch.Tensor, int]] = []
        for r in range(len(cu_q) - 1):
            q_s, q_e = cu_q[r], cu_q[r + 1]
            k_s, k_e = cu_k[r], cu_k[r + 1]
            q_len_r, k_len_r = q_e - q_s, k_e - k_s
            if q_len_r == 0:
                selections.append((torch.empty(0, 0, dtype=torch.int64, device=device), k_len_r))
                continue
            q_r = idx_q[0, :, q_s:q_e, :]  # (h, q_len, d)
            k_r = idx_k_full[k_s:k_e, 0, :]  # (k_len, d)
            # queries occupy absolute positions [k_len - q_len, k_len).
            pos_r = (torch.arange(q_len_r, device=device) + (k_len_r - q_len_r)).unsqueeze(0)
            scores = torch.einsum("hqd,kd->hqk", q_r.float(), k_r.float()).unsqueeze(0)
            block_indices = self._select_blocks(scores, pos_r, k_len_r)  # (1, q_len, topk)
            selections.append((block_indices[0], k_len_r))
        return selections

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        layer_idx: int,
        *,
        dtype: torch.dtype,
        stream_name: str = "index_k",
    ) -> list[torch.Tensor]:
        """Cache-aware selection -> one additive block mask per request.

        `select_cached` picks the blocks; this expands each request's selection
        into the `(q_len_r, 1, k_len_r)` additive mask (`packed_attention`'s
        `block_mask` format). The torch-oracle path of the main attention.
        """
        device = hidden_states.device
        masks: list[torch.Tensor] = []
        for block_indices, k_len_r in self.select_cached(
            hidden_states,
            cos,
            sin,
            past_key_values,
            cu_seqlens_q,
            layer_idx,
            stream_name=stream_name,
        ):
            q_len_r = block_indices.shape[0]
            if q_len_r == 0:
                masks.append(torch.zeros(0, 1, k_len_r, dtype=dtype, device=device))
                continue
            pos_r = (torch.arange(q_len_r, device=device) + (k_len_r - q_len_r)).unsqueeze(0)
            mask_full = self.build_block_mask(
                block_indices.unsqueeze(0), k_len_r, pos_r, dtype=dtype
            )
            masks.append(mask_full[0, 0].unsqueeze(1))  # (q_len, 1, k_len)
        return masks
