"""Grouped-Query Attention with native packed-varlen path.

This block replaces the HF-attention monkey-patch we used to apply at
runtime. The forward computes Q/K/V projections, applies RoPE,
appends new K/V to the paged cache, dispatches the packed varlen
attention kernel, and applies the output projection. One layer
forward = one call to `packed_attention_forward`.

Parameter names (`q_proj`, `k_proj`, `v_proj`, `o_proj`) align with HF
Llama/Qwen2 so weight loading via `state_dict` is identity-rename.

Tensor parallelism
------------------
Q/K/V projections are column-parallel along the head axis (each rank
holds a contiguous slice of heads). The output projection is
row-parallel along its input. This is the standard Megatron pairing
for self-attention: one all-reduce per block, none in between. The
softmax / SDPA kernel itself operates per-head and so is unchanged
under TP. At `world_size=1` both wrappers reduce to plain `nn.Linear`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.cache.packed_attention import packed_attention_forward
from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mini_infer.models.blocks.rope import apply_rotary_pos_emb

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_bias: bool,
        layer_idx: int,
        query_scale: float | None = None,
        q_norm: nn.Module | None = None,
        k_norm: nn.Module | None = None,
        attention_k_eq_v: bool = False,
        v_norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.attention_k_eq_v = attention_k_eq_v
        # Q/K/V are column-parallel: each rank gets `num_*_heads // world_size`
        # heads. Both `num_q_heads` and `num_kv_heads` therefore must divide
        # `world_size`; ColumnParallelLinear validates that internally
        # (because it sees `num_*_heads * head_dim` as the output dim and
        # each head is `head_dim` columns).
        self.q_proj = ColumnParallelLinear(hidden_size, num_q_heads * head_dim, bias=qkv_bias)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        # Gemma 4 full layers (`attention_k_eq_v=True`) reuse the post-`k_proj`
        # tensor as V — there is no separate v_proj parameter. We keep the
        # attribute as `None` so weight-load filters can detect the absence.
        if attention_k_eq_v:
            self.v_proj: ColumnParallelLinear | None = None
        else:
            self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        # Row-parallel output projection: input dim is the sharded
        # head-merged activation, output is the replicated hidden state.
        # Triggers exactly one all-reduce per attention block.
        self.o_proj = RowParallelLinear(num_q_heads * head_dim, hidden_size, bias=False)
        # Optional per-head norm on Q and K AFTER projection and BEFORE RoPE.
        # Gemma 3+ uses GemmaRMSNorm(head_dim) here; the names `q_norm` and
        # `k_norm` align with HF's parameter naming so weight loading is
        # identity rename. None = pass-through (Qwen2, Llama).
        self.q_norm = q_norm
        self.k_norm = k_norm
        # Gemma 4 also normalizes V with an unscaled RMSNorm AFTER capturing
        # V from k_proj output (or v_proj output). None = pass-through.
        self.v_norm = v_norm
        # `query_scale` overrides the default `1/sqrt(head_dim)` softmax
        # scale. Gemma 3 uses `1/sqrt(query_pre_attn_scalar)`. Gemma 4 sets
        # this to 1.0 (q_norm/k_norm absorb the magnitude). `None` keeps the
        # default (Qwen2, Llama).
        self._query_scale = query_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (1, total_q, hidden). The leading "1" is the engine's
        # packed-batch convention; per-request boundaries live in cu_seqlens_q.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # (1, total_q, num_*_heads, head_dim) -> (1, num_*_heads, total_q, head_dim)
        # because apply_rotary_pos_emb expects head dim at index 1 with unsqueeze_dim=1.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # Gemma 4 full layers: V reuses the post-projection K tensor BEFORE
        # k_norm and BEFORE RoPE. Capturing here keeps `value_states`
        # pointing at the un-normalized k_proj output even after the
        # `key_states = self.k_norm(...)` reassignment below (Python rebinds
        # the name; the original tensor stays alive via `value_states`).
        if self.v_proj is not None:
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        else:
            value_states = key_states

        # Per-head Q/K norm (Gemma 3+). Acts on the last dim (head_dim).
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # v_norm is applied AFTER V is captured (so for k_eq_v layers, it
        # operates on the un-k_normed, un-RoPE'd K tensor). Gemma 4's v_norm
        # uses `with_scale=False` so it's a pure RMS rescale.
        if self.v_norm is not None:
            value_states = self.v_norm(value_states)

        # Pack to (total_q, num_*_heads, head_dim) for both the cache append and
        # the varlen attention call.
        new_keys_packed = key_states.transpose(1, 2).squeeze(0).contiguous()
        new_values_packed = value_states.transpose(1, 2).squeeze(0).contiguous()
        queries_packed = query_states.transpose(1, 2).squeeze(0).contiguous()

        past_key_values.append_kv_packed(
            new_keys_packed, new_values_packed, cu_seqlens_q, self.layer_idx
        )

        attn_packed = packed_attention_forward(
            queries_packed,
            past_key_values,
            self.layer_idx,
            cu_seqlens_q,
            softmax_scale=self._query_scale,
        )
        # attn_packed: (total_q, num_q_heads, head_dim).

        attn_output = attn_packed.unsqueeze(0).reshape(*input_shape, -1).contiguous()
        out: torch.Tensor = self.o_proj(attn_output)
        return out
