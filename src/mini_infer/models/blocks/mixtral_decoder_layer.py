"""Mixtral decoder layer: same structure as Llama, FFN replaced by MoE.

Identical to `TransformerBlock` in everything except the FFN attribute
name (`block_sparse_moe` instead of `mlp`) and its body (`MoEFFN`
instead of `SwiGLU`). HF Mixtral safetensors store the expert weights
under `block_sparse_moe.*`, so matching that path keeps weight loading
as identity rename.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.models.blocks.gqa import GroupedQueryAttention
from mini_infer.models.blocks.mixtral_moe import MoEFFN
from mini_infer.models.blocks.rmsnorm import RMSNorm

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class MixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        rms_norm_eps: float,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.block_sparse_moe = MoEFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, position_embeddings, past_key_values, cu_seqlens_q)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.block_sparse_moe(x)
        out: torch.Tensor = residual + x
        return out
