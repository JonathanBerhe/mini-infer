"""Standard transformer decoder block: norm -> attn -> residual -> norm -> ffn -> residual.

Used by every Llama-shape model (Qwen2, Llama, Mistral, ...). HF-aligned
attribute names (`input_layernorm`, `self_attn`, `post_attention_layernorm`,
`mlp`) so weight loading is identity-rename.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.models.blocks.gqa import GroupedQueryAttention
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.swiglu import SwiGLU

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        layer_idx: int,
        with_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Qwen3 (and similar) apply per-head RMSNorm on Q and K after the
        # projections, before RoPE. Standard RMSNorm here (not Gemma's
        # offset variant — those models use `GemmaDecoderLayer` instead).
        q_norm = RMSNorm(head_dim, eps=rms_norm_eps) if with_qk_norm else None
        k_norm = RMSNorm(head_dim, eps=rms_norm_eps) if with_qk_norm else None
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            layer_idx=layer_idx,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

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
        x = self.mlp(x)
        out: torch.Tensor = residual + x
        return out
