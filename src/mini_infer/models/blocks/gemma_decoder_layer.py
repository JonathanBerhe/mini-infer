"""Gemma family decoder layer with the sandwich-norm pattern.

Same shape as `TransformerBlock` (norm -> attention -> norm -> FFN) but
with FOUR norms per layer instead of two: a pre+post norm around
attention AND a pre+post norm around the FFN. HF's per-norm parameter
names (`input_layernorm`, `post_attention_layernorm`,
`pre_feedforward_layernorm`, `post_feedforward_layernorm`) match Gemma's
checkpoint convention so weight loading is identity rename.

Used by Gemma 3 and Gemma 4. Gemma 2 also uses this pattern but adds an
attention-logit softcap that we don't currently implement (see Phase 3
plan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.models.blocks.geglu import GeGLU
from mini_infer.models.blocks.gemma_rmsnorm import GemmaRMSNorm
from mini_infer.models.blocks.gqa import GroupedQueryAttention

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class GemmaDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        layer_idx: int,
        query_scale: float | None = None,
        with_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        # Gemma 3 / Gemma 4 apply per-head RMSNorm on Q and K after the
        # projections, before RoPE. The norm operates on the last dim
        # (head_dim). Gemma 2 doesn't have these, so allow opt-out.
        q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps) if with_qk_norm else None
        k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps) if with_qk_norm else None
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            layer_idx=layer_idx,
            query_scale=query_scale,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = GeGLU(hidden_size, intermediate_size)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        # Sandwich norm around attention.
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, position_embeddings, past_key_values, cu_seqlens_q)
        x = self.post_attention_layernorm(x)
        hidden_states = residual + x

        # Sandwich norm around FFN.
        residual = hidden_states
        x = self.pre_feedforward_layernorm(hidden_states)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        out: torch.Tensor = residual + x
        return out
