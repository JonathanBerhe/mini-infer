"""Gemma 4 decoder layer: per-layer-type heterogeneous attention + layer_scalar.

Differs from `GemmaDecoderLayer` (Gemma 3) in three ways that matter:

  1. Norms use plain `RMSNorm` (init=ones, scale = `x * weight`), NOT
     `GemmaRMSNorm` (init=zeros, scale = `(1+weight) * x`). Gemma 4
     ships `Gemma4RMSNorm` derived from `Gemma3nRMSNorm` which uses
     standard semantics, so the weight values on disk are close to 1.
  2. The attention shape is per-layer-type. Sliding layers use
     `(num_kv_heads=16, head_dim=256)` with separate v_proj. Full
     ("global") layers use `(num_kv_heads=4, head_dim=512)` with
     `attention_k_eq_v=True` (no v_proj). Both layer-types carry a
     per-head `q_norm`, `k_norm`, and `v_norm` (the latter unscaled).
  3. The forward applies a learnable per-layer scalar (`layer_scalar`,
     shape `(1,)`, init=1.0) to the residual output — `hidden_states *=
     layer_scalar`. The checkpoint ships this scalar so we register
     it as a buffer and let `load_state_dict` populate it.

Softmax scale is hard-coded to 1.0 in HF's Gemma4TextAttention; the
q_norm/k_norm absorb the magnitude that would otherwise be split across
`1/sqrt(head_dim)`. This is a deliberate Gemma 4 design choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.models.blocks.geglu import GeGLU
from mini_infer.models.blocks.gqa import GroupedQueryAttention
from mini_infer.models.blocks.rmsnorm import RMSNorm

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class Gemma4DecoderLayer(nn.Module):
    """One Gemma 4 transformer block, sandwich-normed and layer-scaled.

    The constructor takes the resolved per-layer attention shape so the
    caller (`Gemma4ForCausalLM`) decides per `layer_types[i]` which
    `(num_kv_heads, head_dim, attention_k_eq_v)` triple to pass.
    """

    layer_scalar: torch.Tensor

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
        attention_k_eq_v: bool,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Per-head Q/K/V norms. q_norm and k_norm are scaled (init=ones);
        # v_norm has no learnable weight (`with_scale=False`).
        q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        v_norm = RMSNorm(head_dim, eps=rms_norm_eps, with_scale=False)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            layer_idx=layer_idx,
            # Gemma 4 sets attention scaling to 1.0 (the q/k/v norms
            # absorb the magnitude that other models put in 1/sqrt(d_h)).
            query_scale=1.0,
            q_norm=q_norm,
            k_norm=k_norm,
            attention_k_eq_v=attention_k_eq_v,
            v_norm=v_norm,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = GeGLU(hidden_size, intermediate_size)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Per-layer learnable scalar applied at the end of the block. The
        # checkpoint ships this as a buffer of shape (1,); HF registers it
        # the same way (`register_buffer("layer_scalar", torch.ones(1))`).
        self.register_buffer("layer_scalar", torch.ones(1))

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
        hidden_states = residual + x

        # Per-layer scalar gates the block output. Init=1.0 → identity.
        out: torch.Tensor = hidden_states * self.layer_scalar
        return out
