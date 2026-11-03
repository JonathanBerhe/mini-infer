"""Standard transformer decoder block: norm -> attn -> residual -> norm -> ffn -> residual.

Used by every Llama-shape model (Qwen2, Llama, Mistral, ...). HF-aligned
attribute names (`input_layernorm`, `self_attn`, `post_attention_layernorm`,
`mlp`) so weight loading is identity-rename.

`run_decoder_layers` runs a stack of these blocks with every residual add
folded into the norm after it: each block returns its FFN output and its
residual without the last add, and the next block's input norm, or the final
norm after the last block, adds the two and normalizes the sum in one call.
The sums are the same `residual + x` as before, so the outputs are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, overload

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
        fused_rms_norm: bool = False,
    ) -> None:
        super().__init__()
        # `fused_rms_norm` runs every norm of the block as one Triton kernel on
        # CUDA (`RMSNorm(fused=True)`, ADR-045); other devices keep the PyTorch path.
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, fused=fused_rms_norm)
        # Qwen3 (and similar) apply per-head RMSNorm on Q and K after the
        # projections, before RoPE. Standard RMSNorm here (not Gemma's
        # offset variant — those models use `GemmaDecoderLayer` instead).
        q_norm = RMSNorm(head_dim, eps=rms_norm_eps, fused=fused_rms_norm) if with_qk_norm else None
        k_norm = RMSNorm(head_dim, eps=rms_norm_eps, fused=fused_rms_norm) if with_qk_norm else None
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
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, fused=fused_rms_norm)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    @overload
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        defer_add: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        defer_add: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        defer_add: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The block output `residual + ffn(...)`, or with `defer_add` the pair.

        A `residual` is the residual of the block before, still to be added to
        `hidden_states` (that block's FFN output); the input norm adds them.
        `defer_add=True` returns `(ffn_output, residual)` without the last add.
        """
        if residual is None:
            residual = hidden_states
            x = self.input_layernorm(hidden_states)
        else:
            x, residual = self.input_layernorm(hidden_states, residual)
        x = self.self_attn(x, position_embeddings, past_key_values, cu_seqlens_q)
        # One call adds the attention output to the residual and normalizes the
        # sum; with the fused norm on CUDA that is one kernel instead of two.
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        if defer_add:
            return x, residual
        out: torch.Tensor = residual + x
        return out


def run_decoder_layers(
    layers: Iterable[nn.Module],
    final_norm: RMSNorm,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: PagedKVCache,
    cu_seqlens_q: torch.Tensor,
    *,
    tap_layers: frozenset[int] | None = None,
    hidden_state_sink: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """`final_norm` of the output of the last layer, each residual add in the norm after it.

    `layers` take the `residual` and `defer_add` arguments of `TransformerBlock`.
    For each index in `tap_layers`, `hidden_state_sink[i]` gets the output of
    layer `i` with its residual added, the value the plain loop gives.
    """
    x = hidden_states
    residual: torch.Tensor | None = None
    for i, layer in enumerate(layers):
        x, residual = layer(
            x, position_embeddings, past_key_values, cu_seqlens_q, residual, defer_add=True
        )
        if tap_layers is not None and i in tap_layers:
            assert hidden_state_sink is not None and residual is not None
            hidden_state_sink[i] = residual + x
    if residual is None:
        out: torch.Tensor = final_norm(x)
        return out
    out, _ = final_norm(x, residual)
    return out
