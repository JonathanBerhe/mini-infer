"""DeepSeek-V4 decoder layer: per-layer CSA-or-HCA attention + SwiGLU FFN.

V4's hybrid attention interleaves two attention modes by layer index:
    - `compression_ratio == 4` -> Compressed Sparse Attention (CSA)
      with a Lightning Indexer + top-k.
    - any other ratio (paper: 128 for V4-Pro/V4-Flash) -> Heavily
      Compressed Attention (HCA), no indexer.

This block dispatches to the right attention class at construction time
based on the per-layer `compression_ratio`. The surrounding plumbing
(pre-norm RMSNorm + residuals + SwiGLU FFN) is the standard
transformer recipe — V4's published `Block` actually uses
Hyper-Connections (Sinkhorn-mixed residuals, V4 paper §2.5) and a MoE
FFN with hash routing, but both are orthogonal to the attention
contribution and live behind a substantial separate body of work. We
ship the attention here and use vanilla pre-norm + SwiGLU so the
backbone is runnable and easy to read.

Layer state lives in a per-request `StateCache`:
    - SWA circular buffer (per layer, every layer).
    - Compressed history (per layer, append-only).
    - Compressor in-flight accumulator (HCA: simple; CSA: doubled
      with overlap slots).
    - Indexer sub-cache for CSA layers only.

Two forward paths:
    - `forward(hidden_states, ..., layer_idx)`: standalone packed
      prefill. Doesn't touch the cache. Used in unit/parity tests
      that don't need cache state.
    - `forward_decode(hidden_state, *, start_pos, state_cache,
      layer_idx, ...)`: single-token decode that reads/writes
      `state_cache.layer(layer_idx)`. Used by the model's
      end-to-end decode path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.models.blocks.csa import CSAAttention
from mini_infer.models.blocks.hca import HCAAttention
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.swiglu import SwiGLU

if TYPE_CHECKING:
    from mini_infer.cache.state_cache import StateCache


class DeepseekV4DecoderLayer(nn.Module):
    """Pre-norm + (CSA or HCA) + residual + pre-norm + SwiGLU + residual."""

    # Paper §4.2.1: CSA layers have compression_ratio == 4. Anything else
    # is HCA. The reference uses ratio==128 for HCA at V4-Pro/Flash scale.
    CSA_COMPRESSION_RATIO = 4

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_head_dim: int,
        rope_head_dim: int,
        num_groups: int,
        o_lora_rank: int,
        window_size: int,
        compression_ratio: int,
        intermediate_size: int,
        rms_norm_eps: float,
        # CSA-only knobs (ignored for HCA layers).
        index_num_heads: int = 0,
        index_head_dim: int = 0,
        index_top_k: int = 0,
    ) -> None:
        super().__init__()
        self.compression_ratio = compression_ratio
        self.is_csa_layer = compression_ratio == self.CSA_COMPRESSION_RATIO

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.self_attn: HCAAttention | CSAAttention
        if self.is_csa_layer:
            if index_num_heads <= 0 or index_head_dim <= 0 or index_top_k <= 0:
                raise ValueError(
                    "CSA layer (compression_ratio == 4) requires positive "
                    f"index_num_heads={index_num_heads}, index_head_dim={index_head_dim}, "
                    f"index_top_k={index_top_k}"
                )
            self.self_attn = CSAAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                q_lora_rank=q_lora_rank,
                kv_head_dim=kv_head_dim,
                rope_head_dim=rope_head_dim,
                num_groups=num_groups,
                o_lora_rank=o_lora_rank,
                window_size=window_size,
                compression_ratio=compression_ratio,
                rms_norm_eps=rms_norm_eps,
                index_num_heads=index_num_heads,
                index_head_dim=index_head_dim,
                index_top_k=index_top_k,
            )
        else:
            self.self_attn = HCAAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                q_lora_rank=q_lora_rank,
                kv_head_dim=kv_head_dim,
                rope_head_dim=rope_head_dim,
                num_groups=num_groups,
                o_lora_rank=o_lora_rank,
                window_size=window_size,
                compression_ratio=compression_ratio,
                rms_norm_eps=rms_norm_eps,
            )

        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Standalone packed prefill — no cache access.

        Args:
            hidden_states: `(B, T, hidden_size)`. `T` must be a multiple
                of `compression_ratio` (the standalone attention forward
                rejects unaligned input).
            token_position_embeddings: `(cos, sin)` for the `T` raw token
                positions; each `(B, T, rope_head_dim)`.
            compressed_position_embeddings: `(cos, sin)` for the
                `T // compression_ratio` compressed positions (block `i`
                -> token position `i * compression_ratio`); each
                `(B, T // compression_ratio, rope_head_dim)`.
        """
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, token_position_embeddings, compressed_position_embeddings)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        out: torch.Tensor = residual + x
        return out

    def forward_decode(
        self,
        hidden_state: torch.Tensor,
        *,
        start_pos: int,
        state_cache: StateCache,
        layer_idx: int,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Single-token decode through this layer. Mutates `state_cache.layer(layer_idx)`."""
        residual = hidden_state
        x = self.input_layernorm(hidden_state)
        x = self.self_attn.forward_decode(
            x,
            start_pos=start_pos,
            state_cache=state_cache,
            layer_idx=layer_idx,
            token_position_embeddings=token_position_embeddings,
            block_position_embeddings=block_position_embeddings,
        )
        hidden_state = residual + x

        residual = hidden_state
        x = self.post_attention_layernorm(hidden_state)
        x = self.mlp(x)
        out: torch.Tensor = residual + x
        return out
