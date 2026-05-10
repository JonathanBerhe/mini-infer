"""DeepSeek-V4 decoder layer: per-layer CSA-or-HCA attention + SwiGLU or MoE FFN.

V4's hybrid attention interleaves two attention modes by layer index:
    - `compression_ratio == 4` -> Compressed Sparse Attention (CSA)
      with a Lightning Indexer + top-k.
    - any other ratio (paper: 128 for V4-Pro/V4-Flash) -> Heavily
      Compressed Attention (HCA), no indexer.

The FFN is one of:
    - `ffn_type="swiglu"` (default): standard SwiGLU. Used by tests that
      don't need real V4 MoE behaviour.
    - `ffn_type="hash_moe"` (V4-faithful): `HashRoutedMoEFFN` with the
      gate's `routing_mode` chosen at construction (`"hash"` for the
      first `n_hash_routed_layers`, `"score_topk"` for the rest, per
      V4 paper §2.2). The decoder's forward methods accept `input_ids`
      and thread them down so hash routing can read its lookup table.

V4's published `Block` also uses Hyper-Connections (Sinkhorn-mixed
residuals, V4 paper §2.5) — orthogonal to the attention + FFN work
shipped here; we use vanilla pre-norm residuals so the backbone is
reviewable, and HC lands as its own primitive when needed.

Layer state lives in a per-request `StateCache`:
    - SWA circular buffer (per layer, every layer).
    - Compressed history (per layer, append-only).
    - Compressor in-flight accumulator (HCA: simple; CSA: doubled
      with overlap slots).
    - Indexer sub-cache for CSA layers only.

Two forward paths:
    - `forward(hidden_states, ..., input_ids=None)`: standalone packed
      prefill. Doesn't touch the cache. `input_ids` is required iff
      the FFN is `"hash_moe"` AND this layer's gate uses hash routing.
    - `forward_decode(hidden_state, *, start_pos, state_cache,
      layer_idx, ..., input_ids=None)`: single-token decode that
      reads/writes `state_cache.layer(layer_idx)`. Same `input_ids`
      contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from mini_infer.models.blocks.csa import CSAAttention
from mini_infer.models.blocks.hash_routed_gate import RoutingMode, ScoreFunction
from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
from mini_infer.models.blocks.hca import HCAAttention
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.swiglu import SwiGLU

if TYPE_CHECKING:
    from mini_infer.cache.state_cache import StateCache

FFNType = Literal["swiglu", "hash_moe"]


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
        # FFN selection. Defaults to SwiGLU (cache-less, no input_ids needed).
        ffn_type: FFNType = "swiglu",
        # MoE FFN knobs (only consulted when `ffn_type == "hash_moe"`).
        moe_intermediate_size: int = 0,
        num_routed_experts: int = 0,
        num_activated_experts: int = 0,
        moe_routing_mode: RoutingMode = "score_topk",
        moe_score_func: ScoreFunction = "softmax",
        moe_route_scale: float = 1.0,
        moe_vocab_size: int | None = None,
        n_shared_experts: int = 0,
    ) -> None:
        super().__init__()
        self.compression_ratio = compression_ratio
        self.is_csa_layer = compression_ratio == self.CSA_COMPRESSION_RATIO
        self.ffn_type = ffn_type

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

        self.mlp: SwiGLU | HashRoutedMoEFFN
        if ffn_type == "swiglu":
            self.mlp = SwiGLU(hidden_size, intermediate_size)
        elif ffn_type == "hash_moe":
            if num_routed_experts <= 0 or num_activated_experts <= 0:
                raise ValueError(
                    "ffn_type='hash_moe' requires positive num_routed_experts and "
                    f"num_activated_experts; got {num_routed_experts=}, {num_activated_experts=}"
                )
            if moe_intermediate_size <= 0:
                raise ValueError(
                    f"ffn_type='hash_moe' requires moe_intermediate_size > 0; "
                    f"got {moe_intermediate_size}"
                )
            if moe_routing_mode == "hash" and (moe_vocab_size is None or moe_vocab_size <= 0):
                raise ValueError(
                    "ffn_type='hash_moe' with moe_routing_mode='hash' requires "
                    f"a positive moe_vocab_size; got {moe_vocab_size}"
                )
            self.mlp = HashRoutedMoEFFN(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
                routing_mode=moe_routing_mode,
                score_func=moe_score_func,
                route_scale=moe_route_scale,
                vocab_size=moe_vocab_size,
                n_shared_experts=n_shared_experts,
            )
        else:  # pragma: no cover — Literal already constrains this
            raise ValueError(f"unknown ffn_type={ffn_type!r}")

    def _apply_ffn(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Run the FFN — SwiGLU ignores `input_ids`, hash MoE consumes it."""
        if isinstance(self.mlp, HashRoutedMoEFFN):
            moe_out: torch.Tensor = self.mlp(hidden_states, input_ids)
            return moe_out
        swiglu_out: torch.Tensor = self.mlp(hidden_states)
        return swiglu_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        input_ids: torch.Tensor | None = None,
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
            input_ids: `(B, T)` token ids — required iff `ffn_type ==
                "hash_moe"` AND the MoE gate uses hash routing. SwiGLU
                FFN ignores it.
        """
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, token_position_embeddings, compressed_position_embeddings)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self._apply_ffn(x, input_ids)
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
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-token decode through this layer. Mutates `state_cache.layer(layer_idx)`.

        `input_ids` shape `(B, 1)` — required iff `ffn_type == "hash_moe"`
        with hash routing. The new token's id drives the FFN's expert lookup.
        """
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
        x = self._apply_ffn(x, input_ids)
        out: torch.Tensor = residual + x
        return out
