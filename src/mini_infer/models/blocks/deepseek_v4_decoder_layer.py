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

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from mini_infer.models.blocks.csa import CSAAttention
from mini_infer.models.blocks.hash_routed_gate import RoutingMode, ScoreFunction
from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
from mini_infer.models.blocks.hca import HCAAttention
from mini_infer.models.blocks.hyper_connections import HyperConnections
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.swa import SWAAttention
from mini_infer.models.blocks.swiglu import SwiGLU

if TYPE_CHECKING:
    from mini_infer.cache.state_cache import StateCache

FFNType = Literal["swiglu", "hash_moe"]

# Type alias for the per-mode attention call wrapped by `_forward_hc`. The
# decoder's `forward` hands a closure over `(token_pe, compressed_pe)`;
# `forward_decode` hands a closure over `(start_pos, state_cache, ...)`. Both
# take a `(B, T, hidden_size)` tensor and return one of the same shape.
_AttnRunner = Callable[[torch.Tensor], torch.Tensor]


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
        expert_dtype: str = "bf16",
        # Hyper-Connections knobs. Defaults disable HC -> vanilla pre-norm
        # residuals are used.
        use_hyper_connections: bool = False,
        hc_mult: int = 0,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.compression_ratio = compression_ratio
        self.is_csa_layer = compression_ratio == self.CSA_COMPRESSION_RATIO
        self.ffn_type = ffn_type
        self.use_hyper_connections = use_hyper_connections

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.self_attn: HCAAttention | CSAAttention | SWAAttention
        if compression_ratio == 0:
            # Pure-SWA layer: no compressor, no indexer (V4-Flash uses these
            # at layers 0, 1 of its 43-layer stack — the reference treats
            # `compress_ratio = 0` as the disable-everything-but-window case).
            self.self_attn = SWAAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                q_lora_rank=q_lora_rank,
                kv_head_dim=kv_head_dim,
                rope_head_dim=rope_head_dim,
                num_groups=num_groups,
                o_lora_rank=o_lora_rank,
                window_size=window_size,
                rms_norm_eps=rms_norm_eps,
            )
        elif self.is_csa_layer:
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
                expert_dtype=expert_dtype,
            )
        else:  # pragma: no cover — Literal already constrains this
            raise ValueError(f"unknown ffn_type={ffn_type!r}")

        # Hyper-Connections (V4 paper §2.5): when enabled, each sublayer is
        # wrapped by a per-instance `(hc_pre, hc_post)` mediator that
        # mixes `hc_mult` residual copies via Sinkhorn-normalized weights.
        # When disabled, the standard pre-norm residual is used.
        if use_hyper_connections:
            if hc_mult <= 0:
                raise ValueError(f"use_hyper_connections=True requires hc_mult > 0; got {hc_mult}")
            hc_kwargs = dict(
                hidden_size=hidden_size,
                hc_mult=hc_mult,
                sinkhorn_iters=hc_sinkhorn_iters,
                hc_eps=hc_eps,
                rms_norm_eps=rms_norm_eps,
            )
            self.hc_attn: HyperConnections | None = HyperConnections(**hc_kwargs)  # type: ignore[arg-type]
            self.hc_ffn: HyperConnections | None = HyperConnections(**hc_kwargs)  # type: ignore[arg-type]
        else:
            self.hc_attn = None
            self.hc_ffn = None

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
            hidden_states: `(B, T, hidden_size)` (vanilla residual mode) OR
                `(B, T, hc_mult, hidden_size)` (HC mode). `T` must be a
                multiple of `compression_ratio`.
            token_position_embeddings: `(cos, sin)` for the `T` raw token
                positions; each `(B, T, rope_head_dim)`.
            compressed_position_embeddings: `(cos, sin)` for the
                `T // compression_ratio` compressed positions; each
                `(B, T // compression_ratio, rope_head_dim)`.
            input_ids: `(B, T)` token ids — required iff `ffn_type ==
                "hash_moe"` AND the MoE gate uses hash routing.
        """
        if self.use_hyper_connections:
            return self._forward_hc(
                hidden_states,
                token_position_embeddings,
                compressed_position_embeddings,
                input_ids=input_ids,
                attn_runner=lambda x: self.self_attn(
                    x, token_position_embeddings, compressed_position_embeddings
                ),
            )

        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, token_position_embeddings, compressed_position_embeddings)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self._apply_ffn(x, input_ids)
        out: torch.Tensor = residual + x
        return out

    def forward_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        *,
        state_cache: StateCache,
        layer_idx: int,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cache-aware prefill through this layer. Mutates `state_cache.layer(layer_idx)`.

        Same shape and output contract as `forward` (`(B, T, hidden_size)` in
        vanilla mode, `(B, T, hc_mult, hidden_size)` in HC mode), but the
        attention sublayer runs its `forward_prefill_with_cache` path: it
        writes the SWA window, compressed history, and in-flight compressor
        state so a later `forward_decode` continues the prompt instead of
        starting from a zeroed cache. `input_ids` `(B, T)` is required iff
        `ffn_type == "hash_moe"` with hash routing.

        SWA (`compression_ratio == 0`) layers dispatch to
        `SWAAttention.forward_prefill_with_cache` (window only, no compressor).

        Caller must `state_cache.advance_start_pos(T)` after the full stack runs.
        """
        attn = self.self_attn
        if self.use_hyper_connections:

            def attn_runner(x: torch.Tensor) -> torch.Tensor:
                return attn.forward_prefill_with_cache(
                    x,
                    token_position_embeddings=token_position_embeddings,
                    compressed_position_embeddings=compressed_position_embeddings,
                    state_cache=state_cache,
                    layer_idx=layer_idx,
                )

            return self._forward_hc(
                hidden_states,
                token_position_embeddings,
                None,
                input_ids=input_ids,
                attn_runner=attn_runner,
            )

        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = attn.forward_prefill_with_cache(
            x,
            token_position_embeddings=token_position_embeddings,
            compressed_position_embeddings=compressed_position_embeddings,
            state_cache=state_cache,
            layer_idx=layer_idx,
        )
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

        `hidden_state`: `(B, 1, hidden_size)` (vanilla) OR `(B, 1, hc_mult,
        hidden_size)` (HC mode). `input_ids` `(B, 1)` required iff
        `ffn_type == "hash_moe"` with hash routing.
        """
        # CSA, HCA, and SWA each implement `forward_decode` with the same
        # signature; SWA ignores `block_position_embeddings` (no compressor).
        attn = self.self_attn
        if self.use_hyper_connections:

            def attn_runner(x: torch.Tensor) -> torch.Tensor:
                return attn.forward_decode(
                    x,
                    start_pos=start_pos,
                    state_cache=state_cache,
                    layer_idx=layer_idx,
                    token_position_embeddings=token_position_embeddings,
                    block_position_embeddings=block_position_embeddings,
                )

            return self._forward_hc(
                hidden_state,
                token_position_embeddings,
                None,
                input_ids=input_ids,
                attn_runner=attn_runner,
            )

        residual = hidden_state
        x = self.input_layernorm(hidden_state)
        x = attn.forward_decode(
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

    def forward_decode_ragged(
        self,
        hidden_state: torch.Tensor,
        *,
        positions: torch.Tensor,
        state_cache: StateCache,
        layer_idx: int,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        n_compressed_max: int | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ragged single-token decode: B requests, each at its own `positions[b]`.

        Per-request counterpart of `forward_decode`. Identical residual / FFN /
        Hyper-Connections structure (those are position-independent); only the
        attention call switches to `forward_decode_ragged`, threading
        `positions` and the per-layer `n_compressed_max`.
        """
        attn = self.self_attn
        if self.use_hyper_connections:

            def attn_runner(x: torch.Tensor) -> torch.Tensor:
                return attn.forward_decode_ragged(
                    x,
                    positions=positions,
                    state_cache=state_cache,
                    layer_idx=layer_idx,
                    token_position_embeddings=token_position_embeddings,
                    block_position_embeddings=block_position_embeddings,
                    n_compressed_max=n_compressed_max,
                )

            return self._forward_hc(
                hidden_state,
                token_position_embeddings,
                None,
                input_ids=input_ids,
                attn_runner=attn_runner,
            )

        residual = hidden_state
        x = self.input_layernorm(hidden_state)
        x = attn.forward_decode_ragged(
            x,
            positions=positions,
            state_cache=state_cache,
            layer_idx=layer_idx,
            token_position_embeddings=token_position_embeddings,
            block_position_embeddings=block_position_embeddings,
            n_compressed_max=n_compressed_max,
        )
        hidden_state = residual + x

        residual = hidden_state
        x = self.post_attention_layernorm(hidden_state)
        x = self._apply_ffn(x, input_ids)
        out: torch.Tensor = residual + x
        return out

    def _forward_hc(
        self,
        hc_state: torch.Tensor,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        input_ids: torch.Tensor | None,
        attn_runner: _AttnRunner,
    ) -> torch.Tensor:
        """HC variant: hidden state shape is `(B, T, hc_mult, dim)`.

        Mirrors the V4 reference's `Block.forward`: for each sublayer we
        run `hc_pre` to reduce `hc_mult` copies into one (sublayer input),
        run RMSNorm + sublayer, then `hc_post` to expand back to `hc_mult`
        copies mixed with the residual via the Sinkhorn comb matrix.
        """
        # `hc_pre` does its own RMS-norm-style rsqrt pre-projection, but
        # the V4 reference still applies the per-sublayer `attn_norm` /
        # `ffn_norm` AFTER the reduction step. We follow the same structure.
        del compressed_position_embeddings  # captured by attn_runner closure
        assert self.hc_attn is not None and self.hc_ffn is not None  # set when HC enabled

        attn_residual = hc_state
        sublayer_input, post, comb = self.hc_attn.hc_pre(hc_state)
        sublayer_input = self.input_layernorm(sublayer_input)
        attn_out = attn_runner(sublayer_input)
        hc_state = self.hc_attn.hc_post(attn_out, attn_residual, post, comb)

        ffn_residual = hc_state
        sublayer_input, post, comb = self.hc_ffn.hc_pre(hc_state)
        sublayer_input = self.post_attention_layernorm(sublayer_input)
        ffn_out = self._apply_ffn(sublayer_input, input_ids)
        hc_state = self.hc_ffn.hc_post(ffn_out, ffn_residual, post, comb)

        return hc_state
