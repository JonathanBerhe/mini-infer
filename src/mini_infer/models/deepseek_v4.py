"""DeepSeek-V4 family: hybrid CSA / HCA attention backbone.

Targets `DeepseekV4ForCausalLM` (paper `huggingface.co/deepseek-ai/
DeepSeek-V4-Pro` once weights are public). The portfolio-relevant piece
is the attention itself (V4 paper §2.3) — `HCAAttention`, `CSAAttention`,
`LightningIndexer`, `TokenLevelCompressor`, `AttentionSink`,
`GroupedOutputProjection`, all bit-parity validated against the upstream
inference reference.

Architectural status vs V4-published:

  - **Hyper-Connections** (V4 paper §2.5): supported. Toggle via
    `cfg.use_hyper_connections=True` + `cfg.hc_mult > 0`. When enabled,
    every decoder layer mediates residuals via Sinkhorn-mixed multi-
    residuals (`HyperConnections`), and an `HCHeadReduction` collapses
    the `hc_mult` copies before the LM head. Default off → vanilla
    pre-norm residuals.
  - **MoE FFN with hash routing** (V4 paper §2.2): supported. Toggle
    via `cfg.use_moe_ffn=True`. The first `cfg.num_hash_routed_layers`
    layers use per-token-id hash routing; the rest use score-topk.
    Default off → SwiGLU FFN.
  - **YaRN long-context scaling** (V4 paper §3.1): supported. Toggle
    via `cfg.yarn_original_seq_len > 0`. Default off → vanilla RoPE.
  - **No on-disk KV** (V4 paper §3.6.2): orthogonal storage layer; not
    in scope for this implementation.

Layer dispatch — per-layer attention type via `compress_ratios`:

  - `compression_ratio == 4` -> CSA (Lightning Indexer + top-k).
  - any other ratio -> HCA (heavy compression, no indexer).
  - V4-Pro/Flash have the first 2 layers as SWA-only (ratio=0); we
    represent that with HCA at the configured ratio (the synthetic
    test configs always start at layer 0 with a real ratio).

Cache state lives in a per-request `StateCache` (NOT `PagedKVCache`):
SWA circular buffer + per-layer compressed history + compressor in-flight
accumulator. Helper `build_state_cache_layer_specs` translates a config
into the per-layer specs `StateCache.__init__` expects.

`load_weights` walks the HF safetensors state_dict, dequantises FP8
(e4m3fn) and NVFP4 (`float4_e2m1fn_x2`, with per-block e8m0fnu scales)
weights to BF16, applies the same MoE expert renames as V2/V3
(`gate_proj/up_proj/down_proj` -> `w1/w2/w3`), then dispatches through
`load_state_dict_with_tp` so each column / row / vocab-parallel layer
slices its rank's share. The full V4-Flash storage format is now
supported end-to-end on the loader path; a Triton-fused FP4 GEMM is
the optimization follow-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from torch import nn

from mini_infer.cache.state_cache import IndexerStateSpec, StateCache, StateLayerSpec
from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.distributed.loader import load_state_dict_with_tp
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks.deepseek_v4_decoder_layer import DeepseekV4DecoderLayer
from mini_infer.models.blocks.hyper_connections import HCHeadReduction
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import RotaryEmbedding

# HF V4 -> mini-infer renames. Same MoE expert structure as V2/V3 (the
# block geometry didn't change across generations), with the additional
# shared-expert collapse: V4 ships `n_shared_experts=1` and we collapse
# `n_shared_experts` sub-blocks into one wider MLP — the rename rules
# below cover the single-shared-expert case.
_V4_RENAME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.mlp\.experts\.(\d+)\.gate_proj\.weight$"), r".mlp.experts.\1.w1.weight"),
    (re.compile(r"\.mlp\.experts\.(\d+)\.down_proj\.weight$"), r".mlp.experts.\1.w2.weight"),
    (re.compile(r"\.mlp\.experts\.(\d+)\.up_proj\.weight$"), r".mlp.experts.\1.w3.weight"),
    (re.compile(r"\.mlp\.shared_experts\.gate_proj\.weight$"), r".mlp.shared_experts.w1.weight"),
    (re.compile(r"\.mlp\.shared_experts\.down_proj\.weight$"), r".mlp.shared_experts.w2.weight"),
    (re.compile(r"\.mlp\.shared_experts\.up_proj\.weight$"), r".mlp.shared_experts.w3.weight"),
]

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class DeepseekV4Config:
    """Owned mini-infer config for DeepSeek-V4-style hybrid attention.

    Field naming follows mini-infer (HF-aligned where possible) rather than
    the paper's `ModelArgs` shorthand. Field-by-field correspondence:

        vocab_size            <- HF vocab_size
        hidden_size           <- HF hidden_size  (paper: dim)
        intermediate_size     <- HF intermediate_size (used by SwiGLU)
        num_hidden_layers     <- HF num_hidden_layers (paper: n_layers)
        num_attention_heads   <- HF num_attention_heads (paper: n_heads)
        q_lora_rank           <- HF q_lora_rank
        kv_head_dim           <- HF kv_head_dim or head_dim
        rope_head_dim         <- HF rope_head_dim
        o_num_groups          <- HF o_num_groups (paper: o_groups)
        o_lora_rank           <- HF o_lora_rank
        window_size           <- HF window_size
        compress_ratios       <- HF compress_ratios (per-layer tuple)
        index_num_heads       <- HF index_num_heads (paper: index_n_heads)
        index_head_dim        <- HF index_head_dim
        index_top_k           <- HF index_top_k (paper: index_topk)
        rms_norm_eps          <- HF rms_norm_eps  (paper: norm_eps)
        rope_theta            <- HF rope_theta
        tie_word_embeddings   <- HF tie_word_embeddings (default False)
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_head_dim: int
    rope_head_dim: int
    o_num_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    index_num_heads: int
    index_head_dim: int
    index_top_k: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    # YaRN long-context RoPE (V4 paper §3.1). Disabled (no-op) when
    # `yarn_original_seq_len == 0`. The reference's `ModelArgs` carries
    # these as `original_seq_len`, `rope_factor`, `beta_fast`, `beta_slow`;
    # we expose mini-infer-style names that the `RotaryEmbedding` block
    # already accepts. Defaults map to "no YaRN".
    yarn_original_seq_len: int = 0
    yarn_scaling_factor: float = 1.0
    yarn_beta_fast: int = 32
    yarn_beta_slow: int = 1
    # MoE FFN (V4 paper §2.2). Defaults to "off" (every layer uses SwiGLU)
    # for backward compatibility — existing tests built before MoE landed
    # leave `use_moe_ffn=False`. When enabled, every decoder layer uses
    # `HashRoutedMoEFFN`; the routing mode is per-layer:
    #   - `layer_idx < num_hash_routed_layers`: hash routing (per-token-id
    #     lookup table). The first few layers route deterministically by
    #     token id — the V4 paper's design choice for hash routing as an
    #     "early identification" pass.
    #   - else: score-topk routing.
    use_moe_ffn: bool = False
    moe_intermediate_size: int = 0
    num_routed_experts: int = 0
    num_activated_experts: int = 0
    num_hash_routed_layers: int = 0
    moe_score_func: str = "softmax"
    moe_route_scale: float = 1.0
    n_shared_experts: int = 0
    # Hyper-Connections (V4 paper §2.5). Defaults disable HC -> vanilla
    # pre-norm residuals, hidden state stays `(B, T, dim)`. When enabled,
    # the embedding output is expanded to `(B, T, hc_mult, dim)`, every
    # decoder layer mediates via Sinkhorn-mixed multi-residuals, and an
    # `HCHeadReduction` reduces back to `(B, T, dim)` before the LM head.
    use_hyper_connections: bool = False
    hc_mult: int = 0
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                f"compress_ratios length ({len(self.compress_ratios)}) must equal "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )
        if self.use_moe_ffn:
            if self.num_routed_experts <= 0:
                raise ValueError(
                    "use_moe_ffn=True requires num_routed_experts > 0; "
                    f"got {self.num_routed_experts}"
                )
            if self.num_activated_experts <= 0:
                raise ValueError(
                    "use_moe_ffn=True requires num_activated_experts > 0; "
                    f"got {self.num_activated_experts}"
                )
            if self.moe_intermediate_size <= 0:
                raise ValueError(
                    "use_moe_ffn=True requires moe_intermediate_size > 0; "
                    f"got {self.moe_intermediate_size}"
                )
            if self.num_hash_routed_layers > self.num_hidden_layers:
                raise ValueError(
                    f"num_hash_routed_layers ({self.num_hash_routed_layers}) cannot exceed "
                    f"num_hidden_layers ({self.num_hidden_layers})"
                )
        if self.use_hyper_connections and self.hc_mult <= 0:
            raise ValueError(f"use_hyper_connections=True requires hc_mult > 0; got {self.hc_mult}")

    def is_csa_layer(self, layer_idx: int) -> bool:
        """CSA layers have compression_ratio == 4 (paper §4.2.1)."""
        return self.compress_ratios[layer_idx] == DeepseekV4DecoderLayer.CSA_COMPRESSION_RATIO

    def is_hash_routed_layer(self, layer_idx: int) -> bool:
        """Layers `[0, num_hash_routed_layers)` use hash routing per V4 §2.2.

        Always False if `use_moe_ffn` is False (no MoE -> no routing distinction).
        """
        return self.use_moe_ffn and layer_idx < self.num_hash_routed_layers

    @classmethod
    def from_hf(cls, hf_config: Any) -> DeepseekV4Config:
        """Parse an HF config into our owned schema.

        Robust to both attribute-style and dict-style access: transformers
        5.x's fallback PretrainedConfig for unknown model_types (like V4
        before transformers ships native support) doesn't always promote
        every dict key to an instance attribute. We look up via attr then
        fall back to `to_dict()` / `__dict__`.
        """

        # Materialise the underlying config dict once so dict-style lookups
        # work even when attribute access doesn't.
        if isinstance(hf_config, dict):
            cfg_dict: dict[str, Any] = dict(hf_config)
        elif hasattr(hf_config, "to_dict"):
            cfg_dict = dict(hf_config.to_dict())
        else:
            cfg_dict = dict(getattr(hf_config, "__dict__", {}))

        def _pick(*candidate_names: str, default_value: Any) -> Any:
            """First non-None value across `candidate_names`, attr-first then dict."""
            for name in candidate_names:
                value = getattr(hf_config, name, None)
                if value is not None:
                    return value
                value = cfg_dict.get(name)
                if value is not None:
                    return value
            return default_value

        # YaRN params land under `rope_scaling` in the real V4-Flash config
        # (`rope_scaling.original_max_position_embeddings`, `rope_scaling.factor`,
        # ...). Older internal configs put them at the top level. Support both.
        rope_scaling = _pick("rope_scaling", "rope_parameters", default_value=None) or {}
        rope_theta_raw = rope_scaling.get("rope_theta") or _pick(
            "rope_theta", default_value=10000.0
        )

        compress_ratios_raw = _pick("compress_ratios", default_value=None)
        if compress_ratios_raw is None:
            raise ValueError(
                "DeepseekV4Config.from_hf: HF config missing `compress_ratios` field"
            )

        # V4-Flash's checkpoint also carries MTP (multi-token-prediction)
        # head layers — `num_nextn_predict_layers` ratios are appended to
        # `compress_ratios` after the standard transformer layers'.
        # mini-infer doesn't implement MTP today, so truncate the tail.
        num_hidden_layers_int = int(_pick("num_hidden_layers", default_value=0))
        compress_ratios_tuple = tuple(int(ratio) for ratio in compress_ratios_raw)
        if len(compress_ratios_tuple) > num_hidden_layers_int > 0:
            compress_ratios_tuple = compress_ratios_tuple[:num_hidden_layers_int]

        return cls(
            vocab_size=int(_pick("vocab_size", default_value=0)),
            hidden_size=int(_pick("hidden_size", default_value=0)),
            intermediate_size=int(_pick("intermediate_size", default_value=0)),
            num_hidden_layers=num_hidden_layers_int,
            num_attention_heads=int(_pick("num_attention_heads", default_value=0)),
            q_lora_rank=int(_pick("q_lora_rank", default_value=0)),
            kv_head_dim=int(_pick("kv_head_dim", "head_dim", default_value=0)),
            # HF: `qk_rope_head_dim`; older paper alias: `rope_head_dim`.
            rope_head_dim=int(_pick("rope_head_dim", "qk_rope_head_dim", default_value=0)),
            # HF: `o_groups`; older paper alias: `o_num_groups`.
            o_num_groups=int(_pick("o_num_groups", "o_groups", default_value=0)),
            o_lora_rank=int(_pick("o_lora_rank", default_value=0)),
            # HF: `sliding_window`; older paper alias: `window_size`.
            window_size=int(_pick("window_size", "sliding_window", default_value=0)),
            compress_ratios=compress_ratios_tuple,
            index_num_heads=int(_pick("index_num_heads", "index_n_heads", default_value=0)),
            index_head_dim=int(_pick("index_head_dim", default_value=0)),
            index_top_k=int(_pick("index_top_k", "index_topk", default_value=0)),
            rms_norm_eps=float(_pick("rms_norm_eps", "norm_eps", default_value=1e-6)),
            rope_theta=float(rope_theta_raw),
            tie_word_embeddings=bool(_pick("tie_word_embeddings", default_value=False)),
            yarn_original_seq_len=int(
                _pick(
                    "yarn_original_seq_len",
                    "original_seq_len",
                    default_value=rope_scaling.get("original_max_position_embeddings", 0),
                )
            ),
            yarn_scaling_factor=float(
                _pick(
                    "yarn_scaling_factor",
                    "rope_factor",
                    default_value=rope_scaling.get("factor", 1.0),
                )
            ),
            yarn_beta_fast=int(
                _pick(
                    "yarn_beta_fast",
                    "beta_fast",
                    default_value=rope_scaling.get("beta_fast", 32),
                )
            ),
            yarn_beta_slow=int(
                _pick(
                    "yarn_beta_slow",
                    "beta_slow",
                    default_value=rope_scaling.get("beta_slow", 1),
                )
            ),
        )


def build_state_cache_layer_specs(
    cfg: DeepseekV4Config, *, max_n_compressed: int
) -> list[StateLayerSpec]:
    """Per-layer state specs derived from the config.

    `max_n_compressed` is the caller-chosen cap on compressed-history slots
    per layer (typically `ceil(max_seq_len / min(compress_ratios))`).
    Indexer state is allocated only for CSA layers.
    """
    layer_specs: list[StateLayerSpec] = []
    for compression_ratio in cfg.compress_ratios:
        is_csa_layer = compression_ratio == DeepseekV4DecoderLayer.CSA_COMPRESSION_RATIO
        layer_specs.append(
            StateLayerSpec(
                kv_head_dim=cfg.kv_head_dim,
                compression_ratio=compression_ratio,
                n_win=cfg.window_size,
                max_n_compressed=max_n_compressed,
                overlap_mode=is_csa_layer,
                indexer=(IndexerStateSpec(head_dim=cfg.index_head_dim) if is_csa_layer else None),
            )
        )
    return layer_specs


class _DeepseekV4InnerModel(nn.Module):
    """Embed + N decoder layers + final RMSNorm (HF-aligned attribute names)."""

    def __init__(self, cfg: DeepseekV4Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [self._build_layer(cfg, layer_idx) for layer_idx in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    @staticmethod
    def _build_layer(cfg: DeepseekV4Config, layer_idx: int) -> DeepseekV4DecoderLayer:
        """Construct one decoder layer, picking SwiGLU vs MoE FFN per the config.

        With `cfg.use_moe_ffn=True`, layer `i` gets:
          - `moe_routing_mode="hash"` if `i < cfg.num_hash_routed_layers`
            (V4 paper §2.2: early layers route by token-id lookup).
          - `moe_routing_mode="score_topk"` otherwise.
        """
        layer_kwargs: dict[str, object] = dict(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_head_dim=cfg.kv_head_dim,
            rope_head_dim=cfg.rope_head_dim,
            num_groups=cfg.o_num_groups,
            o_lora_rank=cfg.o_lora_rank,
            window_size=cfg.window_size,
            compression_ratio=cfg.compress_ratios[layer_idx],
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            index_num_heads=cfg.index_num_heads,
            index_head_dim=cfg.index_head_dim,
            index_top_k=cfg.index_top_k,
        )
        if cfg.use_moe_ffn:
            routing_mode = "hash" if cfg.is_hash_routed_layer(layer_idx) else "score_topk"
            layer_kwargs.update(
                ffn_type="hash_moe",
                moe_intermediate_size=cfg.moe_intermediate_size,
                num_routed_experts=cfg.num_routed_experts,
                num_activated_experts=cfg.num_activated_experts,
                moe_routing_mode=routing_mode,
                moe_score_func=cfg.moe_score_func,
                moe_route_scale=cfg.moe_route_scale,
                moe_vocab_size=cfg.vocab_size if routing_mode == "hash" else None,
                n_shared_experts=cfg.n_shared_experts,
            )
        if cfg.use_hyper_connections:
            layer_kwargs.update(
                use_hyper_connections=True,
                hc_mult=cfg.hc_mult,
                hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                hc_eps=cfg.hc_eps,
            )
        return DeepseekV4DecoderLayer(**layer_kwargs)  # type: ignore[arg-type]


@register_model
class DeepseekV4ForCausalLM(BaseCausalLM):
    """Owned implementation of the hybrid CSA/HCA backbone."""

    HF_ARCHITECTURE: ClassVar[str] = "DeepseekV4ForCausalLM"
    Config: ClassVar[type] = DeepseekV4Config

    def __init__(self, cfg: DeepseekV4Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _DeepseekV4InnerModel(cfg)
        # Single rotary table for all layers (same `rope_theta`). Per-layer
        # `compressed_position_embeddings` are sliced from this table at
        # block positions — see `forward` below.
        # Single rotary table for all layers — YaRN parameters are no-ops
        # when `yarn_original_seq_len == 0` (the default).
        self.rotary_emb = RotaryEmbedding(
            cfg.rope_head_dim,
            base=cfg.rope_theta,
            yarn_original_seq_len=cfg.yarn_original_seq_len,
            yarn_scaling_factor=cfg.yarn_scaling_factor,
            yarn_beta_fast=cfg.yarn_beta_fast,
            yarn_beta_slow=cfg.yarn_beta_slow,
        )
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False, gather_output=True
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # When Hyper-Connections is on, hidden state through the layers
        # carries `hc_mult` copies; the head needs a single (B, T, dim)
        # summary, so this sub-block applies the reference's
        # `ParallelHead.hc_head` reduction.
        self.hc_head_reduction: HCHeadReduction | None
        if cfg.use_hyper_connections:
            self.hc_head_reduction = HCHeadReduction(
                hidden_size=cfg.hidden_size,
                hc_mult=cfg.hc_mult,
                hc_eps=cfg.hc_eps,
                rms_norm_eps=cfg.rms_norm_eps,
            )
        else:
            self.hc_head_reduction = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: PagedKVCache | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standalone packed prefill (no cache).

        Args:
            input_ids: `(B, T)`. `T` must be a multiple of every layer's
                `compression_ratio` — for a hybrid model that means
                `T % lcm(compress_ratios) == 0`.
            position_ids: `(B, T)` or None (defaults to `arange(T)`).

        The `past_key_values` and `cu_seqlens_q` parameters keep the signature
        compatible with the rest of the registry but are ignored — V4 uses
        `StateCache` (not PagedKVCache). Use `forward_decode_with_cache` for
        decoding through cached state.
        """
        batch_size, seqlen = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            )

        hidden_states = self.model.embed_tokens(input_ids)
        # RoPE tables computed BEFORE expanding to hc_mult copies — they only
        # depend on the per-token shape (B, T, ...), not on the residual fan-out.
        cos_for_tokens, sin_for_tokens = self.rotary_emb(hidden_states, position_ids)
        token_position_embeddings = (cos_for_tokens, sin_for_tokens)

        if self.cfg.use_hyper_connections:
            # Expand `(B, T, dim)` -> `(B, T, hc_mult, dim)` by replicating the
            # embedded hidden state across the `hc_mult` axis. `contiguous()`
            # because subsequent layers may slice on the hc axis.
            hidden_states = (
                hidden_states.unsqueeze(2)
                .expand(batch_size, seqlen, self.cfg.hc_mult, self.cfg.hidden_size)
                .contiguous()
            )

        for layer_idx, layer in enumerate(self.model.layers):
            compression_ratio = self.cfg.compress_ratios[layer_idx]
            num_compressed_blocks = seqlen // compression_ratio
            block_positions = (
                (torch.arange(num_compressed_blocks, device=input_ids.device) * compression_ratio)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            cos_for_blocks, sin_for_blocks = self.rotary_emb(
                torch.zeros(batch_size, num_compressed_blocks, device=input_ids.device),
                block_positions,
            )
            compressed_position_embeddings = (cos_for_blocks, sin_for_blocks)
            hidden_states = layer(
                hidden_states,
                token_position_embeddings,
                compressed_position_embeddings,
                input_ids=input_ids if self.cfg.use_moe_ffn else None,
            )

        if self.hc_head_reduction is not None:
            hidden_states = self.hc_head_reduction(hidden_states)

        hidden_states = self.model.norm(hidden_states)
        logits: torch.Tensor = self.lm_head(hidden_states)
        return logits

    def forward_decode_with_cache(
        self,
        input_id: torch.Tensor,
        *,
        start_pos: int,
        state_cache: StateCache,
    ) -> torch.Tensor:
        """One decode step: read state from `state_cache`, emit logits.

        Args:
            input_id: `(B, 1)` integer token ids.
            start_pos: Global token position of this token (0-indexed).
                The caller is responsible for advancing `state_cache.start_pos`
                AFTER this method returns.
            state_cache: Pre-allocated for this model (use
                `build_state_cache_layer_specs` + `StateCache(specs, ...)`).

        Returns:
            `(B, 1, vocab_size)` logits for the new token.
        """
        if input_id.shape[-1] != 1:
            raise ValueError(
                f"forward_decode_with_cache expects shape (B, 1), got {tuple(input_id.shape)}"
            )
        if state_cache.num_layers != self.cfg.num_hidden_layers:
            raise ValueError(
                f"state_cache has {state_cache.num_layers} layers, "
                f"model has {self.cfg.num_hidden_layers}"
            )

        batch_size = input_id.shape[0]
        hidden_state = self.model.embed_tokens(input_id)  # (B, 1, hidden_size)
        token_position = torch.tensor([[start_pos]], device=input_id.device).expand(batch_size, -1)
        cos_for_token, sin_for_token = self.rotary_emb(hidden_state, token_position)
        token_position_embeddings = (cos_for_token, sin_for_token)

        if self.cfg.use_hyper_connections:
            hidden_state = (
                hidden_state.unsqueeze(2)
                .expand(batch_size, 1, self.cfg.hc_mult, self.cfg.hidden_size)
                .contiguous()
            )

        for layer_idx, layer_module in enumerate(self.model.layers):
            # `nn.ModuleList` stores items as `nn.Module`; narrow for mypy so
            # `forward_decode` resolves on the concrete subclass.
            layer = cast(DeepseekV4DecoderLayer, layer_module)
            compression_ratio = self.cfg.compress_ratios[layer_idx]
            block_position_embeddings = None
            if (start_pos + 1) % compression_ratio == 0:
                # This layer flushes a compressed block at this step;
                # the new entry needs RoPE at position `(start_pos // m) * m`.
                flushed_block_position_value = (start_pos // compression_ratio) * compression_ratio
                flushed_block_position = torch.tensor(
                    [[flushed_block_position_value]], device=input_id.device
                ).expand(batch_size, -1)
                cos_for_block, sin_for_block = self.rotary_emb(
                    torch.zeros(batch_size, 1, device=input_id.device), flushed_block_position
                )
                block_position_embeddings = (cos_for_block, sin_for_block)

            hidden_state = layer.forward_decode(
                hidden_state,
                start_pos=start_pos,
                state_cache=state_cache,
                layer_idx=layer_idx,
                token_position_embeddings=token_position_embeddings,
                block_position_embeddings=block_position_embeddings,
                input_ids=input_id if self.cfg.use_moe_ffn else None,
            )

        if self.hc_head_reduction is not None:
            hidden_state = self.hc_head_reduction(hidden_state)

        hidden_state = self.model.norm(hidden_state)
        logits: torch.Tensor = self.lm_head(hidden_state)
        return logits

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        """Reported size — V4 uses StateCache, not PagedKVCache; consumers that
        expect KV-per-token will see `kv_head_dim` and a placeholder head count."""
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=1,
            head_dim=self.cfg.kv_head_dim,
        )

    def required_attention_backend(self) -> str | None:
        """V4 attention is fully orchestrated inside HCA/CSA; the global
        attention dispatcher in `packed_attention.py` is bypassed entirely."""
        return None

    def expected_missing_state_keys(self) -> set[str]:
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    @staticmethod
    def load_weights(
        model: BaseCausalLM,
        hf_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Load HF DeepSeek-V4 weights into the model under tensor parallelism.

        Pipeline:
          1. Optional dtype dequant: V4 ships mixed FP8 (e4m3fn) for
             non-MoE weights and FP4 for MoE expert weights. FP8 -> BF16
             happens here via `.to(bfloat16)`; FP4 storage is packed
             uint8 (NVFP4), which PyTorch doesn't yet ship a native
             dtype for, so the loader raises an informative error and
             refers to the deferred Triton-kernel work. The non-FP4
             portion of V4-Flash still loads end-to-end through this path.
          2. Rename: HF V4 names follow the V2/V3 pattern for MoE experts
             (`mlp.experts.{j}.{gate,up,down}_proj.weight`); we map to
             our `w1/w2/w3` names with the same rules as V2.
          3. TP-aware load: dispatch through `load_state_dict_with_tp`
             so per-rank slicing falls out of the column/row-parallel
             layers automatically.

        At `world_size=1` this is bit-equivalent to a non-TP load — the
        TP layers behave as plain `nn.Linear` / `nn.Embedding`.
        """
        if not isinstance(model, DeepseekV4ForCausalLM):
            raise TypeError(
                f"DeepseekV4ForCausalLM.load_weights expects a DeepseekV4ForCausalLM, "
                f"got {type(model).__name__}"
            )

        # Step 1: dequant.
        # V4-Flash ships mixed precision:
        #   - Non-MoE weights as `float8_e4m3fn` (1 byte, native PyTorch dtype).
        #   - MoE expert weights as `float4_e2m1fn_x2` (packed NVFP4: 2x FP4
        #     per byte), accompanied by a per-block `float8_e8m0fnu` scale
        #     tensor under the matching `*_scale` key.
        # Walk the state_dict, upcast FP8 to BF16, and dequantize NVFP4
        # weight+scale pairs via `dequantize_nvfp4_to_bf16`.
        from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16, is_packed_nvfp4

        dequantized: dict[str, torch.Tensor] = {}
        consumed_scale_keys: set[str] = set()
        for hf_name, tensor in hf_state_dict.items():
            if hf_name.endswith("_scale") and hf_name[:-len("_scale")] in hf_state_dict:
                # Scale tensor belonging to a packed FP4 weight we'll handle
                # alongside the weight itself; mark consumed and skip.
                consumed_scale_keys.add(hf_name)
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                dequantized[hf_name] = tensor.to(torch.bfloat16)
                continue
            scale_key = hf_name + "_scale"
            if is_packed_nvfp4(tensor) and scale_key in hf_state_dict:
                # Native packed-FP4 dtype + matching scale → dequant to BF16.
                dequantized[hf_name] = dequantize_nvfp4_to_bf16(
                    tensor, hf_state_dict[scale_key]
                )
                consumed_scale_keys.add(scale_key)
            else:
                dequantized[hf_name] = tensor

        # Step 2: rename HF MoE expert names to our `w1/w2/w3` convention.
        # Same rules as V2/V3; the V4 reference uses identical per-expert
        # names since the MoE block structure didn't change between
        # generations.
        remapped: dict[str, torch.Tensor] = {}
        for hf_name, tensor in dequantized.items():
            new_name = hf_name
            for pattern, replacement in _V4_RENAME_RULES:
                if pattern.search(new_name):
                    new_name = pattern.sub(replacement, new_name)
                    break
            remapped[new_name] = tensor

        # Step 3: TP-aware load. `load_state_dict_with_tp` slices each
        # full weight via the matching `load_full_weight` / `load_full_logits`
        # / `load_full_wo_a` helper on the TP-aware layers.
        missing, unexpected = load_state_dict_with_tp(model, remapped)
        whitelist = model.expected_missing_state_keys()
        missing = {m for m in missing if m not in whitelist}
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for DeepseekV4ForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
