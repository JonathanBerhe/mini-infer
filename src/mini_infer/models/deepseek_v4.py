"""DeepSeek-V4 family: hybrid CSA / HCA attention backbone.

Targets `DeepseekV4ForCausalLM` (paper `huggingface.co/deepseek-ai/
DeepSeek-V4-Pro` once weights are public). The portfolio-relevant piece
is the attention itself (V4 paper §2.3) — `HCAAttention`, `CSAAttention`,
`LightningIndexer`, `TokenLevelCompressor`, `AttentionSink`,
`GroupedOutputProjection`, all bit-parity validated against the upstream
inference reference.

Architectural deltas from V4-published vs what we ship:

  - **No Hyper-Connections**: V4's `Block` (paper §2.5) replaces the
    standard residual with a Sinkhorn-mixed `(B, T, hc_mult, dim)`
    multi-residual. Orthogonal to the attention contribution; we use
    vanilla pre-norm residuals so the backbone is reviewable.
  - **No MoE FFN**: V4-Pro / V4-Flash use MoE with hash routing for
    the first `n_hash_layers` layers and softmax-topk after. We use a
    plain `SwiGLU` FFN. (mini-infer already has Mixtral-style MoE; the
    V4-specific hash-routing piece is a separate stage.)
  - **No YaRN long-context scaling**: V4's RoPE uses YaRN at >4k
    contexts; we use vanilla RoPE.
  - **No on-disk KV** (V4 paper §3.6.2): orthogonal storage layer.

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

`load_weights` raises until V4 checkpoints are public — the architecture
class is registered so that an HF config with `architectures=
["DeepseekV4ForCausalLM"]` would resolve here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from torch import nn

from mini_infer.cache.state_cache import IndexerStateSpec, StateCache, StateLayerSpec
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks.deepseek_v4_decoder_layer import DeepseekV4DecoderLayer
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import RotaryEmbedding

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

        V4 checkpoints aren't public yet, so the field names are educated
        guesses based on the reference inference code's `ModelArgs`. Adjust
        once the actual HF config schema lands.
        """

        # Pick the first attribute that's set on `hf_config` and not None,
        # falling back to `default_value` if none of the candidates apply.
        # mini-infer + paper field names sometimes diverge (e.g.
        # `index_n_heads` vs `index_num_heads`); this isolates the alias
        # logic from the typed dataclass call below.
        def _pick(*candidate_names: str, default_value: Any) -> Any:
            for name in candidate_names:
                value = getattr(hf_config, name, None)
                if value is not None:
                    return value
            return default_value

        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta_raw = rope_params.get("rope_theta") or _pick("rope_theta", default_value=10000.0)
        compress_ratios_raw = _pick("compress_ratios", default_value=None)
        if compress_ratios_raw is None:
            raise ValueError("DeepseekV4Config.from_hf: HF config missing `compress_ratios` field")
        return cls(
            vocab_size=int(hf_config.vocab_size),
            hidden_size=int(hf_config.hidden_size),
            intermediate_size=int(hf_config.intermediate_size),
            num_hidden_layers=int(hf_config.num_hidden_layers),
            num_attention_heads=int(hf_config.num_attention_heads),
            q_lora_rank=int(hf_config.q_lora_rank),
            kv_head_dim=int(_pick("kv_head_dim", "head_dim", default_value=0)),
            rope_head_dim=int(hf_config.rope_head_dim),
            o_num_groups=int(_pick("o_num_groups", "o_groups", default_value=0)),
            o_lora_rank=int(hf_config.o_lora_rank),
            window_size=int(hf_config.window_size),
            compress_ratios=tuple(int(ratio) for ratio in compress_ratios_raw),
            index_num_heads=int(_pick("index_num_heads", "index_n_heads", default_value=0)),
            index_head_dim=int(_pick("index_head_dim", default_value=0)),
            index_top_k=int(_pick("index_top_k", "index_topk", default_value=0)),
            rms_norm_eps=float(_pick("rms_norm_eps", "norm_eps", default_value=1e-6)),
            rope_theta=float(rope_theta_raw),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
            yarn_original_seq_len=int(
                _pick("yarn_original_seq_len", "original_seq_len", default_value=0)
            ),
            yarn_scaling_factor=float(
                _pick("yarn_scaling_factor", "rope_factor", default_value=1.0)
            ),
            yarn_beta_fast=int(_pick("yarn_beta_fast", "beta_fast", default_value=32)),
            yarn_beta_slow=int(_pick("yarn_beta_slow", "beta_slow", default_value=1)),
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
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
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
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
        cos_for_tokens, sin_for_tokens = self.rotary_emb(hidden_states, position_ids)
        token_position_embeddings = (cos_for_tokens, sin_for_tokens)

        for layer_idx, layer in enumerate(self.model.layers):
            compression_ratio = self.cfg.compress_ratios[layer_idx]
            num_compressed_blocks = seqlen // compression_ratio
            block_positions = (
                (torch.arange(num_compressed_blocks, device=input_ids.device) * compression_ratio)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            cos_for_blocks, sin_for_blocks = self.rotary_emb(
                hidden_states[:, :num_compressed_blocks], block_positions
            )
            compressed_position_embeddings = (cos_for_blocks, sin_for_blocks)
            hidden_states = layer(
                hidden_states,
                token_position_embeddings,
                compressed_position_embeddings,
                input_ids=input_ids if self.cfg.use_moe_ffn else None,
            )

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
        raise NotImplementedError(
            "DeepseekV4ForCausalLM.load_weights: V4 checkpoints aren't public yet, "
            "and the published architecture uses MoE FFN + Hyper-Connections that "
            "this implementation doesn't yet replicate. The class is registered so "
            "the attention pieces (HCAAttention, CSAAttention, LightningIndexer, "
            "TokenLevelCompressor, AttentionSink, GroupedOutputProjection) can be "
            "exercised end-to-end on synthetic configs."
        )
