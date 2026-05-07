"""DeepSeek-V2 family: owned model implementation (text-only).

Targets `DeepseekV2ForCausalLM` — currently `deepseek-ai/DeepSeek-V2-Lite`
(16B, 27 layers, 64+2 expert MoE, MLA attention) and the larger V2 / V3
checkpoints once the low-rank Q path is exercised. The model exposed
here is the dense text decoder; the larger V2 / V3 deviations
(`q_lora_rank=1536`, `routed_scaling_factor=2.5`, YaRN RoPE scaling)
plug into the same class through `from_hf`.

Compared to Mixtral, the architectural deltas are:

  - **MLA attention** (`MLAAttention`): TWO compressed KV streams per
    layer (`kv_latent` 1xkv_lora_rank, `k_rope` 1xqk_rope_head_dim) instead
    of per-head K/V — ~7x smaller cache. See `mla.py` and
    `mla_attention.py`.
  - **Heterogeneous FFN per layer**: layer 0 is dense `SwiGLU`, layers
    `>= first_k_dense_replace` (1 in V2-Lite) use top-k MoE plus shared
    experts that fire on every token.
  - **Asymmetric Q/K vs V head_dim**: V's head_dim differs from Q/K's
    `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`. flash-attn 2 /
    FlashInfer prefill both reject this; we force the materialized
    PyTorch SDPA path via `required_attention_backend()`.
  - **Interleaved RoPE** (DeepSeek convention): pairs `(x[2i], x[2i+1])`
    rotate together rather than `(x[i], x[i+dim/2])` (Llama convention).
    Handled inside `MLAAttention` via `apply_interleaved_rotary_pos_emb`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.cache.block_pool import StreamSpec
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import MLAAttention, MoEFFN, RMSNorm, RotaryEmbedding, SwiGLU

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class DeepseekV2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int | None
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    first_k_dense_replace: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf_config: Any) -> DeepseekV2Config:
        # `rope_theta` moved into `rope_parameters` in newer transformers
        # versions (~4.50+). Fall back to the legacy top-level field for
        # older configs (e.g. the `auto_map`-loaded V2-Lite config which
        # still ships `rope_theta` directly).
        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = rope_params.get("rope_theta") or getattr(hf_config, "rope_theta", 10000.0)
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            kv_lora_rank=hf_config.kv_lora_rank,
            q_lora_rank=hf_config.q_lora_rank,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            n_routed_experts=hf_config.n_routed_experts,
            n_shared_experts=getattr(hf_config, "n_shared_experts", 0) or 0,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", False)),
            first_k_dense_replace=hf_config.first_k_dense_replace,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=float(rope_theta),
            attention_bias=bool(getattr(hf_config, "attention_bias", False)),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace


class _DeepseekV2DecoderLayer(nn.Module):
    """One DeepSeek-V2 transformer block: MLA attention + dense MLP or MoE.

    Standard pre/post-norm decoder shape (NOT Gemma's sandwich): residual
    around attention + post_attention_layernorm + residual around MLP/MoE.
    Layer 0 in V2-Lite is dense (`SwiGLU`); layers 1+ are MoE
    (`MoEFFN` with `n_shared_experts=2`).
    """

    def __init__(self, cfg: DeepseekV2Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            q_lora_rank=cfg.q_lora_rank,
            rms_norm_eps=cfg.rms_norm_eps,
            attention_bias=cfg.attention_bias,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # FFN: dense for first_k_dense_replace layers, MoE after.
        if cfg.is_moe_layer(layer_idx):
            self.mlp: nn.Module = MoEFFN(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.moe_intermediate_size,
                num_experts=cfg.n_routed_experts,
                top_k=cfg.num_experts_per_tok,
                n_shared_experts=cfg.n_shared_experts,
                shared_intermediate_size=cfg.moe_intermediate_size,
                renormalize_topk=cfg.norm_topk_prob,
                routed_scaling_factor=cfg.routed_scaling_factor,
            )
        else:
            self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

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


class _DeepseekV2InnerModel(nn.Module):
    """Embedding + N decoder layers + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: DeepseekV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [_DeepseekV2DecoderLayer(cfg, layer_idx) for layer_idx in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


# Translate HF MoE keys (`mlp.experts.{j}.{w}_proj.weight`) → our
# (`mlp.experts.{j}.w{N}.weight`). Mixtral checkpoints already match
# our block names; DeepSeek's per-expert checkpoint uses
# gate_proj/up_proj/down_proj instead.
_MOE_RENAME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.mlp\.experts\.(\d+)\.gate_proj\.weight$"), r".mlp.experts.\1.w1.weight"),
    (re.compile(r"\.mlp\.experts\.(\d+)\.down_proj\.weight$"), r".mlp.experts.\1.w2.weight"),
    (re.compile(r"\.mlp\.experts\.(\d+)\.up_proj\.weight$"), r".mlp.experts.\1.w3.weight"),
    # Shared experts: HF uses `mlp.shared_experts.{gate_proj,up_proj,down_proj}`,
    # ours is a single MixtralExpert with w1/w2/w3 names.
    (re.compile(r"\.mlp\.shared_experts\.gate_proj\.weight$"), r".mlp.shared_experts.w1.weight"),
    (re.compile(r"\.mlp\.shared_experts\.down_proj\.weight$"), r".mlp.shared_experts.w2.weight"),
    (re.compile(r"\.mlp\.shared_experts\.up_proj\.weight$"), r".mlp.shared_experts.w3.weight"),
]


@register_model
class DeepseekV2ForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "DeepseekV2ForCausalLM"
    Config: ClassVar[type] = DeepseekV2Config

    def __init__(self, cfg: DeepseekV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _DeepseekV2InnerModel(cfg)
        # RoPE only acts on the qk_rope_head_dim slice of Q/K. Default
        # rotation (no YaRN scaling) is used here — the V2-Lite
        # checkpoint configures YaRN, but it only kicks in past
        # `original_max_position_embeddings=4096`; short prompts route
        # through the default path. (YaRN long-context scaling is a
        # separate primitive on the roadmap.)
        self.rotary_emb = RotaryEmbedding(cfg.qk_rope_head_dim, base=cfg.rope_theta)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.model.layers:
            x = layer(x, (cos, sin), past_key_values, cu_seqlens_q)
        x = self.model.norm(x)
        logits: torch.Tensor = self.lm_head(x)
        return logits

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        # Heterogeneous storage descriptor handles the actual MLA shape
        # (`compressed_kv` 1xkv_lora_rank + `k_rope` 1xqk_rope_head_dim).
        # The legacy `kv_cache_dims` reports the larger of the two streams
        # so any consumer that ignores per-stream specs gets a sane size.
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=1,
            head_dim=self.cfg.kv_lora_rank,
        )

    def per_layer_streams(self) -> list[list[StreamSpec]]:
        kv_latent = StreamSpec("kv_latent", 1, self.cfg.kv_lora_rank)
        k_rope = StreamSpec("k_rope", 1, self.cfg.qk_rope_head_dim)
        return [[kv_latent, k_rope] for _ in range(self.cfg.num_hidden_layers)]

    def required_attention_backend(self) -> str | None:
        # Q/K head_dim != V head_dim breaks every flash-attn / FlashInfer
        # prefill kernel; the materialized SDPA reference in
        # `mla_packed_attention_forward` is the only path that handles it.
        # vLLM and SGLang reach the same conclusion (Triton unified
        # attention kernel).
        return "torch"

    def expected_missing_state_keys(self) -> set[str]:
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, DeepseekV2ForCausalLM):
            raise TypeError(
                f"DeepseekV2ForCausalLM.load_weights expects a DeepseekV2ForCausalLM, "
                f"got {type(model).__name__}"
            )
        remapped: dict[str, torch.Tensor] = {}
        for key, tensor in hf_state_dict.items():
            new_key = key
            for pattern, replacement in _MOE_RENAME_RULES:
                if pattern.search(new_key):
                    new_key = pattern.sub(replacement, new_key)
                    break
            remapped[new_key] = tensor
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        whitelist = model.expected_missing_state_keys()
        missing = [m for m in missing if m not in whitelist]
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for DeepseekV2ForCausalLM: "
                f"missing={missing}, unexpected={unexpected}"
            )
