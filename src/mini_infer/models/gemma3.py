"""Gemma 3 family: owned model implementation (text-only).

Gemma 3 alternates sliding-window attention layers with global attention
layers (typically 5 sliding : 1 global), uses dual RoPE (different theta
per attention type), per-head Q/K norm, sandwich norms around both
attention and FFN, GeGLU activations, GemmaRMSNorm, and embedding
scaling by `sqrt(hidden_size)`. Tied embeddings throughout.

Multimodal variants of Gemma 3 (vision/audio) ship under a separate HF
architecture string; this file targets `Gemma3ForCausalLM` only —
the text-only variants on HF Hub.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.cache.block_pool import LayerAttentionSpec
from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import GemmaDecoderLayer, GemmaRMSNorm, RotaryEmbedding

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class Gemma3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta_local: float
    rope_theta_global: float
    sliding_window: int
    layer_types: list[str]
    query_pre_attn_scalar: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf_config: Any) -> Gemma3Config:
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        # Gemma 3 ships per-attention-type RoPE config in `rope_parameters`:
        #   {"sliding_attention": {"rope_theta": 10000, ...},
        #    "full_attention":    {"rope_theta": 1000000, ...}}
        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        local_params = rope_params.get("sliding_attention", {})
        global_params = rope_params.get("full_attention", {})
        rope_theta_local = float(local_params.get("rope_theta", 10000.0))
        rope_theta_global = float(global_params.get("rope_theta", 1000000.0))
        layer_types = list(getattr(hf_config, "layer_types", []))
        if not layer_types:
            raise ValueError(
                "Gemma 3 config missing `layer_types`; cannot determine "
                "per-layer attention pattern."
            )
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            sliding_window=int(hf_config.sliding_window),
            layer_types=layer_types,
            query_pre_attn_scalar=int(hf_config.query_pre_attn_scalar),
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )


class _Gemma3InnerModel(nn.Module):
    """Embedding + N decoder blocks + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: Gemma3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        query_scale = 1.0 / math.sqrt(cfg.query_pre_attn_scalar)
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(
                    hidden_size=cfg.hidden_size,
                    num_q_heads=cfg.num_attention_heads,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    intermediate_size=cfg.intermediate_size,
                    rms_norm_eps=cfg.rms_norm_eps,
                    layer_idx=i,
                    query_scale=query_scale,
                    with_qk_norm=True,
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


@register_model
class Gemma3ForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "Gemma3ForCausalLM"
    Config: ClassVar[type] = Gemma3Config

    def __init__(self, cfg: Gemma3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _Gemma3InnerModel(cfg)
        # Two RoPE tables: sliding-window layers use the local (smaller) base,
        # global layers use the long-context base.
        self.rotary_emb_local = RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta_local)
        self.rotary_emb_global = RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta_global)
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False, gather_output=True
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        # Gemma scales token embeddings by sqrt(hidden_size) once at the input.
        x = self.model.embed_tokens(input_ids) * math.sqrt(self.cfg.hidden_size)
        cos_local, sin_local = self.rotary_emb_local(x, position_ids)
        cos_global, sin_global = self.rotary_emb_global(x, position_ids)
        for layer_idx, layer in enumerate(self.model.layers):
            kind = self.cfg.layer_types[layer_idx]
            pe = (cos_local, sin_local) if kind == "sliding_attention" else (cos_global, sin_global)
            x = layer(x, pe, past_key_values, cu_seqlens_q)
        x = self.model.norm(x)
        logits: torch.Tensor = self.lm_head(x)
        return logits

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
        )

    def per_layer_attention(self) -> list[LayerAttentionSpec]:
        result: list[LayerAttentionSpec] = []
        for kind in self.cfg.layer_types:
            if kind == "sliding_attention":
                result.append(("sliding", self.cfg.sliding_window))
            elif kind == "full_attention":
                result.append("full")
            else:
                raise ValueError(
                    f"unknown Gemma 3 layer_type {kind!r}; "
                    "expected 'sliding_attention' or 'full_attention'"
                )
        return result

    def expected_missing_state_keys(self) -> set[str]:
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, Gemma3ForCausalLM):
            raise TypeError(
                f"Gemma3ForCausalLM.load_weights expects a Gemma3ForCausalLM, "
                f"got {type(model).__name__}"
            )
        from mini_infer.distributed.loader import load_state_dict_with_tp

        missing, unexpected = load_state_dict_with_tp(model, hf_state_dict)
        whitelist = model.expected_missing_state_keys()
        missing = {m for m in missing if m not in whitelist}
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for Gemma3ForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
