"""Qwen3 family: owned model implementation.

Qwen3 0.6B / 1.7B / 4B / 8B / 14B / 32B (and the MoE variants we'll
add later). Architecturally a Llama-shape backbone with two changes
vs Qwen2:

- **No QKV biases** (`attention_bias=False`). Qwen2 had biased Q/K/V
  projections; Qwen3 dropped them.
- **Per-head Q/K norm** before RoPE. Same RMSNorm shape Gemma 3+ uses,
  but with the standard RMSNorm (no `+1` weight offset).

Tied embeddings throughout, single RoPE theta (no dual-RoPE / partial
RoPE complications). HF safetensors keys map to our module hierarchy
identity-rename.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import RMSNorm, RotaryEmbedding, TransformerBlock

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf_config: Any) -> Qwen3Config:
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        rope_params = getattr(hf_config, "rope_parameters", None)
        if rope_params is not None and "rope_theta" in rope_params:
            rope_theta = float(rope_params["rope_theta"])
        else:
            rope_theta = float(getattr(hf_config, "rope_theta", 10000.0))
        if getattr(hf_config, "attention_bias", False):
            raise NotImplementedError(
                "Qwen3 with attention_bias=True is not supported; "
                "this Qwen3 variant uses biased Q/K/V projections that the "
                "current owned implementation doesn't construct."
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
            rope_theta=rope_theta,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )


class _Qwen3InnerModel(nn.Module):
    """Embedding + N decoder blocks + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=cfg.hidden_size,
                    num_q_heads=cfg.num_attention_heads,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    intermediate_size=cfg.intermediate_size,
                    rms_norm_eps=cfg.rms_norm_eps,
                    qkv_bias=False,
                    layer_idx=i,
                    with_qk_norm=True,
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


@register_model
class Qwen3ForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "Qwen3ForCausalLM"
    Config: ClassVar[type] = Qwen3Config

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _Qwen3InnerModel(cfg)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta)
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
        x = self.model.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(x, position_ids)
        for layer in self.model.layers:
            x = layer(x, position_embeddings, past_key_values, cu_seqlens_q)
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

    def expected_missing_state_keys(self) -> set[str]:
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, Qwen3ForCausalLM):
            raise TypeError(
                f"Qwen3ForCausalLM.load_weights expects a Qwen3ForCausalLM, "
                f"got {type(model).__name__}"
            )
        from mini_infer.distributed.loader import load_state_dict_with_tp

        missing, unexpected = load_state_dict_with_tp(model, hf_state_dict)
        whitelist = model.expected_missing_state_keys()
        missing = {m for m in missing if m not in whitelist}
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for Qwen3ForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
