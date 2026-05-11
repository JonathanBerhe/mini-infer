"""Mixtral family: owned model implementation.

Mixtral 8x7B / 8x22B are Llama-shape backbones (RMSNorm + RoPE + GQA)
with the FFN replaced by a top-k sparse MoE: one router + N expert
SwiGLU-shaped MLPs per layer, top-2 of 8 routing.

Composes from `mini_infer.models.blocks.*`. Parameter names match HF
Mixtral's safetensors convention exactly (`block_sparse_moe.gate.weight`,
`block_sparse_moe.experts.<j>.w1/w2/w3.weight`) so weight loading is
identity rename — no per-key remapping table.
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
from mini_infer.models.blocks import MixtralDecoderLayer, RMSNorm, RotaryEmbedding

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class MixtralConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    num_local_experts: int
    num_experts_per_tok: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf_config: Any) -> MixtralConfig:
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        # transformers 5.x stores rope_theta inside `rope_parameters`; 4.x
        # used a top-level attr. Same fallback as Qwen2 / Llama / Gemma.
        rope_params = getattr(hf_config, "rope_parameters", None)
        if rope_params is not None and "rope_theta" in rope_params:
            rope_theta = float(rope_params["rope_theta"])
        else:
            rope_theta = float(getattr(hf_config, "rope_theta", 10000.0))
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
            num_local_experts=hf_config.num_local_experts,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )


class _MixtralInnerModel(nn.Module):
    """Embedding + N decoder blocks + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: MixtralConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                MixtralDecoderLayer(
                    hidden_size=cfg.hidden_size,
                    num_q_heads=cfg.num_attention_heads,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    intermediate_size=cfg.intermediate_size,
                    num_experts=cfg.num_local_experts,
                    top_k=cfg.num_experts_per_tok,
                    rms_norm_eps=cfg.rms_norm_eps,
                    layer_idx=i,
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


@register_model
class MixtralForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "MixtralForCausalLM"
    Config: ClassVar[type] = MixtralConfig

    def __init__(self, cfg: MixtralConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _MixtralInnerModel(cfg)
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
        if not isinstance(model, MixtralForCausalLM):
            raise TypeError(
                f"MixtralForCausalLM.load_weights expects a MixtralForCausalLM, "
                f"got {type(model).__name__}"
            )
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
        whitelist = model.expected_missing_state_keys()
        missing = [m for m in missing if m not in whitelist]
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for MixtralForCausalLM: "
                f"missing={missing}, unexpected={unexpected}"
            )
