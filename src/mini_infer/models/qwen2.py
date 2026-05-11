"""Qwen2 (Qwen2.5) family: owned model implementation.

Llama-shape backbone with one twist: Q/K/V projections have learnable
biases (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`). Everything else is
the same as Llama: RMSNorm + RoPE + GQA + SwiGLU.

Composes from `mini_infer.models.blocks.*`. Parameter names match HF's
`Qwen2ForCausalLM` exactly so weight loading is `model.load_state_dict`
on the HF safetensors with `strict=True`.
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
class Qwen2Config:
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
    def from_hf(cls, hf_config: Any) -> Qwen2Config:
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        # transformers 5.x moved `rope_theta` into a `rope_parameters` dict;
        # 4.x exposed it as a top-level attr. Read whichever is present.
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
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )


class _Qwen2InnerModel(nn.Module):
    """Embedding + N decoder blocks + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: Qwen2Config) -> None:
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
                    qkv_bias=True,
                    layer_idx=i,
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


@register_model
class Qwen2ForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "Qwen2ForCausalLM"
    Config: ClassVar[type] = Qwen2Config

    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _Qwen2InnerModel(cfg)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta)
        # Always construct lm_head as a Linear so downstream code (int8 quant
        # walker, callers reading `model.lm_head`) sees a consistent module.
        # When `tie_word_embeddings` is set, we point lm_head's weight at the
        # embedding's weight; loading either updates both. Quantizing the
        # lm_head later naturally untangles the tie (the new Int8Linear has
        # its own packed weight) which matches HF's behavior.
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
        # Tied embeddings: `lm_head.weight` aliases `model.embed_tokens.weight`.
        # HF's safetensors shard ships only the embed entry, so PyTorch's
        # `load_state_dict` reports `lm_head.weight` as missing even though
        # the value is correct via the tie.
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        # Qwen2's HF parameter names match our module hierarchy directly:
        # `model.embed_tokens.weight`, `model.layers.<i>.self_attn.q_proj.weight`
        # (and `.bias`), `model.layers.<i>.mlp.gate_proj.weight`,
        # `model.layers.<i>.input_layernorm.weight`, `model.norm.weight`,
        # `lm_head.weight`.
        if not isinstance(model, Qwen2ForCausalLM):
            raise TypeError(
                f"Qwen2ForCausalLM.load_weights expects a Qwen2ForCausalLM, "
                f"got {type(model).__name__}"
            )
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
        whitelist = model.expected_missing_state_keys()
        missing = [m for m in missing if m not in whitelist]
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for Qwen2ForCausalLM: "
                f"missing={missing}, unexpected={unexpected}"
            )
