"""Gemma 4 family: owned model implementation (text-only dense).

Targets `Gemma4ForConditionalGeneration` — the multimodal-wrapped Gemma 4
checkpoint published by Google (e.g. `google/gemma-4-31B-it`). The model
exposed here is the TEXT decoder only; vision and audio towers in the
checkpoint are filtered out at load time.

Compared to Gemma 3 the text decoder differs in:

  - **Heterogeneous per-layer attention shape.** Sliding layers carry
    `(num_kv_heads=16, head_dim=256)`; full ("global") layers carry
    `(num_kv_heads=4, head_dim=512)`. This is the first model file to
    actually exercise the heterogeneous-KV BlockPool primitive (Stage
    C1). `per_layer_kv_shape()` reports the per-layer pair.
  - **Dual RoPE with different head_dim per type.** Sliding RoPE is
    sized to head_dim=256 with theta=10000 and full rotation. Full-layer
    RoPE is sized to head_dim=512 with theta=1000000 and proportional
    rotation `partial_rotary_factor=0.25` — only the first 64 dims
    rotate, the rest pass through.
  - **`attention_k_eq_v=True` on full layers.** No separate `v_proj`;
    V reuses the post-`k_proj` tensor (BEFORE `k_norm` and BEFORE RoPE).
  - **`v_norm` per layer.** Unscaled RMSNorm (no learnable weight) on
    V after capture. Affects every layer.
  - **`layer_scalar` per layer.** A `(1,)` buffer applied to the block
    output as `hidden_states *= layer_scalar`.
  - **Standard RMSNorm, not Gemma 3's offset variant.** All norm weights
    are in their "final" (close-to-1) form on disk — no `(1+weight)`
    transform on forward.
  - **Final logit softcap.** `logits = tanh(logits / 30) * 30` after
    `lm_head`.
  - **Softmax scale of 1.0.** `query_scale=1.0` in the GQA dispatch;
    q_norm/k_norm absorb the magnitude.

Variants we do NOT implement here (different code paths):

  - 26B-A4B MoE (`enable_moe_block=True`)
  - E2B / E4B with PLE + shared KV (`hidden_size_per_layer_input>0` or
    `num_kv_shared_layers>0`)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.cache.block_pool import LayerAttentionSpec
from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import Gemma4DecoderLayer, RMSNorm, RotaryEmbedding

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


# Multimodal weight prefixes that get filtered out before `load_state_dict`.
# Vision tower, audio tower, multimodal projector, and the embed_vision
# adapter all live under these — none are part of the text decoder.
_MULTIMODAL_PREFIX_RE = re.compile(
    r"^model\.(embed_vision|vision_tower|audio_tower|multi_modal_projector|embed_audio)"
)
# Per-layer key index for filtering full-layer v_proj weights when
# `attention_k_eq_v` makes them unused.
_LAYER_IDX_RE = re.compile(r"^model\.layers\.(\d+)\.")


@dataclass
class Gemma4Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    # Sliding-layer KV shape.
    num_key_value_heads: int
    head_dim: int
    # Full-layer KV shape.
    num_global_key_value_heads: int
    global_head_dim: int
    attention_k_eq_v: bool
    layer_types: list[str]
    sliding_window: int
    rms_norm_eps: float
    rope_theta_local: float
    rope_theta_global: float
    rope_partial_rotary_factor_global: float
    final_logit_softcapping: float | None
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf_config: Any) -> Gemma4Config:
        """Parse the multimodal-wrapped HF config into our dataclass.

        `hf_config` here is `Gemma4Config` (the multimodal one); the text
        decoder fields live under `hf_config.text_config`.
        """
        tc = getattr(hf_config, "text_config", None) or hf_config
        # `rope_parameters` is keyed by attention type. We rely on default
        # values matching the public 31B checkpoint when fields are absent.
        rope_params = getattr(tc, "rope_parameters", None) or {}
        local_params = rope_params.get("sliding_attention", {}) or {}
        global_params = rope_params.get("full_attention", {}) or {}
        layer_types = list(getattr(tc, "layer_types", []))
        if not layer_types:
            raise ValueError(
                "Gemma 4 config missing `layer_types`; cannot determine "
                "per-layer attention pattern."
            )
        head_dim = int(getattr(tc, "head_dim", None) or (tc.hidden_size // tc.num_attention_heads))
        global_head_dim = int(getattr(tc, "global_head_dim", None) or head_dim)
        return cls(
            vocab_size=tc.vocab_size,
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            num_hidden_layers=tc.num_hidden_layers,
            num_attention_heads=tc.num_attention_heads,
            num_key_value_heads=tc.num_key_value_heads,
            head_dim=head_dim,
            num_global_key_value_heads=int(
                getattr(tc, "num_global_key_value_heads", None) or tc.num_key_value_heads
            ),
            global_head_dim=global_head_dim,
            attention_k_eq_v=bool(getattr(tc, "attention_k_eq_v", False)),
            layer_types=layer_types,
            sliding_window=int(tc.sliding_window),
            rms_norm_eps=float(tc.rms_norm_eps),
            rope_theta_local=float(local_params.get("rope_theta", 10000.0)),
            rope_theta_global=float(global_params.get("rope_theta", 1000000.0)),
            rope_partial_rotary_factor_global=float(
                global_params.get("partial_rotary_factor", 1.0)
            ),
            final_logit_softcapping=getattr(tc, "final_logit_softcapping", None),
            tie_word_embeddings=bool(tc.tie_word_embeddings),
        )

    def is_full_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    def kv_shape_for_layer(self, layer_idx: int) -> tuple[int, int]:
        if self.is_full_layer(layer_idx):
            return (self.num_global_key_value_heads, self.global_head_dim)
        return (self.num_key_value_heads, self.head_dim)


class _Gemma4InnerModel(nn.Module):
    """Embedding + N decoder layers + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: Gemma4Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        layers: list[nn.Module] = []
        for layer_idx in range(cfg.num_hidden_layers):
            num_kv_heads, head_dim = cfg.kv_shape_for_layer(layer_idx)
            # `attention_k_eq_v` only fires on full layers per HF source
            # (`use_alternative_attention = config.attention_k_eq_v and not is_sliding`).
            attn_k_eq_v = cfg.attention_k_eq_v and cfg.is_full_layer(layer_idx)
            layers.append(
                Gemma4DecoderLayer(
                    hidden_size=cfg.hidden_size,
                    num_q_heads=cfg.num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    intermediate_size=cfg.intermediate_size,
                    rms_norm_eps=cfg.rms_norm_eps,
                    layer_idx=layer_idx,
                    attention_k_eq_v=attn_k_eq_v,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


@register_model
class Gemma4ForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "Gemma4ForConditionalGeneration"
    Config: ClassVar[type] = Gemma4Config

    def __init__(self, cfg: Gemma4Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _Gemma4InnerModel(cfg)
        # Two RoPE tables, one per layer-type. Sliding uses head_dim=256
        # with full rotation; full uses head_dim=512 with proportional
        # rotation (`partial_rotary_factor=0.25` ⇒ first 64 of 256 freq
        # pairs rotate, the remaining 192 are zero-padded).
        self.rotary_emb_local = RotaryEmbedding(
            cfg.head_dim,
            base=cfg.rope_theta_local,
            partial_rotary_factor=1.0,
        )
        self.rotary_emb_global = RotaryEmbedding(
            cfg.global_head_dim,
            base=cfg.rope_theta_global,
            partial_rotary_factor=cfg.rope_partial_rotary_factor_global,
        )
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
        # Compute both RoPE tables once; each layer reads the table sized
        # to its own head_dim (256 for sliding, 512 for full).
        cos_local, sin_local = self.rotary_emb_local(x, position_ids)
        cos_global, sin_global = self.rotary_emb_global(x, position_ids)
        for layer_idx, layer in enumerate(self.model.layers):
            pe = (
                (cos_global, sin_global)
                if self.cfg.is_full_layer(layer_idx)
                else (cos_local, sin_local)
            )
            x = layer(x, pe, past_key_values, cu_seqlens_q)
        x = self.model.norm(x)
        logits: torch.Tensor = self.lm_head(x)
        if self.cfg.final_logit_softcapping is not None:
            cap = self.cfg.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        # The pool ultimately uses `per_layer_kv_shape()` for the
        # heterogeneous storage. `kv_cache_dims` reports the sliding
        # shape (the more common layer type) so any consumer that
        # ignores per-layer shape still gets non-zero, sane defaults.
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
        )

    def per_layer_kv_shape(self) -> list[tuple[int, int]]:
        return [self.cfg.kv_shape_for_layer(i) for i in range(self.cfg.num_hidden_layers)]

    def per_layer_attention(self) -> list[LayerAttentionSpec]:
        result: list[LayerAttentionSpec] = []
        for kind in self.cfg.layer_types:
            if kind == "sliding_attention":
                result.append(("sliding", self.cfg.sliding_window))
            elif kind == "full_attention":
                result.append("full")
            else:
                raise ValueError(
                    f"unknown Gemma 4 layer_type {kind!r}; "
                    "expected 'sliding_attention' or 'full_attention'"
                )
        return result

    def expected_missing_state_keys(self) -> set[str]:
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    def required_attention_backend(self) -> str | None:
        # Full layers carry head_dim=512, which FlashAttention 2 and
        # FlashInfer's prefill kernel both reject (FA2 caps at 256;
        # FlashInfer's `BatchPrefillWithPagedKVCache` errors out with an
        # "Invalid configuration" runtime error). Force the materialized
        # PyTorch SDPA path for the whole model — same conclusion vLLM
        # and SGLang reach for Gemma 4 (they force their Triton unified
        # kernel; we don't ship one, so SDPA materialized is our
        # equivalent head_dim-agnostic fallback).
        if max(self.cfg.head_dim, self.cfg.global_head_dim) > 256:
            return "torch"
        return None

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, Gemma4ForCausalLM):
            raise TypeError(
                f"Gemma4ForCausalLM.load_weights expects a Gemma4ForCausalLM, "
                f"got {type(model).__name__}"
            )
        cfg = model.cfg
        remapped: dict[str, torch.Tensor] = {}
        for key, tensor in hf_state_dict.items():
            # Multimodal towers: drop entirely.
            if _MULTIMODAL_PREFIX_RE.match(key):
                continue
            # Strip the `language_model.` segment so HF's
            # `model.language_model.layers.X.foo` becomes our
            # `model.layers.X.foo`.
            if key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model.") :]
            else:
                new_key = key
            # `v_norm.weight`: HF sets `with_scale=False` so the module
            # has no `weight` parameter even though some checkpoints
            # ship the key. Drop it whether present or not.
            if new_key.endswith(".self_attn.v_norm.weight"):
                continue
            # `v_proj.weight` on full layers: our model leaves
            # `self_attn.v_proj` as `None` to match `attention_k_eq_v`
            # semantics; drop checkpoint copies that would otherwise
            # show up as unexpected keys.
            if new_key.endswith(".self_attn.v_proj.weight"):
                m = _LAYER_IDX_RE.match(new_key)
                if m is not None and cfg.is_full_layer(int(m.group(1))):
                    continue
            remapped[new_key] = tensor

        from mini_infer.distributed.loader import load_state_dict_with_tp

        missing, unexpected = load_state_dict_with_tp(model, remapped)
        whitelist = model.expected_missing_state_keys()
        missing = {m for m in missing if m not in whitelist}
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for Gemma4ForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
