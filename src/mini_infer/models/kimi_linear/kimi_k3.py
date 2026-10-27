"""Kimi K3 (moonshotai/Kimi-K3): text-only serving of the vision-language checkpoint.

The released checkpoint is `KimiK3ForConditionalGeneration`, a LLaVA-style
wrapper (`modeling_kimi_k3.py`) around three parts:

  - `vision_tower` (MoonViT-V2, 401M) and `mm_projector`: dropped at load,
    like the Inkling and Gemma 4 ports;
  - `language_model`: a `KimiLinearForCausalLM` built from `text_config`
    (`model_type: kimi_linear`), the hybrid KDA / NoPE-MLA stack of
    `models/kimi_linear/` with the K3 delta switched on by its config
    (SiTU, low-rank Q + MLA output gate, full-rank KDA gate with a bounded
    decay, latent MoE, AttnRes).

So this module is a thin wrapper: the config reads `text_config`, the model
IS the text stack, and `load_weights` translates the checkpoint layout:

  1. drop the vision keys;
  2. strip the `language_model.` prefix;
  3. dequantize the MXFP4 routed experts (`experts.N.w{1,2,3}.weight_packed`
     + `.weight_scale`, compressed-tensors `mxfp4-pack-quantized`, group 32)
     to `.weight` with `quant/mxfp4.py`. Attention, shared experts, the
     dense MLP, the latent projections, the AttnRes queries and `lm_head`
     ship unquantized.

Resident MXFP4 experts (dequant inside the grouped GEMM, as the DeepSeek-V4
FP4 path does) are the optimization step after parity; this loader
materializes BF16 experts, which is what the parity and structural tests
exercise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import torch

from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM
from mini_infer.models.kimi_linear.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM
from mini_infer.quant.mxfp4 import dequantize_mxfp4_to_bf16

_DROP_RE = re.compile(r"^(vision_tower|mm_projector)\.")
_LANGUAGE_PREFIX = "language_model."
_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"


@dataclass
class KimiK3Config:
    """The wrapper config: only `text_config` matters for text-only serving."""

    text: KimiLinearConfig

    @classmethod
    def from_hf(cls, hf_config: Any) -> KimiK3Config:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is None:
            raise ValueError(
                "KimiK3Config requires `text_config` (the Kimi K3 wrapper layout); "
                "a text-only kimi_linear config loads through KimiLinearForCausalLM"
            )
        if isinstance(text_config, dict):
            # `_raw_config_namespace` keeps nested dicts as dicts.
            text_config = SimpleNamespace(**text_config)
        return cls(text=KimiLinearConfig.from_hf(text_config))


@register_model
class KimiK3ForConditionalGeneration(KimiLinearForCausalLM):
    """Text-only Kimi K3: the `kimi_linear` stack plus the checkpoint translation."""

    HF_ARCHITECTURE: ClassVar[str] = "KimiK3ForConditionalGeneration"
    Config: ClassVar[type] = KimiK3Config

    def __init__(self, cfg: KimiK3Config) -> None:
        super().__init__(cfg.text)

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, KimiK3ForConditionalGeneration):
            raise TypeError(
                f"KimiK3ForConditionalGeneration.load_weights expects a "
                f"KimiK3ForConditionalGeneration, got {type(model).__name__}"
            )
        KimiLinearForCausalLM.load_weights(model, text_state_dict_from_k3(hf_state_dict))


def text_state_dict_from_k3(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint tensors -> the `KimiLinearForCausalLM` layout.

    Drops the vision tower and projector, strips `language_model.` (keys
    without it pass through, so an already text-only extraction loads too),
    and replaces every MXFP4 `weight_packed` / `weight_scale` pair by its
    dequantized BF16 `weight`. A packed tensor without its scale (or the
    reverse) is a corrupt shard, so it fails loudly.
    """
    text: dict[str, torch.Tensor] = {}
    packed: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        if _DROP_RE.match(key):
            continue
        name = key[len(_LANGUAGE_PREFIX) :] if key.startswith(_LANGUAGE_PREFIX) else key
        if name.endswith(_PACKED_SUFFIX):
            packed[name[: -len(_PACKED_SUFFIX)]] = tensor
        elif name.endswith(_SCALE_SUFFIX):
            scales[name[: -len(_SCALE_SUFFIX)]] = tensor
        else:
            text[name] = tensor
    if set(packed) != set(scales):
        orphans = sorted(set(packed) ^ set(scales))
        raise ValueError(f"MXFP4 weight_packed / weight_scale pairs do not match: {orphans[:5]}")
    for module_name, packed_weight in packed.items():
        text[f"{module_name}.weight"] = dequantize_mxfp4_to_bf16(packed_weight, scales[module_name])
    return text
