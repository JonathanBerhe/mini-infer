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
IS the text stack, and `load_weights` drops the vision keys and strips the
`language_model.` prefix. The rest of the translation is the text stack's
own loader (`_remap_kimi_state`), so a text-only extraction that names
`KimiLinearForCausalLM` loads the same way. That includes the MXFP4 routed
experts (`experts.N.w{1,2,3}.weight_packed` + `.weight_scale`,
compressed-tensors `mxfp4-pack-quantized`, group 32), which the loader
dequantizes to BF16 `.weight` with `quant/mxfp4.py` when
`quantization_config` declares that format. Attention, shared
experts, the dense MLP, the latent projections, the AttnRes queries and
`lm_head` ship unquantized.
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

_DROP_RE = re.compile(r"^(vision_tower|mm_projector)\.")
_LANGUAGE_PREFIX = "language_model."


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
        # K3 nests `quantization_config` in `text_config`. A wrapper-level one
        # applies to the text model too.
        quant = getattr(hf_config, "quantization_config", None)
        if quant and not getattr(text_config, "quantization_config", None):
            text_config = SimpleNamespace(**{**vars(text_config), "quantization_config": quant})
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
    """Checkpoint tensors -> the text model's checkpoint names.

    Drops the vision tower and projector and strips `language_model.`. Keys
    without the prefix pass through, so an already text-only extraction
    loads too. The MXFP4 pairs stay packed here; `_remap_kimi_state`
    dequantizes them inside `KimiLinearForCausalLM.load_weights`.
    """
    text: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        if _DROP_RE.match(key):
            continue
        name = key[len(_LANGUAGE_PREFIX) :] if key.startswith(_LANGUAGE_PREFIX) else key
        text[name] = tensor
    return text
