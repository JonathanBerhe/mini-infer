"""Mistral family: owned model implementation.

Mistral 7B (v0.1, v0.2, v0.3) is architecturally identical to our
Llama implementation: RMSNorm + RoPE + GQA + SwiGLU, no Q/K/V biases,
no Q/K norm. The only meaningful difference is the HF architecture
key (`MistralForCausalLM`). v0.1/v0.2 ship a `sliding_window` config
field but we don't honor it here — Mistral 7B v0.3 dropped SWA, and
the rare older variants would need explicit per-layer-attention
threading (a follow-up). Mistral Small / Large variants are also
covered by this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mini_infer.models import register_model
from mini_infer.models.llama import LlamaConfig, LlamaForCausalLM


@dataclass
class MistralConfig(LlamaConfig):
    """Identical surface to LlamaConfig; type-distinct for registry clarity."""


@register_model
class MistralForCausalLM(LlamaForCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "MistralForCausalLM"
    Config: ClassVar[type] = MistralConfig
