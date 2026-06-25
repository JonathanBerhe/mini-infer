"""Owned model registry and entry point.

`load_model(name, ...)` looks up an HF checkpoint's architecture in
`REGISTRY`, instantiates our matching model class, and loads HF
safetensors weights into it. Adding a new model means writing a class
under `mini_infer.models.<family>` that inherits `BaseCausalLM`,
declares `HF_ARCHITECTURE`, decorates with `@register_model`, and is
imported from `_register_builtin_models()` below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mini_infer.models.base import BaseCausalLM

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Maps an HF `config.architectures[0]` string to our model class."""

    def __init__(self) -> None:
        self._models: dict[str, type[BaseCausalLM]] = {}

    def register(self, model_cls: type[BaseCausalLM]) -> None:
        """Register `model_cls` under its `HF_ARCHITECTURE` key.

        Logs a warning on duplicate registration so silent overwrites
        (typically caused by accidental double-registration in tests
        or plugins) are visible.
        """
        if not model_cls.HF_ARCHITECTURE:
            raise ValueError(
                f"{model_cls.__name__} must declare a non-empty HF_ARCHITECTURE class attr"
            )
        existing = self._models.get(model_cls.HF_ARCHITECTURE)
        if existing is not None and existing is not model_cls:
            logger.warning(
                "ModelRegistry: %r is being overwritten (was %s, now %s)",
                model_cls.HF_ARCHITECTURE,
                existing.__name__,
                model_cls.__name__,
            )
        self._models[model_cls.HF_ARCHITECTURE] = model_cls

    def lookup(self, hf_architecture: str) -> type[BaseCausalLM]:
        if hf_architecture not in self._models:
            raise ValueError(
                f"no owned model for HF architecture {hf_architecture!r}; "
                f"supported: {sorted(self._models)}"
            )
        return self._models[hf_architecture]


REGISTRY = ModelRegistry()


def register_model(model_cls: type[BaseCausalLM]) -> type[BaseCausalLM]:
    """Class decorator that registers `model_cls` with the global `REGISTRY`.

    Used as `@register_model` directly above the class definition so the
    side effect is visible at the registration site.
    """
    REGISTRY.register(model_cls)
    return model_cls


def load_model(name: str, *, dtype: torch.dtype, device: str) -> BaseCausalLM:
    """Load an HF checkpoint into our owned model implementation.

    Pulls config + weights via HF's APIs, looks up our class in
    `REGISTRY`, instantiates it, then copies the HF state_dict in.
    HF tokenizer is constructed separately by the caller.
    """
    from transformers import AutoConfig

    from mini_infer.models.loader import load_safetensors_state_dict

    hf_config = AutoConfig.from_pretrained(name)
    if not hf_config.architectures:
        raise ValueError(f"HF config for {name!r} has no `architectures`; can't dispatch")
    arch = hf_config.architectures[0]

    model_cls = REGISTRY.lookup(arch)
    config_cls = model_cls.Config
    cfg = config_cls.from_hf(hf_config)  # type: ignore[attr-defined]
    model = model_cls(cfg).to(device=device, dtype=dtype)
    state_dict = load_safetensors_state_dict(name, device=device, dtype=dtype)
    model_cls.load_weights(model, state_dict)
    model.eval()
    return model


def architecture_uses_state_cache(name: str) -> bool:
    """Whether `name`'s architecture decodes via a per-request StateCache.

    Reads `config.json`'s `architectures[0]` directly (transformers drops
    V4-only fields when building a fallback config, but `architectures` is
    preserved) and checks the registered class's `USES_STATE_CACHE` marker.
    The API server uses this to route V4 to the StateCacheContinuousScheduler.
    Returns False (the PagedKVCache path) if the architecture is unregistered.
    """
    import json
    from pathlib import Path

    candidate = Path(name)
    if candidate.is_dir():
        config_path = candidate / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(name, "config.json"))
    with config_path.open() as config_file:
        architectures = json.load(config_file).get("architectures") or []
    if not architectures:
        return False
    try:
        model_cls = REGISTRY.lookup(architectures[0])
    except ValueError:
        return False
    return bool(model_cls.USES_STATE_CACHE)


def _register_builtin_models() -> None:
    """Import builtin model modules so their `@register_model` decorators fire.

    Side-effect-only: each imported module mutates `REGISTRY`. Kept as an
    explicit function (vs a bare bottom-of-file import) so the side-effect
    has a name and a docstring.
    """
    from mini_infer.models import deepseek_v2 as _deepseek_v2  # noqa: F401
    from mini_infer.models import deepseek_v4 as _deepseek_v4  # noqa: F401
    from mini_infer.models import gemma3 as _gemma3  # noqa: F401
    from mini_infer.models import gemma4 as _gemma4  # noqa: F401
    from mini_infer.models import llama as _llama  # noqa: F401
    from mini_infer.models import mistral as _mistral  # noqa: F401
    from mini_infer.models import mixtral as _mixtral  # noqa: F401
    from mini_infer.models import qwen2 as _qwen2  # noqa: F401
    from mini_infer.models import qwen3 as _qwen3  # noqa: F401


_register_builtin_models()


__all__ = [
    "REGISTRY",
    "ModelRegistry",
    "architecture_uses_state_cache",
    "load_model",
    "register_model",
]
