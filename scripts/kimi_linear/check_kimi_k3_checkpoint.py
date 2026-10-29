"""Check a Kimi K3-family checkpoint against our loader, reading only shard headers.

Run this the day a new checkpoint lands (Kimi K3.1 Pro), before any GPU run.
It answers three questions without downloading the weights:

  1. Does the config pass `Config.from_hf`, with all its config-time checks?
  2. What changed in `text_config` against the pinned Kimi K3 release?
  3. After the loader's key translation, does every checkpoint tensor land on
     a model parameter of the same shape, and does every parameter get one?

For step 3 the script reads each tensor's name, dtype and shape from the
safetensors headers (a few MB in all), builds the model on the meta device
(no memory for weights), and runs the loader's own translation on meta
tensors. If that is clean, it ends with the real `load_weights` call. The
loader's value checks (zero `A_log` padding, E8M0 NaN scales) need real
bytes, so they run only at a real load. CPU only, no GPU cost.

A Kimi Linear checkpoint works too; it skips the config diff.

Run with:
    uv run python scripts/kimi_linear/check_kimi_k3_checkpoint.py moonshotai/Kimi-K3.1-Pro
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

# The release `clone_kimi_k3_reference.py` vendors (the parity oracle).
BASELINE_REPO = "moonshotai/Kimi-K3"
BASELINE_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
}
_INDEX_RE = re.compile(r"\.(\d+)\.")


def _collapse(keys: Iterable[str]) -> list[str]:
    """Replace layer and expert indices by N and count each key pattern."""
    counts = Counter(_INDEX_RE.sub(".N.", _INDEX_RE.sub(".N.", key)) for key in keys)
    return [f"{count:7d}  {pattern}" for pattern, count in sorted(counts.items())]


def _config_json(repo: str, revision: str | None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, "config.json", revision=revision)
    with Path(path).open() as config_file:
        config: dict[str, Any] = json.load(config_file)
    return config


def _diff(old: Any, new: Any, path: str = "") -> list[str]:
    """Leaf-level differences between two JSON values."""
    if isinstance(old, dict) and isinstance(new, dict):
        lines: list[str] = []
        for key in sorted(set(old) | set(new)):
            child = f"{path}.{key}" if path else key
            if key not in old:
                lines.append(f"  + {child} = {_short(new[key])}")
            elif key not in new:
                lines.append(f"  - {child} (was {_short(old[key])})")
            else:
                lines.extend(_diff(old[key], new[key], child))
        return lines
    if old != new:
        return [f"  ~ {path}: {_short(old)} -> {_short(new)}"]
    return []


def _short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) <= 100 else text[:97] + "..."


def _meta_state(repo: str, revision: str | None) -> dict[str, torch.Tensor]:
    """Every checkpoint tensor as an empty meta tensor of its real dtype and shape."""
    from huggingface_hub import get_safetensors_metadata

    metadata = get_safetensors_metadata(repo, revision=revision)
    state: dict[str, torch.Tensor] = {}
    for file_metadata in metadata.files_metadata.values():
        for name, info in file_metadata.tensors.items():
            if info.dtype not in _SAFETENSORS_DTYPES:
                raise ValueError(f"{name}: unhandled safetensors dtype {info.dtype}")
            dtype = _SAFETENSORS_DTYPES[info.dtype]
            state[name] = torch.empty(info.shape, dtype=dtype, device="meta")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("repo_id", help="Hugging Face repo, e.g. moonshotai/Kimi-K3.1-Pro")
    parser.add_argument("--revision", default=None, help="branch, tag or commit (default: main)")
    args = parser.parse_args()

    from types import SimpleNamespace

    from mini_infer.models import REGISTRY
    from mini_infer.models.kimi_linear.kimi_k3 import (
        KimiK3ForConditionalGeneration,
        text_state_dict_from_k3,
    )
    from mini_infer.models.kimi_linear.kimi_linear import KimiLinearForCausalLM, _remap_kimi_state

    failures = 0

    print(f"== 1. config: {args.repo_id} @ {args.revision or 'main'}")
    raw = _config_json(args.repo_id, args.revision)
    architecture = (raw.get("architectures") or ["?"])[0]
    model_cls = REGISTRY.lookup(architecture)
    if not issubclass(model_cls, KimiLinearForCausalLM):
        print(f"FAIL: {architecture} is not a Kimi Linear / K3 architecture")
        return 1
    cfg = model_cls.Config.from_hf(SimpleNamespace(**raw))  # type: ignore[attr-defined]
    text_cfg = getattr(cfg, "text", cfg)
    print(
        f"ok: {architecture}, {text_cfg.num_hidden_layers} layers, "
        f"{text_cfg.num_experts} experts, MXFP4 experts: {text_cfg.mxfp4_weights}"
    )

    if model_cls is KimiK3ForConditionalGeneration:
        print(f"\n== 2. text_config against {BASELINE_REPO} @ {BASELINE_REVISION[:8]}")
        baseline = _config_json(BASELINE_REPO, BASELINE_REVISION)
        top_level = _diff(
            {k: v for k, v in baseline.items() if k not in ("text_config", "vision_config")},
            {k: v for k, v in raw.items() if k not in ("text_config", "vision_config")},
        )
        changes = top_level + _diff(baseline["text_config"], raw.get("text_config", {}))
        print("\n".join(changes) if changes else "no changes")

    print("\n== 3. tensors (shard headers only)")
    state = _meta_state(args.repo_id, args.revision)
    dtypes = Counter(str(tensor.dtype).removeprefix("torch.") for tensor in state.values())
    print(f"{len(state)} tensors: {dict(dtypes)}")
    with torch.device("meta"):
        model = model_cls(cfg)
    text_state = (
        text_state_dict_from_k3(state) if model_cls is KimiK3ForConditionalGeneration else state
    )
    remapped = _remap_kimi_state(
        text_state,
        mxfp4_weights=text_cfg.mxfp4_weights,
        kda_num_heads=text_cfg.kda_num_heads,
    )
    expected = model.state_dict()
    missing = set(expected) - set(remapped) - model.expected_missing_state_keys()
    unexpected = set(remapped) - set(expected)
    wrong_shape = [
        f"{key}: checkpoint {tuple(remapped[key].shape)} vs model {tuple(expected[key].shape)}"
        for key in sorted(set(remapped) & set(expected))
        if remapped[key].shape != expected[key].shape
    ]
    raw_bytes = [
        key
        for key in sorted(set(remapped) & set(expected))
        if not remapped[key].dtype.is_floating_point and expected[key].dtype.is_floating_point
    ]
    for title, items in (
        ("missing parameters", _collapse(missing)),
        ("unexpected checkpoint tensors", _collapse(unexpected)),
        ("integer tensors on float parameters (not dequantized)", _collapse(raw_bytes)),
        ("shape mismatches", wrong_shape[:20]),
    ):
        if items:
            failures += 1
            print(f"FAIL: {title}:")
            print("\n".join(f"  {item}" for item in items))
    if failures:
        return 1
    print(f"ok: {len(remapped)} tensors map onto {len(expected)} parameters")

    model_cls.load_weights(model, state)
    print("ok: load_weights accepts the checkpoint layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
