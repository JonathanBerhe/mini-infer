"""HF safetensors -> state_dict loading for owned models.

Resolves the checkpoint via `huggingface_hub.snapshot_download` (already
a transitive dependency through `transformers`), then reads either
`model.safetensors` (single-file) or the sharded `model.safetensors.index.json`
manifest. Returns a flat `dict[str, Tensor]` ready for
`model.load_state_dict(state_dict, strict=True)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load_safetensors_state_dict(
    name_or_path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Return the model's full state_dict as a flat name -> tensor mapping.

    Tensors are placed on `device` and cast to `dtype`. `name_or_path` is
    either a local directory or an HF Hub repo id; the latter triggers a
    cache-respecting download via `snapshot_download`.

    `safetensors.torch.load_file` accepts a `device` arg and loads tensors
    directly into that device's memory (no CPU intermediate), which keeps
    host RAM bounded by one shard for sharded checkpoints. We still cast
    to `dtype` after; large checkpoints stored in bf16 don't need an
    extra cast on bf16-accepting devices, but the explicit cast makes
    the resulting state_dict's dtype contract unambiguous.
    """
    local_dir = _resolve_local_dir(name_or_path)
    index_path = local_dir / "model.safetensors.index.json"
    target_device = torch.device(device)
    # `safetensors.load_file(device=...)` accepts `cuda` / `cpu` strings and
    # `cuda:0` etc.; pass through the canonical form.
    safetensors_device = (
        f"{target_device.type}:{target_device.index}"
        if target_device.index is not None
        else target_device.type
    )

    # Block-quantized dtypes (FP8 e4m3 / e8m0, packed FP4) must NOT be cast
    # to BF16 here — their values only make sense after a per-block dequant
    # that multiplies by the matching scale tensor. Casting prematurely
    # would (a) destroy the per-block scale relationship and (b) blow up
    # storage 2x-4x (FP8/FP4 are 0.5-1 byte/elem, BF16 is 2 bytes).
    quantized_dtypes = {torch.float8_e4m3fn}
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        quantized_dtypes.add(e8m0_dtype)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None:
        quantized_dtypes.add(fp4_dtype)

    def _maybe_cast(t: torch.Tensor) -> torch.Tensor:
        if t.dtype in quantized_dtypes:
            return t
        return t.to(dtype=dtype) if t.dtype != dtype else t

    state_dict: dict[str, torch.Tensor] = {}
    if index_path.exists():
        with index_path.open() as f:
            manifest = json.load(f)
        weight_map = manifest["weight_map"]  # {tensor_name: shard_filename}
        shards = sorted(set(weight_map.values()))
        for shard in shards:
            shard_dict = load_file(str(local_dir / shard), device=safetensors_device)
            for k, v in shard_dict.items():
                state_dict[k] = _maybe_cast(v)
    else:
        single_file = local_dir / "model.safetensors"
        if not single_file.exists():
            raise FileNotFoundError(
                f"no model.safetensors or model.safetensors.index.json under {local_dir}"
            )
        loaded = load_file(str(single_file), device=safetensors_device)
        for k, v in loaded.items():
            state_dict[k] = _maybe_cast(v)
    return state_dict


def _resolve_local_dir(name_or_path: str) -> Path:
    """Return a local dir for the checkpoint; download if `name_or_path` is a Hub id."""
    candidate = Path(name_or_path)
    if candidate.is_dir():
        return candidate
    cached = snapshot_download(name_or_path, allow_patterns=["*.safetensors", "*.json"])
    return Path(cached)
