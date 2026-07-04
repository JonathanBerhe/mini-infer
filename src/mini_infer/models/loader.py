"""HF safetensors -> state_dict loading for owned models.

Resolves the checkpoint via `huggingface_hub.snapshot_download` (already
a transitive dependency through `transformers`), then reads either
`model.safetensors` (single-file) or the sharded `model.safetensors.index.json`
manifest. Two entry points:

- `load_safetensors_state_dict`: the whole checkpoint as one flat dict. Fine
  up to checkpoints that fit host RAM.
- `iter_safetensors_shards`: one shard-sized dict at a time, for streaming
  loads of checkpoints that do NOT fit host RAM (e.g. MiniMax-M3's 854 GB);
  the consumer copies each shard's tensors into (possibly TP-sliced) module
  params and lets the shard free before the next one loads.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
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
    #
    # int8 / uint8 are preserved for the same reason: V4-Flash's safetensors
    # store packed NVFP4 expert weights as raw int8 bytes (the format's
    # `float4_e2m1fn_x2` type isn't carried in safetensors metadata, so two
    # FP4 nibbles arrive packed into one int8 at shape `(out, in // 2)`).
    # The downstream dequant detects them by this int8 dtype; casting to
    # BF16 here both defeats that detection AND leaves the tensor at its
    # packed half-width shape, so the un-dequantized `(out, in // 2)` weight
    # loads silently and only blows up at the first forward matmul.
    quantized_dtypes = _quantized_dtypes()

    def _maybe_cast(t: torch.Tensor) -> torch.Tensor:
        # Skip quantized dtypes (they need a pre-block dequant to be
        # meaningful; the caller's dequant handles them).
        if t.dtype in quantized_dtypes:
            return t
        # Preserve native FP32 sources. The DeepSeek-V4 Hyper-Connections
        # parameters (`hc_*_fn`, `hc_*_base`, `hc_*_scale`) are stored as
        # FP32 in the published safetensors and the V4 reference declares
        # them under `with set_dtype(torch.float32)`; the Sinkhorn math
        # inside `HyperConnections.hc_pre` casts the hidden state up to
        # FP32 and then multiplies against the param. Downcasting the
        # source to BF16 here would (a) silently lose precision and
        # (b) cause a `mat1 != mat2 dtype` error in the eventual
        # `F.linear(fp32_state, bf16_fn)`.
        if t.dtype == torch.float32:
            return t
        return t.to(dtype=dtype) if t.dtype != dtype else t

    state_dict: dict[str, torch.Tensor] = {}
    for shard_dict in _iter_shard_dicts(local_dir, safetensors_device, _maybe_cast):
        state_dict.update(shard_dict)
    return state_dict


def iter_safetensors_shards(
    name_or_path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield the checkpoint one shard-dict at a time (streaming counterpart of
    `load_safetensors_state_dict`, same device/dtype/quantized-dtype rules).

    Peak host memory is one shard (plus whatever the consumer retains), so
    checkpoints far larger than RAM can be loaded incrementally. Shards are
    yielded in sorted filename order; key->shard assignment follows the
    checkpoint's own manifest, so co-located tensors (e.g. an FP8 weight and
    its `weight_scale_inv` scale) arrive together when the writer put them in
    the same shard.
    """
    local_dir = _resolve_local_dir(name_or_path)
    target_device = torch.device(device)
    safetensors_device = (
        f"{target_device.type}:{target_device.index}"
        if target_device.index is not None
        else target_device.type
    )
    quantized_dtypes = _quantized_dtypes()

    def _maybe_cast(t: torch.Tensor) -> torch.Tensor:
        if t.dtype in quantized_dtypes or t.dtype == torch.float32:
            return t
        return t.to(dtype=dtype) if t.dtype != dtype else t

    yield from _iter_shard_dicts(local_dir, safetensors_device, _maybe_cast)


def _quantized_dtypes() -> set[torch.dtype]:
    """Dtypes that must never be blanket-cast (block-quantized storage)."""
    quantized = {torch.float8_e4m3fn, torch.int8, torch.uint8}
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        quantized.add(e8m0_dtype)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None:
        quantized.add(fp4_dtype)
    return quantized


def _iter_shard_dicts(
    local_dir: Path,
    safetensors_device: str,
    maybe_cast: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[dict[str, torch.Tensor]]:
    index_path = local_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            manifest = json.load(f)
        weight_map = manifest["weight_map"]  # {tensor_name: shard_filename}
        shards = sorted(set(weight_map.values()))
        for shard in shards:
            shard_dict = load_file(str(local_dir / shard), device=safetensors_device)
            yield {k: maybe_cast(v) for k, v in shard_dict.items()}
    else:
        single_file = local_dir / "model.safetensors"
        if not single_file.exists():
            raise FileNotFoundError(
                f"no model.safetensors or model.safetensors.index.json under {local_dir}"
            )
        loaded = load_file(str(single_file), device=safetensors_device)
        yield {k: maybe_cast(v) for k, v in loaded.items()}


def _resolve_local_dir(name_or_path: str) -> Path:
    """Return a local dir for the checkpoint; download if `name_or_path` is a Hub id."""
    candidate = Path(name_or_path)
    if candidate.is_dir():
        return candidate
    cached = snapshot_download(name_or_path, allow_patterns=["*.safetensors", "*.json"])
    return Path(cached)
