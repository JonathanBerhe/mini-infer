"""Stage MiniMax-M3 weights into a Modal Volume with routed experts as block-FP8.

The bf16 checkpoint is ~854 GB and the routed experts
(`language_model.model.layers.L.block_sparse_moe.experts.E.{w1,w3,w2}`) are the
bulk of it. Quantizing them to block-FP8 while staging (e4m3 weight + fp32
`weight_scale_inv` scale per [128, 128] tile) roughly halves the volume
(~450 GB) and the later GPU load time. Every other tensor (attention, router,
shared experts, norms, embeddings) is copied through byte-identical. The
transform is the exact inverse of `dequantize_block_fp8_to_bf16_partial` in
src/mini_infer/quant/nvfp4.py, which the loader applies at load time, and
config.json gets a `quantization_config` fp8 marker so the loader detects the
fp8 expert residency.

COST NOTES (read before running):
  - This function is CPU-only (no `gpu=`), so it bills CPU + egress, not GPUs.
    Download + quantize of the full checkpoint takes a few hours; the run is
    resumable (already-staged shards are skipped), so a timeout or preemption
    costs only the in-flight shard.
  - The Volume then STORES ~450 GB persistently, which bills per GB-month
    until you delete it. After the GPU runs are done, free it with:
        uv run modal volume delete minimax-m3-fp8-weights
  - There is no point running this until you intend to fund the GPU run; the
    staged weights are only useful paired with the fp8-resident expert path.

Run with:
    uv run modal run scripts/modal_m3_stage_weights.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import torch

_REPO = "MiniMaxAI/MiniMax-M3"
_REVISION = "bfd6c97f0296da547f10ecb20102c5d51a5c462e"
_VOLUME_NAME = "minimax-m3-fp8-weights"
_MOUNT = "/weights"
_TARGET = f"{_MOUNT}/MiniMax-M3"
_SCRATCH = "/scratch"
_INDEX_NAME = "model.safetensors.index.json"
_COMMIT_EVERY = 4  # shards between volume commits; bounds the uncommitted write buffer
_FP8_BLOCK = 128
_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max

# Routed expert weights (bf16 on disk): quantize these and only these.
# w1/w3 are [3072, 6144] (gate/up), w2 is [6144, 3072] (down); E 0..127, L 3..59.
_EXPERT_WEIGHT_RE = re.compile(
    r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.(w1|w2|w3)\.weight$"
)

app = modal.App("mini-infer-m3-stage")
weights_volume = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU wheel index: the default PyPI wheel drags in multi-GB CUDA libraries
    # that a CPU-only staging container never uses.
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "huggingface_hub>=0.20",
        "hf-transfer>=0.1.6",  # parallel, fast LFS downloads
        "safetensors>=0.4",
    )
)


def _quant_block_fp8(w: torch.Tensor, block: int = _FP8_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a 2-D weight to e4m3 plus a per-tile fp32 scale.

    Exact inverse of `dequantize_block_fp8_to_bf16_partial`
    (src/mini_infer/quant/nvfp4.py): the scale grid is
    `(ceil(rows / 128), ceil(cols / 128))`, each scale is the tile absmax
    divided by the e4m3 max (448), and dequant multiplies q by the scale.
    Vectorized so a full shard quantizes in seconds, but bit-identical to the
    per-tile reference loop in tests/unit/test_glm_fp8_load.py: the fp64
    divide before the fp32 store mirrors its `float(blk.abs().max()) / 448.0`.
    """
    import torch

    rows, cols = w.shape
    nbm = -(-rows // block)
    nbn = -(-cols // block)
    # Zero-pad partial edge tiles; abs-max over a tile is unaffected by zeros.
    padded = torch.zeros((nbm * block, nbn * block), dtype=torch.float32)
    padded[:rows, :cols] = w.float()
    tiles = padded.view(nbm, block, nbn, block)
    absmax = tiles.abs().amax(dim=(1, 3))
    scale = (absmax.double() / _E4M3_MAX).float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    expanded = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    q = (padded[:rows, :cols] / expanded[:rows, :cols]).to(torch.float8_e4m3fn)
    return q, scale


def transform_shard(in_path: str, out_path: str) -> dict:
    """Rewrite one safetensors shard, quantizing routed-expert weights to block-FP8.

    Tensors matching `_EXPERT_WEIGHT_RE` become an e4m3 `<key>` plus an fp32
    `<key>_scale_inv` scale grid; every other tensor is copied through
    byte-identical. Streams input tensors one at a time so peak memory stays
    near the size of the (already halved) output shard.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    out: dict[str, torch.Tensor] = {}
    tensors = 0
    quantized = 0
    with safe_open(in_path, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 (safe_open is not a dict)
            tensor = f.get_tensor(key)
            tensors += 1
            if _EXPERT_WEIGHT_RE.match(key):
                q, scale = _quant_block_fp8(tensor)
                out[key] = q
                out[key + "_scale_inv"] = scale
                quantized += 1
            else:
                out[key] = tensor
    save_file(out, out_path, metadata={"format": "pt"})
    return {
        "tensors": tensors,
        "quantized": quantized,
        "bytes_in": os.path.getsize(in_path),
        "bytes_out": os.path.getsize(out_path),
    }


def _index_with_scales(index: dict) -> dict:
    """Copy the index, mapping each new `weight_scale_inv` key to its weight's shard."""
    weight_map = dict(index["weight_map"])
    for key, shard in index["weight_map"].items():
        if _EXPERT_WEIGHT_RE.match(key):
            weight_map[key + "_scale_inv"] = shard
    new_index = dict(index)
    new_index["weight_map"] = weight_map
    return new_index


def _inject_fp8_config(config: dict) -> dict:
    """Mark the config as block-FP8 quantized so the loader detects expert residency.

    The loader reads `quantization_config` from the top level or from
    `text_config` (the repo is a composite VLM config), so inject into both.
    """
    marker = {"quant_method": "fp8", "weight_block_size": [_FP8_BLOCK, _FP8_BLOCK]}
    config = dict(config)
    config["quantization_config"] = marker
    if isinstance(config.get("text_config"), dict):
        text = dict(config["text_config"])
        text["quantization_config"] = marker
        config["text_config"] = text
    return config


@app.function(
    image=image,
    volumes={_MOUNT: weights_volume},
    timeout=24 * 3600,  # large checkpoint; the loop is resumable if it still times out
    cpu=8.0,
    memory=32768,  # MiB; one transformed shard is held in RAM before save_file
)
def stage() -> dict:
    """Download, quantize, and stage the checkpoint shard-by-shard (resumable)."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(_TARGET, exist_ok=True)
    os.makedirs(_SCRATCH, exist_ok=True)

    index_path = hf_hub_download(_REPO, _INDEX_NAME, revision=_REVISION, local_dir=_SCRATCH)
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))

    processed = skipped = quantized = 0
    bytes_written = 0
    for i, shard in enumerate(shards):
        final = os.path.join(_TARGET, shard)
        # Finished shards exist under their final name (writes land on a .tmp
        # path first), so existence + nonzero size means complete: skip on resume.
        if os.path.exists(final) and os.path.getsize(final) > 0:
            skipped += 1
            continue
        local = hf_hub_download(_REPO, shard, revision=_REVISION, local_dir=_SCRATCH)
        stats = transform_shard(local, final + ".tmp")
        os.replace(final + ".tmp", final)
        os.remove(local)  # bound scratch disk to one shard at a time
        processed += 1
        quantized += stats["quantized"]
        bytes_written += stats["bytes_out"]
        print(
            f"[{i + 1}/{len(shards)}] {shard}: "
            f"{stats['bytes_in'] / 1e9:.2f} GB -> {stats['bytes_out'] / 1e9:.2f} GB "
            f"({stats['quantized']} expert tensors quantized)",
            flush=True,
        )
        if processed % _COMMIT_EVERY == 0:
            weights_volume.commit()

    # Sidecar files: config.json (with the fp8 marker injected) + tokenizer assets.
    api = HfApi()
    for name in api.list_repo_files(_REPO, revision=_REVISION):
        top_level = "/" not in name
        wanted = name.endswith((".json", ".txt", ".model", ".jinja"))
        if not top_level or not wanted or name == _INDEX_NAME:
            continue
        local = hf_hub_download(_REPO, name, revision=_REVISION, local_dir=_SCRATCH)
        dest = os.path.join(_TARGET, name)
        if name == "config.json":
            with open(local) as f:
                cfg = json.load(f)
            with open(dest, "w") as f:
                json.dump(_inject_fp8_config(cfg), f, indent=2)
        else:
            shutil.copyfile(local, dest)

    new_index = _index_with_scales(index)
    total_shard_bytes = sum(os.path.getsize(os.path.join(_TARGET, s)) for s in shards)
    metadata = dict(new_index.get("metadata") or {})
    metadata["total_size"] = total_shard_bytes
    new_index["metadata"] = metadata
    with open(os.path.join(_TARGET, _INDEX_NAME), "w") as f:
        json.dump(new_index, f, indent=2)

    weights_volume.commit()
    return {
        "shards_processed": processed,
        "shards_skipped": skipped,
        "expert_tensors_quantized": quantized,
        "bytes_written": bytes_written,
        "volume_gb": round(total_shard_bytes / 1e9, 1),
    }


@app.local_entrypoint()
def main() -> None:
    info = stage.remote()
    print(f"Staged {_REPO} -> Volume {_VOLUME_NAME!r} at {_TARGET}")
    print(f"  shards processed: {info['shards_processed']}, skipped: {info['shards_skipped']}")
    print(f"  expert tensors quantized: {info['expert_tensors_quantized']}")
    print(
        f"  bytes written this run: {info['bytes_written'] / 1e9:.1f} GB; "
        f"volume now {info['volume_gb']} GB"
    )
    print(
        f"Remember to `uv run modal volume delete {_VOLUME_NAME}` when done "
        "(storage bills per GB-month)."
    )
