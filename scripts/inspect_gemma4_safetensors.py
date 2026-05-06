"""Inspect the Gemma 4 safetensors index without downloading the weights.

Walks `model.safetensors.index.json` on HF Hub (a few KB) to confirm the
weight layout we'll target with `Gemma4ForCausalLM.load_weights`. The
inspection answers three uncertain questions before we pay for a Modal
GPU smoke run:

  1. Does `v_proj.weight` exist on every layer or only on sliding ones?
     (HF source skips it for full layers when `attention_k_eq_v=True`.)
  2. Does `v_norm.weight` exist? (HF source uses `with_scale=False` for
     `v_norm` so no parameter is constructed; the disk layout might
     differ.)
  3. Is `lm_head.weight` absent? (`tie_word_embeddings=True` means it
     should be missing from the index and the model loads via the
     embedding alias.)

Run with:
    HF_TOKEN=$(hf auth token | tail -1) \\
        uv run python scripts/inspect_gemma4_safetensors.py google/gemma-4-31B-it
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _collapse_layer_indices(keys: Iterable[str]) -> dict[str, list[int]]:
    """Group safetensors keys by their pattern with layer indices abstracted out.

    Returns `{ "model.language_model.layers.*.self_attn.q_proj.weight": [0, 1, 2, ...] }`.
    Patterns without a `layers.<N>.` segment map to a single empty-list entry.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        match = LAYER_RE.search(key)
        if match is None:
            groups.setdefault(key, [])
            continue
        layer_idx = int(match.group(1))
        pattern = key[: match.start()] + ".layers.*." + key[match.end() :]
        groups[pattern].append(layer_idx)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="google/gemma-4-31B-it",
        help="HF repo id to inspect (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed; pip install huggingface-hub", file=sys.stderr)
        return 1

    print(f"Fetching model.safetensors.index.json for {args.repo_id} (no model download)...")
    index_path = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map: dict[str, str] = index["weight_map"]
    metadata = index.get("metadata", {})

    print()
    print(f"Total tensors: {len(weight_map)}")
    if "total_size" in metadata:
        gb = metadata["total_size"] / 1e9
        print(f"Total size:    {gb:.2f} GB")

    # Collapse by pattern.
    groups = _collapse_layer_indices(weight_map)

    # --- Q1: full-layer v_proj ---
    v_proj_pattern = "model.language_model.layers.*.self_attn.v_proj.weight"
    v_proj_layers = sorted(groups.get(v_proj_pattern, []))
    print()
    print(f"v_proj.weight present on {len(v_proj_layers)} layers")
    if v_proj_layers:
        # Identify gaps (which we expect on full-attention layers if HF source
        # is the source of truth).
        all_layers = sorted({i for layers in groups.values() for i in layers})
        if all_layers:
            missing = [i for i in all_layers if i not in v_proj_layers]
            print(f"  Layers with v_proj:    {v_proj_layers[:8]}... (total {len(v_proj_layers)})")
            print(f"  Layers WITHOUT v_proj: {missing}")

    # --- Q2: v_norm ---
    v_norm_pattern = "model.language_model.layers.*.self_attn.v_norm.weight"
    v_norm_layers = groups.get(v_norm_pattern, [])
    print()
    print(
        f"v_norm.weight present on {len(v_norm_layers)} layers (expected: 0 if `with_scale=False`)"
    )
    if v_norm_layers:
        print(f"  Layers with v_norm: {sorted(v_norm_layers)[:8]}...")

    # --- Q3: lm_head tie ---
    print()
    has_lm_head = any(k.startswith("lm_head") for k in weight_map)
    print(
        f"lm_head.weight present in index: {has_lm_head} "
        "(expected: False with tie_word_embeddings=True)"
    )

    # --- layer_scalar (we register it as a buffer, so the index should ship it). ---
    layer_scalar_pattern = "model.language_model.layers.*.layer_scalar"
    layer_scalar_count = len(groups.get(layer_scalar_pattern, []))
    print()
    print(f"layer_scalar present on {layer_scalar_count} layers")

    # --- Multimodal keys to filter ---
    print()
    print("Top-level prefixes (collapsed):")
    seen: set[str] = set()
    for pattern in sorted(groups):
        head = pattern.split(".", 2)[:2]
        prefix = ".".join(head)
        if prefix in seen:
            continue
        seen.add(prefix)
        print(f"  {prefix}.*")

    # --- Per-layer key inventory for layer 0 (sliding) and layer 5 (full) ---
    print()
    for label, layer_idx in (("layer 0 (expected sliding)", 0), ("layer 5 (expected full)", 5)):
        print(f"{label}:")
        prefix = f"model.language_model.layers.{layer_idx}."
        layer_keys = sorted(k for k in weight_map if k.startswith(prefix))
        for k in layer_keys:
            print(f"  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
