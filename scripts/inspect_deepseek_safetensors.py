"""Inspect the DeepSeek-V2 safetensors index without downloading the weights.

Confirms the weight layout we'll target with `DeepseekV2ForCausalLM.load_weights`
before paying for a Modal smoke run. The questions:

  1. Are MoE experts stored per-expert (`experts.{j}.gate_proj`) or as a
     concatenated tensor (`experts.gate_up_proj`)? HF's `DeepseekV2Experts`
     class uses the concatenated form; some Lite checkpoints predate this.
  2. Which layers are dense MLP vs MoE? Confirms `first_k_dense_replace`.
  3. Do shared experts ship under `mlp.shared_experts.{gate_proj,up_proj,
     down_proj}`?
  4. Is `lm_head.weight` present (untied) or absent (tied)?

Run with:
    HF_TOKEN=$(hf auth token | tail -1) \\
        uv run python scripts/inspect_deepseek_safetensors.py deepseek-ai/DeepSeek-V2-Lite
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
    """Group safetensors keys by their pattern with layer indices abstracted out."""
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
        default="deepseek-ai/DeepSeek-V2-Lite",
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

    groups = _collapse_layer_indices(weight_map)

    # --- Dense MLP vs MoE per layer ---
    print()
    print("Per-layer FFN structure (sample of keys for layers 0, 1, 2):")
    for layer_idx in (0, 1, 2):
        prefix = f"model.layers.{layer_idx}."
        keys = sorted(k for k in weight_map if k.startswith(prefix) and ".mlp." in k)
        print(f"  layer {layer_idx}: {len(keys)} mlp keys")
        for k in keys[:6]:
            print(f"    {k}")
        if len(keys) > 6:
            print(f"    ... +{len(keys) - 6} more")

    # --- Distinct top-level prefixes ---
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

    # --- lm_head tie check ---
    print()
    has_lm_head = any(k.startswith("lm_head") for k in weight_map)
    print(f"lm_head.weight present in index: {has_lm_head}")

    # --- MoE expert layout ---
    print()
    has_concat_experts = any("experts.gate_up_proj" in k for k in weight_map)
    has_per_expert = any(re.search(r"experts\.\d+\.gate_proj", k) for k in weight_map)
    print(
        f"MoE experts: concatenated (gate_up_proj)={has_concat_experts}, "
        f"per-expert (experts.{{j}}.gate_proj)={has_per_expert}"
    )
    has_shared = any("shared_experts" in k for k in weight_map)
    print(f"shared_experts present: {has_shared}")

    # --- Per-attention-layer key count ---
    print()
    print("Sample MLA attention keys for layer 0:")
    attn_keys = sorted(
        k for k in weight_map if k.startswith("model.layers.0.") and ".self_attn." in k
    )
    for k in attn_keys:
        print(f"  {k}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
