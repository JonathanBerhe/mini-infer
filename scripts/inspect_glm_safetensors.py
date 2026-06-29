"""Inspect the GLM-5.2-FP8 safetensors index without downloading the weights.

Confirms the block-FP8 layout `GlmMoeDsaForCausalLM.load_weights` must target
before paying for any GPU run. The questions:

  1. What is the FP8 scale-companion key naming? (`weight_scale_inv` per the
     DeepSeek-V3 / HF-fp8 convention, vs V4's `.scale`?)
  2. Are routed experts stacked (`experts.gate_up_proj`) or per-expert? Does the
     stacked tensor carry a scale companion (and at what key)?
  3. Which modules ship FP8 (have a scale) vs BF16 (excluded: norms, gate,
     indexer, embed, lm_head)?
  4. Total size (confirms the ~753 GB figure driving the GPU sizing).

Downloads only `model.safetensors.index.json` (a few MB; hf_hub_download
resolves the LFS pointer), so this is a CPU/$0 step.

Run with:
    uv run python scripts/inspect_glm_safetensors.py zai-org/GLM-5.2-FP8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _collapse(keys: Iterable[str]) -> set[str]:
    """Abstract out layer indices so distinct key *shapes* are visible."""
    patterns: set[str] = set()
    for key in keys:
        match = LAYER_RE.search(key)
        patterns.add(
            key if match is None else key[: match.start()] + ".layers.N." + key[match.end() :]
        )
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("repo_id", nargs="?", default="zai-org/GLM-5.2-FP8")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    print(f"Fetching index for {args.repo_id} (index only, no weights)...")
    index_path = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]
    meta = index.get("metadata", {})

    print(f"\nTotal tensors: {len(weight_map)}")
    if "total_size" in meta:
        print(f"Total size:    {meta['total_size'] / 1e9:.1f} GB")

    scale_keys = [k for k in weight_map if "scale" in k.lower()]
    print(f"\nScale-companion tensors: {len(scale_keys)}")
    print("Distinct scale key patterns:")
    for p in sorted(_collapse(scale_keys)):
        print(f"  {p}")

    print("\nDistinct WEIGHT key patterns for one MoE layer (last layer):")
    last = max(int(m.group(1)) for k in weight_map if (m := LAYER_RE.search(k)))
    for k in sorted(weight_map):
        if f".layers.{last}." in k and (".mlp." in k or ".self_attn." in k):
            print(f"  {k}")

    per_expert_re = re.compile(r"experts\.\d+\.gate_proj")
    stacked = any("experts.gate_up_proj" in k for k in weight_map)
    per_expert = any(per_expert_re.search(k) for k in weight_map)
    lm_head = any(k.startswith("lm_head") for k in weight_map)
    lm_head_scale = any("lm_head" in k and "scale" in k for k in weight_map)
    embed_scale = any("embed_tokens" in k and "scale" in k for k in weight_map)
    indexer_scale = any("indexer" in k and "scale" in k for k in weight_map)
    print("\nLayout checks:")
    print(f"  stacked experts (experts.gate_up_proj): {stacked}")
    print(f"  per-expert (experts.N.gate_proj):       {per_expert}")
    print(f"  lm_head present:                        {lm_head}")
    print(f"  lm_head has scale (FP8?):               {lm_head_scale}")
    print(f"  embed_tokens has scale (FP8?):          {embed_scale}")
    print(f"  indexer has scale (FP8?):               {indexer_scale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
