"""Local (no-GPU, no-Modal) feasibility profile for the V4-Flash load path.

Answers two questions the expensive 2x B200 runs should have been preceded
by, both computable on the M1:

  1. Does the dequantized-to-BF16 V4-Flash model fit 2x B200 (384 GB HBM)?
     The loader currently dequantizes FP4 experts to BF16 at load time;
     FP4 -> BF16 is a 4x storage blow-up on the part of the model that
     dominates the parameter count.

  2. How slow is the FP4 dequant itself on CPU? (Times one real-sized
     expert weight and extrapolates to the full model.)

Reads `config.json` from the Hub (a few KB, no weights download) so the
sizes are exact. Run:

    uv run --with huggingface_hub python scripts/profile_v4_dequant.py
"""

from __future__ import annotations

import json
import time

import torch
from huggingface_hub import hf_hub_download

from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16

_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_B200_HBM_GB = 192.0
_GiB = 1024**3


def _load_config() -> dict:
    return json.load(open(hf_hub_download(_MODEL, "config.json")))


def _expert_params_per_layer(cfg: dict) -> int:
    # One expert = w1 (gate) + w3 (up): (moe_inter, hidden) each,
    # + w2 (down): (hidden, moe_inter). All three are moe_inter * hidden.
    hidden = cfg["hidden_size"]
    moe_inter = cfg["moe_intermediate_size"]
    per_expert = 3 * moe_inter * hidden
    return cfg["n_routed_experts"] * per_expert


def main() -> None:
    cfg = _load_config()
    layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    moe_inter = cfg["moe_intermediate_size"]
    n_routed = cfg["n_routed_experts"]
    n_shared = cfg["n_shared_experts"]
    vocab = cfg["vocab_size"]

    # Param counts (routed experts dominate; attention is low-rank MLA and
    # small by comparison, included only as a rough remainder).
    routed = layers * _expert_params_per_layer(cfg)
    shared = layers * n_shared * 3 * moe_inter * hidden
    embed = vocab * hidden  # tied lm_head per config
    expert_total = routed + shared

    print(f"=== V4-Flash size profile (from {_MODEL} config.json) ===")
    print(
        f"layers={layers} hidden={hidden} moe_inter={moe_inter} "
        f"routed_experts={n_routed} shared_experts={n_shared}"
    )
    print()
    print(f"routed-expert params : {routed / 1e9:7.1f} B")
    print(f"shared-expert params : {shared / 1e9:7.1f} B")
    print(f"embedding params     : {embed / 1e9:7.1f} B")
    print()

    # Storage: experts are FP4 (0.5 byte/param) on disk; dequant -> BF16 is
    # 2 bytes/param (a 4x blow-up). Compare BF16 expert storage to HBM.
    fp4_expert_gb = expert_total * 0.5 / _GiB
    bf16_expert_gb = expert_total * 2.0 / _GiB
    two_b200_gb = 2 * _B200_HBM_GB

    print(f"experts as FP4 (on disk)      : {fp4_expert_gb:7.1f} GiB")
    print(f"experts dequantized to BF16   : {bf16_expert_gb:7.1f} GiB")
    print(f"2x B200 total HBM             : {two_b200_gb:7.1f} GiB")
    print(
        f"per-rank BF16 experts (ws=2)  : {bf16_expert_gb / 2:7.1f} GiB  "
        f"(one B200 = {_B200_HBM_GB:.0f} GiB)"
    )
    print()
    fits = bf16_expert_gb <= two_b200_gb
    print(
        f"VERDICT: dequantized-BF16 experts {'FIT' if fits else 'DO NOT FIT'} "
        f"2x B200 ({bf16_expert_gb:.0f} vs {two_b200_gb:.0f} GiB)"
    )
    print()

    # CPU staging blow-up: load_weights builds a second full dict of BF16
    # tensors while the packed source dict is still alive.
    print(
        f"CPU staging peak (packed + bf16 both live) ~ "
        f"{fp4_expert_gb + bf16_expert_gb:.0f} GiB before any GPU slice"
    )
    print()

    # Time one real-sized expert weight dequant on CPU, extrapolate.
    out_dim, in_dim = moe_inter, hidden  # w1 shape (moe_inter, hidden)
    packed = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8).view(torch.int8)
    scale = torch.ones(out_dim, in_dim // 32, dtype=torch.float32)
    # warmup
    dequantize_nvfp4_to_bf16(packed, scale)
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        dequantize_nvfp4_to_bf16(packed, scale)
    per_weight_s = (time.perf_counter() - t0) / reps

    weights_total = layers * (n_routed + n_shared) * 3  # w1/w2/w3 per expert
    est_full_s = per_weight_s * weights_total
    print("dequant timing (CPU, this machine):")
    print(f"  one expert weight ({out_dim}x{in_dim}) : {per_weight_s * 1e3:7.1f} ms")
    print(
        f"  full model ({weights_total} expert weights) : "
        f"{est_full_s:7.0f} s  (~{est_full_s / 60:.0f} min), single-threaded extrapolation"
    )


if __name__ == "__main__":
    main()
