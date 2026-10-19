"""CPU run of the compressor-projection dtype probe on a synthetic DeepSeek-V4.

Random weights at small dims, bf16 on CPU: shows the MECHANISM (how far bf16
rounding of the projected kv / score entries moves the compressed entries,
the indexer's top-k sets and the logits), not production magnitudes. The real
measurement is `scripts/modal_v4_projection_dtype_probe.py` on V4-Flash.

    uv run python scripts/v4_projection_dtype_probe_synthetic.py --seq-len 512
"""

from __future__ import annotations

import argparse
import json

import torch

from mini_infer.bench.v4_projection_dtype_probe import (
    FP32_OUT,
    NATIVE_OUT,
    format_report,
    make_synthetic_hybrid_config,
    run_paired_prefill,
)
from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=str, default=None, help="also write the report here")
    args = parser.parse_args()

    cfg = make_synthetic_hybrid_config(num_layers=args.num_layers, hidden_size=args.hidden_size)
    torch.manual_seed(args.seed)
    model = DeepseekV4ForCausalLM(cfg).to(_DTYPES[args.dtype]).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len), dtype=torch.long)  # (1, T)

    report = run_paired_prefill(
        model,
        input_ids,
        baseline=FP32_OUT,
        alternate=NATIVE_OUT,
        max_new_tokens=args.max_new_tokens,
        position_stride=max(1, args.seq_len // 16),
        position_tail=32,
    )
    print(format_report(report))
    if args.json is not None:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        print(f"\nreport written to {args.json}")


if __name__ == "__main__":
    main()
