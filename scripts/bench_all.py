"""Single entry point: benchmark every technique on one workload, one table.

Runs the SAME workload (model, prompts, concurrency levels, max_tokens) through
each registered technique and prints a comparable tok/s + per-request-latency
table. Replaces the per-technique `scripts/bench_*` scripts, whose differing
workloads made their numbers impossible to line up.

Techniques the current machine can't run (CUDA-only KV formats / attention
backends on an M1, the multi-process TP path, not-yet-wired paths) are listed
as skipped with a reason, never silently dropped. So the free CPU run measures
the CPU-runnable subset and a GPU run (via `scripts/modal_bench_all.py`) fills
in the rest, against the same workload.

The harness also doubles as a cross-technique correctness check: greedy
decoding is deterministic, so every lossless technique must emit identical
tokens. A divergence is reported as a parity violation (a real bug); lossy
quant (INT8, quantized KV) is exempt by design.

Run:
    uv run python scripts/bench_all.py
    uv run python scripts/bench_all.py --concurrency 1,4,8 --max-tokens 16
    uv run python scripts/bench_all.py --techniques baseline,int8_w8a16 --device cpu

The technique registry is shared with the Modal GPU wrapper; see
`mini_infer.bench.registry`.
"""

from __future__ import annotations

import argparse
import logging

from mini_infer.bench import (
    DEFAULT_PROMPTS,
    Workload,
    build_registry,
    format_table,
    parity_violations,
    run_suite,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", default="1,2,4", help="comma-separated, default 1,2,4")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto (default cpu)")
    parser.add_argument(
        "--attention-backend",
        default="flash_attn",
        help="backend for techniques that don't pin one: flash_attn | flashinfer | torch",
    )
    parser.add_argument(
        "--techniques",
        default="",
        help="comma-separated technique names to include (default: all registered)",
    )
    args = parser.parse_args()

    workload = Workload(
        model=args.model,
        prompts=DEFAULT_PROMPTS,
        concurrency_levels=_parse_int_list(args.concurrency),
        max_tokens=args.max_tokens,
        device=args.device,
        attention_backend=args.attention_backend,
    )

    registry = build_registry()
    if args.techniques:
        wanted = {name.strip() for name in args.techniques.split(",") if name.strip()}
        unknown = wanted - {technique.name for technique in registry}
        if unknown:
            parser.error(f"unknown technique(s): {sorted(unknown)}")
        registry = [technique for technique in registry if technique.name in wanted]

    print(
        f"Workload: model={workload.model} device={workload.device} "
        f"concurrency={workload.concurrency_levels} max_tokens={workload.max_tokens}"
    )
    results, skipped = run_suite(workload, registry)

    print()
    print(format_table(results, skipped))

    violations = parity_violations(results)
    print()
    if violations:
        print("PARITY VIOLATIONS (lossless techniques decoded different tokens; a bug):")
        for ref, other in violations:
            print(f"  - {ref} != {other}")
    else:
        print("Parity: all lossless techniques agree (lossy quant excluded by design).")


if __name__ == "__main__":
    main()
