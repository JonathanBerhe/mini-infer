"""Single entry point: benchmark every technique on one workload, one table.

Runs the SAME workload (model, prompts, concurrency levels, max_tokens) through
each registered technique and prints a comparable tok/s + per-request-latency
table. Replaces the per-technique `scripts/bench_*` scripts, whose differing
workloads made their numbers impossible to line up.

Techniques the current machine can't run (CUDA-only KV formats / attention
backends on an M1, the multi-process TP path, not-yet-wired paths) are listed
as skipped with a reason, never silently dropped. So the free CPU run measures
the CPU-runnable subset and a GPU run (via `scripts/bench/modal_bench_all.py`) fills
in the rest, against the same workload.

The harness also doubles as a cross-technique correctness check: greedy
decoding is deterministic, so every lossless technique must emit identical
tokens. A divergence is reported as a parity violation (a real bug); lossy
quant (INT8, quantized KV) is exempt by design.

Run:
    uv run python scripts/bench/bench_all.py
    uv run python scripts/bench/bench_all.py --concurrency 1,4,8 --max-tokens 16
    uv run python scripts/bench/bench_all.py --techniques baseline,int8_w8a16 --device cpu
    uv run python scripts/bench/bench_all.py --techniques baseline --concurrency 1 --profile

`--profile` runs each technique once more at its first concurrency level
under `torch.profiler`, with every scheduler step and model module labelled
from the outside (`mini_infer.bench.profile_scheduler_steps`), and prints the
per-step and per-part tables after the main table. On a CPU the tables show
the wall time per step and the host time by part of the model; on a GPU they
add the GPU time, the kernels and the host syncs. `--profile-trace DIR` keeps
one Chrome trace per technique in DIR.

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
    format_step_profiles,
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="after the timed sweep of a technique, profile its scheduler steps at the "
        "first concurrency level with torch.profiler and print the tables",
    )
    parser.add_argument(
        "--profile-trace",
        default=None,
        metavar="DIR",
        help="keep one Chrome trace per profiled technique in DIR (default: not kept)",
    )
    args = parser.parse_args()

    workload = Workload(
        model=args.model,
        prompts=DEFAULT_PROMPTS,
        concurrency_levels=_parse_int_list(args.concurrency),
        max_tokens=args.max_tokens,
        device=args.device,
        attention_backend=args.attention_backend,
        profile=args.profile or args.profile_trace is not None,
        profile_trace_dir=args.profile_trace,
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

    profiles = format_step_profiles(results)
    if profiles:
        print()
        print(profiles)


if __name__ == "__main__":
    main()
