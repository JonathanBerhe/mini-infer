"""Concurrency-sweep bench for `PDScheduler`.

Runs N concurrent requests through three scheduler configurations and
reports throughput (tokens / second) + per-request latency:

  - `ContinuousScheduler` (the non-PD baseline)
  - `PDScheduler(mode="serial")` (single-engine-thread PD)
  - `PDScheduler(mode="parallel")` (two-thread PD)

Concurrency levels (default): 1, 2, 4, 8. Per-request `max_tokens`: 32.

Output is a CSV-shaped table on stdout. Same workload on each config
so the columns are comparable; greedy decoding so the token counts
are identical across configs by parity contract.

What this bench *can* validate on CPU:
  - Correctness (every config produces the same tokens).
  - Relative speedups between configurations at the Python overhead
    + small-matmul scale where M1 lives.
  - Cross-mode timing differences in PDScheduler (serial vs parallel).

What this bench *cannot* validate on CPU:
  - The parallel mode's real win, which is cross-GPU phase overlap on
    a multi-GPU host. On CPU both phases share the same compute, so
    "parallel" doesn't help (and may hurt slightly due to threading
    overhead). The number we want is from a 2x H100 Modal run with
    each worker on its own GPU; see `roadmap-2026.md` for the budget
    posture on Modal runs.

Run with:
    uv run python scripts/bench_pd_scheduler.py
    uv run python scripts/bench_pd_scheduler.py --concurrency 1,4,8 --max-tokens 16

The script accepts `--model` (default Qwen2.5-0.5B-Instruct).
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
from mini_infer.scheduler.request_state import Request
from mini_infer.workers import PDScheduler

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "A common greeting in English is",
    "The chemical symbol for water is",
    "An example of a fruit is",
    "The first president of the United States was",
    "Python is a programming language created by",
    "The opposite of hot is",
]


@dataclass
class BenchResult:
    config: str
    concurrency: int
    total_seconds: float
    total_tokens: int
    per_request_latencies: list[float]

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def median_latency(self) -> float:
        return statistics.median(self.per_request_latencies)


def _run_scheduler(
    scheduler: ContinuousScheduler | PDScheduler,
    prompts: list[str],
    max_tokens: int,
) -> BenchResult:
    """Submit `prompts` concurrently, drain to completion, time everything.

    Returns a `BenchResult` with wall-time, total decoded tokens, and
    per-request latencies (submit → terminal).
    """
    scheduler.start()
    try:
        submit_times = []
        handles = []
        wall_start = time.perf_counter()
        for prompt in prompts:
            submit_times.append(time.perf_counter())
            handles.append(
                scheduler.submit(
                    Request(
                        prompt=prompt,
                        sampling_params=SamplingParams(),  # greedy
                        max_tokens=max_tokens,
                    )
                )
            )
        results = [h.wait() for h in handles]
        wall_end = time.perf_counter()
        finish_times = [wall_end] * len(
            prompts
        )  # finer per-request timing needs a stream-drain hook
        per_request = [finish_times[i] - submit_times[i] for i in range(len(prompts))]
    finally:
        scheduler.stop()
    return BenchResult(
        config="(unset)",
        concurrency=len(prompts),
        total_seconds=wall_end - wall_start,
        total_tokens=sum(len(r.tokens) for r in results),
        per_request_latencies=per_request,
    )


def _bench_one_config(
    config: str,
    factory: Callable[[], ContinuousScheduler | PDScheduler],
    prompts: list[str],
    max_tokens: int,
) -> BenchResult:
    """Run one sweep through a fresh scheduler from `factory`."""
    scheduler = factory()
    result = _run_scheduler(scheduler, prompts, max_tokens)
    result.config = config
    return result


def _parse_concurrency(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def _build_prompts(concurrency: int, sources: Iterable[str]) -> list[str]:
    """Replicate the source prompts to fill `concurrency` slots.

    Greedy is deterministic, so duplicates produce identical tokens —
    fine for a throughput bench. Real workloads would use distinct
    prompts; for throughput measurement the prompt text doesn't matter.
    """
    sources_list = list(sources)
    out: list[str] = []
    while len(out) < concurrency:
        out.append(sources_list[len(out) % len(sources_list)])
    return out


def _print_table(rows: list[BenchResult]) -> None:
    header = ["config", "concurrency", "tok/s", "total tokens", "wall s", "median latency s"]
    print(",".join(header))
    for r in rows:
        print(
            ",".join(
                [
                    r.config,
                    str(r.concurrency),
                    f"{r.tokens_per_sec:.2f}",
                    str(r.total_tokens),
                    f"{r.total_seconds:.2f}",
                    f"{r.median_latency:.2f}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="comma-separated concurrency levels (default 1,2,4,8)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="per-request max output tokens (default 32)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run a 1-request warmup pass per config to amortize first-call costs",
    )
    args = parser.parse_args()

    levels = _parse_concurrency(args.concurrency)
    print(f"Loading {args.model} (this is the only model load; all configs share it)...")
    runner = ModelRunner.from_pretrained(args.model)
    print("Model loaded.")

    factories: dict[str, Callable[[], ContinuousScheduler | PDScheduler]] = {
        "ContinuousScheduler": lambda: ContinuousScheduler(runner),
        "PDScheduler[serial]": lambda: PDScheduler(runner, mode="serial"),
        "PDScheduler[parallel]": lambda: PDScheduler(runner, mode="parallel"),
    }

    if args.warmup:
        warmup_prompts = _build_prompts(1, DEFAULT_PROMPTS)
        for name, factory in factories.items():
            print(f"Warmup: {name}...")
            _bench_one_config(name, factory, warmup_prompts, max_tokens=4)

    rows: list[BenchResult] = []
    for concurrency in levels:
        prompts = _build_prompts(concurrency, DEFAULT_PROMPTS)
        for name, factory in factories.items():
            print(f"Running {name} @ concurrency={concurrency}...", end=" ", flush=True)
            result = _bench_one_config(name, factory, prompts, args.max_tokens)
            print(
                f"tok/s={result.tokens_per_sec:.2f} "
                f"wall={result.total_seconds:.2f}s "
                f"median_latency={result.median_latency:.2f}s"
            )
            rows.append(result)

    print()
    print("=== Results table ===")
    _print_table(rows)


if __name__ == "__main__":
    main()
