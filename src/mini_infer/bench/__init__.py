"""Uniform benchmarking: run one workload through many techniques, one table.

See `harness.py` for the design. The concrete technique registry (which knows
about `ModelRunner` and the schedulers) lives in `scripts/bench_all.py`, the
single CLI entry point.
"""

from mini_infer.bench.harness import (
    BenchEnv,
    BenchResult,
    SkippedTechnique,
    Technique,
    Workload,
    drive_scheduler,
    format_table,
    make_scheduler_technique,
    parity_violations,
    run_suite,
)
from mini_infer.bench.registry import DEFAULT_PROMPTS, build_registry

__all__ = [
    "DEFAULT_PROMPTS",
    "BenchEnv",
    "BenchResult",
    "SkippedTechnique",
    "Technique",
    "Workload",
    "build_registry",
    "drive_scheduler",
    "format_table",
    "make_scheduler_technique",
    "parity_violations",
    "run_suite",
]
