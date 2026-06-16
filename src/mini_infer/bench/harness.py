"""Uniform benchmark harness: one workload, many techniques, one table.

Every technique runs the SAME `Workload` (model, prompts, concurrency levels,
max_tokens) and reports the SAME `BenchResult` (tok/s, total tokens, per-request
latency), so the table columns are directly comparable. This is the lift the
roadmap calls for: the per-technique `scripts/bench_*` scripts each used their
own workload, so their numbers never lined up.

Techniques whose requirements the current environment doesn't meet (a CUDA-only
FlashInfer KV format on an M1, multi-GPU tensor parallelism, a path not yet
wired) are skipped with a recorded reason rather than silently dropped, so the
output always shows the full technique surface and what was and wasn't measured.

The harness core is deliberately model-agnostic and import-light (torch is
touched only in `BenchEnv.detect`); the concrete technique registry that knows
about `ModelRunner` / schedulers lives in `scripts/bench_all.py`.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Workload:
    """The single workload every technique runs, so results compare directly."""

    model: str
    prompts: list[str]
    concurrency_levels: list[int]
    max_tokens: int
    device: str = "cpu"
    # Attention backend for techniques that don't pin their own. The FlashInfer
    # KV-quant techniques force "flashinfer"; everything else uses this. Default
    # "flash_attn" (CPU falls back to SDPA automatically); the GPU wrapper sets
    # "torch" because its image doesn't ship flash-attn.
    attention_backend: str = "flash_attn"

    def prompts_for(self, concurrency: int) -> list[str]:
        """Replicate the source prompts to fill `concurrency` slots.

        Greedy decoding is deterministic, so duplicated prompts produce
        identical tokens. That is fine for a throughput measurement and is
        what makes the cross-technique parity check meaningful.
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1; got {concurrency}")
        if not self.prompts:
            raise ValueError("workload has no prompts")
        return [self.prompts[i % len(self.prompts)] for i in range(concurrency)]


@dataclass(frozen=True)
class BenchEnv:
    """Detected execution environment, used to gate technique applicability."""

    device: str
    cuda_available: bool
    cuda_device_count: int

    @classmethod
    def detect(cls, device: str) -> BenchEnv:
        import torch

        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        return cls(device=device, cuda_available=available, cuda_device_count=count)


@dataclass
class BenchResult:
    """One technique's measurement at one concurrency level."""

    technique: str
    concurrency: int
    total_seconds: float
    total_tokens: int
    per_request_latencies: list[float]
    # Per-request generated token ids. Not printed; used by `parity_violations`
    # to assert every LOSSLESS technique decoded the same tokens (greedy => they
    # must). Lossy techniques (INT8, quantized KV) are exempt; see `lossless`.
    output_signature: tuple[tuple[int, ...], ...]
    # False for techniques that deliberately change the math (INT8 weights,
    # quantized KV); those are expected to diverge from the baseline and are
    # excluded from the parity check.
    lossless: bool = True

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def median_latency(self) -> float:
        return statistics.median(self.per_request_latencies) if self.per_request_latencies else 0.0


@dataclass(frozen=True)
class SkippedTechnique:
    """A technique that did not run, with the reason (env gap or failure)."""

    technique: str
    reason: str


# The runner and scheduler are duck-typed `Any` at this boundary: they span
# into the heavy model layer (`ModelRunner`, `ContinuousScheduler`,
# `PDScheduler`, spec-decode), whose concrete types the harness core
# deliberately does not import. The contract is structural: a scheduler exposes
# `start()` / `submit(request) -> handle` / `stop()`, a handle exposes
# `wait() -> result`, and a result exposes `.tokens`.


@dataclass
class Technique:
    """One benchmarkable configuration.

    `run` executes the full workload (every concurrency level) and returns one
    `BenchResult` per level. `requires_cuda` / `min_cuda_devices` gate
    applicability against the detected `BenchEnv`; `pending` marks a technique
    that is registered for visibility but not yet wired. A technique that does
    not fit the environment is skipped (and recorded), never silently dropped.
    """

    name: str
    run: Callable[[Workload, BenchEnv], list[BenchResult]]
    requires_cuda: bool = False
    min_cuda_devices: int = 0
    pending: str = ""
    note: str = ""
    # See `BenchResult.lossless`: False exempts the technique from the parity
    # check (lossy quant is expected to diverge from the baseline).
    lossless: bool = True

    def applicability(self, env: BenchEnv) -> tuple[bool, str]:
        """Return `(can_run, reason_if_not)` for this technique under `env`."""
        if self.pending:
            return False, f"pending: {self.pending}"
        if self.requires_cuda and not env.cuda_available:
            detail = f" ({self.note})" if self.note else ""
            return False, f"requires CUDA{detail}"
        if self.min_cuda_devices and env.cuda_device_count < self.min_cuda_devices:
            return False, (
                f"requires >= {self.min_cuda_devices} CUDA devices (found {env.cuda_device_count})"
            )
        return True, ""


def drive_scheduler(
    scheduler: Any,
    prompts: list[str],
    max_tokens: int,
) -> tuple[float, int, list[float], tuple[tuple[int, ...], ...]]:
    """Submit `prompts` concurrently, drain to completion, time the whole sweep.

    Returns `(wall_seconds, total_tokens, per_request_latencies, signature)`,
    where `signature` is the per-request generated token ids (for parity).
    """
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler.request_state import Request

    scheduler.start()
    try:
        submit_times: list[float] = []
        handles: list[Any] = []
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
        results = [handle.wait() for handle in handles]
        wall_end = time.perf_counter()
    finally:
        scheduler.stop()

    latencies = [wall_end - submitted for submitted in submit_times]
    total_tokens = sum(len(result.tokens) for result in results)
    signature = tuple(tuple(result.tokens) for result in results)
    return wall_end - wall_start, total_tokens, latencies, signature


def make_scheduler_technique(
    name: str,
    build_runner: Callable[[Workload], Any],
    make_scheduler: Callable[[Any], Any],
    *,
    requires_cuda: bool = False,
    min_cuda_devices: int = 0,
    note: str = "",
    lossless: bool = True,
) -> Technique:
    """Build a `Technique` for a scheduler-driven configuration.

    The runner (the model load) is built ONCE and reused across every
    concurrency level; a fresh scheduler is created per level so each sweep
    starts from a clean queue. `build_runner` applies the technique's
    distinguishing `ModelRunner.from_pretrained` kwargs (quant, kv_quant,
    attention_backend, prefix_cache, ...); `make_scheduler` wraps the runner
    in the right scheduler (`ContinuousScheduler`, `PDScheduler`, ...).
    """

    def _run(workload: Workload, env: BenchEnv) -> list[BenchResult]:
        runner = build_runner(workload)
        results: list[BenchResult] = []
        for concurrency in workload.concurrency_levels:
            prompts = workload.prompts_for(concurrency)
            seconds, total_tokens, latencies, signature = drive_scheduler(
                make_scheduler(runner), prompts, workload.max_tokens
            )
            results.append(
                BenchResult(
                    technique=name,
                    concurrency=concurrency,
                    total_seconds=seconds,
                    total_tokens=total_tokens,
                    per_request_latencies=latencies,
                    output_signature=signature,
                    lossless=lossless,
                )
            )
        return results

    return Technique(
        name=name,
        run=_run,
        requires_cuda=requires_cuda,
        min_cuda_devices=min_cuda_devices,
        note=note,
        lossless=lossless,
    )


def run_suite(
    workload: Workload,
    techniques: Sequence[Technique],
    env: BenchEnv | None = None,
) -> tuple[list[BenchResult], list[SkippedTechnique]]:
    """Run every applicable technique through `workload`; collect results.

    Inapplicable techniques (env gap or `pending`) are recorded in the skipped
    list with a reason. A technique that raises mid-run is caught and recorded
    as skipped rather than aborting the whole suite, so one broken config does
    not cost the rest of the table.
    """
    if env is None:
        env = BenchEnv.detect(workload.device)

    results: list[BenchResult] = []
    skipped: list[SkippedTechnique] = []
    for technique in techniques:
        can_run, reason = technique.applicability(env)
        if not can_run:
            logger.info("Skipping %r: %s", technique.name, reason)
            skipped.append(SkippedTechnique(technique.name, reason))
            continue
        try:
            results.extend(technique.run(workload, env))
        except Exception as exc:  # one bad config must not kill the rest of the suite
            logger.warning("Technique %r failed: %s", technique.name, exc)
            skipped.append(SkippedTechnique(technique.name, f"failed: {exc}"))
    return results, skipped


def parity_violations(results: Sequence[BenchResult]) -> list[tuple[str, str]]:
    """Return lossless-technique pairs whose tokens differ at the same concurrency.

    Greedy decoding is deterministic, so every LOSSLESS technique (cache
    strategy, scheduler, attention-backend swap, tensor parallelism, exact spec
    decode) must produce identical tokens on the same workload; a mismatch
    there is a real correctness bug. Lossy techniques (INT8 weights, quantized
    KV) deliberately change the math and are expected to diverge from the
    baseline, so `lossless=False` results are excluded from this check.
    """
    by_concurrency: dict[int, list[BenchResult]] = {}
    for result in results:
        if not result.lossless:
            continue
        by_concurrency.setdefault(result.concurrency, []).append(result)

    violations: list[tuple[str, str]] = []
    for group in by_concurrency.values():
        reference = group[0]
        for other in group[1:]:
            if other.output_signature != reference.output_signature:
                violations.append((reference.technique, other.technique))
    return violations


def format_table(results: Sequence[BenchResult], skipped: Sequence[SkippedTechnique]) -> str:
    """Render results as an aligned table plus a footnote of skipped techniques."""
    header = ["technique", "concurrency", "tok/s", "tokens", "wall_s", "median_lat_s"]
    rows = [
        [
            result.technique,
            str(result.concurrency),
            f"{result.tokens_per_sec:.2f}",
            str(result.total_tokens),
            f"{result.total_seconds:.3f}",
            f"{result.median_latency:.3f}",
        ]
        for result in results
    ]
    widths = [
        max(len(header[col]), *(len(row[col]) for row in rows)) if rows else len(header[col])
        for col in range(len(header))
    ]

    def _fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(cells))

    lines = [_fmt(header), _fmt(["-" * w for w in widths])]
    lines.extend(_fmt(row) for row in rows)
    if skipped:
        lines.append("")
        lines.append("Skipped:")
        lines.extend(f"  - {item.technique}: {item.reason}" for item in skipped)
    return "\n".join(lines)
