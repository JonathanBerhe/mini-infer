"""Concurrency sweep for DSpark speculative decoding inside `ContinuousScheduler`.

One measurement point runs a fixed list of prompts through a fresh scheduler
whose `max_concurrent` is the concurrency under test. All prompts are
submitted at once. The scheduler admits them up to that cap and starts the
next one as soon as a slot frees, so `concurrency` requests are in flight
while any prompt still waits (a closed loop). After the last admission the
batch drains, and the in-flight count falls to zero. Every point of a sweep
uses the same prompts, so a change between points is a change of
concurrency, not of workload. The in-process driver is chosen over the HTTP
open-loop benchmark for this: the server cannot enable dspark, that client
counts stream chunks as tokens, and its arrival rate does not control the
batch size that the cost curve needs.

What a point reports:

- throughput over the whole run and over the steady part only (the steps in
  which a prompt still waited), plus the time-weighted mean in-flight count,
  all counted from each request's own tokens, which is exact even when a
  verify step streams several tokens as one step;
- `SpecSummary` pooled over every request that speculated, for tau and
  per-position survival, plus the number of requests that did NOT speculate.
  A quiet admission downgrade would otherwise make a "DSpark" point measure
  the plain path;
- one record per packed forward (requests, q-tokens, verify windows, queued
  requests, tokens streamed so far), and with `timed=True` wall time per
  phase (sample, draft, forward, verify), the raw material for the cost curve.

Timing honesty on a GPU: kernels run asynchronously, so a host timer charges
GPU work to whichever later phase first synchronizes. A timed point therefore
calls `sync` (pass `torch.cuda.synchronize`) at every phase boundary. Those
barriers slow the step, so a timed point attributes time and an untimed point
measures throughput; one run cannot do both.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from mini_infer.engine.dspark.drafter import Qwen3DSparkDrafter
from mini_infer.engine.dspark.spec import DSparkSpec, SpecSummary
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
from mini_infer.scheduler.request_state import (
    GenerationResult,
    GenerationStep,
    Request,
    RunningRequest,
)


@dataclasses.dataclass
class PhaseTimes:
    """Wall seconds spent in each phase of the step, summed over a run."""

    sample_s: float = 0.0
    draft_s: float = 0.0
    forward_s: float = 0.0
    verify_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """One packed forward: how wide it was, what was waiting, how far the run was.

    `batch` counts requests in the forward and `total_q` counts its q-tokens.
    Their difference is what speculation adds: a verify window feeds
    `1 + len(offered)` tokens where plain decoding feeds 1. `waiting` is how
    many requests were still queued, so `waiting > 0` marks a step with every
    slot busy. `end_s` (seconds since the point started) and `emitted`
    (tokens streamed so far) locate the step in the run.

    `forward_s` is trustworthy only in a timed point: without a GPU barrier a
    host timer measures kernel launches, not kernels.
    """

    batch: int
    total_q: int
    n_verify: int
    waiting: int
    forward_s: float
    end_s: float
    emitted: int


class InstrumentedScheduler(ContinuousScheduler):
    """A `ContinuousScheduler` that records every forward, and can time phases.

    For benchmarks only. It wraps the private phase methods, so it is coupled
    to `_step`'s structure on purpose, and it lives next to the benchmark
    rather than in the scheduler so that the serving path carries none of it.

    Every forward gets a `StepRecord`, at the cost of a clock read and a queue
    size, with no barrier. With `timed=True` it also sums wall time per phase,
    calling `sync` (pass `torch.cuda.synchronize`) at every phase boundary.
    Those barriers slow the step, so a timed point attributes time and an
    untimed point measures throughput. Read `phases` and `steps` after
    `stop()`: the engine thread writes them.
    """

    def __init__(
        self,
        runner: ModelRunner,
        *,
        timed: bool = False,
        sync: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(runner, **kwargs)
        self._timed = timed
        self._sync = sync if timed else None
        self.phases = PhaseTimes()
        self.steps: list[StepRecord] = []
        self._t0: float | None = None
        self._emitted = 0
        self._this_forward: tuple[int, int, int, int, float] | None = None

    def mark_start(self) -> None:
        """Start the point's clock. Call once, right before the first submit."""
        self._t0 = time.perf_counter()

    def _emit(self, req: RunningRequest, step: GenerationStep) -> None:
        self._emitted += step.num_tokens
        super()._emit(req, step)

    def _step(self) -> None:
        self._this_forward = None
        super()._step()
        if self._this_forward is not None and self._t0 is not None:
            batch, total_q, n_verify, waiting, forward_s = self._this_forward
            self.steps.append(
                StepRecord(
                    batch=batch,
                    total_q=total_q,
                    n_verify=n_verify,
                    waiting=waiting,
                    forward_s=forward_s,
                    end_s=time.perf_counter() - self._t0,
                    emitted=self._emitted,
                )
            )

    def _phase(self, name: str, fn: Callable[[], Any]) -> Any:
        if not self._timed:
            return fn()
        if self._sync is not None:
            self._sync()
        t0 = time.perf_counter()
        out = fn()
        if self._sync is not None:
            self._sync()
        setattr(self.phases, name, getattr(self.phases, name) + time.perf_counter() - t0)
        return out

    def _sample_decoders(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = self._phase("sample_s", super()._sample_decoders)
        return out

    def _draft_speculators(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = self._phase("draft_s", super()._draft_speculators)
        return out

    def _packed_forward(
        self, alive: list[RunningRequest], sampled_decode_tokens: dict[int, list[int]]
    ) -> tuple[torch.Tensor, list[int], dict[int, torch.Tensor] | None]:
        # Counted before the call: the verify phase clears `offered` later.
        n_verify = sum(1 for r in alive if r.spec is not None and r.spec.offered is not None)
        waiting = self._waiting.qsize()
        t0 = time.perf_counter()
        out: tuple[torch.Tensor, list[int], dict[int, torch.Tensor] | None] = self._phase(
            "forward_s",
            lambda: super(InstrumentedScheduler, self)._packed_forward(
                alive, sampled_decode_tokens
            ),
        )
        self._this_forward = (len(alive), out[1][-1], n_verify, waiting, time.perf_counter() - t0)
        return out

    def _verify_speculators(
        self,
        alive: list[RunningRequest],
        packed_logits: torch.Tensor,
        cu_seqlens_q: list[int],
        sink: dict[int, torch.Tensor] | None,
    ) -> None:
        self._phase(
            "verify_s",
            lambda: super(InstrumentedScheduler, self)._verify_speculators(
                alive, packed_logits, cu_seqlens_q, sink
            ),
        )


@dataclasses.dataclass(frozen=True)
class SweepPoint:
    """The outcome of one measurement point. JSON-ready through `to_json`.

    Two throughput numbers, because a finite batch drains. `tokens_per_s`
    covers the whole run, start to last result. But once no prompt waits, the
    in-flight count falls below `concurrency`, and the two arms fall unevenly:
    plain requests of one length finish together, while speculative requests
    finish at different rounds. `steady_tokens_per_s` covers only the steps in
    which a prompt was still waiting, so every slot was busy, and compares the
    arms at the concurrency the label names. `mean_in_flight` says how far the
    whole run fell short of it.
    """

    arm: str
    concurrency: int
    confidence_threshold: float | None
    max_tokens: int
    n_requests: int
    wall_s: float
    generated_tokens: int
    tokens: list[list[int]]
    spec: SpecSummary | None
    n_not_speculating: int
    finish_reasons: dict[str, int]
    phases: PhaseTimes | None
    steps: list[StepRecord]

    @property
    def tokens_per_s(self) -> float:
        return self.generated_tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def mean_in_flight(self) -> float:
        """Requests per forward, weighted by how long each step lasted."""
        total = 0.0
        weighted = 0.0
        previous = 0.0
        for step in self.steps:
            dt = step.end_s - previous
            weighted += step.batch * dt
            total += dt
            previous = step.end_s
        return weighted / total if total > 0 else 0.0

    @property
    def steady_tokens_per_s(self) -> float | None:
        """Throughput up to the last step in which a prompt was still waiting.

        None when no step had a prompt waiting, which happens when the point
        has no more prompts than `concurrency`.
        """
        steady = [s for s in self.steps if s.waiting > 0]
        if not steady:
            return None
        last = steady[-1]
        return last.emitted / last.end_s if last.end_s > 0 else None

    def to_json(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["tokens_per_s"] = self.tokens_per_s
        out["steady_tokens_per_s"] = self.steady_tokens_per_s
        out["mean_in_flight"] = self.mean_in_flight
        out["tau"] = self.spec.tau if self.spec is not None else None
        out["tau_unclamped"] = self.spec.tau_unclamped if self.spec is not None else None
        return out


def run_point(
    runner: ModelRunner,
    prompts: Sequence[str],
    *,
    concurrency: int,
    max_tokens: int,
    dspark: DSparkSpec | None = None,
    sampling: SamplingParams | None = None,
    timed: bool = False,
    sync: Callable[[], None] | None = None,
    timeout_s: float = 1800.0,
) -> SweepPoint:
    """Run `prompts` through a fresh scheduler with `concurrency` in flight.

    Fails fast when the engine thread dies. Any exception other than
    `OutOfMemoryError` ends that thread, and then no request in flight ever
    gets its terminal step. The handle notices within one liveness poll and
    raises `EngineNotRunningError`; the traceback is in the log. `timeout_s`
    still bounds the whole point, for an engine that lives but stalls.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    if not prompts:
        raise ValueError("a sweep point needs at least one prompt")
    params = sampling if sampling is not None else SamplingParams(temperature=0.0)
    sched = InstrumentedScheduler(
        runner, timed=timed, sync=sync, max_concurrent=concurrency, dspark=dspark
    )

    sched.start()
    try:
        sched.mark_start()
        start = time.perf_counter()
        handles = [
            sched.submit(Request(prompt=p, sampling_params=params, max_tokens=max_tokens))
            for p in prompts
        ]
        # Drained in submission order. A later request's queue fills while an
        # earlier one is drained, but a step carries a whole committed block,
        # so `max_tokens` well under the queue cap cannot overflow it.
        deadline = start + timeout_s
        results: list[GenerationResult] = []
        for handle in handles:
            try:
                results.append(handle.wait(timeout=max(0.0, deadline - time.perf_counter())))
            except TimeoutError:
                raise TimeoutError(
                    f"sweep point (concurrency={concurrency}) exceeded {timeout_s:.0f}s"
                ) from None
        wall_s = time.perf_counter() - start
    finally:
        sched.stop()

    summaries = [r.spec_summary for r in results if r.spec_summary is not None]
    return SweepPoint(
        arm="dspark" if dspark is not None else "plain",
        concurrency=concurrency,
        confidence_threshold=dspark.confidence_threshold if dspark is not None else None,
        max_tokens=max_tokens,
        n_requests=len(results),
        wall_s=wall_s,
        generated_tokens=sum(len(r.tokens) for r in results),
        tokens=[list(r.tokens) for r in results],
        spec=SpecSummary.pool(summaries) if summaries else None,
        n_not_speculating=(len(results) - len(summaries)) if dspark is not None else 0,
        finish_reasons={
            reason: sum(1 for r in results if r.finish_reason == reason)
            for reason in sorted({r.finish_reason for r in results})
        },
        phases=sched.phases if timed else None,
        steps=list(sched.steps),
    )


@dataclasses.dataclass(frozen=True)
class TokenAgreement:
    """How closely one arm's tokens follow another's, request by request.

    On CPU fp32 greedy speculation must match the plain path exactly. On a
    GPU in bf16 it need not: a different matmul shape flips a near-tie and the
    two outputs part ways. So the bf16 gate is statistical, and these are its
    numbers: how many requests match exactly, and how far each one agrees
    before its first different token.
    """

    n_requests: int
    exact_matches: int
    common_prefix: list[int]
    reference_lengths: list[int]

    @property
    def mean_prefix_fraction(self) -> float:
        fractions = [
            p / n if n else 1.0
            for p, n in zip(self.common_prefix, self.reference_lengths, strict=True)
        ]
        return sum(fractions) / len(fractions) if fractions else 1.0


def token_agreement(reference: list[list[int]], candidate: list[list[int]]) -> TokenAgreement:
    """Compare two arms' per-request tokens; the lists must be in the same order."""
    if len(reference) != len(candidate):
        raise ValueError(f"{len(reference)} reference requests vs {len(candidate)} candidates")
    prefixes: list[int] = []
    for ref, cand in zip(reference, candidate, strict=True):
        n = 0
        for a, b in zip(ref, cand, strict=False):
            if a != b:
                break
            n += 1
        prefixes.append(n)
    return TokenAgreement(
        n_requests=len(reference),
        exact_matches=sum(1 for r, c in zip(reference, candidate, strict=True) if r == c),
        common_prefix=prefixes,
        reference_lengths=[len(r) for r in reference],
    )


def run_sweep(
    runner: ModelRunner,
    drafter: Qwen3DSparkDrafter,
    prompts_by_dataset: dict[str, list[str]],
    *,
    out_dir: Path,
    max_tokens: int,
    concurrencies: Sequence[int],
    thresholds: Sequence[float],
    timed_concurrencies: Sequence[int] = (),
    sync: Callable[[], None] | None = None,
    on_checkpoint: Callable[[], None] | None = None,
    timeout_s: float = 1800.0,
    max_spec_prompt_tokens: int = 1024,
    identity: str = "",
) -> dict[str, dict[str, Any]]:
    """Measure every point of the grid, one JSON file per point in `out_dir`.

    Per dataset and concurrency: the plain arm, then the dspark arm at each
    confidence threshold, with tau and token agreement against the plain arm
    at the same level. Then a timed pass (plain and dspark at threshold 0) at
    each of `timed_concurrencies`.

    Resumable: a point whose file already exists is skipped, so a rerun after
    a dropped connection or a failed point measures only what is missing.
    `on_checkpoint` runs after each file write (a volume commit, on Modal).
    Each file carries a fingerprint of what produced it: the dataset's
    prompts, `max_tokens`, the speculation limit, and `identity`, which names
    the models and settings that the folder name does not show. A stored point
    with a different fingerprint raises instead of mixing an older grid's
    numbers into this one, and a new folder is the way to measure a new grid.

    Raises if a dspark point has any request that did not speculate, because
    that point would then be measuring the plain path under the dspark label.
    Raises as well if any request of any point was cancelled: admission
    rejects a request that can never fit the pool, and the point would then
    measure fewer requests than the grid claims.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only this grid's points, not every file in the folder: a rerun may share
    # the folder with other files, such as run metadata or an older grid.
    points: dict[str, dict[str, Any]] = {}

    fingerprint = ""

    def stored(key: str) -> dict[str, Any] | None:
        path = out_dir / f"{key}.json"
        loaded: dict[str, Any] | None = json.loads(path.read_text()) if path.exists() else None
        if loaded is not None:
            if loaded.get("fingerprint") != fingerprint:
                raise ValueError(f"{path} was measured with a different grid; use a new out_dir")
            points[key] = loaded
        return loaded

    def store(key: str, payload: dict[str, Any]) -> None:
        payload["fingerprint"] = fingerprint
        (out_dir / f"{key}.json").write_text(json.dumps(payload))
        points[key] = payload
        if on_checkpoint is not None:
            on_checkpoint()

    def spec_for(threshold: float) -> DSparkSpec:
        return DSparkSpec(
            drafter,
            confidence_threshold=threshold,
            max_spec_prompt_tokens=max_spec_prompt_tokens,
        )

    for name, prompts in prompts_by_dataset.items():
        fingerprint = _fingerprint(prompts, max_tokens, max_spec_prompt_tokens, identity)
        for c in concurrencies:
            plain_key = f"{name}-plain-c{c}"
            plain = stored(plain_key)
            if plain is None:
                plain_point = run_point(
                    runner, prompts, concurrency=c, max_tokens=max_tokens, timeout_s=timeout_s
                )
                _require_complete(plain_key, plain_point)
                plain = plain_point.to_json()
                store(plain_key, plain)
            for t in thresholds:
                key = f"{name}-dspark-t{t}-c{c}"
                if stored(key) is not None:
                    continue
                point = run_point(
                    runner,
                    prompts,
                    concurrency=c,
                    max_tokens=max_tokens,
                    dspark=spec_for(t),
                    timeout_s=timeout_s,
                )
                _require_complete(key, point)
                payload = point.to_json()
                agreement = token_agreement(plain["tokens"], point.tokens)
                payload["agreement_vs_plain"] = {
                    "exact_matches": agreement.exact_matches,
                    "n_requests": agreement.n_requests,
                    "mean_prefix_fraction": agreement.mean_prefix_fraction,
                }
                plain_rate = plain["tokens_per_s"]
                payload["speedup_vs_plain"] = (
                    point.tokens_per_s / plain_rate if plain_rate else None
                )
                # The steady speedup compares the arms while every slot was
                # busy, where the whole-run ratio also mixes in the drain.
                steady, plain_steady = point.steady_tokens_per_s, plain["steady_tokens_per_s"]
                payload["steady_speedup_vs_plain"] = (
                    steady / plain_steady if steady is not None and plain_steady else None
                )
                store(key, payload)

        for c in timed_concurrencies:
            for arm, dspark in (("plain", None), ("dspark", spec_for(0.0))):
                key = f"{name}-timed-{arm}-c{c}"
                if stored(key) is not None:
                    continue
                timed = run_point(
                    runner,
                    prompts,
                    concurrency=c,
                    max_tokens=max_tokens,
                    dspark=dspark,
                    timed=True,
                    sync=sync,
                    timeout_s=timeout_s,
                )
                _require_complete(key, timed)
                store(key, timed.to_json())

    return points


def _require_complete(key: str, point: SweepPoint) -> None:
    """Refuse a point that did not measure what its label says.

    A cancelled request means admission rejected it (it can never fit the
    pool), so the point covers fewer requests than the grid claims. A dspark
    point with a request that did not speculate measures the plain path for
    that request under the dspark label.
    """
    cancelled = point.finish_reasons.get("cancelled", 0)
    if cancelled:
        raise AssertionError(f"{key}: {cancelled} of {point.n_requests} requests were cancelled")
    if point.n_not_speculating:
        raise AssertionError(f"{key}: {point.n_not_speculating} requests did not speculate")


def _fingerprint(prompts: Sequence[str], max_tokens: int, spec_limit: int, identity: str) -> str:
    """A short hash of everything that makes two points comparable."""
    blob = json.dumps(
        {"prompts": list(prompts), "max_tokens": max_tokens, "limit": spec_limit, "id": identity}
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
