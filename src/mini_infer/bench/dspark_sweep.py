"""Concurrency sweep for DSpark speculative decoding inside `ContinuousScheduler`.

A point runs one arm, "plain" or "dspark", on a new scheduler with
`max_concurrent` set to the concurrency under test. All prompts are submitted
at once. So while a prompt waits, `concurrency` requests are in flight if the
block pool has room (a closed loop), and then the batch drains. All points of
a dataset use the same prompts, so only the concurrency or the arm changes.
Throughput counts tokens, not stream steps, because a verify step streams
several tokens as one step. The sweep runs in-process because the server
cannot enable DSpark. See ADR-031.

GPU kernels run asynchronously, so a host timer charges GPU work to the next
phase that synchronizes. A timed point calls `sync` (pass `torch.cuda.synchronize`)
at every phase boundary, and these barriers slow the step. So a timed point
gives the time per phase, and an untimed point gives the throughput.
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
    """One packed forward: its width, the queue behind it, and its place in the run.

    `batch` counts requests and `total_q` counts query tokens. Prefill chunks
    and verify windows make `total_q` larger. A verify window feeds
    `1 + len(offered)` tokens, and `n_verify` counts those windows. `waiting > 0`
    marks a step with full admission: every slot was busy, or the pool had no
    room. `end_s` is seconds since the point started, and `emitted` counts
    tokens streamed so far. `forward_s` is correct only in a timed point.
    Without a barrier, a host timer measures only kernel launches.
    """

    batch: int
    total_q: int
    n_verify: int
    waiting: int
    forward_s: float
    end_s: float
    emitted: int


class InstrumentedScheduler(ContinuousScheduler):
    """A `ContinuousScheduler` that records every forward and can time each phase.

    For benchmarks only. It wraps the private phase methods, so it depends on
    the structure of `_step`. An untimed record costs a few clock reads and one
    queue size read, with no barrier. Read `phases` and `steps` only after
    `stop()`, because the engine thread writes them.
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
        # The verify phase clears `offered` after this method, so count it here.
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
    """The outcome of one measurement point. `to_json` gives a JSON-ready dict.

    A finite batch drains at the end, and speculative requests finish at
    different rounds, so the drain differs between arms. `tokens_per_s` covers
    the whole run. `steady_tokens_per_s` covers only the steps in which a prompt
    still waited, so it compares the arms while admission was full.
    `mean_in_flight` shows how far the whole run fell below `concurrency`.
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
        """Throughput up to the last step in which a prompt waited, or None if none did."""
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
    """Run `prompts` through a new scheduler with `concurrency` requests in flight.

    Raises `EngineNotRunningError` within one liveness poll if the engine thread
    dies, and the log holds the traceback. See ADR-028. `timeout_s` bounds the
    whole point, for an engine that is alive but stalls.
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
        # The loop drains handles in submission order, so later queues fill meanwhile.
        # Each step carries at least one token, so a queue holds at most
        # `max_tokens` steps plus the terminal one. That must stay under the queue cap.
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
    """How closely the tokens of one arm follow those of another, per request.

    Greedy speculation must match the plain path exactly in CPU fp32. In GPU
    bf16 a new matmul shape can flip a near-tie, so the pass criterion is statistical.
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
    """Compare the tokens of two arms per request. Both lists must use the same request order."""
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
    """Measure every point of the grid, and write one JSON file per point to `out_dir`.

    Per dataset and concurrency, it runs the plain arm, then the dspark arm at
    each threshold, and stores the dspark speedup and token agreement against
    plain. Then a timed pass runs both arms (dspark at threshold 0) at each of
    `timed_concurrencies`.

    The sweep resumes: it skips a point whose file exists. `on_checkpoint` runs
    after each file write (for example a Modal volume commit). Each file stores
    a fingerprint of the prompts, `max_tokens`, `max_spec_prompt_tokens`, and
    `identity` (models and settings that the folder name omits). A stored point
    with another fingerprint raises, so an older grid never mixes in. A point
    that does not measure what its label says also raises (`_require_complete`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Return only this grid. The folder can also hold run metadata or older grids.
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
    """Raise if a point did not measure what its label says.

    A cancelled request (rejected at admission or preempted on OOM) raises. A dspark
    request that admission moved to the plain path, with only an INFO log, also raises.
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
