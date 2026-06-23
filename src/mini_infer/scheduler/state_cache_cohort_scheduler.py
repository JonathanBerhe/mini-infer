"""Lockstep cohort scheduler for StateCache models (DeepSeek-V4).

Serves a *cohort* of requests through one batched forward, in contrast to
`StateCacheScheduler` which serves one request at a time. Requests whose prompts
tokenize to the same length and share sampling parameters are grouped and
decoded together in lockstep: one prefill, then a shared decode loop where every
step advances all sequences by one token at a single position. A sequence stops
at its own EOS or `max_tokens` while the cohort keeps stepping for the rest; the
cohort drains when all its sequences finish.

This is the lockstep / static-batching form: positions are shared, so a cohort
is fixed once formed. Requests do not join or leave mid-flight, and a request
that arrives while a cohort is running waits for the next one. Ragged
per-request positions (continuous batching) are a separate path. Equal-length
grouping is deliberate; padding mixed lengths to batch them is a follow-up.

The interface (start / stop / submit / run / stream) matches
`StateCacheScheduler` and `ContinuousScheduler`, so the HTTP server can treat
all three interchangeably.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Iterator

from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.scheduler.request_state import (
    FinishReason,
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RunningRequest,
)

logger = logging.getLogger(__name__)

# A cohort groups by (prompt length, sampling params), so two requests batch
# together only if they decode in true lockstep: same number of prefill
# positions and the same sampling behavior on every step.
_CohortKey = tuple[int, tuple[float, int, float]]


class StateCacheCohortScheduler:
    """Serves equal-length StateCache requests as lockstep cohorts (one batched forward)."""

    # Same rationale as ContinuousScheduler / StateCacheScheduler.
    DEFAULT_OUTPUT_QUEUE_SIZE = 256
    IDLE_SLEEP_SECONDS = 0.005
    DEFAULT_MAX_COHORT_SIZE = 8

    def __init__(
        self,
        generator: StateCacheGenerator,
        *,
        max_cohort_size: int = DEFAULT_MAX_COHORT_SIZE,
    ) -> None:
        if max_cohort_size <= 0:
            raise ValueError(f"max_cohort_size must be positive, got {max_cohort_size}")
        self._generator = generator
        self._max_cohort_size = max_cohort_size
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the engine thread. Idempotent if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-state-cache-cohort-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the engine thread to exit and join it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, request: Request) -> RequestHandle:
        """Enqueue a request and return a handle the caller drains for output."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("scheduler not running; call start() before submit()")
        running = RunningRequest(
            request=request,
            output_queue=queue.Queue(maxsize=self.DEFAULT_OUTPUT_QUEUE_SIZE),
        )
        self._waiting.put(running)
        return RequestHandle(running)

    def run(self, request: Request) -> GenerationResult:
        """Submit and block until the request completes; return the result."""
        return self.submit(request).wait()

    def stream(self, request: Request) -> Iterator[GenerationStep]:
        """Submit and yield each generation step as the engine produces it."""
        yield from self.submit(request).steps()

    def _engine_loop(self) -> None:
        """Drain queued requests, group them into cohorts, and serve each cohort."""
        try:
            while not self._stop_event.is_set():
                try:
                    first = self._waiting.get(timeout=self.IDLE_SLEEP_SECONDS)
                except queue.Empty:
                    continue
                # Take everything already queued so we batch as wide as the
                # current arrivals allow before forming cohorts.
                batch = [first]
                while True:
                    try:
                        batch.append(self._waiting.get_nowait())
                    except queue.Empty:
                        break
                for cohort in self._form_cohorts(batch):
                    self._serve_cohort(cohort)
        except Exception:
            logger.exception("state-cache cohort engine thread crashed")
            raise

    def _form_cohorts(self, batch: list[RunningRequest]) -> list[list[RunningRequest]]:
        """Group requests into lockstep cohorts by (prompt length, sampling params).

        Tokenizes each request and buckets by the key, splitting each bucket into
        cohorts of at most `max_cohort_size`. Group order follows first arrival.
        Requests already cancelled, or whose prompt tokenizes to empty, are
        finished here rather than placed in a cohort (an empty prompt cannot
        prefill, and one bad member must not poison the cohort's batched
        forward).
        """
        tokenizer = self._generator.tokenizer
        groups: dict[_CohortKey, list[RunningRequest]] = {}
        order: list[_CohortKey] = []
        for running in batch:
            if running.cancel_event.is_set():
                self._finish(running, "cancelled")
                continue
            running.prompt_token_ids = tokenizer.encode(running.request.prompt)
            if not running.prompt_token_ids:
                self._finish(running, "stop")
                continue
            params = running.request.sampling_params
            key = (len(running.prompt_token_ids), (params.temperature, params.top_k, params.top_p))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(running)
        cohorts: list[list[RunningRequest]] = []
        for key in order:
            members = groups[key]
            for start in range(0, len(members), self._max_cohort_size):
                cohorts.append(members[start : start + self._max_cohort_size])
        return cohorts

    def _serve_cohort(self, cohort: list[RunningRequest]) -> None:
        """Decode one cohort in lockstep, streaming each sequence's deltas.

        Each sequence stops at its own `max_tokens` (length) or EOS (stop); a
        cancelled request drops out without disturbing the rest. Any error
        finishes every not-yet-finished member as cancelled so no consumer
        hangs.
        """
        tokenizer = self._generator.tokenizer
        prompts = [running.prompt_token_ids for running in cohort]
        cohort_max_tokens = max(running.request.max_tokens for running in cohort)
        finalized = [False] * len(cohort)
        # Reason recorded when this scheduler stops a sequence via should_cancel;
        # None means the iterator stopped it on EOS, which finalizes as "stop".
        pending_reason: list[FinishReason | None] = [None] * len(cohort)

        def should_cancel(index: int) -> bool:
            running = cohort[index]
            if running.cancel_event.is_set():
                pending_reason[index] = "cancelled"
                return True
            if len(running.tokens_generated) >= running.request.max_tokens:
                pending_reason[index] = "length"
                return True
            return False

        def finalize(index: int, reason: FinishReason) -> None:
            if finalized[index]:
                return
            finalized[index] = True
            self._finish(cohort[index], reason)

        try:
            for step in self._generator.iter_generate_ids_batched(
                prompts,
                max_new_tokens=cohort_max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                sampling_params=cohort[0].request.sampling_params,
                should_cancel=should_cancel,
            ):
                for index, token in enumerate(step):
                    if finalized[index]:
                        continue
                    if token is None:
                        # Stopped this step: reason set by should_cancel, else EOS.
                        finalize(index, pending_reason[index] or "stop")
                        continue
                    running = cohort[index]
                    running.tokens_generated.append(token)
                    # Decode-and-diff so multi-byte UTF-8 emits correctly.
                    current_text = tokenizer.decode(running.tokens_generated)
                    delta = current_text[len(running.last_text) :]
                    running.last_text = current_text
                    self._emit(running, GenerationStep(text=delta))
        except Exception:
            logger.exception("cohort generation failed; finishing remaining as cancelled")
            for index in range(len(cohort)):
                finalize(index, "cancelled")
            return
        # Sequences still open when the iterator exhausts ran to the length cap.
        for index in range(len(cohort)):
            finalize(index, "length")

    def _emit(self, running: RunningRequest, step: GenerationStep) -> None:
        """Non-blocking emit; drop the oldest buffered step if the queue is full."""
        try:
            running.output_queue.put_nowait(step)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                running.output_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                running.output_queue.put_nowait(step)

    def _finish(self, running: RunningRequest, reason: FinishReason) -> None:
        """Emit the terminal step (the consumer's end-of-stream signal)."""
        running.finish_reason = reason
        terminal = GenerationStep(text="", finish_reason=reason)
        try:
            running.output_queue.put_nowait(terminal)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                running.output_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                running.output_queue.put_nowait(terminal)
