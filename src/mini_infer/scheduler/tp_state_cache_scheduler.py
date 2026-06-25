"""Rank-0 scheduler for tensor-parallel StateCache serving (DeepSeek-V4-Flash).

Runs only on the leader (rank 0). Same start / stop / submit / run / stream
interface as the other schedulers (`ContinuousScheduler`,
`StateCacheContinuousScheduler`), so the HTTP endpoint is unchanged, but it
drives a `TensorParallelStateCacheServer`: each request's generation runs in
lockstep across all ranks (the leader broadcasts the prompt and each sampled
token). Requests are served one at a time, since TP generation is inherently a
single synchronized stream across the ranks.

Cancellation is wired through the server's `should_cancel` hook rather than by
abandoning a generator mid-stream: the leader checks it each step and
broadcasts a stop, so the follower ranks never block waiting for a token that
will not come (which would deadlock them).

The follower ranks do NOT use this scheduler; they call
`TensorParallelStateCacheServer.run_follower_loop()`.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Iterator

from mini_infer.engine.tp_state_cache_server import TensorParallelStateCacheServer
from mini_infer.scheduler.request_state import (
    FinishReason,
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RunningRequest,
)

logger = logging.getLogger(__name__)


class TensorParallelStateCacheScheduler:
    """FIFO scheduler serving one TP request at a time via the rank-0 leader."""

    DEFAULT_OUTPUT_QUEUE_SIZE = 256
    IDLE_SLEEP_SECONDS = 0.005

    def __init__(self, server: TensorParallelStateCacheServer) -> None:
        if not server.is_leader:
            raise RuntimeError(
                "TensorParallelStateCacheScheduler runs on the leader (rank 0) only; "
                "followers call server.run_follower_loop()"
            )
        self._server = server
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-tp-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, request: Request) -> RequestHandle:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("scheduler not running; call start() before submit()")
        running = RunningRequest(
            request=request,
            output_queue=queue.Queue(maxsize=self.DEFAULT_OUTPUT_QUEUE_SIZE),
        )
        self._waiting.put(running)
        return RequestHandle(running)

    def run(self, request: Request) -> GenerationResult:
        return self.submit(request).wait()

    def stream(self, request: Request) -> Iterator[GenerationStep]:
        yield from self.submit(request).steps()

    def _engine_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    running = self._waiting.get(timeout=self.IDLE_SLEEP_SECONDS)
                except queue.Empty:
                    continue
                self._serve(running)
        except Exception:
            logger.exception("tensor-parallel engine thread crashed")
            raise

    def _serve(self, running: RunningRequest) -> None:
        if running.cancel_event.is_set():
            self._finish(running, "cancelled")
            return
        tokenizer = self._server.tokenizer
        running.prompt_token_ids = tokenizer.encode(running.request.prompt)

        def emit(token: int) -> None:
            running.tokens_generated.append(token)
            current_text = tokenizer.decode(running.tokens_generated)
            delta = current_text[len(running.last_text) :]
            running.last_text = current_text
            self._emit(running, GenerationStep(text=delta))

        try:
            self._server.generate_ids(
                running.prompt_token_ids,
                max_new_tokens=running.request.max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                sampling_params=running.request.sampling_params,
                emit=emit,
                should_cancel=running.cancel_event.is_set,
            )
        except Exception:
            logger.exception("tensor-parallel generation failed; finishing as cancelled")
            self._finish(running, "cancelled")
            return

        if running.cancel_event.is_set():
            finish_reason: FinishReason = "cancelled"
        elif len(running.tokens_generated) >= running.request.max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"
        self._finish(running, finish_reason)

    def _emit(self, running: RunningRequest, step: GenerationStep) -> None:
        try:
            running.output_queue.put_nowait(step)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                running.output_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                running.output_queue.put_nowait(step)

    def _finish(self, running: RunningRequest, reason: FinishReason) -> None:
        running.finish_reason = reason
        terminal = GenerationStep(text="", finish_reason=reason)
        try:
            running.output_queue.put_nowait(terminal)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                running.output_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                running.output_queue.put_nowait(terminal)
