"""Sequential (queue-based) scheduler for StateCache models (DeepSeek-V4).

V4 keeps per-request attention state in a `StateCache` and decodes from a
single position, so it does not fit the packed-varlen continuous batching
`ContinuousScheduler` runs over `PagedKVCache`. This scheduler serves one
request at a time from a FIFO queue via `StateCacheGenerator`: correct,
streaming, cancellable serving of concurrent clients, without batched
throughput.

Batched throughput over a StateCache (ragged per-request positions in the
decode) is a deliberate non-goal here, consistent with the project not
competing on throughput; it can be layered on later if it ever matters.

The interface (start / stop / submit / run / stream) matches
`ContinuousScheduler` so the HTTP server treats the two interchangeably.
"""

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


class StateCacheScheduler:
    """FIFO scheduler that serves one StateCache request at a time."""

    # See ContinuousScheduler for the rationale on these; same values.
    DEFAULT_OUTPUT_QUEUE_SIZE = 256
    IDLE_SLEEP_SECONDS = 0.005

    def __init__(self, generator: StateCacheGenerator) -> None:
        self._generator = generator
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the engine thread. Idempotent if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-state-cache-engine", daemon=True
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
        """Pull queued requests and serve each to completion, FIFO."""
        try:
            while not self._stop_event.is_set():
                try:
                    running = self._waiting.get(timeout=self.IDLE_SLEEP_SECONDS)
                except queue.Empty:
                    continue
                self._serve(running)
        except Exception:
            logger.exception("state-cache engine thread crashed")
            raise

    def _serve(self, running: RunningRequest) -> None:
        """Generate the full response for one request, streaming per-token deltas.

        Cancellation (set on the handle when a client disconnects) is checked
        after each token, so an abandoned generation stops within one token.
        Any generation error is logged and the request is finished with a
        terminal step so the consumer never hangs.
        """
        if running.cancel_event.is_set():
            self._finish(running, "cancelled")
            return

        tokenizer = self._generator.tokenizer
        running.prompt_token_ids = tokenizer.encode(running.request.prompt)
        finish_reason: FinishReason = "stop"
        try:
            cancelled = False
            for token in self._generator.iter_generate_ids(
                running.prompt_token_ids,
                max_new_tokens=running.request.max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                sampling_params=running.request.sampling_params,
            ):
                if running.cancel_event.is_set():
                    cancelled = True
                    break
                running.tokens_generated.append(token)
                # Decode-and-diff so multi-byte UTF-8 sequences emit correctly:
                # re-decode all tokens, emit only the new suffix.
                current_text = tokenizer.decode(running.tokens_generated)
                delta = current_text[len(running.last_text) :]
                running.last_text = current_text
                self._emit(running, GenerationStep(text=delta))

            if cancelled:
                finish_reason = "cancelled"
            elif len(running.tokens_generated) >= running.request.max_tokens:
                finish_reason = "length"
            else:
                finish_reason = "stop"
        except Exception:
            logger.exception("generation failed for request; finishing as cancelled")
            finish_reason = "cancelled"
        self._finish(running, finish_reason)

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
