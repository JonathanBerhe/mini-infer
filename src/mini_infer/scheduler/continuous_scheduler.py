"""Continuous batching scheduler with a dedicated engine thread.

The scheduler accepts requests asynchronously, admits them when the KV block pool
has room, drives them through prefill and decode, and frees blocks when each
completes. A single engine thread owns the model and the running batch; API
threads only enqueue requests and drain per-request output queues.

The current implementation processes one request per forward pass within a step;
when a batched-decode attention path is available, the same scheduler structure
can dispatch a single batched call without changing the public API.
"""

import logging
import math
import queue
import threading
from collections.abc import Iterator

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import sample
from mini_infer.scheduler.request_state import (
    FinishReason,
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RequestState,
    RunningRequest,
)

logger = logging.getLogger(__name__)


class ContinuousScheduler:
    """FIFO multi-request scheduler with admission control.

    A single engine thread owns the model and the running batch. Submitted
    requests join a waiting queue; each engine step admits new requests if the
    block pool has capacity, prefills new admits, decodes each running request
    by one token, and reaps any that have finished.

    Lifecycle: call `start()` once before `submit()`; call `stop()` at shutdown
    to join the engine thread. The FastAPI lifespan does this for the HTTP
    server; tests do it via a fixture.
    """

    # Per-request output queue cap. Each item is a small GenerationStep; in
    # normal operation the consumer (SSE stream / wait()) drains as fast as the
    # engine produces, so the queue rarely holds more than 1-2 items. 256 is
    # generous headroom for transient consumer hiccups (network blip on a
    # streaming HTTP client) without burning meaningful memory.
    DEFAULT_OUTPUT_QUEUE_SIZE = 256

    # Extra blocks reserved beyond the prompt at admission time, so a request
    # has decode runway before the engine hits the pool again. 8 blocks * 16
    # tokens/block = 128 tokens of decode capacity. If a request needs more, we
    # currently allocate during decode (and may OOM if the pool is full).
    # Production engines preempt instead; that's tracked as a follow-up.
    DEFAULT_DECODE_HEADROOM_BLOCKS = 8

    # Engine-thread idle poll interval. When there's no work, the loop waits
    # this long before checking again. 5 ms = 200 Hz: low enough that newly
    # submitted requests start within ~5 ms (imperceptible for HTTP), high
    # enough not to spin the CPU. A signal-based wake on submit would be better
    # but adds complexity for negligible payoff at our scale.
    IDLE_SLEEP_SECONDS = 0.005

    def __init__(
        self,
        runner: ModelRunner,
        max_concurrent: int = 16,
        decode_headroom_blocks: int = DEFAULT_DECODE_HEADROOM_BLOCKS,
    ) -> None:
        self._runner = runner
        self._max_concurrent = max_concurrent
        self._decode_headroom = decode_headroom_blocks
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._running: list[RunningRequest] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the engine thread. Idempotent if the thread is already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the engine thread to exit and join it (with the given timeout)."""
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
        """Submit and block until the request completes; returns the aggregated result."""
        return self.submit(request).wait()

    def stream(self, request: Request) -> Iterator[GenerationStep]:
        """Submit and yield each generation step as the engine produces it."""
        yield from self.submit(request).steps()

    def _engine_loop(self) -> None:
        """Engine-thread main loop; runs `_step()` until `stop()` is signaled."""
        try:
            while not self._stop_event.is_set():
                self._step()
                # When fully idle, wait briefly so we don't spin the CPU.
                if not self._running and self._waiting.empty():
                    self._stop_event.wait(timeout=self.IDLE_SLEEP_SECONDS)
        except Exception:
            logger.exception("engine thread crashed")
            raise

    def _step(self) -> None:
        """One scheduler iteration: admit, prefill new admits, decode running, reap done."""
        self._admit_waiting()
        for req in list(self._running):
            if req.state == RequestState.PREFILLING:
                self._prefill(req)
        for req in list(self._running):
            if req.state == RequestState.DECODING:
                self._decode_one(req)
        # Reap completed requests and return their blocks to the pool.
        for req in [r for r in self._running if r.state == RequestState.DONE]:
            if req.cache is not None:
                req.cache.free()
            self._running.remove(req)

    def _admit_waiting(self) -> None:
        """Move waiting requests into the running batch while the block pool has room."""
        while len(self._running) < self._max_concurrent:
            try:
                running = self._waiting.get_nowait()
            except queue.Empty:
                return

            running.prompt_token_ids = self._runner.tokenizer.encode(running.request.prompt)
            block_size = self._runner.block_pool.block_size
            required_blocks = (
                math.ceil(len(running.prompt_token_ids) / block_size) + self._decode_headroom
            )

            if self._runner.block_pool.num_free_blocks < required_blocks:
                # Pool is too full to safely admit this request. Re-enqueue and
                # stop admitting this step. Note: the re-enqueue breaks strict
                # FIFO if the queue contains smaller requests behind this one;
                # acceptable trade for now, can be revisited if it matters.
                self._waiting.put(running)
                return

            running.state = RequestState.PREFILLING
            self._running.append(running)

    def _prefill(self, req: RunningRequest) -> None:
        """Run prefill once for a newly admitted request, then transition to DECODING."""
        cache, logits = self._runner.prefill(req.prompt_token_ids)
        req.cache = cache
        req.last_logits = logits
        req.state = RequestState.DECODING

    def _decode_one(self, req: RunningRequest) -> None:
        """Sample one token, emit a streaming step, advance the cache for the next step."""
        assert req.cache is not None
        assert req.last_logits is not None
        tokenizer = self._runner.tokenizer
        next_token = sample(req.last_logits, req.request.sampling_params)

        if next_token == tokenizer.eos_token_id:
            self._finish(req, "stop")
            return

        req.tokens_generated.append(next_token)
        # Decode-and-diff: re-decode all tokens so far and emit the new suffix.
        # Handles multi-byte UTF-8 sequences correctly (one tokenizer call,
        # one string slice) at the cost of O(n) work per step.
        current_text = tokenizer.decode(req.tokens_generated)
        delta = current_text[len(req.last_text) :]
        req.last_text = current_text
        req.output_queue.put(GenerationStep(text=delta))

        if len(req.tokens_generated) >= req.request.max_tokens:
            self._finish(req, "length")
            return

        req.cache, req.last_logits = self._runner.decode(req.cache, next_token)

    def _finish(self, req: RunningRequest, reason: FinishReason) -> None:
        """Mark a request DONE and emit the terminal step with its finish_reason."""
        req.finish_reason = reason
        req.state = RequestState.DONE
        req.output_queue.put(GenerationStep(text="", finish_reason=reason))
