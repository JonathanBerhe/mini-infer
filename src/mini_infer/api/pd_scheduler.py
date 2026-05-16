"""PD-backed scheduler adapter for the HTTP API.

The FastAPI server in `mini_infer.api.server` expects a scheduler with the
shape `ContinuousScheduler` exposes:

  - `start()` / `stop()` lifecycle.
  - `submit(request) -> RequestHandle` for streaming responses (the API
    layer drains `RequestHandle.get_step()` and pushes each step to the
    SSE stream).
  - `run(request) -> GenerationResult` for non-streaming responses.

`PDStreamingScheduler` is that surface, backed by `PrefillWorker` +
`DecodeWorker` instead of the continuous-batching engine thread. It
runs an internal worker thread that pulls submitted requests from a
queue, runs each through the PD pipeline, decodes tokens to text, and
pushes per-token `GenerationStep`s to the handle's output queue.

What this is and is not
-----------------------

This is a single-request serial scheduler: one request runs at a time
through `Orchestrator.run_stream` on the engine thread, even though the
underlying workers expose `prefill_batch` / `decode_batch`. A multi-
request scheduler that admits concurrently and batches handoffs is a
natural follow-up; the surface here lines up exactly with what the API
server already calls, so the swap is one env-var toggle for now.

Cancellation works the same way as `ContinuousScheduler`: API thread
calls `handle.cancel()`, the engine thread notices at the next token
boundary and emits a terminal step with `finish_reason="cancelled"`.
"""

from __future__ import annotations

import logging
import queue
import threading

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.scheduler.request_state import (
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RequestState,
    RunningRequest,
)
from mini_infer.workers import DecodeWorker, Orchestrator, PrefillWorker

logger = logging.getLogger(__name__)

# Idle poll interval for the engine thread's queue check. Matches
# ContinuousScheduler's default; low enough that submitted requests
# start within ~5 ms, high enough not to burn CPU at idle.
_IDLE_SLEEP_SECONDS = 0.005


class PDStreamingScheduler:
    """`ContinuousScheduler`-shaped wrapper around a PD `Orchestrator`.

    The shape that matters for the API server is exactly:

      - `start()` -> None
      - `stop(timeout=...)` -> None
      - `submit(request: Request) -> RequestHandle`
      - `run(request: Request) -> GenerationResult`

    Internally a single engine thread pulls requests off a queue and runs
    each through the PD pipeline:

      1. `PrefillWorker.prefill(request)` -> `KVHandoff`.
      2. `DecodeWorker.decode(handoff)` yields token ids one at a time.
      3. Each token is decoded to text and pushed as a `GenerationStep`
         to the handle's output queue.
      4. On completion or cancellation, a terminal `GenerationStep` with
         a `finish_reason` is pushed; the API thread sees it and ends
         the SSE stream.

    Block-pool ownership is shared between prefill and decode (same
    underlying `ModelRunner` for now). Each PD call allocates + frees
    its own request slot, so pool state is clean between requests.
    """

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        # Both workers wrap the same runner: one model load, one cache pool.
        # The handoff is still serialized (KV is copied into the handoff via
        # `.clone()`) so the workers don't share slot state, just allocator.
        self._orchestrator = Orchestrator(
            prefill_worker=PrefillWorker(runner),
            decode_worker=DecodeWorker(runner),
        )
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the engine thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-pd-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the engine thread to exit; join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---- API surface --------------------------------------------------------

    def submit(self, request: Request) -> RequestHandle:
        """Enqueue a request for the engine thread; return a handle.

        The handle's `get_step()` blocks until the engine produces the
        next token; the final step carries a `finish_reason` indicating
        normal completion, EOS, max-tokens, or client-cancel.
        """
        # We tokenize ahead so `RunningRequest.prompt_token_ids` is
        # populated (the API layer reports `prompt_tokens` from this).
        prompt_token_ids = self._runner.tokenizer.encode(request.prompt)
        running = RunningRequest(
            request=request,
            prompt_token_ids=prompt_token_ids,
            output_queue=queue.Queue(maxsize=256),
            cancel_event=threading.Event(),
            state=RequestState.WAITING,
        )
        self._waiting.put(running)
        return RequestHandle(running)

    def run(self, request: Request) -> GenerationResult:
        """Submit + drain to completion. Convenience for non-streaming callers."""
        return self.submit(request).wait()

    # ---- engine thread ------------------------------------------------------

    def _engine_loop(self) -> None:
        """Pull requests, run each through the PD pipeline, push steps to its handle."""
        while not self._stop_event.is_set():
            try:
                running = self._waiting.get(timeout=_IDLE_SLEEP_SECONDS)
            except queue.Empty:
                continue
            try:
                self._run_one_request(running)
            except Exception as exc:
                logger.exception("PD engine error for request")
                # Best-effort: emit a terminal step so the API caller's
                # `get_step()` doesn't hang forever.
                self._push_terminal(running, text=f"engine error: {exc}", reason="cancelled")

    def _run_one_request(self, running: RunningRequest) -> None:
        """Drive one request through `Orchestrator.run_stream`, streaming tokens out."""
        running.state = RequestState.DECODING  # PD has no chunked-prefill state
        max_tokens = running.request.max_tokens
        eos_token_id = self._runner.tokenizer.eos_token_id

        token_stream = self._orchestrator.run_stream(running.request)
        finish_reason = "length"
        for emitted_idx, token_id in enumerate(token_stream):
            if running.cancel_event.is_set():
                finish_reason = "cancelled"
                break
            running.tokens_generated.append(token_id)
            text = self._runner.tokenizer.decode([token_id])
            running.last_text = (running.last_text or "") + text
            try:
                running.output_queue.put(
                    GenerationStep(text=text, finish_reason=None), timeout=10.0
                )
            except queue.Full:
                # API consumer is gone; drop the request.
                finish_reason = "cancelled"
                break
            if token_id == eos_token_id:
                finish_reason = "stop"
                break
            if emitted_idx + 1 >= max_tokens:
                finish_reason = "length"
                break

        self._push_terminal(running, text="", reason=finish_reason)

    def _push_terminal(self, running: RunningRequest, text: str, reason: str) -> None:
        """Push the final `GenerationStep` with a finish_reason; idempotent-friendly."""
        running.finish_reason = reason  # type: ignore[assignment]
        running.state = RequestState.DONE
        # `output_queue.put` with a timeout: the API consumer should be
        # draining; if they're gone we just drop the last marker.
        try:
            running.output_queue.put(
                GenerationStep(text=text, finish_reason=reason),  # type: ignore[arg-type]
                timeout=5.0,
            )
        except queue.Full:
            logger.warning("PD terminal step dropped: output_queue full")
