"""Multi-request scheduler over the PD pipeline.

`PDScheduler` is the continuous-batching scheduler for the disaggregated
pipeline. Same `start / stop / submit / run` surface as
`ContinuousScheduler` and `PDStreamingScheduler`, but internally drives
multiple concurrent requests through:

  - **Admission queue** for waiting requests.
  - **Batched prefill** via `PrefillWorker.prefill_batch` (one forward
    over all newly-admitted prompts).
  - **Batched decode** via a long-lived `DecodeSession` (one forward
    per step over every in-flight decoder).
  - **Per-request termination tracking** (EOS, `max_tokens`, cancel).
  - **Streaming output** to each handle's per-request queue.

Two threading modes, selected by the `mode` constructor parameter:

- **`mode="serial"`** (default): one engine thread runs both phases
  in sequence — admit + batched prefill + batched decode step + reap.
  Simple correctness story; bit-equivalent test surface. Prefill and
  decode share the same thread, so on a 2-GPU host they don't
  overlap (the prefill GPU idles while decode runs and vice versa).
- **`mode="parallel"`**: two engine threads — a prefill thread admits
  + runs `prefill_batch`, a decode thread reads handoffs and drives
  the session. The threads are connected by a bounded handoff queue;
  the queue's `maxsize=max_concurrent` provides backpressure. On a
  2-GPU host the phases overlap: the prefill GPU produces while the
  decode GPU consumes.

Greedy output is identical between modes (the threading shape only
affects timing). The two-thread mode's value shows up on real
multi-GPU hardware where the two phases run on different devices.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Literal

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.scheduler.request_state import (
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RequestState,
    RunningRequest,
)
from mini_infer.workers.decode_worker import DecodeSession, DecodeWorker
from mini_infer.workers.kv_handoff import KVHandoff
from mini_infer.workers.prefill_worker import PrefillWorker

logger = logging.getLogger(__name__)

# Idle poll interval for the engine thread. Matches `ContinuousScheduler`.
# Low enough that newly-submitted requests start within ~5 ms; high enough
# not to burn CPU at idle.
_IDLE_SLEEP_SECONDS = 0.005


class PDScheduler:
    """`ContinuousScheduler`-shaped multi-request scheduler over the PD pipeline.

    The shape that matters for the API server is exactly:

      - `start()` -> None
      - `stop(timeout=...)` -> None
      - `submit(request) -> RequestHandle`
      - `run(request) -> GenerationResult`

    Internally an engine thread runs a mixed loop that batches prefill
    and decode across concurrent requests. Block-pool ownership is
    shared between prefill and decode (same underlying `ModelRunner`
    for now); the prefill side allocates + frees its own slots inside
    `prefill_batch`, and the decode side owns its long-lived
    `DecodeSession`.
    """

    DEFAULT_MAX_CONCURRENT = 16

    def __init__(
        self,
        runner: ModelRunner,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        mode: Literal["serial", "parallel"] = "serial",
    ) -> None:
        self._runner = runner
        self._prefill_worker = PrefillWorker(runner)
        self._decode_worker = DecodeWorker(runner)
        self._max_concurrent = max_concurrent
        self._mode = mode

        # API-side state (mutated from API thread + engine threads).
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        # Per-mode threads. Serial uses `_thread` only; parallel uses both.
        self._thread: threading.Thread | None = None
        self._decode_thread: threading.Thread | None = None

        # Engine-thread-only state. In serial mode the single engine thread
        # owns everything below. In parallel mode the prefill thread owns
        # `_prefilling`; the decode thread owns `_decoding`, `_handoffs`,
        # and `_session`. The two threads communicate exclusively via
        # `_waiting` and `_handoff_queue`, both thread-safe `queue.Queue`s.
        #
        # - `_prefilling`: requests that have been admitted but haven't
        #   been through `prefill_batch` yet. Drained each prefill cycle.
        # - `_decoding`: slot_id -> RunningRequest for every in-flight
        #   decode slot. Updated as `session.add_handoff` / `remove_slot`.
        # - `_handoffs`: slot_id -> KVHandoff so the engine knows the
        #   sampling_params / eos_token_id / max_tokens for each slot.
        # - `_session`: the long-lived `DecodeSession`. Lazy-init on first
        #   handoff. Stays alive until shutdown.
        # - `_handoff_queue` (parallel mode only): prefill thread puts
        #   `(running_request, handoff)` tuples; decode thread gets them.
        #   `maxsize = max_concurrent` provides backpressure: when the
        #   decode pool fills up, the prefill thread blocks on `put` and
        #   stops admitting from `_waiting`.
        self._prefilling: list[RunningRequest] = []
        self._decoding: dict[int, RunningRequest] = {}
        self._handoffs: dict[int, KVHandoff] = {}
        self._session: DecodeSession | None = None
        self._handoff_queue: queue.Queue[tuple[RunningRequest, KVHandoff]] | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the engine thread(s). Idempotent.

        Serial mode: one thread (`_thread`) running `_engine_loop`.
        Parallel mode: two threads (`_thread` running the prefill loop,
        `_decode_thread` running the decode loop), connected by a
        bounded `_handoff_queue`.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        if self._mode == "serial":
            self._thread = threading.Thread(
                target=self._engine_loop, name="mini-infer-pd-scheduler", daemon=True
            )
            self._thread.start()
        elif self._mode == "parallel":
            # Lazy-init the handoff queue at start, NOT at __init__: a
            # fresh queue every start/stop cycle avoids stale state.
            self._handoff_queue = queue.Queue(maxsize=self._max_concurrent)
            self._thread = threading.Thread(
                target=self._prefill_thread_loop,
                name="mini-infer-pd-prefill",
                daemon=True,
            )
            self._decode_thread = threading.Thread(
                target=self._decode_thread_loop,
                name="mini-infer-pd-decode",
                daemon=True,
            )
            self._thread.start()
            self._decode_thread.start()
        else:
            raise ValueError(f"unknown mode {self._mode!r}; expected 'serial' or 'parallel'")

    def stop(self, timeout: float = 10.0) -> None:
        """Signal engine thread(s) to exit; join; drain remaining state.

        Any in-flight slots are released so the block pool returns to
        fully-free state. Final terminal steps with
        `finish_reason="cancelled"` are pushed to every still-active
        request so API consumers don't hang on `get_step()`.
        """
        self._stop_event.set()
        # Wake up any thread blocked on a queue.get(timeout=...). The poll
        # loops in the engine threads will see the stop event and return.
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._decode_thread is not None:
            self._decode_thread.join(timeout=timeout)
            self._decode_thread = None

        # Drain remaining state in dependency order: free decode slots
        # first (so the block pool releases), then emit terminal steps
        # for any still-active handles.
        if self._session is not None:
            for slot_id in list(self._decoding.keys()):
                if self._session.is_active(slot_id):
                    self._session.remove_slot(slot_id)
        for r in list(self._decoding.values()):
            self._emit_terminal(r, reason="cancelled")
        self._decoding.clear()
        self._handoffs.clear()
        # In parallel mode the handoff queue may hold un-decoded handoffs;
        # they were prefilled but never reached the decode session. Their
        # requests still need a terminal step.
        if self._handoff_queue is not None:
            while True:
                try:
                    r, _handoff = self._handoff_queue.get_nowait()
                except queue.Empty:
                    break
                self._emit_terminal(r, reason="cancelled")
        for r in self._prefilling:
            self._emit_terminal(r, reason="cancelled")
        self._prefilling.clear()
        self._session = None
        self._handoff_queue = None

    # ── API surface ──────────────────────────────────────────────────

    def submit(self, request: Request) -> RequestHandle:
        """Enqueue a request for the engine thread; return a handle.

        The handle's `get_step()` blocks until the engine produces the
        next token; the final step carries a `finish_reason` indicating
        normal completion, EOS, max-tokens, or client-cancel.
        """
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

    # ── engine thread ────────────────────────────────────────────────

    def _engine_loop(self) -> None:
        """Forever loop: admit + batched prefill + batched decode step + reap."""
        while not self._stop_event.is_set():
            self._admit_waiting()
            self._run_prefill_batch()
            self._run_decode_step()
            self._reap_cancelled()

            idle = not self._prefilling and not self._decoding and self._waiting.empty()
            # When idle, park the thread briefly so we don't burn CPU.
            # The wait returns True if the stop event fired during the wait.
            if idle and self._stop_event.wait(_IDLE_SLEEP_SECONDS):
                return

    def _admit_waiting(self) -> None:
        """Pull from waiting queue into PREFILLING state, up to max_concurrent."""
        in_flight = len(self._prefilling) + len(self._decoding)
        while in_flight < self._max_concurrent:
            try:
                running = self._waiting.get_nowait()
            except queue.Empty:
                return
            if running.cancel_event.is_set():
                # Cancelled before we even admitted; emit terminal + skip.
                self._emit_terminal(running, reason="cancelled")
                continue
            running.state = RequestState.PREFILLING
            self._prefilling.append(running)
            in_flight += 1

    def _run_prefill_batch(self) -> None:
        """Run one `prefill_batch` over all currently-prefilling requests.

        Each handoff is added to the decode session immediately so the
        first token (which came from prefill, not from a decode forward)
        is emitted in the same iteration. The next decode step then
        produces token #2 onward for every newly-admitted request.
        """
        if not self._prefilling:
            return
        prefilling = self._prefilling
        self._prefilling = []
        requests = [r.request for r in prefilling]
        try:
            handoffs = self._prefill_worker.prefill_batch(requests)
        except Exception as exc:
            logger.exception("PDScheduler: prefill_batch failed; terminating batch")
            for r in prefilling:
                self._emit_terminal(r, reason="cancelled", error=str(exc))
            return

        if self._session is None:
            self._session = self._decode_worker.start_session()

        for r, handoff in zip(prefilling, handoffs, strict=True):
            slot_id = self._session.add_handoff(handoff)
            r.state = RequestState.DECODING
            self._decoding[slot_id] = r
            self._handoffs[slot_id] = handoff
            # First token comes from the handoff (not from a decode forward).
            first_token = handoff.first_sampled_token_id
            self._emit_token(r, first_token)
            self._check_termination(slot_id, r, first_token)

    def _run_decode_step(self) -> None:
        """One batched decode forward over the session; emit per-slot tokens.

        If the session is empty (no active slots), this is a no-op.
        Cancellations between this iteration and the next are picked up
        by `_reap_cancelled`.
        """
        if self._session is None or self._session.num_active_slots == 0:
            return
        slot_to_token = self._session.step()
        for slot_id, token in slot_to_token.items():
            r = self._decoding.get(slot_id)
            if r is None or r.state == RequestState.DONE:
                # Slot was reaped during this same step. Shouldn't normally
                # happen (we reap after the step), but defensive guard.
                continue
            if r.cancel_event.is_set():
                # Will be reaped in `_reap_cancelled`; skip emission.
                continue
            self._emit_token(r, token)
            self._check_termination(slot_id, r, token)

    def _reap_cancelled(self) -> None:
        """Remove slots whose `cancel_event` fired since the last step.

        Called once per iteration after the step. Termination from EOS /
        max_tokens is handled inline in `_check_termination` (which calls
        `_terminate_slot` directly); cancellation is asynchronous (the
        API thread sets the event) so it lands here.
        """
        cancelled_slots = [
            slot_id
            for slot_id, r in self._decoding.items()
            if r.cancel_event.is_set() and r.state != RequestState.DONE
        ]
        for slot_id in cancelled_slots:
            self._terminate_slot(slot_id, self._decoding[slot_id], reason="cancelled")

    # ── parallel mode: two threads ───────────────────────────────────

    def _prefill_thread_loop(self) -> None:
        """Prefill thread main loop (parallel mode).

        Pull from the waiting queue, batch up to `max_concurrent`
        requests at a time, run `prefill_batch`, push
        `(running_request, handoff)` tuples to the handoff queue. The
        handoff queue is bounded (`maxsize = max_concurrent`); a full
        queue blocks the put, which naturally backpressures admission
        when the decode side can't keep up.

        Cancellation: requests with `cancel_event` set are skipped
        (terminal emitted) without going through prefill.

        Shutdown: when `_stop_event` is set, the loop exits at its
        next iteration boundary. Any partially-claimed prefill batch
        in `_prefilling` is cleaned up by `stop()`.
        """
        assert self._handoff_queue is not None
        while not self._stop_event.is_set():
            self._admit_waiting()
            if not self._prefilling:
                if self._stop_event.wait(_IDLE_SLEEP_SECONDS):
                    return
                continue
            prefilling = self._prefilling
            self._prefilling = []
            requests = [r.request for r in prefilling]
            try:
                handoffs = self._prefill_worker.prefill_batch(requests)
            except Exception as exc:
                logger.exception("PDScheduler (parallel): prefill_batch failed; terminating batch")
                for r in prefilling:
                    self._emit_terminal(r, reason="cancelled", error=str(exc))
                continue
            for r, handoff in zip(prefilling, handoffs, strict=True):
                if r.cancel_event.is_set():
                    # Cancelled between admit and prefill completion; emit
                    # terminal without enqueuing for decode.
                    self._emit_terminal(r, reason="cancelled")
                    continue
                # Block on put if the handoff queue is full — this is the
                # backpressure mechanism that bounds total in-flight to
                # roughly `2 * max_concurrent` worst case (in-prefill +
                # in-decode). Poll the stop event so we can exit cleanly.
                while not self._stop_event.is_set():
                    try:
                        self._handoff_queue.put((r, handoff), timeout=_IDLE_SLEEP_SECONDS)
                        break
                    except queue.Full:
                        continue
                else:
                    # Stop requested while the put was blocked; terminal
                    # for this handoff (the decode thread won't see it).
                    self._emit_terminal(r, reason="cancelled")

    def _decode_thread_loop(self) -> None:
        """Decode thread main loop (parallel mode).

        Pull handoffs from the handoff queue and add them to the
        long-lived `DecodeSession`. Each iteration: drain any
        immediately-available handoffs (without blocking), run one
        batched decode step, emit per-slot tokens, reap terminations
        (EOS / max_tokens / cancel).

        Shutdown: when `_stop_event` is set, the loop exits at its
        next iteration boundary. Slots are freed by `stop()`.
        """
        assert self._handoff_queue is not None
        self._session = self._decode_worker.start_session()
        while not self._stop_event.is_set():
            # Drain new handoffs without blocking. Adding handoffs to the
            # session emits the first token (from the handoff itself) and
            # may immediately terminate slots that have `max_tokens == 1`.
            self._drain_new_handoffs()

            if self._session.num_active_slots > 0:
                self._run_decode_step()
                self._reap_cancelled()
            else:
                # No active slots; park briefly while we wait for handoffs.
                # Use the handoff queue's blocking get with a timeout so we
                # don't busy-spin and don't miss handoff arrivals.
                try:
                    item = self._handoff_queue.get(timeout=_IDLE_SLEEP_SECONDS)
                except queue.Empty:
                    continue
                self._absorb_handoff(item)

    def _drain_new_handoffs(self) -> None:
        """Move every immediately-available handoff into the session.

        Non-blocking. Called once per decode-thread iteration so newly-
        prefilled requests join the next decode forward.
        """
        assert self._handoff_queue is not None
        while True:
            try:
                item = self._handoff_queue.get_nowait()
            except queue.Empty:
                return
            self._absorb_handoff(item)

    def _absorb_handoff(self, item: tuple[RunningRequest, KVHandoff]) -> None:
        """Add a (request, handoff) pair to the decode session.

        Emits the first token (which came from the prefill worker, not
        from a decode forward) and checks termination immediately. If
        the request was cancelled between prefill and absorb, emit a
        terminal step instead of adding to the session.
        """
        r, handoff = item
        if r.cancel_event.is_set():
            self._emit_terminal(r, reason="cancelled")
            return
        assert self._session is not None
        slot_id = self._session.add_handoff(handoff)
        r.state = RequestState.DECODING
        self._decoding[slot_id] = r
        self._handoffs[slot_id] = handoff
        first_token = handoff.first_sampled_token_id
        self._emit_token(r, first_token)
        self._check_termination(slot_id, r, first_token)

    # ── per-request bookkeeping ──────────────────────────────────────

    def _emit_token(self, r: RunningRequest, token: int) -> None:
        """Append the token to the request's history and push a `GenerationStep`."""
        r.tokens_generated.append(token)
        text = self._runner.tokenizer.decode([token])
        r.last_text = (r.last_text or "") + text
        try:
            r.output_queue.put(
                GenerationStep(text=text, finish_reason=None),
                timeout=10.0,
            )
        except queue.Full:
            # Consumer is gone; cancel and let `_reap_cancelled` clean up.
            r.cancel_event.set()

    def _check_termination(self, slot_id: int, r: RunningRequest, last_token: int) -> None:
        """Reap the slot if it hit `max_tokens` or emitted EOS."""
        handoff = self._handoffs[slot_id]
        if len(r.tokens_generated) >= handoff.max_tokens:
            self._terminate_slot(slot_id, r, reason="length")
            return
        if handoff.eos_token_id is not None and last_token == handoff.eos_token_id:
            self._terminate_slot(slot_id, r, reason="stop")
            return

    def _terminate_slot(self, slot_id: int, r: RunningRequest, reason: str) -> None:
        """Free the session slot, drop bookkeeping, emit the terminal step."""
        if self._session is not None and self._session.is_active(slot_id):
            self._session.remove_slot(slot_id)
        self._decoding.pop(slot_id, None)
        self._handoffs.pop(slot_id, None)
        self._emit_terminal(r, reason=reason)

    def _emit_terminal(self, r: RunningRequest, reason: str, *, error: str | None = None) -> None:
        """Push the final `GenerationStep` with a `finish_reason`."""
        if r.state == RequestState.DONE:
            # Already terminated; idempotent.
            return
        text = f"engine error: {error}" if error else ""
        r.state = RequestState.DONE
        r.finish_reason = reason  # type: ignore[assignment]
        try:
            r.output_queue.put(
                GenerationStep(text=text, finish_reason=reason),  # type: ignore[arg-type]
                timeout=5.0,
            )
        except queue.Full:
            logger.warning("PDScheduler terminal step dropped: output_queue full")
