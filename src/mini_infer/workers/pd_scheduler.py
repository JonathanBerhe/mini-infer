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

Single engine thread drives the loop. Each iteration:

  1. Admit waiting requests up to `max_concurrent` total in-flight.
  2. If any requests are in PREFILLING state, run one `prefill_batch`
     over them. The resulting handoffs go to a `DecodeSession`; each
     request transitions to DECODING and emits its first token (which
     came from the handoff, not from a decode forward).
  3. If the decode session has any active slots, run one batched
     `session.step()`. Per slot: emit the new token, check termination,
     reap if terminated.

Same threading shape as `ContinuousScheduler`: one engine thread,
queue-based admission, API thread submits + drains handles. The
cross-phase coordination (PD's "different GPU for different phase")
is preserved because the underlying `PrefillWorker` and `DecodeWorker`
each carry their own paged cache; the scheduler just orchestrates the
calls.

Slice 1 (this file's first commit) is the single-thread variant. The
two-thread variant (prefill + decode on separate threads with a
handoff queue between them) is a follow-up that gives real cross-GPU
overlap. See `docs/plans/pd-scheduler.md`.
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
    ) -> None:
        self._runner = runner
        self._prefill_worker = PrefillWorker(runner)
        self._decode_worker = DecodeWorker(runner)
        self._max_concurrent = max_concurrent

        # API-side state (mutated from API thread + engine thread).
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Engine-thread-only state (no locks needed):
        # - `_prefilling`: requests that have been admitted but haven't
        #   been through `prefill_batch` yet. Drained each step.
        # - `_decoding`: slot_id -> RunningRequest for every in-flight
        #   decode slot. Updated as `session.add_handoff` / `remove_slot`.
        # - `_handoffs`: slot_id -> KVHandoff so the engine knows the
        #   sampling_params / eos_token_id / max_tokens for each slot.
        #   (RunningRequest carries the original Request; the handoff has
        #   the post-prefill bits.)
        # - `_session`: the long-lived `DecodeSession`. Lazy-init on first
        #   prefill batch. Stays alive across batches; empty when no slots.
        self._prefilling: list[RunningRequest] = []
        self._decoding: dict[int, RunningRequest] = {}
        self._handoffs: dict[int, KVHandoff] = {}
        self._session: DecodeSession | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the engine thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-pd-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the engine thread to exit; join.

        Any in-flight slots are released so the block pool returns to
        fully-free state. Already-emitted tokens stay in their per-request
        output queues; final terminal steps with `finish_reason="cancelled"`
        are pushed so API consumers don't hang on `get_step()`.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Drain any remaining state. Slots first (so the block pool is
        # released), then push terminal steps to the still-waiting handles.
        if self._session is not None:
            for slot_id in list(self._decoding.keys()):
                if self._session.is_active(slot_id):
                    self._session.remove_slot(slot_id)
        for r in list(self._decoding.values()):
            self._emit_terminal(r, reason="cancelled")
        self._decoding.clear()
        self._handoffs.clear()
        for r in self._prefilling:
            self._emit_terminal(r, reason="cancelled")
        self._prefilling.clear()
        self._session = None

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
