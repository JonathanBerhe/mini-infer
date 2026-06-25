"""Tensor-parallel ragged continuous-batching scheduler for StateCache models.

The HTTP-facing leader (rank 0) counterpart of `StateCacheContinuousScheduler`:
same dynamic admit / evict engine, but its prefill and decode forwards go through
a `TensorParallelStateCacheContinuousServer`, which broadcasts each forward so the
follower ranks mirror it. This lets a model too big for one GPU (V4-Flash) be
served with continuous batching over `/v1/completions` across ranks.

Division of labour: the server owns the per-rank (replicated) batched StateCache
and the broadcast; this scheduler owns the request -> slot mapping, per-slot
position + next token, the waiting queue, and streaming. Runs on the leader only;
follower ranks call `TensorParallelStateCacheContinuousServer.run_follower_loop()`.

`stop()` broadcasts a shutdown after the engine thread exits, so the followers
leave their loop instead of blocking forever on the next broadcast.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import queue
import threading
from collections.abc import Iterator

from mini_infer.engine.sampler import sample
from mini_infer.engine.tp_state_cache_continuous_server import (
    TensorParallelStateCacheContinuousServer,
)
from mini_infer.scheduler.request_state import (
    FinishReason,
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RunningRequest,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Slot:
    """A row of the server's batched cache and the request occupying it."""

    request: RunningRequest
    position: int
    next_token: int


class TensorParallelStateCacheContinuousScheduler:
    """Leader-side continuous-batching scheduler over a TP StateCache server."""

    DEFAULT_OUTPUT_QUEUE_SIZE = 256
    IDLE_SLEEP_SECONDS = 0.005

    def __init__(self, server: TensorParallelStateCacheContinuousServer) -> None:
        if not server.is_leader:
            raise RuntimeError(
                "TensorParallelStateCacheContinuousScheduler runs on the leader (rank 0) "
                "only; followers call server.run_follower_loop()"
            )
        self._server = server
        self._max_batch_size = server.max_batch_size
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._slots: list[_Slot | None] = [None] * server.max_batch_size

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._slots = [None] * self._max_batch_size
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-tp-continuous-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # The engine thread has exited, so no broadcast is in flight; tell the
        # followers to leave run_follower_loop (else they block on the next op).
        self._server.shutdown()

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

    # ---- engine (leader) ----

    def _engine_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._admit()
                if not any(self._slots):
                    try:
                        running = self._waiting.get(timeout=self.IDLE_SLEEP_SECONDS)
                    except queue.Empty:
                        continue
                    self._waiting.put(running)
                    continue
                self._decode_step()
        except Exception:
            logger.exception("tp continuous engine thread crashed")
            raise

    def _admit(self) -> None:
        for slot_idx in range(self._max_batch_size):
            if self._slots[slot_idx] is not None:
                continue
            try:
                running = self._waiting.get_nowait()
            except queue.Empty:
                return
            if running.cancel_event.is_set():
                self._finish(running, "cancelled")
                continue
            self._prefill_into_slot(running, slot_idx)

    def _prefill_into_slot(self, running: RunningRequest, slot_idx: int) -> None:
        tokenizer = self._server.tokenizer
        running.prompt_token_ids = tokenizer.encode(running.request.prompt)
        prompt_ids = running.prompt_token_ids
        if not prompt_ids:
            self._finish(running, "stop")
            return
        if len(prompt_ids) + running.request.max_tokens > self._server.max_seq_len:
            self._finish(running, "length")
            return
        prefill_logits = self._server.prefill_into_slot(prompt_ids, slot_idx)
        first_token = sample(prefill_logits, running.request.sampling_params)
        self._slots[slot_idx] = _Slot(
            request=running, position=len(prompt_ids), next_token=first_token
        )

    def _decode_step(self) -> None:
        tokenizer = self._server.tokenizer
        eos = tokenizer.eos_token_id

        decode_slots: list[int] = []
        for slot_idx, slot in enumerate(self._slots):
            if slot is None:
                continue
            request = slot.request
            if request.cancel_event.is_set():
                self._finish(request, "cancelled")
                self._slots[slot_idx] = None
                continue
            if eos is not None and slot.next_token == eos:
                self._finish(request, "stop")
                self._slots[slot_idx] = None
                continue
            request.tokens_generated.append(slot.next_token)
            current_text = tokenizer.decode(request.tokens_generated)
            delta = current_text[len(request.last_text) :]
            request.last_text = current_text
            self._emit(request, GenerationStep(text=delta))
            if len(request.tokens_generated) >= request.request.max_tokens:
                self._finish(request, "length")
                self._slots[slot_idx] = None
                continue
            decode_slots.append(slot_idx)

        if not decode_slots:
            return

        input_tokens = [0] * self._max_batch_size
        positions = [0] * self._max_batch_size
        for slot_idx, slot in enumerate(self._slots):
            if slot is not None:
                input_tokens[slot_idx] = slot.next_token
                positions[slot_idx] = slot.position
        logits = self._server.decode_batch(input_tokens, positions)
        for slot_idx in decode_slots:
            slot = self._slots[slot_idx]
            assert slot is not None
            slot.next_token = sample(logits[slot_idx, -1, :], slot.request.request.sampling_params)
            slot.position += 1

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
