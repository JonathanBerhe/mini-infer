"""Continuous (ragged) batching scheduler for StateCache models (DeepSeek-V4).

True continuous batching: a running batch of up to `max_batch_size` requests,
each at its OWN position. New requests are admitted into free slots as soon as
they are available (prefilled, then merged into the batch); finished requests
free their slot immediately. Every step issues ONE ragged forward over the
running batch (`forward_decode_with_cache_ragged`), so requests of different
lengths, at different points in their own generation, share the model weights
and attention kernels.

Contrast with the other StateCache schedulers:
  - `StateCacheScheduler`: one request at a time.
  - `StateCacheCohortScheduler`: lockstep cohorts (equal-length, no mid-flight
    join / leave).
  - this one: ragged, per-request positions, dynamic admit / evict.

The decode runs over all `max_batch_size` rows of a single batched `StateCache`.
Free rows compute throwaway output the engine ignores; a later admit overwrites
the row. At steady state (a full batch) there is no waste. Per-request output
equals running that request alone through `StateCacheGenerator` (the ragged
decode is bit-parity self-consistent with the scalar path).

Same start / stop / submit / run / stream interface as the other schedulers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import queue
import threading
from collections.abc import Iterator

import torch

from mini_infer.cache.state_cache import StateCache
from mini_infer.engine.sampler import sample
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.deepseek_v4 import build_state_cache_layer_specs
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
    """A row of the batched StateCache and the request occupying it."""

    request: RunningRequest
    position: int  # global position of this request's next token to feed
    next_token: int  # the token to emit, then feed into the next forward


class StateCacheContinuousScheduler:
    """Ragged continuous batching for V4: dynamic admit / evict, one forward per step."""

    DEFAULT_OUTPUT_QUEUE_SIZE = 256
    IDLE_SLEEP_SECONDS = 0.005

    def __init__(
        self,
        generator: StateCacheGenerator,
        *,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self._generator = generator
        self._model = generator.model
        self._device = generator.device
        self._dtype = generator.dtype
        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cache: StateCache | None = None
        self._slots: list[_Slot | None] = [None] * max_batch_size

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._cache = StateCache(
            build_state_cache_layer_specs(self._model.cfg, max_seq_len=self._max_seq_len),
            batch_size=self._max_batch_size,
            device=self._device,
            dtype=self._dtype,
        )
        self._slots = [None] * self._max_batch_size
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-state-cache-continuous-engine", daemon=True
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

    # ---- engine ----

    def _engine_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._admit()
                if not any(self._slots):
                    # Idle: block briefly for an arrival rather than spin, then
                    # requeue it so the next pass admits it.
                    try:
                        running = self._waiting.get(timeout=self.IDLE_SLEEP_SECONDS)
                    except queue.Empty:
                        continue
                    self._waiting.put(running)
                    continue
                self._decode_step()
        except Exception:
            logger.exception("state-cache continuous engine thread crashed")
            raise

    def _admit(self) -> None:
        """Prefill waiting requests into free slots until full or the queue drains."""
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
        tokenizer = self._generator.tokenizer
        running.prompt_token_ids = tokenizer.encode(running.request.prompt)
        prompt_ids = running.prompt_token_ids
        if not prompt_ids:
            self._finish(running, "stop")
            return
        if len(prompt_ids) + running.request.max_tokens > self._max_seq_len:
            self._finish(running, "length")
            return

        # Prefill in a temporary single-row cache, then copy the row into our slot.
        temp = StateCache(
            build_state_cache_layer_specs(self._model.cfg, max_seq_len=self._max_seq_len),
            batch_size=1,
            device=self._device,
            dtype=self._dtype,
        )
        input_ids = torch.tensor([prompt_ids], device=self._device, dtype=torch.long)
        with torch.inference_mode():
            logits = self._model.forward_prefill_with_cache(input_ids, state_cache=temp)
            first_token = sample(logits[0, -1, :], running.request.sampling_params)
        self._copy_row(temp, src=0, dst=slot_idx)
        self._slots[slot_idx] = _Slot(
            request=running, position=len(prompt_ids), next_token=first_token
        )

    def _copy_row(self, src_cache: StateCache, *, src: int, dst: int) -> None:
        """Copy one request's full per-layer state from `src_cache[src]` into the
        batched cache row `dst` (including the CSA indexer sub-state)."""
        assert self._cache is not None
        for layer_idx in range(self._cache.num_layers):
            src_layer = src_cache.layer(layer_idx)
            dst_layer = self._cache.layer(layer_idx)
            dst_layer.swa_kv[dst] = src_layer.swa_kv[src]
            dst_layer.compressed_kv[dst] = src_layer.compressed_kv[src]
            dst_layer.cmp_kv_state[dst] = src_layer.cmp_kv_state[src]
            dst_layer.cmp_score_state[dst] = src_layer.cmp_score_state[src]
            if dst_layer.indexer is not None and src_layer.indexer is not None:
                dst_layer.indexer.compressed_kv[dst] = src_layer.indexer.compressed_kv[src]
                dst_layer.indexer.cmp_kv_state[dst] = src_layer.indexer.cmp_kv_state[src]
                dst_layer.indexer.cmp_score_state[dst] = src_layer.indexer.cmp_score_state[src]

    def _decode_step(self) -> None:
        assert self._cache is not None
        tokenizer = self._generator.tokenizer
        eos = tokenizer.eos_token_id

        # 1. Emit each active slot's current token; finish + free the slots that stop.
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
                self._finish(request, "stop")  # EOS is not emitted
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

        # 2. One ragged forward over all rows (free rows compute ignored output).
        input_tokens = torch.zeros(self._max_batch_size, 1, dtype=torch.long)
        positions = torch.zeros(self._max_batch_size, dtype=torch.long)
        for slot_idx, slot in enumerate(self._slots):
            if slot is not None:
                input_tokens[slot_idx, 0] = slot.next_token
                positions[slot_idx] = slot.position
        with torch.inference_mode():
            logits = self._model.forward_decode_with_cache_ragged(
                input_tokens.to(self._device),
                positions=positions.to(self._device),
                state_cache=self._cache,
            )

        # 3. Sample each still-active slot's next token; advance its position.
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
