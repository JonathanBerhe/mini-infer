"""Request-related types: input Request, output Result/Step, internal scheduler state."""

import asyncio
import contextlib
import dataclasses
import enum
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Literal

import torch

from mini_infer.engine.sampler import SamplingParams
from mini_infer.exceptions import EngineNotRunningError, RequestFailedError

if TYPE_CHECKING:
    from mini_infer.engine.dspark.spec import SpecDecodeState, SpecSummary

# `cancelled` covers both API-side cancellation (client disconnect on a
# streaming request) and engine-side rejection (admission can't be satisfied
# by the current pool). Both paths emit a final GenerationStep with this
# finish_reason so consumers see a clean terminal step rather than a
# never-completing stream.
FinishReason = Literal["stop", "length", "cancelled"]


@dataclasses.dataclass(frozen=True)
class Request:
    prompt: str
    sampling_params: SamplingParams
    max_tokens: int


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    """The aggregated outcome of one request.

    `spec_summary` is a fixed-size summary of the DSpark rounds, or None when the
    request ran the plain path (speculation off, or downgraded at admission).
    """

    text: str
    tokens: list[int]
    finish_reason: FinishReason
    prompt_tokens: int
    spec_summary: "SpecSummary | None" = None


@dataclasses.dataclass(frozen=True)
class GenerationStep:
    """One step of streaming generation; finish_reason set on the final step only.

    `num_tokens` is the number of generated tokens in the step, 0 on a terminal
    step. A speculative verify step carries its whole committed block as one
    step, so a consumer must sum this field, not count steps. `ContinuousScheduler`
    and the StateCache schedulers do not emit EOS. `PDScheduler` emits it as a
    one-token step before its terminal step.

    `failed` marks the terminal step of a request that the engine failed. It
    carries `finish_reason="cancelled"`, but a `RequestHandle` raises
    `RequestFailedError` instead of returning it. See ADR-030.
    """

    text: str
    finish_reason: FinishReason | None = None
    num_tokens: int = 0
    failed: bool = False


class StepQueue(queue.Queue[GenerationStep]):
    """The output queue of one request: a `queue.Queue` that can wake an asyncio reader.

    Each put calls the listener on the engine thread, outside the queue lock, so the
    listener must return at once. A `RuntimeError` from it (the reader's loop closed)
    is ignored. No wakeup is lost: a put reads the listener after the step is in the
    queue, and a reader sets the listener before it reads the queue. See ADR-029.
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self._listener: Callable[[], object] | None = None

    def set_listener(self, listener: Callable[[], object] | None) -> None:
        with self.mutex:
            self._listener = listener

    def put(self, item: GenerationStep, block: bool = True, timeout: float | None = None) -> None:
        # `put_nowait` calls this too, so both put paths call the listener.
        super().put(item, block, timeout)
        listener = self._listener
        if listener is not None:
            with contextlib.suppress(RuntimeError):
                listener()


class RequestState(enum.StrEnum):
    """Engine-side state machine for a request in flight.

    Lifecycle:
        WAITING -> PREFILLING (admitted, slot allocated, no tokens processed)
                -> CHUNKED_PREFILLING (one or more chunks dispatched, more remain)
                -> DECODING (entire prompt processed, last_logits ready)
                -> DONE
    PREFILLING is a transient one-step state set on admission; the actual prefill
    work happens in CHUNKED_PREFILLING, possibly across many engine steps. A
    request leaves CHUNKED_PREFILLING when `tokens_prefilled == len(prompt_token_ids)`.
    """

    WAITING = "waiting"
    PREFILLING = "prefilling"
    CHUNKED_PREFILLING = "chunked_prefilling"
    DECODING = "decoding"
    DONE = "done"


@dataclasses.dataclass
class RunningRequest:
    """Engine-side state for a request currently in the scheduler.

    Mutated only by the engine thread. The output_queue + completion via the
    final GenerationStep gives the API-side handle a happens-before edge to
    read the final tokens_generated / finish_reason fields safely.

    `batch_idx` is the request's slot in the scheduler's shared batched
    `PagedKVCache`. None until prefill is merged in; shifts down by one each
    time an earlier-slot request finishes and is removed from the cache.

    `cancel_event` is set by the API thread (`RequestHandle.cancel`) when a
    client disconnects on a streaming request. The engine checks it at safe
    points (before sampling each decoder, before admitting from the queue)
    and finishes the request with finish_reason="cancelled" when set.

    `pending_decode_tokens` holds tokens that the engine sampled and emitted but
    did not yet feed through a forward. A retry after a mid-step OOM reuses them,
    so it does not sample `last_logits` again and emit a duplicate token. The
    forward that consumes them clears the field.

    `spec` is the DSpark state of a request chosen for speculation at admission, else
    None. One field keeps DSpark members out of this type, which every scheduler shares.
    `_reap_done` clears it at finish, because the API handle keeps the `RunningRequest`
    alive. `spec_summary` survives the clear: `_finish` writes it before the terminal
    step, so the API thread can read it after the stream ends.
    """

    request: Request
    output_queue: queue.Queue[GenerationStep]
    cancel_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    state: RequestState = RequestState.WAITING
    batch_idx: int | None = None
    prompt_token_ids: list[int] = dataclasses.field(default_factory=list)
    tokens_prefilled: int = 0
    tokens_generated: list[int] = dataclasses.field(default_factory=list)
    last_logits: torch.Tensor | None = None
    last_text: str = ""
    pending_decode_tokens: list[int] | None = None
    spec: "SpecDecodeState | None" = None
    spec_summary: "SpecSummary | None" = None
    finish_reason: FinishReason | None = None


class RequestHandle:
    """API-side handle for a submitted request; drained from the API thread.

    `engine_alive` reports whether a step can still arrive. Each scheduler binds it
    at `submit()` to the engine threads that run then, because a thread from a later
    `start()` can lack the request. It returns True while `stop()` runs, because
    `stop()` sends the 'cancelled' steps after the thread exits. See ADR-028.

    The handle calls it only after the queue stays empty for one poll slice, so buffered
    steps come first. An exited thread makes the handle raise `EngineNotRunningError`,
    but a stuck thread looks alive (for example a tensor-parallel leader that waits on a
    dead follower). `engine_alive=None` turns the check off, for tests.
    """

    # Read at call time, not bound as a default argument, so a test can lower it.
    LIVENESS_POLL_SECONDS = 0.5

    def __init__(
        self,
        running_request: RunningRequest,
        *,
        engine_alive: Callable[[], bool] | None = None,
    ) -> None:
        self._req = running_request
        self._engine_alive = engine_alive

    def steps(self) -> Iterator[GenerationStep]:
        """Yield GenerationStep as the engine produces them; ends after the final step.

        Raises `EngineNotRunningError` if the engine exits before the terminal step.
        """
        while True:
            step = self._next_step(deadline=None)
            yield step
            if step.finish_reason is not None:
                return

    def wait(self, timeout: float | None = None) -> GenerationResult:
        """Block until complete; return the aggregated result.

        If `timeout` (seconds) runs out on the whole wait, raises `TimeoutError`. A dead
        engine raises `EngineNotRunningError` within one poll slice, timeout or not.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        text_parts: list[str] = []
        while True:
            step = self._next_step(deadline)
            if step.finish_reason is not None:
                return self.build_result("".join(text_parts), step.finish_reason)
            text_parts.append(step.text)

    def build_result(self, text: str, finish_reason: FinishReason) -> GenerationResult:
        """Build the result of a finished request from the step text the caller read.

        Call only after the terminal step arrives: its queue put makes the
        engine-side fields safe to read from this thread.
        """
        return GenerationResult(
            text=text,
            tokens=list(self._req.tokens_generated),
            finish_reason=finish_reason,
            prompt_tokens=len(self._req.prompt_token_ids),
            spec_summary=self._req.spec_summary,
        )

    def cancel(self) -> None:
        """Signal the engine to stop generating for this request.

        Idempotent and safe to call from any thread. The engine notices the
        flag at safe points (between sample/forward iterations) and emits a
        terminal `GenerationStep(finish_reason="cancelled")` to drain
        consumers. Resources (KV blocks, batch slot) are reclaimed on the
        next `_reap_done` pass.
        """
        self._req.cancel_event.set()

    def get_step(self, timeout: float | None = None) -> GenerationStep:
        """Block until the next GenerationStep arrives. Thread-safe.

        Raises `TimeoutError` on timeout and `EngineNotRunningError` if the engine is dead.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        return self._next_step(deadline)

    async def get_step_async(self) -> GenerationStep:
        """`get_step()` for a caller on an asyncio event loop, without a thread.

        The `StepQueue` wakes the loop on each put. `asyncio.to_thread` holds a pool thread
        per waiting request, so requests without a thread can drop steps. See ADR-029.
        """
        out = self._req.output_queue
        if not isinstance(out, StepQueue):
            raise TypeError("get_step_async() needs a StepQueue as the output queue")
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()
        # The listener runs on the engine thread, and `asyncio.Event` is not thread-safe.
        out.set_listener(lambda: loop.call_soon_threadsafe(ready.set))
        try:
            while True:
                # Clear before the read, so that a put after the read sets it again.
                ready.clear()
                try:
                    return self._take(out.get_nowait())
                except queue.Empty:
                    pass
                try:
                    async with asyncio.timeout(self.LIVENESS_POLL_SECONDS):
                        await ready.wait()
                    continue
                except TimeoutError:
                    pass
                step = self._after_empty_poll()
                if step is not None:
                    return step
        finally:
            out.set_listener(None)

    def _next_step(self, deadline: float | None) -> GenerationStep:
        """Return the next step, polling engine liveness while the queue is empty.

        The queue comes first, so buffered steps reach the caller before an error.
        Liveness comes before the deadline, so a dead engine raises `EngineNotRunningError`.
        """
        while True:
            timeout: float | None = None
            if self._engine_alive is not None:
                timeout = self.LIVENESS_POLL_SECONDS
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                timeout = remaining if timeout is None else min(timeout, remaining)
            try:
                return self._take(self._req.output_queue.get(timeout=timeout))
            except queue.Empty:
                pass
            step = self._after_empty_poll()
            if step is not None:
                return step
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("request did not finish within the timeout")

    def _after_empty_poll(self) -> GenerationStep | None:
        """Handle an empty poll slice: return None while the engine can still send a step.

        When `engine_alive` returns False, read the queue once more. The thread can put
        its last step between the empty read and the liveness read. An exited thread
        made all its puts visible, so an empty second read means no step will come.
        """
        if self._engine_alive is None or self._engine_alive():
            return None
        try:
            return self._take(self._req.output_queue.get_nowait())
        except queue.Empty:
            # A restarted scheduler or a live PD prefill thread can still hold the
            # request. Cancel it, so no engine works for output that nobody reads.
            self.cancel()
            raise EngineNotRunningError(
                "engine not running: the engine thread exited before this request "
                "finished (see the log)"
            ) from None

    @staticmethod
    def _take(step: GenerationStep) -> GenerationStep:
        """Return the step, or raise `RequestFailedError` if it is `failed`.

        Every queue read calls this, so no caller sees a failure as a cancellation.
        """
        if step.failed:
            raise RequestFailedError(
                "the engine failed while processing this request (see the log)"
            )
        return step
