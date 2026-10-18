"""Request-related types: input Request, output Result/Step, internal scheduler state."""

import dataclasses
import enum
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Literal

import torch

from mini_infer.engine.sampler import SamplingParams
from mini_infer.exceptions import EngineNotRunningError

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

    `spec_summary` is set only when the request speculated: it is the fixed-size
    summary of its DSpark rounds (see `SpecSummary`). None means the request ran
    the plain path, either because speculation was off or because admission
    downgraded it, which is how a benchmark can tell the two cases apart.
    """

    text: str
    tokens: list[int]
    finish_reason: FinishReason
    prompt_tokens: int
    spec_summary: "SpecSummary | None" = None


@dataclasses.dataclass(frozen=True)
class GenerationStep:
    """One step of streaming generation; finish_reason set on the final step only.

    `num_tokens` is how many generated tokens this step carries. Every
    scheduler sets it on the steps that carry tokens: one per step on the
    single-token paths, and the whole committed block on a speculative verify
    step, which arrives as ONE step with one text delta. Terminal steps carry
    0. How EOS shows up depends on the scheduler: `ContinuousScheduler` and the
    StateCache schedulers drop it, while `PDScheduler` streams it as an
    ordinary one-token step before its terminal step. A consumer that counts
    steps instead of summing this field undercounts speculative output by the
    accepted length.
    """

    text: str
    finish_reason: FinishReason | None = None
    num_tokens: int = 0


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

    `pending_decode_tokens` holds tokens that have been sampled and emitted but
    not yet fed through a forward. Sampling happens before the forward while a
    forward can fail recoverably (a mid-step OOM the engine retries), so the
    step has to be idempotent: without this, the retry would sample the same
    `last_logits` again and emit a duplicate of a token the client already has.
    Cleared by the forward that consumes them.

    `spec` is the DSpark speculative-decoding state, present only on requests
    the scheduler decided to speculate for at admission. Kept as one optional
    field so a type every scheduler shares carries no technique-specific
    members; the reaper clears it when the request finishes, since the API
    handle keeps the `RunningRequest` itself alive.

    `spec_summary` is what survives that clearing: `_finish` reduces the
    request's `DSparkStats` to a fixed-size `SpecSummary` just before it puts
    the terminal step, so the API thread can read it after the stream ends.
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

    `engine_alive` reports whether a step can still arrive for this request.
    Each scheduler binds it at `submit()` to the thread object it has at that
    moment, not to whatever thread it has later: after a crash and a quick
    `start()`, the new thread can lack the request (the StateCache schedulers
    reset their slots), and the handle must not wait on it.
    It also stays True while `stop()` runs, because `stop()` finishes the
    requests left in flight as 'cancelled' after the thread has exited.

    Liveness is read only when the queue stays empty for one poll slice
    (`LIVENESS_POLL_SECONDS`). Steps already buffered are always handed out
    first, and a dead engine is reported as `EngineNotRunningError` within
    one slice. A thread that is alive but stuck (for example a tensor-parallel
    leader blocked in a collective whose follower died) still looks alive;
    this check covers only a thread that exited.

    `engine_alive=None` keeps the old unbounded blocking, for tests that build
    a handle over a bare `RunningRequest` with no engine behind it.
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

        Raises `EngineNotRunningError` after the last buffered step when the
        engine thread exits without a terminal step.
        """
        while True:
            step = self._next_step(deadline=None)
            yield step
            if step.finish_reason is not None:
                return

    def wait(self, timeout: float | None = None) -> GenerationResult:
        """Block until complete; return the aggregated result.

        `timeout` bounds the whole wait, in seconds, and raises `TimeoutError`
        when it runs out while the engine is still alive. A dead engine raises
        `EngineNotRunningError` instead, within one poll slice, with or
        without a timeout.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        text_parts: list[str] = []
        while True:
            step = self._next_step(deadline)
            if step.finish_reason is not None:
                return self.build_result("".join(text_parts), step.finish_reason)
            text_parts.append(step.text)

    def build_result(self, text: str, finish_reason: FinishReason) -> GenerationResult:
        """Aggregate a finished request: the caller's text plus the engine's counts.

        `text` is the concatenation of the non-terminal steps the caller read.
        A terminal step's text is not part of it: `PDScheduler` puts an error
        string there. Call this only after the terminal step arrived, because
        the queue put of that step is what makes the engine-side fields
        (`tokens_generated`, `spec_summary`) safe to read from this thread.
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

        `timeout` raises `TimeoutError` when it runs out while the engine is
        still alive. A dead engine raises `EngineNotRunningError`.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        return self._next_step(deadline)

    def _next_step(self, deadline: float | None) -> GenerationStep:
        """Return the next step, polling engine liveness while the queue is empty.

        The order of the checks matters. The queue comes first, so buffered
        steps always reach the caller before an error. Liveness comes before
        the deadline, so a dead engine is reported as dead and not as slow.
        After liveness reads False, one more non-blocking read is needed: the
        thread can put its last step between the timed-out `get` and the
        liveness read, and a thread that exited made all its puts visible.
        """
        while True:
            timeout: float | None = None
            if self._engine_alive is not None:
                timeout = self.LIVENESS_POLL_SECONDS
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                timeout = remaining if timeout is None else min(timeout, remaining)
            try:
                return self._req.output_queue.get(timeout=timeout)
            except queue.Empty:
                pass
            if self._engine_alive is not None and not self._engine_alive():
                try:
                    return self._req.output_queue.get_nowait()
                except queue.Empty:
                    # A restarted scheduler or a surviving PD prefill thread
                    # can still hold this request; cancel it so that the
                    # engine does not spend work on output nobody reads.
                    self.cancel()
                    raise EngineNotRunningError(
                        "engine not running: the engine thread exited before this request "
                        "finished (it was stopped or it crashed; see the log)"
                    ) from None
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("request did not finish within the timeout")
