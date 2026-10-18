"""Request-related types: input Request, output Result/Step, internal scheduler state."""

import dataclasses
import enum
import queue
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import torch

from mini_infer.engine.sampler import SamplingParams

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
    """API-side handle for a submitted request; drained from the API thread."""

    def __init__(self, running_request: RunningRequest) -> None:
        self._req = running_request

    def steps(self) -> Iterator[GenerationStep]:
        """Yield GenerationStep as the engine produces them; ends after the final step."""
        while True:
            step = self._req.output_queue.get()
            yield step
            if step.finish_reason is not None:
                return

    def wait(self, timeout: float | None = None) -> GenerationResult:
        """Block until complete; return the aggregated result.

        `timeout` bounds the whole wait, in seconds, and raises `TimeoutError`
        when it runs out. The engine thread dies on any exception other than
        `OutOfMemoryError`, and after that no terminal step ever arrives, so
        an unbounded wait blocks its caller forever. `None` keeps that
        unbounded behavior for callers that own the engine's lifetime.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        text_parts: list[str] = []
        finish_reason: FinishReason = "length"
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                step = self._req.output_queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"request did not finish within {timeout}s; the engine thread may have died"
                ) from None
            if step.finish_reason is not None:
                finish_reason = step.finish_reason
                break
            text_parts.append(step.text)
        return GenerationResult(
            text="".join(text_parts),
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

    def get_step(self) -> GenerationStep:
        """Block until the next GenerationStep arrives. Thread-safe."""
        return self._req.output_queue.get()
