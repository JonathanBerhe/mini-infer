"""Request-related types: input Request, output Result/Step, internal scheduler state."""

import dataclasses
import enum
import queue
import threading
from collections.abc import Iterator
from typing import Literal

import torch

from mini_infer.engine.sampler import SamplingParams

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
    text: str
    tokens: list[int]
    finish_reason: FinishReason
    prompt_tokens: int


@dataclasses.dataclass(frozen=True)
class GenerationStep:
    """One step of streaming generation; finish_reason set on the final step only."""

    text: str
    finish_reason: FinishReason | None = None


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

    def wait(self) -> GenerationResult:
        """Block until complete; return the aggregated result."""
        text_parts: list[str] = []
        finish_reason: FinishReason = "length"
        for step in self.steps():
            if step.finish_reason is not None:
                finish_reason = step.finish_reason
                break
            text_parts.append(step.text)
        return GenerationResult(
            text="".join(text_parts),
            tokens=list(self._req.tokens_generated),
            finish_reason=finish_reason,
            prompt_tokens=len(self._req.prompt_token_ids),
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
