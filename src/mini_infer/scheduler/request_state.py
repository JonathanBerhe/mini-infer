"""Request-related types: input Request, output Result/Step, internal scheduler state."""

import dataclasses
import enum
import queue
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import torch

from mini_infer.engine.sampler import SamplingParams

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


FinishReason = Literal["stop", "length"]


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
    """Engine-side state machine for a request in flight."""

    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"


@dataclasses.dataclass
class RunningRequest:
    """Engine-side state for a request currently in the scheduler.

    Mutated only by the engine thread. The output_queue + completion via the
    final GenerationStep gives the API-side handle a happens-before edge to
    read the final tokens_generated / finish_reason fields safely.

    `cache` is None until prefill runs; the engine populates it from
    `runner.prefill()`'s return value.
    """

    request: Request
    output_queue: queue.Queue[GenerationStep]
    cache: "PagedKVCache | None" = None
    state: RequestState = RequestState.WAITING
    prompt_token_ids: list[int] = dataclasses.field(default_factory=list)
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
