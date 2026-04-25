import dataclasses
from collections.abc import Iterator
from typing import Literal

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams, sample

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


class Scheduler:
    """Single-request scheduler; Phase 2 will add continuous batching."""

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    def run(self, request: Request) -> GenerationResult:
        tokenizer = self._runner.tokenizer
        prompt_ids = tokenizer.encode(request.prompt)
        cache, logits = self._runner.prefill(prompt_ids)
        try:
            tokens: list[int] = []
            finish_reason: FinishReason = "length"
            for _ in range(request.max_tokens):
                next_token = sample(logits, request.sampling_params)
                if next_token == tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                tokens.append(next_token)
                cache, logits = self._runner.decode(cache, next_token)

            return GenerationResult(
                text=tokenizer.decode(tokens),
                tokens=tokens,
                finish_reason=finish_reason,
                prompt_tokens=len(prompt_ids),
            )
        finally:
            cache.free()

    def stream(self, request: Request) -> Iterator[GenerationStep]:
        """Yield one GenerationStep per token; final yield carries the finish_reason."""
        # Decode-and-diff so multi-byte UTF-8 boundaries don't fragment text deltas.
        # O(n^2) total decode work across the loop; fine at Phase 1 scale.
        tokenizer = self._runner.tokenizer
        prompt_ids = tokenizer.encode(request.prompt)
        cache, logits = self._runner.prefill(prompt_ids)

        all_tokens: list[int] = []
        last_text = ""

        try:
            for _ in range(request.max_tokens):
                next_token = sample(logits, request.sampling_params)
                if next_token == tokenizer.eos_token_id:
                    yield GenerationStep(text="", finish_reason="stop")
                    return

                all_tokens.append(next_token)
                current_text = tokenizer.decode(all_tokens)
                delta = current_text[len(last_text) :]
                last_text = current_text

                yield GenerationStep(text=delta)
                cache, logits = self._runner.decode(cache, next_token)

            yield GenerationStep(text="", finish_reason="length")
        finally:
            cache.free()
