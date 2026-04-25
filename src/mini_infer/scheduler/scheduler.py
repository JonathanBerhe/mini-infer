import dataclasses
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


class Scheduler:
    """Single-request scheduler; Phase 2 will add continuous batching."""

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    def run(self, request: Request) -> GenerationResult:
        tokenizer = self._runner.tokenizer
        prompt_ids = tokenizer.encode(request.prompt)
        cache, logits = self._runner.prefill(prompt_ids)

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
        )
