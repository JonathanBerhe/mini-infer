import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import Request, Scheduler

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def scheduler() -> Scheduler:
    runner = ModelRunner.from_pretrained(MODEL_NAME)
    return Scheduler(runner)


@pytest.mark.requires_model
def test_run_returns_paris_for_france_prompt(scheduler: Scheduler) -> None:
    result = scheduler.run(
        Request(
            prompt="The capital of France is",
            sampling_params=SamplingParams(),
            max_tokens=8,
        )
    )
    assert "Paris" in result.text
    assert len(result.tokens) > 0
    assert result.finish_reason in {"stop", "length"}


@pytest.mark.requires_model
def test_run_respects_max_tokens(scheduler: Scheduler) -> None:
    result = scheduler.run(
        Request(
            prompt="Once upon a time",
            sampling_params=SamplingParams(),
            max_tokens=2,
        )
    )
    assert len(result.tokens) <= 2
