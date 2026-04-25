import pytest

from mini_infer.engine.model_runner import ModelRunner

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_runner() -> ModelRunner:
    return ModelRunner.from_pretrained(MODEL_NAME)


@pytest.mark.requires_model
def test_device_resolves_to_supported_backend(qwen_runner: ModelRunner) -> None:
    assert qwen_runner.device in {"mps", "cuda", "cpu"}


@pytest.mark.requires_model
def test_greedy_generation_knows_capital_of_france(qwen_runner: ModelRunner) -> None:
    output = qwen_runner.generate(
        "The capital of France is",
        max_tokens=8,
    )
    assert "Paris" in output
