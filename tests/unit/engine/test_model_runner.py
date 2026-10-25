import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_runner() -> ModelRunner:
    return ModelRunner.from_pretrained(MODEL_NAME)


@pytest.mark.requires_model
def test_device_resolves_to_supported_backend(qwen_runner: ModelRunner) -> None:
    assert qwen_runner.device in {"mps", "cuda", "cpu"}


@pytest.mark.requires_model
def test_prefill_then_decode_advances_cache_by_one(qwen_runner: ModelRunner) -> None:
    prompt_ids = qwen_runner.tokenizer.encode("Hello, world!")
    cache, logits = qwen_runner.prefill(prompt_ids)
    assert cache.get_seq_length() == len(prompt_ids)
    assert logits.ndim == 1
    assert logits.shape[0] > 1000  # vocab is not tiny

    next_token = int(torch.argmax(logits).item())
    cache_after, logits_after = qwen_runner.decode(cache, next_token)
    assert cache_after.get_seq_length() == len(prompt_ids) + 1
    assert logits_after.shape == logits.shape
