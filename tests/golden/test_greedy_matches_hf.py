import json
from pathlib import Path
from typing import Any

import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GOLDEN_PATH = Path(__file__).parent / "golden_data" / "qwen_0_5b_instruct.json"

_GOLDEN_SAMPLES: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text())["samples"]


@pytest.fixture(scope="module")
def cpu_runner() -> ModelRunner:
    return ModelRunner.from_pretrained(MODEL_NAME, device="cpu", dtype=torch.float32)


@pytest.mark.requires_model
@pytest.mark.parametrize("sample", _GOLDEN_SAMPLES, ids=lambda s: s["prompt"][:30])
def test_greedy_matches_hf_reference(cpu_runner: ModelRunner, sample: dict[str, Any]) -> None:
    prompt_ids = cpu_runner.tokenizer.encode(sample["prompt"])
    cache, logits = cpu_runner.prefill(prompt_ids)

    our_tokens: list[int] = []
    for _ in range(sample["max_new_tokens"]):
        next_token = int(torch.argmax(logits).item())
        our_tokens.append(next_token)
        if next_token == cpu_runner.tokenizer.eos_token_id:
            break
        cache, logits = cpu_runner.decode(cache, next_token)

    assert our_tokens == sample["expected_tokens"], (
        f"Token divergence on prompt {sample['prompt']!r}; "
        f"ours={our_tokens}, golden={sample['expected_tokens']}"
    )
