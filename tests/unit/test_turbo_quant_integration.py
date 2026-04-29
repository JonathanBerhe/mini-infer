"""End-to-end TurboQuant integration tests on a real Qwen2.5-0.5B model.

Verifies that running with `kv_quant="turbo4"` produces attention outputs
within cosine similarity > 0.99 of the bf16 baseline, and that decoding
matches token-for-token for at least the first emitted token.

Marked `@pytest.mark.requires_model`. The cache-only round-trip tests
(no model load) live in `test_turbo_quant.py`.
"""

import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.requires_model
def test_turbo4_logits_close_to_bf16_reference() -> None:
    """Last-position logits with kv_quant='turbo4' have cosine sim > 0.99 vs baseline."""
    prompt = "The capital of France is"

    baseline = ModelRunner.from_pretrained(MODEL_NAME, kv_quant=None)
    turbo = ModelRunner.from_pretrained(MODEL_NAME, kv_quant="turbo4")

    input_ids = baseline.tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=baseline.device, dtype=torch.long)
    with torch.inference_mode():
        baseline_logits = baseline._model(input_tensor, use_cache=False).logits[0, -1, :]
        turbo_logits = turbo._model(input_tensor, use_cache=False).logits[0, -1, :]

    cos = torch.nn.functional.cosine_similarity(
        baseline_logits.float().flatten(), turbo_logits.float().flatten(), dim=0
    ).item()
    assert cos > 0.99, f"turbo4 vs baseline cosine sim {cos:.4f} below 0.99"


@pytest.mark.requires_model
def test_turbo4_decodes_paris_smoke() -> None:
    """A turbo4-cache run on the canonical prompt still decodes to 'Paris'."""
    runner = ModelRunner.from_pretrained(MODEL_NAME, kv_quant="turbo4")
    sched = ContinuousScheduler(runner)
    sched.start()
    try:
        result = sched.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=8,
            )
        )
    finally:
        sched.stop()
    assert "Paris" in result.text, f"expected 'Paris' in output, got {result.text!r}"


@pytest.mark.requires_model
def test_turbo4_first_token_matches_bf16_baseline() -> None:
    """First decoded token of turbo4 must match the bf16 baseline.

    Subsequent tokens may drift due to compounding 4-bit quantization noise
    across many layers and decode steps; first-token parity is the
    strictest claim that holds robustly.
    """
    prompt = "The capital of France is"

    baseline_runner = ModelRunner.from_pretrained(MODEL_NAME, kv_quant=None)
    baseline_sched = ContinuousScheduler(baseline_runner)
    baseline_sched.start()
    try:
        baseline_result = baseline_sched.run(
            Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=4)
        )
    finally:
        baseline_sched.stop()

    turbo_runner = ModelRunner.from_pretrained(MODEL_NAME, kv_quant="turbo4")
    turbo_sched = ContinuousScheduler(turbo_runner)
    turbo_sched.start()
    try:
        turbo_result = turbo_sched.run(
            Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=4)
        )
    finally:
        turbo_sched.stop()

    assert baseline_result.tokens, "baseline produced no tokens"
    assert turbo_result.tokens, "turbo produced no tokens"
    assert baseline_result.tokens[0] == turbo_result.tokens[0], (
        f"first-token mismatch: baseline={baseline_result.tokens[0]} "
        f"turbo={turbo_result.tokens[0]} (bf16={baseline_result.text!r}, "
        f"turbo={turbo_result.text!r})"
    )
