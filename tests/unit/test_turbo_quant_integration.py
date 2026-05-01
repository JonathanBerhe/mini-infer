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
def test_turbo3_logits_close_to_bf16_reference() -> None:
    """Last-position logits with kv_quant='turbo3' have cosine sim > 0.99 vs baseline.

    turbo3 = full V3 (rotation + polar + Lloyd-Max + QJL + asymmetric K3V4).
    The richer recipe should match (or beat) turbo4's > 0.99 cosine sim
    despite K only having 3 bits of codebook precision (the QJL residual
    sign bit closes the gap to ~4-bit fidelity).
    """
    prompt = "The capital of France is"

    baseline = ModelRunner.from_pretrained(MODEL_NAME, kv_quant=None)
    turbo3 = ModelRunner.from_pretrained(MODEL_NAME, kv_quant="turbo3")

    input_ids = baseline.tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=baseline.device, dtype=torch.long)
    with torch.inference_mode():
        baseline_logits = baseline._model(input_tensor, use_cache=False).logits[0, -1, :]
        turbo3_logits = turbo3._model(input_tensor, use_cache=False).logits[0, -1, :]

    cos = torch.nn.functional.cosine_similarity(
        baseline_logits.float().flatten(), turbo3_logits.float().flatten(), dim=0
    ).item()
    assert cos > 0.99, f"turbo3 vs baseline cosine sim {cos:.4f} below 0.99"


@pytest.mark.requires_model
def test_turbo3_produces_coherent_output() -> None:
    """A turbo3-cache run produces grammatically coherent output (non-degenerate).

    Unlike turbo4 (which preserves argmax exactly on 0.5B), turbo3's
    aggressive 3-bit K compression can flip argmax even when logit
    cosine sim is > 0.99. The contract is "high-fidelity logits, but
    not argmax parity"; this test asserts the model still produces a
    non-empty, coherent completion (no NaN, no empty output, no infinite
    repeat).
    """
    runner = ModelRunner.from_pretrained(MODEL_NAME, kv_quant="turbo3")
    sched = ContinuousScheduler(runner)
    sched.start()
    try:
        result = sched.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=12,
            )
        )
    finally:
        sched.stop()
    # Coherence checks: the model emitted some tokens, decoded text isn't
    # empty, and doesn't degenerate into single-token repetition.
    assert result.tokens, "turbo3 produced no tokens"
    assert result.text.strip(), f"turbo3 output is empty: {result.text!r}"
    # Reject obvious degenerate output (same token > 6 times in a row).
    counts: dict[int, int] = {}
    max_run = 1
    current_run = 1
    last_tok = result.tokens[0]
    for tok in result.tokens[1:]:
        counts[tok] = counts.get(tok, 0) + 1
        current_run = current_run + 1 if tok == last_tok else 1
        max_run = max(max_run, current_run)
        last_tok = tok
    assert max_run < 6, (
        f"turbo3 output has degenerate {max_run}-token repeat: "
        f"tokens={result.tokens}, text={result.text!r}"
    )


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
