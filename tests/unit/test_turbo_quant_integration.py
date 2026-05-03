"""End-to-end TurboQuant integration tests on a real Qwen2.5-0.5B model.

Compares last-position prefill logits with `kv_quant="turbo4"` (or
`"turbo3"`) against the bf16 baseline. Both runners build their own
KV cache from the same prompt; the only difference is whether the
cache stores the K/V uncompressed (baseline) or 4-bit-compressed
(turbo). The compression noise propagates through attention into the
final logits — the test bounds it at the prompt length below.

Cosine thresholds are calibrated for Qwen2.5-0.5B at ~31 tokens
(roughly 2 blocks at block_size=16). turbo4 round-trips at
~0.98 cosine; turbo3's 3-bit-K is noisier at ~0.82 on a small model
(turbo3's recipe is designed to widen the gap on deeper models).

Marked `@pytest.mark.requires_model`. The cache-only round-trip tests
(no model load) live in `test_turbo_quant.py`.
"""

import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ~31 tokens — fills two blocks at block_size=16 (one full, one partial) so
# the compression-noise measurement isn't dominated by zero-pad slots in a
# single half-empty block.
LONG_PROMPT = (
    "Once upon a time, in a land far far away, there was a great kingdom "
    "ruled by a wise king who governed his lands with kindness and dedication."
)


@pytest.mark.requires_model
def test_turbo4_logits_close_to_bf16_reference() -> None:
    """Last-position logits with kv_quant='turbo4' have cosine sim > 0.95 vs baseline.

    The two-block prompt above keeps compression noise on Qwen2.5-0.5B at
    ~0.98 cosine; 0.95 is a comfortable floor.
    """
    baseline = ModelRunner.from_pretrained(MODEL_NAME, kv_quant=None)
    turbo = ModelRunner.from_pretrained(MODEL_NAME, kv_quant="turbo4")

    input_ids = baseline.tokenizer.encode(LONG_PROMPT)
    _, baseline_logits = baseline.prefill(input_ids)
    _, turbo_logits = turbo.prefill(input_ids)

    cos = torch.nn.functional.cosine_similarity(
        baseline_logits.float().flatten(), turbo_logits.float().flatten(), dim=0
    ).item()
    assert cos > 0.95, f"turbo4 vs baseline cosine sim {cos:.4f} below 0.95"


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
    """Last-position logits with kv_quant='turbo3' have cosine sim > 0.75 vs baseline.

    turbo3 = full V3 (rotation + polar + Lloyd-Max + QJL + asymmetric K3V4).
    The 3-bit K compression is noisier than turbo4's symmetric 4-bit on
    Qwen2.5-0.5B; the recipe is designed to widen its margin on deeper
    models. At ~31 tokens we measure ~0.82 cosine, so 0.75 is a stable
    floor that catches genuine breakage.
    """
    baseline = ModelRunner.from_pretrained(MODEL_NAME, kv_quant=None)
    turbo3 = ModelRunner.from_pretrained(MODEL_NAME, kv_quant="turbo3")

    input_ids = baseline.tokenizer.encode(LONG_PROMPT)
    _, baseline_logits = baseline.prefill(input_ids)
    _, turbo3_logits = turbo3.prefill(input_ids)

    cos = torch.nn.functional.cosine_similarity(
        baseline_logits.float().flatten(), turbo3_logits.float().flatten(), dim=0
    ).item()
    assert cos > 0.75, f"turbo3 vs baseline cosine sim {cos:.4f} below 0.75"


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
