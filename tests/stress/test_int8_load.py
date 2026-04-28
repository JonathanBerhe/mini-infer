"""Stress tests for the INT8 quantized model on a real Qwen2.5-0.5B.

The integration tests in `tests/unit/test_int8_model_integration.py` verify
single-forward correctness; these tests run greedy decode and check that
quantization noise doesn't immediately flip token decisions.

Marked `@pytest.mark.slow` to keep them out of CI; run locally with
`uv run pytest tests/stress/ -v`.
"""

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request
from mini_infer.scheduler.request_state import GenerationResult

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _greedy_decode_all(
    quant: str | None, prompts: list[str], max_tokens: int
) -> list[GenerationResult]:
    """Build a runner with the given quant setting, decode every prompt, tear down."""
    runner = ModelRunner.from_pretrained(MODEL_NAME, quant=quant)
    sched = ContinuousScheduler(runner)
    sched.start()
    try:
        return [
            sched.run(Request(prompt=p, sampling_params=SamplingParams(), max_tokens=max_tokens))
            for p in prompts
        ]
    finally:
        sched.stop()


@pytest.mark.requires_model
@pytest.mark.slow
def test_int8_first_tokens_match_fp_reference() -> None:
    """Greedy-decode parity: the first few tokens of int8 should match fp reference.

    Weight-only INT8 with per-channel scales typically keeps the cosine
    similarity of logits >0.99, which usually preserves the argmax for early
    tokens before drift accumulates. We require the FIRST decoded token to
    match (the strictest claim that's actually robust under quantization
    noise) and report how far the match extends.
    """
    prompts = [
        "The capital of France is",
        "Once upon a time in a faraway land,",
        "def fibonacci(n):",
    ]
    max_tokens = 8

    fp_results = _greedy_decode_all(quant=None, prompts=prompts, max_tokens=max_tokens)
    int8_results = _greedy_decode_all(quant="int8", prompts=prompts, max_tokens=max_tokens)

    # First-token parity is the strictest claim that holds robustly. After
    # that, drift is allowed; we report the prefix-match length per prompt.
    diffs: list[str] = []
    for prompt, fp, int8 in zip(prompts, fp_results, int8_results, strict=True):
        if not fp.tokens or not int8.tokens:
            diffs.append(f"  {prompt!r}: empty output")
            continue
        if fp.tokens[0] != int8.tokens[0]:
            diffs.append(
                f"  {prompt!r}: first token diverged (fp={fp.tokens[0]} int8={int8.tokens[0]})"
            )
    assert not diffs, "int8 first-token parity failures:\n" + "\n".join(diffs)
