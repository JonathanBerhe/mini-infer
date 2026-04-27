"""Stress tests for chunked prefill under realistic load.

These tests are slow (load model + run many requests) and excluded from CI via
`@pytest.mark.slow`. Run locally with `uv run pytest tests/stress/ -v`.
"""

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.requires_model
@pytest.mark.slow
def test_correctness_under_mixed_length_load() -> None:
    """8 concurrent mixed-length requests, chunked prefill + batched decode.

    Asserts each output matches a serial reference (same scheduler, max_concurrent=1).
    Token-for-token equality with greedy sampling proves the batched flow doesn't
    silently corrupt per-request state under concurrent prefill + decode pressure.
    """
    short_prompts = ["Hi", "What is 2+2?", "The capital of"]
    medium_prompts = [
        "The quick brown fox " * 8,  # ~32 tokens
        "Once upon a time in a faraway land " * 4,  # ~28 tokens
    ]
    long_prompts = [
        "The quick brown fox jumps over the lazy dog. " * 16,  # ~144 tokens
        "Once upon a time in a faraway land " * 16,  # ~112 tokens
        "In the beginning was the Word and the Word was with " * 16,  # ~144 tokens
    ]
    prompts = short_prompts + medium_prompts + long_prompts

    chunk_size = 32  # small enough to force multi-chunk prefills on the long prompts
    max_tokens = 8

    runner = ModelRunner.from_pretrained(MODEL_NAME)

    sched_concurrent = ContinuousScheduler(runner, max_concurrent=16, chunk_size=chunk_size)
    sched_concurrent.start()
    try:
        handles = [
            sched_concurrent.submit(
                Request(prompt=p, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
            for p in prompts
        ]
        concurrent_results = [h.wait() for h in handles]
    finally:
        sched_concurrent.stop()

    sched_serial = ContinuousScheduler(runner, max_concurrent=1, chunk_size=chunk_size)
    sched_serial.start()
    try:
        serial_results = [
            sched_serial.run(
                Request(prompt=p, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
            for p in prompts
        ]
    finally:
        sched_serial.stop()

    diffs: list[str] = []
    for prompt, c, s in zip(prompts, concurrent_results, serial_results, strict=True):
        if c.tokens != s.tokens:
            diffs.append(f"  {prompt[:40]!r}: concurrent={c.tokens} vs serial={s.tokens}")
    assert not diffs, "concurrent != serial under mixed-length load:\n" + "\n".join(diffs)


@pytest.mark.requires_model
@pytest.mark.slow
def test_memory_pressure_with_small_block_pool() -> None:
    """Configure a tiny block pool; submit more requests than fit; all must complete."""
    runner = ModelRunner.from_pretrained(MODEL_NAME, num_blocks=32, block_size=16)
    sched = ContinuousScheduler(runner, max_concurrent=8, chunk_size=32)
    sched.start()
    try:
        prompts = [
            "The quick brown fox jumps over " * 4,
            "Once upon a time in a faraway " * 4,
            "What is the meaning of life? " * 4,
            "The capital of France is Paris " * 4,
            "Two roads diverged in a yellow " * 4,
            "Roses are red and violets are " * 4,
        ]
        handles = [
            sched.submit(Request(prompt=p, sampling_params=SamplingParams(), max_tokens=4))
            for p in prompts
        ]
        results = [h.wait() for h in handles]
    finally:
        sched.stop()

    assert all(r.finish_reason in {"stop", "length"} for r in results), (
        f"some requests crashed: finish_reasons = {[r.finish_reason for r in results]}"
    )
    # Block pool should be fully released after all requests finish.
    assert runner.block_pool.num_free_blocks == 32, (
        f"block leak: {32 - runner.block_pool.num_free_blocks} blocks still allocated"
    )
