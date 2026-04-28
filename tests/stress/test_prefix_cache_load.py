"""Stress tests for prefix caching under realistic shared-prefix workloads.

Marked `@pytest.mark.slow` to keep them out of CI; run locally with
`uv run pytest tests/stress/ -v`.
"""

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.requires_model
@pytest.mark.slow
def test_shared_system_prompt_parity() -> None:
    """Concurrent requests with a shared system prompt match a no-cache reference.

    Realistic chat-template workload: every request opens with the same long
    system prompt and adds a unique user turn. With prefix caching enabled, the
    system prompt's blocks should be computed once and reused for every later
    request. Output tokens must match a no-cache run token-for-token; if the
    cache silently returned wrong K/V, the divergence would be visible here.
    """
    system_prompt = (
        "You are a helpful, harmless, honest assistant. Answer concisely. "
        "Today's date is January 1st 2025. Use plain English; no emojis. "
        "Decline questions about prior conversations; you have no memory. "
    )
    user_questions = [
        "What is the capital of France?",
        "How many planets are in the solar system?",
        "Name three primary colors.",
        "Who wrote Hamlet?",
        "What is 7 times 8?",
        "What is the chemical symbol for gold?",
    ]
    prompts = [system_prompt + q for q in user_questions]
    sampling = SamplingParams()
    max_tokens = 8

    # Reference: separate runner with prefix caching off.
    runner_no_cache = ModelRunner.from_pretrained(MODEL_NAME, prefix_cache=False)
    sched_no_cache = ContinuousScheduler(runner_no_cache)
    sched_no_cache.start()
    try:
        ref_results = [
            sched_no_cache.run(Request(prompt=p, sampling_params=sampling, max_tokens=max_tokens))
            for p in prompts
        ]
    finally:
        sched_no_cache.stop()

    runner_with_cache = ModelRunner.from_pretrained(MODEL_NAME, prefix_cache=True)
    sched_with_cache = ContinuousScheduler(runner_with_cache, max_concurrent=4)
    sched_with_cache.start()
    try:
        handles = [
            sched_with_cache.submit(
                Request(prompt=p, sampling_params=sampling, max_tokens=max_tokens)
            )
            for p in prompts
        ]
        cached_results = [h.wait() for h in handles]
    finally:
        sched_with_cache.stop()

    diffs: list[str] = []
    for prompt, cached, ref in zip(prompts, cached_results, ref_results, strict=True):
        if cached.tokens != ref.tokens:
            tail = prompt[-48:]
            diffs.append(f"  ...{tail!r}: cached={cached.tokens} vs ref={ref.tokens}")
    assert not diffs, "prefix-cached output diverged from no-cache reference:\n" + "\n".join(diffs)

    # Sanity: the cache populated. Without this, the test would still pass
    # trivially (no cached blocks => no possible mismatch).
    pf = runner_with_cache.block_pool.prefix_cache
    assert pf is not None
    assert pf.num_cached > 0


@pytest.mark.requires_model
@pytest.mark.slow
def test_eviction_under_unique_prompts() -> None:
    """Many unique prompts on a small pool: pool reclaims via LRU; nothing crashes.

    Stresses the BlockPool fallback path (free list exhausted -> evict_lru).
    Each prompt is unique, so the cache fills with non-shared blocks; the pool
    must reclaim them on demand. Pass criterion: every request finishes, no OOM.
    """
    runner = ModelRunner.from_pretrained(
        MODEL_NAME, num_blocks=64, block_size=16, prefix_cache=True
    )
    sched = ContinuousScheduler(runner, max_concurrent=4, chunk_size=32)
    sched.start()
    try:
        prompts = [
            f"Write one sentence about topic {i}: the year {2000 + i} and the number {i * 13}."
            for i in range(20)
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
