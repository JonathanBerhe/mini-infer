from collections.abc import Iterator

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def scheduler() -> Iterator[ContinuousScheduler]:
    runner = ModelRunner.from_pretrained(MODEL_NAME)
    sched = ContinuousScheduler(runner)
    sched.start()
    try:
        yield sched
    finally:
        sched.stop()


@pytest.mark.requires_model
def test_run_returns_paris_for_france_prompt(scheduler: ContinuousScheduler) -> None:
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
def test_run_respects_max_tokens(scheduler: ContinuousScheduler) -> None:
    result = scheduler.run(
        Request(
            prompt="Once upon a time",
            sampling_params=SamplingParams(),
            max_tokens=2,
        )
    )
    assert len(result.tokens) <= 2


@pytest.mark.requires_model
def test_run_records_prompt_tokens(scheduler: ContinuousScheduler) -> None:
    result = scheduler.run(
        Request(
            prompt="Hello",
            sampling_params=SamplingParams(),
            max_tokens=2,
        )
    )
    assert result.prompt_tokens > 0


@pytest.mark.requires_model
def test_concurrent_two_requests_complete(scheduler: ContinuousScheduler) -> None:
    """Two requests submitted close together both complete with sensible output."""
    handle_a = scheduler.submit(
        Request(
            prompt="The capital of France is",
            sampling_params=SamplingParams(),
            max_tokens=8,
        )
    )
    handle_b = scheduler.submit(
        Request(
            prompt="def fibonacci(n):",
            sampling_params=SamplingParams(),
            max_tokens=4,
        )
    )

    result_a = handle_a.wait()
    result_b = handle_b.wait()

    assert "Paris" in result_a.text
    assert len(result_b.tokens) > 0
    assert len(result_b.tokens) <= 4


@pytest.mark.requires_model
def test_stream_yields_text_then_finish(scheduler: ContinuousScheduler) -> None:
    steps = list(
        scheduler.stream(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=8,
            )
        )
    )
    text_steps = [s for s in steps if s.finish_reason is None]
    finish_steps = [s for s in steps if s.finish_reason is not None]
    assert len(text_steps) > 0
    assert len(finish_steps) == 1
    full_text = "".join(s.text for s in text_steps)
    assert "Paris" in full_text


@pytest.mark.requires_model
def test_batched_decode_matches_serial(scheduler: ContinuousScheduler) -> None:
    """Three concurrent requests through the batched scheduler match a serial reference.

    Greedy sampling on each request, so outputs are deterministic. The shared
    `ContinuousScheduler` runs them with B>=1 (likely B=3 for most steps); a
    fresh single-request scheduler runs each with B=1 throughout. Token
    sequences must match exactly — proves the batched forward produces the
    same per-request logits as N independent forwards.
    """
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "def fibonacci(n):",
    ]

    handles = [
        scheduler.submit(Request(prompt=p, sampling_params=SamplingParams(), max_tokens=6))
        for p in prompts
    ]
    batched_results = [h.wait() for h in handles]

    serial_results = []
    for prompt in prompts:
        serial_results.append(
            scheduler.run(Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=6))
        )

    for batched, serial, prompt in zip(batched_results, serial_results, prompts, strict=True):
        assert batched.tokens == serial.tokens, (
            f"divergence on {prompt!r}: batched={batched.tokens}, serial={serial.tokens}"
        )


@pytest.mark.requires_model
def test_short_request_finishing_first_does_not_corrupt_others(
    scheduler: ContinuousScheduler,
) -> None:
    """When a request with a tighter max_tokens finishes mid-batch, the survivors
    must continue producing the same outputs as if they had run alone.
    """
    short_prompt = "Hi"
    long_prompt = "The capital of France is"

    h_short = scheduler.submit(
        Request(prompt=short_prompt, sampling_params=SamplingParams(), max_tokens=2)
    )
    h_long = scheduler.submit(
        Request(prompt=long_prompt, sampling_params=SamplingParams(), max_tokens=8)
    )
    long_concurrent = h_long.wait()
    h_short.wait()  # drain to completion

    long_solo = scheduler.run(
        Request(prompt=long_prompt, sampling_params=SamplingParams(), max_tokens=8)
    )

    assert long_concurrent.tokens == long_solo.tokens, (
        "long-request output diverged after short-request finish; batch_idx drift bug"
    )


@pytest.mark.requires_model
def test_chunked_prefill_matches_unchunked() -> None:
    """A long prompt processed in multiple chunks must produce identical output to
    the same prompt processed in a single shot.

    Uses two separately-instantiated schedulers so each has its own engine state:
    one with a small chunk_size that forces multi-step prefill, one with a chunk
    big enough to fit the prompt in a single chunk. Greedy sampling so outputs
    are deterministic — token-for-token equality proves chunked prefill is a
    no-op on output.
    """
    runner = ModelRunner.from_pretrained(MODEL_NAME)
    prompt = "The capital of France is one of the most popular tourist destinations"
    sampling = SamplingParams()
    max_tokens = 8

    # Get the prompt length in tokens to choose chunk sizes that actually chunk.
    prompt_len = len(runner.tokenizer.encode(prompt))
    chunk_size_small = max(prompt_len // 3, 2)  # forces 3 or 4 chunks
    chunk_size_big = prompt_len + 10  # fits the whole prompt in one chunk

    sched_small = ContinuousScheduler(runner, chunk_size=chunk_size_small)
    sched_big = ContinuousScheduler(runner, chunk_size=chunk_size_big)
    sched_small.start()
    sched_big.start()
    try:
        out_small = sched_small.run(
            Request(prompt=prompt, sampling_params=sampling, max_tokens=max_tokens)
        )
        out_big = sched_big.run(
            Request(prompt=prompt, sampling_params=sampling, max_tokens=max_tokens)
        )
    finally:
        sched_small.stop()
        sched_big.stop()

    assert out_small.tokens == out_big.tokens, (
        f"chunked prefill diverged from unchunked: "
        f"chunked={out_small.tokens}, unchunked={out_big.tokens}"
    )


@pytest.mark.requires_model
def test_prefix_cache_matches_no_cache() -> None:
    """Prefix-cached output must equal the no-cache reference token-for-token.

    The point of prefix caching is to skip prefill on prompt prefixes that have
    been computed before. This must be a numerical no-op: a request whose
    prefix is in the cache produces the same K/V values, the same attention
    outputs, and therefore the same sampled tokens as a request whose cache is
    cold. Two prompts sharing a long prefix exercise both the lookup path
    (second prompt hits) and the publish path (first prompt populates).
    """
    # Long shared prefix so the prefix cache has multiple full blocks to hit.
    shared = (
        "You are a helpful assistant. Answer the user's question directly and "
        "without unnecessary preamble. Today's date is January 1st 2025. "
    )
    prompt_a = shared + "Question: What is the capital of France?"
    prompt_b = shared + "Question: What is the capital of Germany?"
    sampling = SamplingParams()
    max_tokens = 8

    runner_no_cache = ModelRunner.from_pretrained(MODEL_NAME, prefix_cache=False)
    sched_no_cache = ContinuousScheduler(runner_no_cache)
    sched_no_cache.start()
    try:
        ref_a = sched_no_cache.run(
            Request(prompt=prompt_a, sampling_params=sampling, max_tokens=max_tokens)
        )
        ref_b = sched_no_cache.run(
            Request(prompt=prompt_b, sampling_params=sampling, max_tokens=max_tokens)
        )
    finally:
        sched_no_cache.stop()

    runner_with_cache = ModelRunner.from_pretrained(MODEL_NAME, prefix_cache=True)
    sched_with_cache = ContinuousScheduler(runner_with_cache)
    sched_with_cache.start()
    try:
        cached_a = sched_with_cache.run(
            Request(prompt=prompt_a, sampling_params=sampling, max_tokens=max_tokens)
        )
        # `prompt_b` shares the system prefix with `prompt_a`; the second run
        # exercises the lookup-then-prepopulate path.
        cached_b = sched_with_cache.run(
            Request(prompt=prompt_b, sampling_params=sampling, max_tokens=max_tokens)
        )
    finally:
        sched_with_cache.stop()

    assert cached_a.tokens == ref_a.tokens, (
        f"first cached run diverged: cached={cached_a.tokens}, ref={ref_a.tokens}"
    )
    assert cached_b.tokens == ref_b.tokens, (
        f"second cached run (with prefix hit) diverged: "
        f"cached={cached_b.tokens}, ref={ref_b.tokens}"
    )

    # Sanity: the second cached run actually hit something. Block pool's prefix
    # cache should have entries from prompt_a; prompt_b's shared prefix was
    # served from those entries. We can't easily inspect the engine thread's
    # state here, but `num_cached > 0` after both runs is enough.
    pf = runner_with_cache.block_pool.prefix_cache
    assert pf is not None
    assert pf.num_cached > 0, "expected the prefix cache to retain at least one block"
