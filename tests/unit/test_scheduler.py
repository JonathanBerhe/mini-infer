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
