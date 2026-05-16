"""Multi-request tests for `PDScheduler` (Slice 1: single-thread variant).

The contract: PD's greedy output for N concurrent requests is
token-for-token identical to `ContinuousScheduler`'s output on the
same requests. Greedy because temperature=0 collapses to argmax;
any drift between schedulers shows up as a different token.

Plus: per-request termination (heterogeneous `max_tokens`, EOS),
cancellation mid-stream, and block-pool cleanliness after the run.

Marked `requires_model`: needs a real model load. Each test
constructs both schedulers over the same `ModelRunner` (one
in-memory model, two scheduler instances).
"""

from __future__ import annotations

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
from mini_infer.scheduler.request_state import Request
from mini_infer.workers import PDScheduler

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_runner() -> ModelRunner:
    return ModelRunner.from_pretrained(MODEL_NAME)


def _continuous_greedy_tokens(runner: ModelRunner, prompt: str, max_tokens: int) -> list[int]:
    """Run one greedy request through `ContinuousScheduler`; return token ids.

    Used as the bit-parity reference for `PDScheduler` output.
    """
    scheduler = ContinuousScheduler(runner)
    scheduler.start()
    try:
        result = scheduler.run(
            Request(
                prompt=prompt,
                sampling_params=SamplingParams(),
                max_tokens=max_tokens,
            )
        )
        return list(result.tokens)
    finally:
        scheduler.stop()


@pytest.mark.requires_model
def test_pd_scheduler_single_request_matches_continuous(
    qwen_runner: ModelRunner,
) -> None:
    """One request through PDScheduler should match ContinuousScheduler.

    Same as the existing `PDStreamingScheduler` contract but via the new
    multi-request engine. The single-request case is the degenerate batch.
    """
    request = Request(
        prompt="The capital of France is",
        sampling_params=SamplingParams(),
        max_tokens=8,
    )
    baseline = _continuous_greedy_tokens(qwen_runner, request.prompt, request.max_tokens)

    scheduler = PDScheduler(qwen_runner)
    scheduler.start()
    try:
        result = scheduler.run(request)
    finally:
        scheduler.stop()

    assert list(result.tokens) == baseline


@pytest.mark.requires_model
def test_pd_scheduler_concurrent_requests_match_continuous(
    qwen_runner: ModelRunner,
) -> None:
    """N=3 concurrent requests through PDScheduler produce the same tokens as
    the same N requests through ContinuousScheduler.

    The PD scheduler batches them via `prefill_batch` + `DecodeSession.step`;
    the ContinuousScheduler runs them through its chunked-prefill + packed
    decode forward. Greedy + same model + same prompt -> same tokens, even
    though the batching shape inside each engine differs.
    """
    prompts = [
        "The capital of France is",
        "The largest planet is",
        "A common greeting is",
    ]
    max_tokens = 6

    baseline: list[list[int]] = [
        _continuous_greedy_tokens(qwen_runner, p, max_tokens) for p in prompts
    ]

    scheduler = PDScheduler(qwen_runner)
    scheduler.start()
    try:
        handles = [
            scheduler.submit(
                Request(
                    prompt=p,
                    sampling_params=SamplingParams(),
                    max_tokens=max_tokens,
                )
            )
            for p in prompts
        ]
        results = [h.wait() for h in handles]
    finally:
        scheduler.stop()

    actual = [list(r.tokens) for r in results]
    assert actual == baseline, (
        f"PDScheduler diverged from ContinuousScheduler:\n"
        f"  PD       : {actual}\n"
        f"  baseline : {baseline}"
    )


@pytest.mark.requires_model
def test_pd_scheduler_heterogeneous_max_tokens(qwen_runner: ModelRunner) -> None:
    """Different `max_tokens` per request; each stops at its own budget.

    Verifies per-slot termination in the scheduler's `_check_termination`
    path. The shorter-budget requests must terminate cleanly and free
    their slot; the longer-budget requests keep running uninterrupted.
    """
    prompts_and_budgets = [
        ("The capital of France is", 3),
        ("The capital of France is", 8),
        ("Once upon a time", 5),
    ]

    scheduler = PDScheduler(qwen_runner)
    scheduler.start()
    try:
        handles = [
            scheduler.submit(
                Request(
                    prompt=p,
                    sampling_params=SamplingParams(),
                    max_tokens=mt,
                )
            )
            for p, mt in prompts_and_budgets
        ]
        results = [h.wait() for h in handles]
    finally:
        scheduler.stop()

    for (_, mt), result in zip(prompts_and_budgets, results, strict=True):
        assert len(result.tokens) <= mt, f"request exceeded max_tokens: {len(result.tokens)} > {mt}"
        assert result.finish_reason in {"length", "stop"}, result.finish_reason


@pytest.mark.requires_model
def test_pd_scheduler_cancel_mid_stream_emits_terminal(
    qwen_runner: ModelRunner,
) -> None:
    """`handle.cancel()` makes the engine emit a terminal step and free the slot.

    Concurrent: cancel one request while two others keep streaming.
    The cancelled request gets a terminal step; the others run to
    completion. Block pool returns to fully free after stop.
    """
    scheduler = PDScheduler(qwen_runner)
    scheduler.start()
    pool = qwen_runner.block_pool
    free_before = pool.num_free_blocks
    try:
        h_to_cancel = scheduler.submit(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=32,
            )
        )
        h_keep_a = scheduler.submit(
            Request(
                prompt="The largest planet is",
                sampling_params=SamplingParams(),
                max_tokens=4,
            )
        )
        h_keep_b = scheduler.submit(
            Request(
                prompt="A common greeting is",
                sampling_params=SamplingParams(),
                max_tokens=4,
            )
        )

        # Read one step from the cancel-target so we know it's mid-decode,
        # then cancel it.
        first_step = h_to_cancel.get_step()
        assert first_step.finish_reason is None
        h_to_cancel.cancel()

        # Drain the cancelled handle until terminal.
        terminal_reason = None
        for _ in range(64):
            step = h_to_cancel.get_step()
            if step.finish_reason is not None:
                terminal_reason = step.finish_reason
                break
        assert terminal_reason in {"cancelled", "length", "stop"}

        # The other two should complete normally.
        result_a = h_keep_a.wait()
        result_b = h_keep_b.wait()
        assert len(result_a.tokens) <= 4
        assert len(result_b.tokens) <= 4
    finally:
        scheduler.stop()

    # Block pool returned to free state (no leaked slots).
    assert pool.num_free_blocks == free_before, (
        f"PDScheduler leaked blocks: {free_before} free before, {pool.num_free_blocks} after"
    )


@pytest.mark.requires_model
def test_pd_scheduler_releases_blocks_on_completion(
    qwen_runner: ModelRunner,
) -> None:
    """Block pool returns to fully free after a multi-request run.

    Catches leaked slots from a missing `remove_slot` (the most likely
    bug surface): if any slot leaks, repeated multi-request runs would
    starve the pool.
    """
    pool = qwen_runner.block_pool
    free_before = pool.num_free_blocks

    scheduler = PDScheduler(qwen_runner)
    scheduler.start()
    try:
        handles = [
            scheduler.submit(
                Request(
                    prompt=p,
                    sampling_params=SamplingParams(),
                    max_tokens=4,
                )
            )
            for p in [
                "The capital of France is",
                "The largest planet is",
                "A common greeting is",
            ]
        ]
        for h in handles:
            h.wait()
    finally:
        scheduler.stop()

    assert pool.num_free_blocks == free_before, (
        f"PDScheduler leaked blocks: {free_before} free before, {pool.num_free_blocks} after"
    )


@pytest.mark.requires_model
def test_pd_scheduler_lifecycle_idempotent(qwen_runner: ModelRunner) -> None:
    """`start()` is idempotent; `stop()` after `start()` is safe and repeatable."""
    scheduler = PDScheduler(qwen_runner)
    scheduler.start()
    scheduler.start()  # idempotent
    scheduler.stop()
    scheduler.stop()  # idempotent
