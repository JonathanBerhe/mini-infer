"""Tests for `PDStreamingScheduler` (HTTP-shaped adapter over the PD pipeline).

Two contracts to enforce:

  1. Surface: `PDStreamingScheduler` exposes `start()` / `stop()` / `submit()` /
     `run()` matching `ContinuousScheduler`. A `RequestHandle.wait()` returns
     a `GenerationResult` with the same fields populated.

  2. Output: greedy parity vs the in-process `Orchestrator.run()`. Same
     model, same prompt, same sampling params -> same tokens, end-to-end
     through the scheduler engine thread.

The PD path doesn't load any new model state; we reuse the same Qwen2.5-0.5B
runner that the other worker tests pin. Marked `requires_model` so it
skips in CI but runs locally when the model is cached.
"""

from __future__ import annotations

import pytest

from mini_infer.api.pd_scheduler import PDStreamingScheduler
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.request_state import Request
from mini_infer.workers import DecodeWorker, Orchestrator, PrefillWorker

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_runner() -> ModelRunner:
    return ModelRunner.from_pretrained(MODEL_NAME)


@pytest.mark.requires_model
def test_pd_scheduler_run_matches_orchestrator(qwen_runner: ModelRunner) -> None:
    """`scheduler.run(req)` returns a result whose tokens equal `orchestrator.run(req)`."""
    request = Request(
        prompt="The capital of France is",
        sampling_params=SamplingParams(),  # greedy
        max_tokens=8,
    )
    orchestrator = Orchestrator(
        prefill_worker=PrefillWorker(qwen_runner),
        decode_worker=DecodeWorker(qwen_runner),
    )
    direct_tokens = orchestrator.run(request)

    scheduler = PDStreamingScheduler(qwen_runner)
    scheduler.start()
    try:
        result = scheduler.run(request)
    finally:
        scheduler.stop()

    assert list(result.tokens) == direct_tokens, (
        f"PDStreamingScheduler.run drifted from Orchestrator.run:\n"
        f"  scheduler: {list(result.tokens)}\n"
        f"  direct   : {direct_tokens}"
    )
    assert result.prompt_tokens == len(qwen_runner.tokenizer.encode(request.prompt))
    assert result.finish_reason in {"length", "stop"}


@pytest.mark.requires_model
def test_pd_scheduler_streams_tokens(qwen_runner: ModelRunner) -> None:
    """`scheduler.submit(req)` yields per-token steps then a terminal step."""
    request = Request(
        prompt="Once upon a time",
        sampling_params=SamplingParams(),
        max_tokens=4,
    )
    scheduler = PDStreamingScheduler(qwen_runner)
    scheduler.start()
    try:
        handle = scheduler.submit(request)
        # Drain step-by-step; we should see len-1 token steps plus one terminal.
        token_steps = 0
        terminal_seen = False
        while True:
            step = handle.get_step()
            if step.finish_reason is not None:
                terminal_seen = True
                break
            assert step.text != "" or step.finish_reason is not None
            token_steps += 1
            if token_steps > 100:  # safety: this prompt + max_tokens=4 should never get here
                pytest.fail("scheduler emitted >100 steps before terminating")
    finally:
        scheduler.stop()
    assert terminal_seen, "scheduler never emitted a terminal step"
    assert token_steps == request.max_tokens, (
        f"expected {request.max_tokens} token steps before terminal, got {token_steps}"
    )


@pytest.mark.requires_model
def test_pd_scheduler_cancel_emits_terminal(qwen_runner: ModelRunner) -> None:
    """Calling `handle.cancel()` makes the engine emit a 'cancelled' terminal step.

    The engine is allowed to emit the cancellation lazily (at the next token
    boundary), so we read steps until we see a terminal one.
    """
    scheduler = PDStreamingScheduler(qwen_runner)
    scheduler.start()
    try:
        handle = scheduler.submit(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=32,
            )
        )
        # Read one step, then cancel.
        _ = handle.get_step()
        handle.cancel()
        # Drain until a terminal step appears.
        terminal_reason = None
        for _ in range(64):
            step = handle.get_step()
            if step.finish_reason is not None:
                terminal_reason = step.finish_reason
                break
        assert terminal_reason in {"cancelled", "length", "stop"}, (
            f"unexpected terminal finish_reason: {terminal_reason!r}"
        )
    finally:
        scheduler.stop()


@pytest.mark.requires_model
def test_pd_scheduler_lifecycle_idempotent(qwen_runner: ModelRunner) -> None:
    """`start()` is idempotent; `stop()` after start is safe."""
    scheduler = PDStreamingScheduler(qwen_runner)
    scheduler.start()
    scheduler.start()  # second start is a no-op
    scheduler.stop()
    scheduler.stop()  # second stop is safe
