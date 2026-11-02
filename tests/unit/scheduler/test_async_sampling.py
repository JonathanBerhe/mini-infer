"""Async sampling on the CPU: the ids of a forward feed the next one before the host reads them.

The micro Qwen3 of the DSpark doubles stands in for a model, with its
character tokenizer. Random weights are enough, because each test compares the
async scheduler against the sync one on the same weights, token for token.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from mini_infer.engine.device_tokens import DeviceTokens
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, GenerationResult, Request
from mini_infer.scheduler.request_state import RequestHandle
from tests.unit._dspark_doubles import EOS, CharTokenizer, micro_setup

_PROMPTS = ("the quick brown fox", "jumps over", "the lazy dog by the river")


class _RecordingRunner(ModelRunner):
    """A micro-model runner that records where each forward took its ids.

    The random model never emits EOS on its own, so every request runs to its
    `max_tokens`. `eos_for_slot_0_at` makes the forward with that ordinal put
    EOS first in the last row of slot 0, so one request stops at a known step.
    """

    def __init__(self, seed: int, *, eos_for_slot_0_at: int | None = None) -> None:
        micro, _ = micro_setup(seed=seed)
        super().__init__(micro._model, CharTokenizer(), "cpu", micro.block_pool)  # type: ignore[arg-type]
        self.sources: list[str] = []
        self.forwards = 0
        self._eos_at = eos_for_slot_0_at

    def forward_step_packed(  # type: ignore[override]
        self,
        cache: Any,
        packed_input_ids: Any,
        cu_seqlens_q: list[int],
        position_offsets: list[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        self.forwards += 1
        self.sources.append("device" if isinstance(packed_input_ids, DeviceTokens) else "host")
        logits = (
            super()
            .forward_step_packed(cache, packed_input_ids, cu_seqlens_q, position_offsets, **kwargs)
            .clone()
        )
        logits[..., EOS] = -1e9
        if self._eos_at is not None and self.forwards == self._eos_at:
            logits[0, cu_seqlens_q[1] - 1, EOS] = logits.max() + 10.0
        return logits


def _request(prompt: str, max_tokens: int) -> Request:
    return Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=max_tokens)


def _run_all(scheduler: ContinuousScheduler, requests: list[Request]) -> list[GenerationResult]:
    scheduler.start()
    try:
        handles = [scheduler.submit(request) for request in requests]
        return [handle.wait(timeout=60) for handle in handles]
    finally:
        scheduler.stop()


def _same_results(got: list[GenerationResult], expected: list[GenerationResult]) -> None:
    """Same ids and the same end. The text is the ids decoded, and `wait` only
    collects the steps that nobody read before it, so it is not compared."""
    assert [r.tokens for r in got] == [r.tokens for r in expected]
    assert [r.finish_reason for r in got] == [r.finish_reason for r in expected]
    assert [r.prompt_tokens for r in got] == [r.prompt_tokens for r in expected]


def _wait_for_tokens(handle: RequestHandle, count: int) -> None:
    seen = 0
    deadline = time.monotonic() + 30
    while seen < count:
        step = handle.get_step(timeout=max(0.1, deadline - time.monotonic()))
        assert step.finish_reason is None, "the request ended before it produced enough tokens"
        seen += step.num_tokens


def test_off_the_gpu_async_sampling_is_off_by_default() -> None:
    runner = _RecordingRunner(seed=0)
    assert ContinuousScheduler(runner)._async_sampling is False
    assert ContinuousScheduler(runner, async_sampling=True)._async_sampling is True


def test_async_steps_take_their_ids_from_the_device_and_decode_the_same_tokens() -> None:
    """Three requests with staggered ends, one of them on EOS: tokens, text and reasons match.

    After the prefill forward, every forward of the async scheduler takes its
    ids from the device. A request that finished still has a row in the next
    forward, which runs anyway for the others; only the last request to finish
    costs a forward of its own, so the async run counts one forward more.
    """
    requests = [
        _request(_PROMPTS[0], 10),
        _request(_PROMPTS[1], 7),
        _request(_PROMPTS[2], 12),
    ]
    sync_runner = _RecordingRunner(seed=3, eos_for_slot_0_at=4)
    expected = _run_all(ContinuousScheduler(sync_runner, async_sampling=False), requests)
    assert expected[0].finish_reason == "stop" and len(expected[0].tokens) == 3
    assert [r.finish_reason for r in expected[1:]] == ["length", "length"]

    async_runner = _RecordingRunner(seed=3, eos_for_slot_0_at=4)
    got = _run_all(ContinuousScheduler(async_runner, async_sampling=True), requests)
    _same_results(got, expected)

    assert set(sync_runner.sources) == {"host"}
    assert async_runner.sources[0] == "host"
    assert set(async_runner.sources[1:]) == {"device"}
    # Only the last request's unread forward ran for nothing else.
    assert async_runner.forwards == sync_runner.forwards + 1


def test_a_new_arrival_reads_the_ids_on_the_host_for_its_mixed_step_only() -> None:
    """A request admitted mid-decode makes one host-side step; the decode then goes async again."""
    first = _request(_PROMPTS[2], 12)
    second = _request(_PROMPTS[0], 6)
    alone = [
        _run_all(ContinuousScheduler(_RecordingRunner(seed=5), async_sampling=False), [first]),
        _run_all(ContinuousScheduler(_RecordingRunner(seed=5), async_sampling=False), [second]),
    ]

    runner = _RecordingRunner(seed=5)
    scheduler = ContinuousScheduler(runner, async_sampling=True)
    scheduler.start()
    try:
        handle_first = scheduler.submit(first)
        _wait_for_tokens(handle_first, 3)
        handle_second = scheduler.submit(second)
        results = [handle_first.wait(timeout=60), handle_second.wait(timeout=60)]
    finally:
        scheduler.stop()
    _same_results(results, [alone[0][0], alone[1][0]])

    sources = runner.sources
    first_device = sources.index("device")
    later_host = [i for i, source in enumerate(sources) if source == "host" and i > first_device]
    assert later_host, "the mixed prefill step of the second request took host ids"
    assert "device" in sources[later_host[-1] + 1 :], "the decode went back to device ids"


def test_a_cancelled_request_leaves_the_others_on_the_device_path() -> None:
    survivor = _request(_PROMPTS[1], 9)
    alone = _run_all(
        ContinuousScheduler(_RecordingRunner(seed=8), async_sampling=False), [survivor]
    )

    runner = _RecordingRunner(seed=8)
    scheduler = ContinuousScheduler(runner, async_sampling=True)
    scheduler.start()
    try:
        doomed = scheduler.submit(_request(_PROMPTS[0], 40))
        kept = scheduler.submit(survivor)
        _wait_for_tokens(doomed, 2)
        doomed.cancel()
        cancelled = doomed.wait(timeout=60)
        result = kept.wait(timeout=60)
    finally:
        scheduler.stop()
    assert cancelled.finish_reason == "cancelled"
    _same_results([result], alone)
    assert "device" in runner.sources


def test_sampling_rows_keep_their_request_when_a_prefill_shares_the_forward() -> None:
    """The deferred ids of a mixed forward come from each decoder's own last row."""
    runner = _RecordingRunner(seed=11)
    scheduler = ContinuousScheduler(runner, async_sampling=True, chunk_size=4)
    scheduler.start()
    try:
        short = scheduler.submit(_request("ab", 8))
        _wait_for_tokens(short, 2)
        # A long prompt with a small chunk size prefills over several forwards
        # while `short` decodes beside it.
        long = scheduler.submit(_request(_PROMPTS[2] * 2, 5))
        results = [short.wait(timeout=60), long.wait(timeout=60)]
    finally:
        scheduler.stop()
    expected = [
        _run_all(ContinuousScheduler(_RecordingRunner(seed=11), async_sampling=False), [req])[0]
        for req in (_request("ab", 8), _request(_PROMPTS[2] * 2, 5))
    ]
    _same_results(results, expected)


@pytest.mark.parametrize("async_sampling", [False, True])
def test_results_do_not_depend_on_the_mode_for_a_single_request(async_sampling: bool) -> None:
    result = _run_all(
        ContinuousScheduler(_RecordingRunner(seed=1), async_sampling=async_sampling),
        [_request(_PROMPTS[0], 5)],
    )[0]
    assert len(result.tokens) == 5 or result.finish_reason == "stop"
    assert result.prompt_tokens == len(CharTokenizer().encode(_PROMPTS[0]))
