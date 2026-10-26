"""`DecodeStepTimer`: per-step forward time and the host gap between forwards.

On the CPU the timer uses host-clock events, so sleeps inside and between
the timed blocks set lower bounds on the reported times. The CUDA test times
real kernels and skips without CUDA.
"""

from __future__ import annotations

import itertools
import time

import pytest
import torch

from mini_infer.engine.decode_step_timer import DecodeStepTimer


def test_first_step_has_no_gap_and_later_steps_do() -> None:
    timer = DecodeStepTimer("cpu")
    for _ in range(3):
        with timer.step():
            time.sleep(0.02)
        time.sleep(0.01)
    steps = timer.drain()
    assert len(steps) == 3
    assert steps[0].gap_ms is None and steps[0].wall_ms is None
    for step in steps:
        assert step.forward_ms >= 19.0
    for previous, step in itertools.pairwise(steps):
        assert step.gap_ms is not None and step.gap_ms >= 9.0
        # One wall interval spans the previous forward and the gap after it.
        assert step.wall_ms == pytest.approx(previous.forward_ms + step.gap_ms, abs=0.5)


def test_skip_gap_drops_the_gap_of_the_next_step_only() -> None:
    timer = DecodeStepTimer("cpu")
    for skip in (False, False, True, False):
        if skip:
            timer.skip_gap()
        with timer.step():
            pass
    gaps = [step.gap_ms for step in timer.drain()]
    assert gaps[0] is None and gaps[2] is None
    assert gaps[1] is not None and gaps[3] is not None


def test_drain_forgets_the_steps_it_returned() -> None:
    timer = DecodeStepTimer("cpu")
    assert timer.drain() == []
    with timer.step():
        pass
    assert len(timer.drain()) == 1
    assert timer.drain() == []
    with timer.step():
        pass
    # The gap of a later step still reaches back to the drained one.
    (step,) = timer.drain()
    assert step.gap_ms is not None


def test_max_steps_keeps_the_newest_steps() -> None:
    timer = DecodeStepTimer("cpu", max_steps=2)
    for sleep_seconds in (0.0, 0.0, 0.03):
        with timer.step():
            time.sleep(sleep_seconds)
    steps = timer.drain()
    assert len(steps) == 2
    assert steps[-1].forward_ms >= 29.0
    assert steps[0].gap_ms is not None


@pytest.mark.requires_cuda
def test_cuda_events_time_the_kernels_without_a_host_sync() -> None:
    a = torch.randn(2048, 2048, device="cuda")
    timer = DecodeStepTimer("cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(4):
            with timer.step():
                for _ in range(8):
                    a = (a @ a).tanh()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    steps = timer.drain()
    assert len(steps) == 4
    assert all(step.forward_ms > 0 for step in steps)
    assert all(step.gap_ms is not None and step.gap_ms >= 0 for step in steps[1:])
