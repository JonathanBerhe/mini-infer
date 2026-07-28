"""The multi-process test harness's own failure handling.

This exists because the harness is what every TP test's diagnostics come
from, and its error paths were the reason three tests once failed with a bare
`_queue.Empty` that said nothing about which of several very different things
had gone wrong. Untested error handling tends to be wrong error handling, so
the paths are exercised here directly: a crashing child, a silent child, and
the load-versus-hang timeout.

The workers are module-level functions because `spawn` re-imports this module
in the child and cannot pickle a closure or a local.
"""

from __future__ import annotations

import os
import time
from unittest import mock

import pytest

from tests.unit._distributed_test_utils import (
    _DEFAULT_TIMEOUT_SCALE,
    _TIMEOUT_CEILING_SEC,
    _TIMEOUT_SCALE_ENV,
    _effective_timeout,
    is_multi_process_available,
    run_multi_process,
)

pytestmark = pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process / gloo unavailable in this environment",
)


def _ok_worker(rank: int, world_size: int) -> int:
    del world_size
    return rank * 10


def _hard_exit_worker(rank: int, world_size: int) -> int:
    """Die without unwinding, so nothing reaches the result queue.

    `os._exit` skips the harness's `except`/`finally`, which is what a
    segfault or an OOM kill looks like from the parent's side.
    """
    del world_size
    if rank == 1:
        os._exit(7)
    return rank


def _raising_worker(rank: int, world_size: int) -> int:
    del world_size
    if rank == 1:
        raise RuntimeError("worker blew up on purpose")
    return rank


def _slow_worker(rank: int, world_size: int) -> int:
    """Outlive a deliberately tiny deadline, then exit on its own promptly.

    The sleep is only a few seconds rather than a minute so the children are
    gone shortly after the timeout fires. An earlier version slept 30s, which
    meant `run_multi_process`'s cleanup had to SIGTERM them mid-gloo, and a
    later test in the same session could then hang in `init_distributed`,
    presumably on the port-reuse race `_find_free_port` documents. Letting
    them finish naturally keeps this test from corrupting its neighbours.
    """
    del world_size
    time.sleep(3)
    return rank


# --- timeout scaling -------------------------------------------------------


def test_budget_is_scaled_for_load() -> None:
    assert _effective_timeout(60.0) == 60.0 * _DEFAULT_TIMEOUT_SCALE


def test_scale_is_env_overridable() -> None:
    with mock.patch.dict(os.environ, {_TIMEOUT_SCALE_ENV: "2.0"}):
        assert _effective_timeout(60.0) == 120.0


def test_ceiling_bounds_the_wait() -> None:
    """No scale or budget may push a single test past the ceiling."""
    with mock.patch.dict(os.environ, {_TIMEOUT_SCALE_ENV: "1000"}):
        assert _effective_timeout(600.0) == _TIMEOUT_CEILING_SEC


@pytest.mark.parametrize("bad", ["abc", "0", "-1"])
def test_invalid_scale_is_rejected(bad: str) -> None:
    """Fail loudly rather than silently falling back to the default."""
    with mock.patch.dict(os.environ, {_TIMEOUT_SCALE_ENV: bad}), pytest.raises(ValueError):
        _effective_timeout(60.0)


# --- the failure paths the harness has to tell apart -----------------------


def test_happy_path_returns_per_rank_results() -> None:
    assert run_multi_process(2, _ok_worker, timeout_sec=30.0) == [0, 10]


def test_crashed_child_is_reported_as_a_crash_not_a_timeout() -> None:
    """A child that dies without reporting must name the exit code, and fast.

    This is the case that used to consume the whole budget and then surface as
    a bare `queue.Empty`, since a hard exit never reaches the queue.
    """
    started = time.monotonic()
    with pytest.raises(AssertionError) as excinfo:
        run_multi_process(2, _hard_exit_worker, timeout_sec=30.0)

    message = str(excinfo.value)
    assert "died without reporting" in message
    assert "exit 7" in message
    assert "not a timeout" in message
    # Detection is driven by the exit code, so it must not wait out the budget
    # (30s base, 150s scaled at the default).
    assert time.monotonic() - started < 30.0, "should fail fast, not wait for the deadline"


def test_worker_exception_still_arrives_with_its_traceback() -> None:
    """An ordinary raise is forwarded, and must not be mistaken for a crash."""
    with pytest.raises(AssertionError) as excinfo:
        run_multi_process(2, _raising_worker, timeout_sec=30.0)

    message = str(excinfo.value)
    assert "rank 1 failed" in message
    assert "worker blew up on purpose" in message
    assert "died without reporting" not in message


def test_timeout_names_load_as_a_possibility_and_points_at_the_knob() -> None:
    """Children still alive at the deadline is the one genuinely ambiguous case."""
    # Scale down so the test is quick: 1s base * 0.5 = 0.5s against a 30s worker.
    with (
        mock.patch.dict(os.environ, {_TIMEOUT_SCALE_ENV: "0.5"}),
        pytest.raises(AssertionError) as excinfo,
    ):
        run_multi_process(2, _slow_worker, timeout_sec=1.0)

    message = str(excinfo.value)
    assert "timed out" in message
    assert "still" in message and "running" in message
    assert "loaded host" in message
    assert _TIMEOUT_SCALE_ENV in message, "must say which knob to turn"
