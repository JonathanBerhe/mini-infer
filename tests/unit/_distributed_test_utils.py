"""Shared utilities for multi-process TP unit tests.

Spawns N child processes, runs `target_fn(rank, world_size, *args)` in each
under a freshly-initialised gloo PG, collects the per-rank return values
through a `multiprocessing.Queue`, and returns them sorted by rank.

We use gloo (CPU) so these tests run in CI without GPU. The child function
*must not* raise: any exception is caught and forwarded through the queue
so the parent process gets a useful traceback instead of a hung wait.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import traceback
from collections.abc import Callable
from queue import Empty
from typing import Any

import torch
import torch.multiprocessing as mp

from mini_infer.distributed.group import destroy_distributed, init_distributed

# Callers pass a `timeout_sec` sized for their test on an idle machine (60s for
# a KV round-trip, 240s for a full worker cohort). Those budgets are useful
# relative information, so rather than replacing them we multiply them by a
# headroom factor: a busy host slows every test roughly in proportion.
#
# The default of 5 comes from a measured worst case. One full unit run took
# 1069s while Modal jobs shared the machine and three multi-process tests
# failed on their 60s budgets; the same suite unloaded took 74s, a factor of
# ~14 in wall time. 5x the per-test budget clears that comfortably given the
# budgets already carry slack (the KV round-trip needs ~5s of the 60 it asks
# for), without inflating the wait for a test that is genuinely wedged.
_TIMEOUT_SCALE_ENV = "MINI_INFER_TEST_MP_TIMEOUT_SCALE"
_DEFAULT_TIMEOUT_SCALE = 5.0
# Absolute ceiling so a pathological scale or budget cannot wedge CI for an
# unbounded stretch. 240s (the largest budget in the suite) still lands under
# this at the default scale.
_TIMEOUT_CEILING_SEC = 900.0
# How often to wake and check whether the children are still alive while
# waiting for results.
_POLL_SEC = 0.5
# `mp.Queue.put` hands off to a feeder thread, so a child can exit before its
# result is readable. When every child has exited we wait this long for the
# queue to drain before concluding that nothing was reported.
_DRAIN_GRACE_SEC = 5.0


def _effective_timeout(timeout_sec: float) -> float:
    """Scale a caller's idle-machine budget for host load, bounded by a ceiling."""
    raw = os.environ.get(_TIMEOUT_SCALE_ENV)
    scale = _DEFAULT_TIMEOUT_SCALE
    if raw is not None:
        try:
            scale = float(raw)
        except ValueError as exc:
            raise ValueError(f"{_TIMEOUT_SCALE_ENV} must be a number, got {raw!r}") from exc
        if scale <= 0:
            raise ValueError(f"{_TIMEOUT_SCALE_ENV} must be positive, got {scale}")
    return min(timeout_sec * scale, _TIMEOUT_CEILING_SEC)


def _find_free_port() -> int:
    """Bind a socket to port 0 to grab an OS-assigned free port, then close.

    There's a small race between us closing the socket and the child binding
    it, but for unit tests run serially this is good enough. The real fix
    (use `dist.TCPStore` directly with port=0) would complicate the test
    harness for vanishing benefit.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _child_entry(
    rank: int,
    world_size: int,
    master_port: int,
    target_fn: Callable[..., Any],
    queue: mp.Queue,
    args: tuple,
    kwargs: dict,
) -> None:
    """Process entry-point: init PG, run target, push (rank, result-or-error)."""
    try:
        init_distributed(
            world_size=world_size,
            rank=rank,
            backend="gloo",
            master_addr="127.0.0.1",
            master_port=master_port,
        )
        result = target_fn(rank, world_size, *args, **kwargs)
        queue.put(("ok", rank, result))
    except Exception:
        # We can't pickle exceptions cleanly across processes in all cases
        # (some have un-picklable attached state); ship the formatted
        # traceback string instead.
        queue.put(("err", rank, traceback.format_exc()))
    finally:
        destroy_distributed()


def run_multi_process(
    world_size: int,
    target_fn: Callable[..., Any],
    *args: Any,
    timeout_sec: float = 60.0,
    **kwargs: Any,
) -> list[Any]:
    """Run `target_fn` in `world_size` child processes and return per-rank
    results sorted by rank.

    `timeout_sec` is the budget for an IDLE machine; the real deadline is that
    scaled for host load (see `_effective_timeout`). A shared runner used to
    block on `queue.get(timeout=...)` for the whole budget and let a bare
    `queue.Empty` escape, which conflated three very different situations and
    named none of them. They are now separated:

      - **A child crashed.** Detected from a non-zero exit code, reported with
        that code, and raised as soon as it happens rather than after the full
        budget. A child killed by a signal (an OOM kill shows up as -9) never
        reaches the queue, so this used to surface as an inscrutable timeout.
      - **Every child exited without reporting.** Distinct from a crash, and
        usually means a result went missing rather than a worker dying.
      - **Children still running at the deadline.** The genuinely ambiguous
        case: a wedged test and a heavily loaded host look alike from here, so
        the message says so and points at the scale knob.

    Deliberately NOT retried on timeout. These tests exist to catch races in
    distributed code, and re-running until one passes is precisely how a real
    intermittent race gets mistaken for load. Generous headroom plus fast
    crash detection addresses the flakiness without hiding that class of bug.

    Notes:
      - We force the spawn start method (rather than fork) for portability;
        fork on macOS with PyTorch is fragile.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    deadline_sec = _effective_timeout(timeout_sec)
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    master_port = _find_free_port()

    processes: list[mp.Process] = []
    try:
        for rank in range(world_size):
            p = ctx.Process(
                target=_child_entry,
                args=(rank, world_size, master_port, target_fn, result_queue, args, kwargs),
            )
            p.start()
            processes.append(p)

        results = _collect_results(result_queue, processes, world_size, deadline_sec, timeout_sec)
        results.sort(key=lambda triple: triple[1])

        for status, rank, payload in results:
            if status == "err":
                raise AssertionError(f"rank {rank} failed:\n{payload}")

        return [payload for _, _, payload in results]
    finally:
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


def _describe_exits(processes: list[mp.Process]) -> str:
    """Per-rank liveness and exit codes, for failure messages."""
    parts = []
    for rank, p in enumerate(processes):
        code = p.exitcode
        parts.append(f"rank {rank}: {'running' if code is None else f'exit {code}'}")
    return ", ".join(parts)


def _collect_results(
    result_queue: mp.Queue,
    processes: list[mp.Process],
    world_size: int,
    deadline_sec: float,
    requested_sec: float,
) -> list[tuple[str, int, Any]]:
    """Gather `world_size` results, failing fast and specifically on a dead child."""
    results: list[tuple[str, int, Any]] = []
    deadline = time.monotonic() + deadline_sec

    while len(results) < world_size:
        try:
            results.append(result_queue.get(timeout=_POLL_SEC))
            continue
        except Empty:
            pass

        crashed = [
            (rank, p.exitcode)
            for rank, p in enumerate(processes)
            if p.exitcode is not None and p.exitcode != 0
        ]
        if crashed:
            detail = ", ".join(f"rank {rank} exit {code}" for rank, code in crashed)
            raise AssertionError(
                f"child process died without reporting a result ({detail}). "
                f"A negative code is a signal (-9 is typically an OOM kill). "
                f"This is a crash, not a timeout. Full state: "
                f"{_describe_exits(processes)}"
            )

        if all(p.exitcode is not None for p in processes):
            # Everyone finished cleanly but we are short results. Most likely
            # the queue simply has not drained yet, so give the feeder thread
            # a moment before calling it a failure.
            try:
                results.append(result_queue.get(timeout=_DRAIN_GRACE_SEC))
                continue
            except Empty:
                raise AssertionError(
                    f"all {world_size} children exited cleanly but only "
                    f"{len(results)} of {world_size} results arrived, after a "
                    f"{_DRAIN_GRACE_SEC}s drain grace. A result was lost rather "
                    f"than a worker crashing. Full state: {_describe_exits(processes)}"
                ) from None

        if time.monotonic() >= deadline:
            raise AssertionError(
                f"timed out after {deadline_sec:.0f}s with "
                f"{len(results)}/{world_size} results and children still "
                f"running ({_describe_exits(processes)}). No child crashed, so "
                f"this is either a genuine hang or a heavily loaded host. The "
                f"budget is {requested_sec:.0f}s scaled by "
                f"{_TIMEOUT_SCALE_ENV} (default {_DEFAULT_TIMEOUT_SCALE}, "
                f"ceiling {_TIMEOUT_CEILING_SEC:.0f}s); raise that env var if "
                f"the machine is busy."
            )

    return results


def is_multi_process_available() -> bool:
    """True iff we can run multi-process tests in this environment.

    We require:
      - `torch.distributed` is built with gloo support.
      - We can spawn subprocesses (true on macOS / Linux, false in some
        sandboxes).

    Tests skip themselves when this returns False rather than fail, since
    "the dev's machine doesn't allow subprocess spawn" is not a code defect.
    """
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_gloo_available():
        return False
    # Probe ability to spawn: `mp.get_context("spawn")` raises on platforms
    # that don't support it. The actual `spawn` is exercised by every test
    # that runs, so a probe-only check here is sufficient.
    try:
        mp.get_context("spawn")
    except Exception:
        return False
    # On some sandboxed CI machines stdin is closed which breaks spawn.
    # Cheap probe: can we open /dev/null?
    try:
        with open(os.devnull, "rb"):
            pass
    except OSError:
        return False
    # Spawn imports the test module via __main__; if pytest hasn't put us
    # on a real filesystem path the children can't re-import. Fall back
    # gracefully.
    return bool(getattr(sys, "argv", None))
