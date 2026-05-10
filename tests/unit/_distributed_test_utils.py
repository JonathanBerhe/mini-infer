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
import traceback
from collections.abc import Callable
from typing import Any

import torch
import torch.multiprocessing as mp

from mini_infer.distributed.group import destroy_distributed, init_distributed


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

    Notes:
      - We force the spawn start method (rather than fork) for portability;
        fork on macOS with PyTorch is fragile.
      - Children are joined with a per-process timeout; on timeout we
        terminate them and raise so a hung test doesn't wedge CI.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    master_port = _find_free_port()

    processes: list[mp.Process] = []
    try:
        for rank in range(world_size):
            p = ctx.Process(
                target=_child_entry,
                args=(rank, world_size, master_port, target_fn, queue, args, kwargs),
            )
            p.start()
            processes.append(p)

        # Collect exactly `world_size` results; sort by rank for determinism.
        results: list[tuple[str, int, Any]] = []
        for _ in range(world_size):
            results.append(queue.get(timeout=timeout_sec))
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
