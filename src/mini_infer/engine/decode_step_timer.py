"""Time each decode forward and the host gap between forwards.

A CUDA graph replays the whole decode step with one launch. Between two
replays the GPU waits for the host: the host reads the sampled tokens, does
its bookkeeping, sends the next inputs to the other ranks and copies them in.
`DecodeStepTimer` measures the two parts on the device clock without a host
sync per step. It records an event before and after each forward, and
`drain()` reads the events later, after it waits for the last one.

Each step reports three numbers:

- `forward_ms`: device time from the start to the end of the forward.
- `gap_ms`: device time from the end of the previous forward to the start of
  this one. The GPU runs no model work in it, so it is the host gap.
- `wall_ms`: host time from the start of the previous forward to the start of
  this one.

`gap_ms` and `wall_ms` are None for the first step, and for the first step
after `skip_gap()`. A caller that runs other work between two decode steps (a
prefill, for example) calls `skip_gap()`, so that work does not count as
host gap.

On a device without CUDA, host-clock stand-ins replace the CUDA events, so
the CPU tests see the same fields.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import time
from collections.abc import Iterator
from typing import Any

import torch

from mini_infer.device import is_cuda_device


@dataclasses.dataclass(frozen=True)
class DecodeStepTiming:
    """The timing of one decode forward. See the module docstring."""

    forward_ms: float
    gap_ms: float | None
    wall_ms: float | None


class _HostEvent:
    """A host-clock stand-in for `torch.cuda.Event` on a device without CUDA."""

    def __init__(self) -> None:
        self._seconds = time.perf_counter()

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end: _HostEvent) -> float:
        return (end._seconds - self._seconds) * 1e3


@dataclasses.dataclass(frozen=True)
class _PendingStep:
    previous_end: Any
    start: Any
    end: Any
    wall_ms: float | None


class DecodeStepTimer:
    """Records the decode forwards of one rank. Use it from one thread."""

    def __init__(self, device: torch.device | str, *, max_steps: int = 4096) -> None:
        self._device = device
        self._on_cuda = is_cuda_device(device)
        # Old steps fall out when nobody drains, so the probe stays bounded.
        self._pending: collections.deque[_PendingStep] = collections.deque(maxlen=max_steps)
        self._previous_end: Any = None
        self._previous_start_seconds = 0.0

    def _record(self) -> Any:
        if not self._on_cuda:
            return _HostEvent()
        event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        event.record(torch.cuda.current_stream(self._device))
        return event

    @contextlib.contextmanager
    def step(self) -> Iterator[None]:
        """Time the forward that the `with` block launches."""
        start_seconds = time.perf_counter()
        start = self._record()
        yield
        end = self._record()
        wall_ms = None
        if self._previous_end is not None:
            wall_ms = (start_seconds - self._previous_start_seconds) * 1e3
        self._pending.append(_PendingStep(self._previous_end, start, end, wall_ms))
        self._previous_end = end
        self._previous_start_seconds = start_seconds

    def skip_gap(self) -> None:
        """Report no gap for the next step."""
        self._previous_end = None

    def drain(self) -> list[DecodeStepTiming]:
        """Wait for the recorded steps, return their timings and forget them."""
        if not self._pending:
            return []
        self._pending[-1].end.synchronize()
        steps = [
            DecodeStepTiming(
                forward_ms=pending.start.elapsed_time(pending.end),
                gap_ms=(
                    None
                    if pending.previous_end is None
                    else pending.previous_end.elapsed_time(pending.start)
                ),
                wall_ms=pending.wall_ms,
            )
            for pending in self._pending
        ]
        self._pending.clear()
        return steps
