"""Measure the target's `CapacityCurve` before the scheduler starts.

The paper profiles SPS(B) once at engine start and keeps it as a small table.
`profile_capacity_curve` does the same with the runner's own packed forward,
so the curve includes the attention backend, the hidden-state taps, and the
Python work of a step.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.dspark.verify_scheduler import CapacityCurve
from mini_infer.engine.model_runner import ModelRunner


def default_token_counts(max_requests: int, block_size: int) -> list[int]:
    """Every width up to 64 tokens, then every eighth, up to `max_requests * (block_size + 1)`.

    The largest width is a step in which every request verifies a full block.
    GPU step times jump at tile boundaries, so the grid is dense where steps are short.
    """
    if max_requests < 1 or block_size < 1:
        raise ValueError(
            f"max_requests and block_size must be positive, got {max_requests}, {block_size}"
        )
    top = max_requests * (block_size + 1)
    counts = set(range(1, min(top, 64) + 1))
    counts.update(range(64, top + 1, 8))
    counts.add(top)
    return sorted(counts)


def profile_capacity_curve(
    runner: ModelRunner,
    *,
    token_counts: Sequence[int],
    max_requests: int,
    context_len: int,
    tap_layers: frozenset[int] | None = None,
    warmup: int = 2,
    repeats: int = 5,
    sync: Callable[[], None] | None = None,
    filler_token_id: int = 1,
) -> CapacityCurve:
    """Time one target forward at each width in `token_counts`.

    A width of `B` tokens runs as `min(B, max_requests)` decoding requests. Each
    request has `context_len` tokens of K/V in its slot, and the `B` query tokens
    spread over the requests as evenly as possible. The paper also assumes that
    the time depends mainly on `B` (its footnote 2). Pass the drafter's
    `tap_layers`, because the scheduler's verify forwards collect them.

    Each width reports the median of `repeats` timed forwards after `warmup`
    untimed ones. Pass `sync` (for example `torch.cuda.synchronize`) on a device
    that runs kernels asynchronously. A forward does not get faster with more
    tokens, so each time is raised to the largest time at a smaller width. Then
    timer noise cannot give the scheduler a free token.

    The slots borrow blocks from the runner's pool and give them back at the
    end. Call this before the scheduler starts, while the pool is empty.
    """
    counts = sorted(set(token_counts))
    if not counts or counts[0] < 1:
        raise ValueError(f"token_counts must be positive, got {list(token_counts)}")
    if max_requests < 1:
        raise ValueError(f"max_requests must be positive, got {max_requests}")
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}")
    if warmup < 0 or repeats < 1:
        raise ValueError(f"need warmup >= 0 and repeats >= 1, got {warmup}, {repeats}")

    slots = min(max_requests, counts[-1])
    cache = PagedKVCache(runner.block_pool)
    measured: dict[int, float] = {}
    try:
        for _ in range(slots):
            cache.add_request_slot()
        if context_len > 0:
            runner.forward_step_packed(
                cache,
                [filler_token_id] * (slots * context_len),
                [i * context_len for i in range(slots + 1)],
                [0] * slots,
            )
        # Widest first: a width below the slot count drops slots from the end.
        for width in reversed(counts):
            while cache.batch_size > width:
                cache.remove_request(cache.batch_size - 1)
            measured[width] = _time_width(
                runner,
                cache,
                width,
                context_len=context_len,
                tap_layers=tap_layers,
                warmup=warmup,
                repeats=repeats,
                sync=sync,
                filler_token_id=filler_token_id,
            )
    finally:
        cache.free()

    seconds: list[float] = []
    for width in counts:
        seconds.append(max(measured[width], seconds[-1]) if seconds else measured[width])
    return CapacityCurve(token_counts=tuple(counts), step_seconds=tuple(seconds))


def _time_width(
    runner: ModelRunner,
    cache: PagedKVCache,
    width: int,
    *,
    context_len: int,
    tap_layers: frozenset[int] | None,
    warmup: int,
    repeats: int,
    sync: Callable[[], None] | None,
    filler_token_id: int,
) -> float:
    """The median time of a forward of `width` tokens over every slot of `cache`."""
    requests = cache.batch_size
    per_request, extra = divmod(width, requests)
    cu_seqlens_q = [0]
    for i in range(requests):
        cu_seqlens_q.append(cu_seqlens_q[-1] + per_request + (1 if i < extra else 0))
    input_ids = [filler_token_id] * width
    offsets = [context_len] * requests

    times: list[float] = []
    for i in range(warmup + repeats):
        kwargs: dict[str, Any] = {}
        if tap_layers is not None:
            sink: dict[int, torch.Tensor] = {}
            kwargs = {"tap_layers": tap_layers, "hidden_state_sink": sink}
        if sync is not None:
            sync()
        start = time.perf_counter()
        runner.forward_step_packed(cache, input_ids, cu_seqlens_q, offsets, **kwargs)
        if sync is not None:
            sync()
        elapsed = time.perf_counter() - start
        for slot in range(requests):
            cache.truncate_to(slot, context_len)
        if i >= warmup:
            times.append(elapsed)
    return statistics.median(times)
