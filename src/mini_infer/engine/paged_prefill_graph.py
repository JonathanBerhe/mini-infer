"""Replay a prefill or mixed step of the paged path from a CUDA graph, one graph per token bucket.

The decode graph (`paged_decode_graph.py`) serves steps where every request
adds one token. A step that admits a prompt, or that carries a prefill chunk
next to the decoders, ran eagerly: several hundred launches for a few
milliseconds of GPU work, so the host set its time. `PagedPrefillGraph`
records such a step once per bucket of new tokens and replays it with one
launch. The cache does the host work before the replay, in
`PagedKVCache.begin_static_step`, and FlashInfer plans the attention once per
step on a wrapper with fixed buffers, in `flashinfer_backend.plan_static_step`.

A bucket of `rows` new tokens has `min(max_requests + 1, rows)` request
slots. The cache's requests take the first slots and their tokens the first
rows; the other slots hold the padding rows, which write into a scratch block
and attend to it. A step with fewer tokens than its bucket needs a free slot
for the padding rows. The slots never outnumber the rows, because FlashInfer
bounds the longest request of a graph plan by `rows - slots + 1`.

The graph needs what the decode graph needs (`paged_decode_graph_blocker`).
A step with hidden-state taps, with more requests than a bucket holds, or
with more tokens than the largest bucket runs eagerly through
`ModelRunner.forward_step_packed`.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch
from torch.profiler import record_function

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.flashinfer_backend import plan_static_step
from mini_infer.cache.paged_kv_cache import PagedKVCache, StaticStep
from mini_infer.device import is_cuda_device
from mini_infer.engine.paged_decode_graph import paged_decode_graph_blocker

DEFAULT_PREFILL_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
DEFAULT_MAX_REQUESTS = 32
# Eager runs before the capture, so lazy initialization (cuBLAS handles,
# kernel selection, the FlashInfer plan parameters) happens outside the graph.
_WARMUP_STEPS = 2


class PagedPrefillGraph:
    """Captured packed steps of a paged-cache model, one per bucket of new tokens.

    A step replays the smallest bucket that holds its tokens and its requests.
    The graphs share the pool's storage, the model and one CUDA memory pool, so
    one instance serves every `PagedKVCache` of the block pool.
    """

    def __init__(
        self,
        model: Any,
        block_pool: BlockPool,
        device: torch.device | str,
        *,
        buckets: tuple[int, ...] = DEFAULT_PREFILL_BUCKETS,
        max_requests: int = DEFAULT_MAX_REQUESTS,
    ) -> None:
        blocker = paged_decode_graph_blocker(block_pool, device, "prefill graph")
        if blocker is not None:
            raise ValueError(blocker)
        if not buckets or min(buckets) < 1:
            raise ValueError(f"buckets must be positive token counts, got {buckets}")
        if max_requests < 1:
            raise ValueError(f"max_requests must be positive, got {max_requests}")
        self._model = model
        self._pool = block_pool
        self._device = torch.device(device)
        self._buckets = tuple(sorted(set(buckets)))
        self._max_requests = max_requests
        # Padding rows write their K/V here and attend to it.
        self._scratch_block = block_pool.allocate()
        self._steps: dict[int, StaticStep] = {}
        # (input ids (1, rows), position ids (1, rows), last row of each slot (slots,),
        #  pinned host staging (3, rows))
        self._inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # A model whose forward takes `logits_rows` runs its head on the last rows only.
        self._selects_rows = "logits_rows" in inspect.signature(model.forward).parameters
        self._inputs_copied: dict[int, Any] = {}
        self._graphs: dict[int, Any] = {}
        self._logits: dict[int, torch.Tensor] = {}
        self._memory_pool: Any = None
        # Largest first: the smaller graphs then reuse the memory of the largest.
        for rows in reversed(self._buckets):
            self._capture(rows)

    @property
    def buckets(self) -> tuple[int, ...]:
        return self._buckets

    @property
    def max_tokens(self) -> int:
        return self._buckets[-1]

    def slots(self, rows: int) -> int:
        """The request slots of the bucket of `rows` tokens."""
        return min(self._max_requests + 1, rows)

    def bucket(self, total: int, requests: int) -> int | None:
        """The smallest bucket that holds `total` new tokens of `requests` requests, or None."""
        for rows in self._buckets:
            if rows < total:
                continue
            slots = self.slots(rows)
            if requests < slots or (requests == slots and total == rows):
                return rows
        return None

    def accepts(self, cu_seqlens_q: list[int]) -> bool:
        """Whether a step with these query boundaries fits a bucket."""
        requests = len(cu_seqlens_q) - 1
        total = cu_seqlens_q[-1]
        if requests < 1 or total < 1:
            return False
        return self.bucket(total, requests) is not None

    def replay(
        self,
        cache: PagedKVCache,
        token_ids: Sequence[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
    ) -> torch.Tensor:
        """Run one packed step: the tokens of request `b` start at `position_offsets[b]`.

        Takes the arguments of `ModelRunner.forward_step_packed`. Returns the
        logits of each request's last row, `(1, requests, vocab)`, in a fresh
        tensor, so the caller can keep rows across the next replay. The
        captured head runs on one row per slot, not on every row of the bucket.
        """
        requests = len(cu_seqlens_q) - 1
        total = cu_seqlens_q[-1]
        if len(token_ids) != total:
            raise ValueError(f"{len(token_ids)} token ids for {total} new tokens")
        if len(position_offsets) != requests:
            raise ValueError(f"{len(position_offsets)} position offsets for {requests} requests")
        rows = self.bucket(total, requests)
        if rows is None:
            raise ValueError(f"no bucket holds {total} tokens of {requests} requests")
        step = self._steps[rows]
        input_ids, position_ids, last_rows, staging = self._inputs[rows]
        positions: list[int] = []
        for batch_idx, offset in enumerate(position_offsets):
            count = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
            positions.extend(range(offset, offset + count))
        on_cuda = is_cuda_device(self._device)
        with record_function("prefill_graph.inputs"):
            copied = self._inputs_copied.get(rows)
            if copied is not None:
                copied.synchronize()  # the previous copy out of `staging` must be done
            staging.zero_()
            staging[0, :total] = torch.tensor(list(token_ids), dtype=torch.long)
            staging[1, :total] = torch.tensor(positions, dtype=torch.long)
            # Padding slots keep row 0, a valid row whose logits nobody reads.
            staging[2, :requests] = torch.tensor(cu_seqlens_q[1:], dtype=torch.long) - 1
            input_ids.copy_(staging[0].unsqueeze(0), non_blocking=on_cuda)
            position_ids.copy_(staging[1].unsqueeze(0), non_blocking=on_cuda)
            last_rows.copy_(staging[2, : last_rows.shape[0]], non_blocking=on_cuda)
            if on_cuda:
                event = torch.cuda.Event()  # type: ignore[no-untyped-call]
                event.record()
                self._inputs_copied[rows] = event
        # Inference mode throughout, like the capture: the plan writes into
        # buffers born inside a forward, which are inference tensors.
        with torch.inference_mode():
            with record_function("prefill_graph.begin"):
                cache.begin_static_step(step, cu_seqlens_q)
            try:
                with record_function("prefill_graph.plan"):
                    plan_static_step(cache, self._device)
                with record_function("prefill_graph.replay"):
                    self._graphs[rows].replay()
                logits = self._logits[rows][:, :requests].clone()
            except BaseException:
                cache.end_static_step(publish=False)
                raise
            cache.end_static_step()
        return logits

    def _forward(self, rows: int, cache: PagedKVCache) -> torch.Tensor:
        """The step's forward with the logits of one row per slot."""
        input_ids, position_ids, last_rows, _ = self._inputs[rows]
        extra = {"logits_rows": last_rows} if self._selects_rows else {}
        logits: torch.Tensor = self._model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=self._steps[rows].qo_indptr,
            **extra,
        )
        if not self._selects_rows:
            logits = logits.index_select(1, last_rows)
        return logits

    def _capture(self, rows: int) -> None:
        """Warm up and capture the step of `rows` tokens, all of them padding."""
        step = StaticStep.allocate(
            rows, self.slots(rows), self._scratch_block, self._pool.num_blocks, self._device
        )
        self._steps[rows] = step
        self._inputs[rows] = (
            torch.zeros(1, rows, dtype=torch.long, device=self._device),
            torch.zeros(1, rows, dtype=torch.long, device=self._device),
            torch.zeros(step.slots, dtype=torch.long, device=self._device),
            torch.zeros(3, rows, dtype=torch.long, pin_memory=True),
        )
        # An empty cache: every slot is padding, so the capture touches no slot.
        cache = PagedKVCache(self._pool)
        main_stream = torch.cuda.current_stream(self._device)
        side_stream = torch.cuda.Stream(device=self._device)  # type: ignore[no-untyped-call]
        side_stream.wait_stream(main_stream)
        with torch.inference_mode(), torch.cuda.stream(side_stream):
            for _ in range(_WARMUP_STEPS):
                cache.begin_static_step(step, [0])
                try:
                    plan_static_step(cache, self._device)
                    self._forward(rows, cache)
                finally:
                    cache.end_static_step(publish=False)
        main_stream.wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            cache.begin_static_step(step, [0])
            try:
                plan_static_step(cache, self._device)
                with torch.cuda.graph(graph, pool=self._memory_pool):
                    self._logits[rows] = self._forward(rows, cache)
            finally:
                cache.end_static_step(publish=False)
        if self._memory_pool is None:
            self._memory_pool = graph.pool()
        self._graphs[rows] = graph
