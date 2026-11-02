"""Replay a decode step of the paged path from a CUDA graph, one graph per batch size.

A decode step of a dense model launches more than a thousand small kernels,
and at small batch the host cannot issue them as fast as the GPU runs them.
A CUDA graph records the step once per batch-size bucket and replays it with
one launch. The paged cache does its host work (slot counts, block
allocation, the write and page indices) before the replay, in
`PagedKVCache.begin_static_decode`, and FlashInfer plans the attention
kernel once per step on a wrapper with fixed buffers, in
`flashinfer_backend.plan_static_decode`. Rows above the real batch are
padding: they write into one scratch block, attend to it, and their logits
are dropped.

The graph needs CUDA, the FlashInfer attention backend, an uncompressed KV
pool and a model whose layers all run full attention with the same head
counts, so that one plan serves every layer. A step that is not decode-only,
carries hidden state taps, or exceeds the largest bucket runs eagerly
through `ModelRunner.forward_step_packed`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.profiler import record_function

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.flashinfer_backend import plan_static_decode
from mini_infer.cache.paged_kv_cache import PagedKVCache, StaticDecodeStep
from mini_infer.device import is_cuda_device
from mini_infer.engine.device_tokens import DeviceTokens

DEFAULT_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
# Eager runs before the capture, so lazy initialization (cuBLAS handles,
# kernel selection, the FlashInfer plan parameters) happens outside the graph.
_WARMUP_STEPS = 2


def paged_decode_graph_blocker(block_pool: BlockPool, device: torch.device | str) -> str | None:
    """Why `PagedDecodeGraph` cannot serve this pool on this device, or None when it can."""
    if not is_cuda_device(device):
        return "the decode graph needs a CUDA device"
    if block_pool.attention_backend != "flashinfer":
        return "the decode graph needs the FlashInfer attention backend"
    if block_pool.kv_quant is not None:
        return "the decode graph writes the uncompressed KV pool only"
    if any(spec != "full" for spec in block_pool.layer_attention):
        return "the decode graph plans one attention call per step, so every layer must be full"
    shapes = {
        (block_pool.num_kv_heads_for_layer(i), block_pool.head_dim_for_layer(i))
        for i in range(block_pool.num_layers)
    }
    if len(shapes) != 1:
        return "the decode graph needs the same KV head count and head size in every layer"
    return None


class PagedDecodeGraph:
    """Captured decode steps of a paged-cache model, one per batch-size bucket.

    A step of `active` rows replays the smallest bucket that holds them. The
    graphs share the pool's storage and the model, so one instance serves every
    `PagedKVCache` of the pool.
    """

    def __init__(
        self,
        model: Any,
        block_pool: BlockPool,
        device: torch.device | str,
        *,
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    ) -> None:
        blocker = paged_decode_graph_blocker(block_pool, device)
        if blocker is not None:
            raise ValueError(blocker)
        if not buckets or min(buckets) < 1:
            raise ValueError(f"buckets must be positive batch sizes, got {buckets}")
        self._model = model
        self._pool = block_pool
        self._device = torch.device(device)
        self._buckets = tuple(sorted(set(buckets)))
        # Padding rows write their K/V here and attend to it.
        self._scratch_block = block_pool.allocate()
        self._steps: dict[int, StaticDecodeStep] = {}
        # (input ids (1, batch), position ids (1, batch), pinned host staging (2, batch))
        self._inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._inputs_copied: dict[int, Any] = {}
        self._graphs: dict[int, Any] = {}
        self._logits: dict[int, torch.Tensor] = {}
        for batch in self._buckets:
            self._capture(batch)

    @property
    def buckets(self) -> tuple[int, ...]:
        return self._buckets

    @property
    def max_batch(self) -> int:
        return self._buckets[-1]

    def accepts(self, cu_seqlens_q: list[int]) -> bool:
        """Whether a step with these query boundaries is a decode step the graph serves."""
        batch = len(cu_seqlens_q) - 1
        if not 0 < batch <= self.max_batch:
            return False
        return all(cu_seqlens_q[i + 1] - cu_seqlens_q[i] == 1 for i in range(batch))

    def bucket(self, active: int) -> int:
        """The smallest bucket that holds `active` rows."""
        for batch in self._buckets:
            if batch >= active:
                return batch
        raise ValueError(f"{active} rows exceed the largest bucket {self.max_batch}")

    def replay(
        self, cache: PagedKVCache, token_ids: Sequence[int] | DeviceTokens, positions: list[int]
    ) -> torch.Tensor:
        """Run one decode step: token `token_ids[b]` at `positions[b]` for every slot `b`.

        With a `DeviceTokens`, the ids go from the sampled tensor into the
        input buffer on the device, and the host never waits for them. The
        padding rows then keep the ids of an earlier step, which are valid
        ids whose logits nobody reads.

        Returns `(1, len(token_ids), vocab)` logits in a fresh tensor, so the
        caller can keep rows across the next replay.
        """
        active = len(token_ids)
        if len(positions) != active:
            raise ValueError(f"{active} tokens but {len(positions)} positions")
        batch = self.bucket(active)
        step = self._steps[batch]
        input_ids, position_ids, staging = self._inputs[batch]
        on_cuda = is_cuda_device(self._device)
        # The phases carry profiler labels, so the step profile shows the host
        # time of each one next to the replay.
        with record_function("decode_graph.inputs"):
            copied = self._inputs_copied.get(batch)
            if copied is not None:
                copied.synchronize()  # the previous copy out of `staging` must be done
            staging.zero_()
            staging[1, :active] = torch.tensor(positions, dtype=torch.long)
            position_ids.copy_(staging[1].unsqueeze(0), non_blocking=on_cuda)
            if isinstance(token_ids, DeviceTokens):
                input_ids[0, :active].copy_(token_ids.gather())
            else:
                staging[0, :active] = torch.tensor(token_ids, dtype=torch.long)
                input_ids.copy_(staging[0].unsqueeze(0), non_blocking=on_cuda)
            if on_cuda:
                event = torch.cuda.Event()  # type: ignore[no-untyped-call]
                event.record()
                self._inputs_copied[batch] = event
        # Inference mode throughout, like the capture: the plan writes into
        # buffers born inside a forward, and torch 2.5 fills the graph's RNG
        # state on replay. Both are inference tensors.
        with torch.inference_mode():
            with record_function("decode_graph.begin"):
                cache.begin_static_decode(step, active)
            try:
                with record_function("decode_graph.plan"):
                    plan_static_decode(cache, self._device)
                with record_function("decode_graph.replay"):
                    self._graphs[batch].replay()
                logits = self._logits[batch][:, :active].clone()
            finally:
                cache.end_static_decode()
        return logits

    def _forward(self, batch: int, cache: PagedKVCache) -> torch.Tensor:
        input_ids, position_ids, _ = self._inputs[batch]
        logits: torch.Tensor = self._model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=self._steps[batch].qo_indptr,
        )
        return logits

    def _capture(self, batch: int) -> None:
        """Warm up and capture the step of `batch` rows, all of them padding."""
        step = StaticDecodeStep.allocate(
            batch, self._scratch_block, self._pool.num_blocks, self._device
        )
        self._steps[batch] = step
        self._inputs[batch] = (
            torch.zeros(1, batch, dtype=torch.long, device=self._device),
            torch.zeros(1, batch, dtype=torch.long, device=self._device),
            torch.zeros(2, batch, dtype=torch.long, pin_memory=True),
        )
        # An empty cache: every row is padding, so the capture touches no slot.
        cache = PagedKVCache(self._pool)
        main_stream = torch.cuda.current_stream(self._device)
        side_stream = torch.cuda.Stream(device=self._device)  # type: ignore[no-untyped-call]
        side_stream.wait_stream(main_stream)
        with torch.inference_mode(), torch.cuda.stream(side_stream):
            for _ in range(_WARMUP_STEPS):
                cache.begin_static_decode(step, active=0)
                try:
                    plan_static_decode(cache, self._device)
                    self._forward(batch, cache)
                finally:
                    cache.end_static_decode()
        main_stream.wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            cache.begin_static_decode(step, active=0)
            try:
                plan_static_decode(cache, self._device)
                with torch.cuda.graph(graph):
                    self._logits[batch] = self._forward(batch, cache)
            finally:
                cache.end_static_decode()
        self._graphs[batch] = graph
