"""The paged prefill graph without a GPU.

Its gate, its bucket choice, the runner's routing, and replays whose "captured"
step runs the eager forward on the static buffers and must match the eager
step bit for bit. Such a replay needs a step that fills its bucket exactly,
because padding rows read their K/V through FlashInfer. The capture itself and
the padded buckets run on CUDA only.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache, StaticStep
from mini_infer.engine.device_tokens import DeviceTokens
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.paged_prefill_graph import PagedPrefillGraph
from tests.unit._dspark_doubles import micro_setup


class _EagerStandIn:
    """Stands in for a captured graph: `replay` runs the forward on the static buffers."""

    def __init__(self, graph: PagedPrefillGraph, rows: int) -> None:
        self.graph = graph
        self.rows = rows
        self.cache: PagedKVCache | None = None

    def replay(self) -> None:
        assert self.cache is not None and self.cache.static_step is not None
        self.graph._logits[self.rows] = self.graph._forward(self.rows, self.cache)


def _fake_graph(
    model: Any, pool: BlockPool, buckets: tuple[int, ...], max_requests: int
) -> PagedPrefillGraph:
    """A graph object on the CPU whose steps run eagerly instead of replaying."""
    graph = PagedPrefillGraph.__new__(PagedPrefillGraph)
    graph._model = model
    graph._pool = pool
    graph._device = torch.device("cpu")
    graph._buckets = tuple(sorted(buckets))
    graph._max_requests = max_requests
    graph._scratch_block = pool.allocate()
    graph._steps = {}
    graph._inputs = {}
    graph._inputs_copied = {}
    graph._graphs = {}
    graph._logits = {}
    graph._memory_pool = None
    for rows in graph._buckets:
        graph._steps[rows] = StaticStep.allocate(
            rows, graph.slots(rows), graph._scratch_block, pool.num_blocks, graph._device
        )
        graph._inputs[rows] = (
            torch.zeros(1, rows, dtype=torch.long),
            torch.zeros(1, rows, dtype=torch.long),
            torch.zeros(2, rows, dtype=torch.long),
        )
        graph._graphs[rows] = _EagerStandIn(graph, rows)
    return graph


def _two_slot_cache(seed: int) -> tuple[Any, PagedKVCache]:
    runner, _ = micro_setup(seed=seed)
    cache = PagedKVCache(runner.block_pool)
    cache.add_request_slot()
    cache.add_request_slot()
    return runner, cache


def test_blocker_names_the_prefill_graph() -> None:
    runner, _ = micro_setup()
    with pytest.raises(ValueError, match="the prefill graph needs a CUDA device"):
        PagedPrefillGraph(runner._model, runner.block_pool, "cpu")


def test_bucket_takes_the_smallest_that_holds_the_tokens_and_the_requests() -> None:
    runner, _ = micro_setup()
    graph = _fake_graph(runner._model, runner.block_pool, (16, 32, 64), max_requests=32)
    assert [graph.slots(rows) for rows in (16, 32, 64)] == [16, 32, 33]
    assert graph.bucket(10, 3) == 16
    assert graph.bucket(16, 16) == 16  # every slot taken and every row used: no padding
    assert graph.bucket(15, 16) == 32  # one padding row needs a free slot
    assert graph.bucket(64, 33) == 64
    assert graph.bucket(63, 33) is None
    assert graph.bucket(65, 1) is None  # more tokens than the largest bucket
    assert graph.accepts([0, 6, 9])
    assert not graph.accepts([0])
    assert not graph.accepts([0, 0])
    assert not graph.accepts([0, 70])


def test_replay_matches_the_eager_prefill_and_mixed_steps() -> None:
    """A prefill step of two prompts, then a step with a decoder next to a prefill
    chunk: the static steps and the eager steps agree bit for bit."""
    eager_runner, eager_cache = _two_slot_cache(seed=3)
    graph_runner, graph_cache = _two_slot_cache(seed=3)
    # max_requests=1 gives two slots per bucket, so a two-request step that fills
    # its bucket has no padding.
    graph = _fake_graph(graph_runner._model, graph_runner.block_pool, (4, 9), max_requests=1)
    for stand_in in graph._graphs.values():
        assert isinstance(stand_in, _EagerStandIn)
        stand_in.cache = graph_cache
    steps = [
        (list(range(10, 19)), [0, 6, 9], [0, 0]),
        ([5, 7, 8, 9], [0, 1, 4], [6, 3]),
    ]
    assert not torch.is_inference_mode_enabled()
    for tokens, cu_seqlens_q, offsets in steps:
        expected = eager_runner.forward_step_packed(eager_cache, tokens, cu_seqlens_q, offsets)
        got = graph.replay(graph_cache, tokens, cu_seqlens_q, offsets)
        assert torch.equal(got, expected)
        assert graph_cache.static_step is None
        assert graph_cache.seq_lens_list() == eager_cache.seq_lens_list()
    for layer_idx in range(graph_runner.block_pool.num_layers):
        graph_k, graph_v, _, _ = graph_cache.materialize_packed_kv(layer_idx)
        eager_k, eager_v, _, _ = eager_cache.materialize_packed_kv(layer_idx)
        assert torch.equal(graph_k, eager_k) and torch.equal(graph_v, eager_v)


def test_padded_replay_needs_flashinfer_and_leaves_the_step() -> None:
    runner, cache = _two_slot_cache(seed=0)
    graph = _fake_graph(runner._model, runner.block_pool, (16,), max_requests=4)
    stand_in = graph._graphs[16]
    assert isinstance(stand_in, _EagerStandIn)
    stand_in.cache = cache
    with pytest.raises(RuntimeError, match="padded static step"):
        graph.replay(cache, list(range(10, 19)), [0, 6, 9], [0, 0])
    assert cache.static_step is None  # the step closed on the way out


class _RecordingGraph:
    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.calls: list[tuple[Any, ...]] = []

    def accepts(self, cu_seqlens_q: list[int]) -> bool:
        return self.accept

    def replay(self, cache: PagedKVCache, *args: Any) -> torch.Tensor:
        self.calls.append(args)
        total = len(args[0])
        return torch.zeros(1, total, 7)


def test_runner_routes_steps_to_the_prefill_graph_after_the_decode_graph() -> None:
    micro, cache = _two_slot_cache(seed=1)
    micro.forward_step_packed(cache, list(range(10, 19)), [0, 6, 9], [0, 0])
    runner = ModelRunner(micro._model, None, "cpu", micro.block_pool)  # type: ignore[arg-type]
    prefill = _RecordingGraph(accept=True)
    runner._prefill_graph = prefill  # type: ignore[assignment]
    assert runner.prefill_graph is prefill

    out = runner.forward_step_packed(cache, [5, 7, 8], [0, 2, 3], [6, 3])
    assert prefill.calls == [([5, 7, 8], [0, 2, 3], [6, 3])] and out.shape == (1, 3, 7)

    # A decode-only step goes to the decode graph when there is one.
    decode = _RecordingGraph(accept=True)
    runner._decode_graph = decode  # type: ignore[assignment]
    runner.forward_step_packed(cache, [5, 7], [0, 1, 2], [8, 4])
    assert len(decode.calls) == 1 and len(prefill.calls) == 1

    # Device-side ids and hidden-state taps stay off the prefill graph.
    runner._decode_graph = None
    eager = runner.forward_step_packed(
        cache, DeviceTokens(torch.tensor([5, 7]), [0, 1]), [0, 1, 2], [9, 5]
    )
    assert len(prefill.calls) == 1 and eager.shape[1] == 2
    sink: dict[int, torch.Tensor] = {}
    tapped = runner.forward_step_packed(
        cache, [3, 4, 6], [0, 2, 3], [10, 6], tap_layers=frozenset({1}), hidden_state_sink=sink
    )
    assert len(prefill.calls) == 1 and tapped.shape[1] == 3 and set(sink) == {1}
