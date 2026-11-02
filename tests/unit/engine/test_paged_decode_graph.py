"""The paged decode graph without a GPU.

Its gate, its bucket choice, the runner's routing, and a replay whose "captured"
step runs the eager forward on the static buffers and must match the eager step
bit for bit. The capture itself and the padded buckets run on CUDA only.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache, StaticDecodeStep
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.paged_decode_graph import PagedDecodeGraph, paged_decode_graph_blocker
from tests.unit._dspark_doubles import micro_setup


class _EagerStandIn:
    """Stands in for a captured graph: `replay` runs the forward on the static buffers."""

    def __init__(self, graph: PagedDecodeGraph, batch: int) -> None:
        self.graph = graph
        self.batch = batch
        self.cache: PagedKVCache | None = None

    def replay(self) -> None:
        assert self.cache is not None and self.cache.static_step is not None
        self.graph._logits[self.batch] = self.graph._forward(self.batch, self.cache)


def _fake_graph(model: Any, pool: BlockPool, buckets: tuple[int, ...]) -> PagedDecodeGraph:
    """A graph object on the CPU whose steps run eagerly instead of replaying."""
    graph = PagedDecodeGraph.__new__(PagedDecodeGraph)
    graph._model = model
    graph._pool = pool
    graph._device = torch.device("cpu")
    graph._buckets = tuple(sorted(buckets))
    graph._scratch_block = pool.allocate()
    graph._steps = {}
    graph._inputs = {}
    graph._inputs_copied = {}
    graph._graphs = {}
    graph._logits = {}
    for batch in graph._buckets:
        graph._steps[batch] = StaticDecodeStep.allocate(
            batch, graph._scratch_block, pool.num_blocks, graph._device
        )
        graph._inputs[batch] = (
            torch.zeros(1, batch, dtype=torch.long),
            torch.zeros(1, batch, dtype=torch.long),
            torch.zeros(2, batch, dtype=torch.long),
        )
        graph._graphs[batch] = _EagerStandIn(graph, batch)
    return graph


def _prefilled(seed: int) -> tuple[Any, PagedKVCache]:
    runner, _ = micro_setup(seed=seed)
    cache = PagedKVCache(runner.block_pool)
    cache.add_request_slot()
    cache.add_request_slot()
    runner.forward_step_packed(cache, list(range(10, 19)), [0, 6, 9], [0, 0])
    return runner, cache


def test_blocker_names_what_is_missing() -> None:
    runner, _ = micro_setup()
    pool = runner.block_pool
    assert paged_decode_graph_blocker(pool, "cpu") == "the decode graph needs a CUDA device"
    # The micro pool uses the torch attention backend.
    assert "FlashInfer" in str(paged_decode_graph_blocker(pool, "cuda"))
    with pytest.raises(ValueError, match="CUDA device"):
        PagedDecodeGraph(runner._model, pool, "cpu")


def test_accepts_decode_only_steps_up_to_the_largest_bucket() -> None:
    runner, _ = micro_setup()
    graph = _fake_graph(runner._model, runner.block_pool, (1, 2, 4))
    assert graph.accepts([0, 1, 2])
    assert graph.accepts([0, 1, 2, 3, 4])
    assert not graph.accepts([0, 2])  # a two-token row is not a decode row
    assert not graph.accepts([0])
    assert not graph.accepts([0, 1, 2, 3, 4, 5])  # five rows exceed the largest bucket
    assert graph.bucket(3) == 4 and graph.bucket(1) == 1
    with pytest.raises(ValueError):
        graph.bucket(5)


def test_replay_matches_the_eager_decode_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same prompts, same decode tokens: the static step and the eager step agree bit for bit.

    The plan before a replay runs in inference mode, because the FlashInfer
    buffers it writes are born inside a forward and are inference tensors.
    """
    import mini_infer.engine.paged_decode_graph as module

    plans_in_inference_mode: list[bool] = []
    real_plan = module.plan_static_decode

    def checking_plan(cache: PagedKVCache, device: torch.device) -> None:
        plans_in_inference_mode.append(torch.is_inference_mode_enabled())
        real_plan(cache, device)

    monkeypatch.setattr(module, "plan_static_decode", checking_plan)
    eager_runner, eager_cache = _prefilled(seed=3)
    graph_runner, graph_cache = _prefilled(seed=3)
    graph = _fake_graph(graph_runner._model, graph_runner.block_pool, (2,))
    stand_in = graph._graphs[2]
    assert isinstance(stand_in, _EagerStandIn)
    stand_in.cache = graph_cache
    tokens, offsets = [5, 7], [6, 3]
    assert not torch.is_inference_mode_enabled()
    for _ in range(3):
        expected = eager_runner.forward_step_packed(eager_cache, tokens, [0, 1, 2], offsets)
        got = graph.replay(graph_cache, tokens, offsets)
        assert torch.equal(got, expected)
        assert graph_cache.static_step is None
        assert graph_cache.seq_lens_list() == eager_cache.seq_lens_list()
        offsets = [offset + 1 for offset in offsets]
    assert plans_in_inference_mode == [True, True, True]


def test_replay_with_device_ids_matches_the_replay_with_host_ids() -> None:
    """The ids of a `DeviceTokens` land in the input buffer like the host ids do."""
    from mini_infer.engine.device_tokens import DeviceTokens

    host_runner, host_cache = _prefilled(seed=6)
    device_runner, device_cache = _prefilled(seed=6)
    host_graph = _fake_graph(host_runner._model, host_runner.block_pool, (2,))
    device_graph = _fake_graph(device_runner._model, device_runner.block_pool, (2,))
    for graph, cache in ((host_graph, host_cache), (device_graph, device_cache)):
        stand_in = graph._graphs[2]
        assert isinstance(stand_in, _EagerStandIn)
        stand_in.cache = cache
    sampled = torch.tensor([4, 7, 5])  # slot 0 takes row 2, slot 1 takes row 1
    expected = host_graph.replay(host_cache, [5, 7], [6, 3])
    got = device_graph.replay(device_cache, DeviceTokens(sampled, [2, 1]), [6, 3])
    assert torch.equal(got, expected)
    assert torch.equal(device_graph._inputs[2][0], host_graph._inputs[2][0])


def test_padded_replay_needs_flashinfer_on_this_pool() -> None:
    runner, cache = _prefilled(seed=0)
    graph = _fake_graph(runner._model, runner.block_pool, (4,))
    stand_in = graph._graphs[4]
    assert isinstance(stand_in, _EagerStandIn)
    stand_in.cache = cache
    with pytest.raises(RuntimeError, match="padded static decode step"):
        graph.replay(cache, [5, 7], [6, 3])
    assert cache.static_step is None  # the step closed on the way out


class _RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int]]] = []

    def accepts(self, cu_seqlens_q: list[int]) -> bool:
        batch = len(cu_seqlens_q) - 1
        return batch <= 2 and all(cu_seqlens_q[i + 1] - cu_seqlens_q[i] == 1 for i in range(batch))

    def replay(
        self, cache: PagedKVCache, token_ids: list[int], positions: list[int]
    ) -> torch.Tensor:
        self.calls.append((token_ids, positions))
        return torch.zeros(1, len(token_ids), 7)


def test_runner_routes_decode_only_steps_to_the_graph() -> None:
    micro, cache = _prefilled(seed=1)
    runner = ModelRunner(micro._model, None, "cpu", micro.block_pool)  # type: ignore[arg-type]
    fake = _RecordingGraph()
    runner._decode_graph = fake  # type: ignore[assignment]
    assert runner.decode_graph is fake

    out = runner.forward_step_packed(cache, [5, 7], [0, 1, 2], [6, 3])
    assert fake.calls == [([5, 7], [6, 3])] and out.shape == (1, 2, 7)

    # A step with a two-token row stays eager.
    eager = runner.forward_step_packed(cache, [5, 7, 8], [0, 2, 3], [6, 3])
    assert len(fake.calls) == 1 and eager.shape[1] == 3
