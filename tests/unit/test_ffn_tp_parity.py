"""Multi-process TP parity tests for FFN modules.

Three primitives:
  - `SwiGLU`: column/row pair on (gate, up, down). Standard Megatron MLP.
  - `MoEFFN`: expert-parallel; each rank owns `num_experts // world_size`
    experts. Gate is replicated. Partial routed sums all-reduced.
  - `HashRoutedMoEFFN`: same expert-parallel scheme; gate (including the
    hash routing's `tid2eid` table) is replicated, shared expert is
    replicated and added after the reduce.

Parity contract: at `world_size=2` the per-rank outputs (which are
already replicated for row-parallel / all-reduced for MoE) match the
single-device reference.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process

# ----------------------------- world_size=1 contract -----------------------------


def test_swiglu_constructs_with_tp_linears_at_world_size_1() -> None:
    """SwiGLU's gate/up are ColumnParallelLinear; down is RowParallelLinear.
    At ws=1 they're bit-equivalent to plain nn.Linear so existing tests stay green."""
    from mini_infer.models.blocks.swiglu import SwiGLU

    ffn = SwiGLU(hidden_size=16, intermediate_size=32)
    assert isinstance(ffn.gate_proj, ColumnParallelLinear)
    assert isinstance(ffn.up_proj, ColumnParallelLinear)
    assert isinstance(ffn.down_proj, RowParallelLinear)
    # ws=1: weight shape equals the un-sharded size.
    assert ffn.gate_proj.weight.shape == (32, 16)
    assert ffn.down_proj.weight.shape == (16, 32)


def test_swiglu_world_size_1_matches_plain_nn_linear() -> None:
    """A SwiGLU with copied weights is bit-identical to the manual formula."""
    from mini_infer.models.blocks.swiglu import SwiGLU

    torch.manual_seed(0)
    hidden = 16
    intermediate = 32
    ffn = SwiGLU(hidden, intermediate)

    g = nn.Linear(hidden, intermediate, bias=False)
    u = nn.Linear(hidden, intermediate, bias=False)
    d = nn.Linear(intermediate, hidden, bias=False)
    with torch.no_grad():
        g.weight.copy_(ffn.gate_proj.weight)
        u.weight.copy_(ffn.up_proj.weight)
        d.weight.copy_(ffn.down_proj.weight)

    x = torch.randn(2, 5, hidden)
    expected = d(torch.nn.functional.silu(g(x)) * u(x))
    torch.testing.assert_close(ffn(x), expected, rtol=0, atol=0)


def test_moe_ffn_constructs_with_local_experts_at_world_size_1() -> None:
    """At ws=1, MoEFFN holds all experts locally (num_experts_per_rank == num_experts)."""
    from mini_infer.models.blocks.mixtral_moe import MoEFFN

    moe = MoEFFN(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
    )
    assert moe.num_experts == 4
    assert moe.num_experts_per_rank == 4
    assert len(moe.experts) == 4
    assert moe.local_expert_start == 0
    assert moe.local_expert_end == 4


# ----------------------------- SwiGLU world_size=2 -----------------------------


def _swiglu_worker(
    rank: int,
    world_size: int,
    hidden: int,
    intermediate: int,
    full_gate_weight: torch.Tensor,
    full_up_weight: torch.Tensor,
    full_down_weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    from mini_infer.models.blocks.swiglu import SwiGLU

    ffn = SwiGLU(hidden, intermediate)
    ffn.gate_proj.load_full_weight(full_gate_weight)
    ffn.up_proj.load_full_weight(full_up_weight)
    ffn.down_proj.load_full_weight(full_down_weight)
    return ffn(x).detach().cpu()


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_swiglu_world_size_2_matches_reference() -> None:
    """SwiGLU column->row pairing: per-rank output equals the reference."""
    torch.manual_seed(0)
    hidden = 16
    intermediate = 32  # 32 / 2 = 16 per rank

    full_gate_weight = torch.randn(intermediate, hidden)
    full_up_weight = torch.randn(intermediate, hidden)
    full_down_weight = torch.randn(hidden, intermediate)
    x = torch.randn(2, 5, hidden)

    expected = nn.functional.linear(
        nn.functional.silu(nn.functional.linear(x, full_gate_weight))
        * nn.functional.linear(x, full_up_weight),
        full_down_weight,
    )

    per_rank_outputs = run_multi_process(
        2,
        _swiglu_worker,
        hidden,
        intermediate,
        full_gate_weight,
        full_up_weight,
        full_down_weight,
        x,
    )
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            expected,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} swiglu mismatch: {m}",
        )


# ----------------------------- MoEFFN world_size=2 -----------------------------


def _moe_ffn_worker(
    rank: int,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    top_k: int,
    gate_weight: torch.Tensor,
    per_expert_w1: list[torch.Tensor],
    per_expert_w2: list[torch.Tensor],
    per_expert_w3: list[torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Build a MoEFFN at this rank, load its local-expert slice, run forward.

    The full per-expert weight lists are passed in (all `num_experts` of
    them); each rank picks its own slice via `local_expert_start`.
    """
    from mini_infer.models.blocks.mixtral_moe import MoEFFN

    moe = MoEFFN(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=num_experts,
        top_k=top_k,
    )
    with torch.no_grad():
        moe.gate.weight.copy_(gate_weight)
        for local_idx in range(moe.num_experts_per_rank):
            global_idx = moe.local_expert_start + local_idx
            moe.experts[local_idx].w1.weight.copy_(per_expert_w1[global_idx])
            moe.experts[local_idx].w2.weight.copy_(per_expert_w2[global_idx])
            moe.experts[local_idx].w3.weight.copy_(per_expert_w3[global_idx])
    return moe(x).detach().cpu()


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_moe_ffn_world_size_2_matches_single_device() -> None:
    """Expert-parallel MoEFFN at world_size=2 produces the same output as
    the single-device reference (which holds all experts on rank 0)."""
    torch.manual_seed(0)
    hidden = 16
    intermediate = 24
    num_experts = 4  # 4 / 2 = 2 experts per rank
    top_k = 2

    gate_weight = torch.randn(num_experts, hidden)
    per_expert_w1 = [torch.randn(intermediate, hidden) for _ in range(num_experts)]
    per_expert_w2 = [torch.randn(hidden, intermediate) for _ in range(num_experts)]
    per_expert_w3 = [torch.randn(intermediate, hidden) for _ in range(num_experts)]
    x = torch.randn(1, 6, hidden)

    # Single-device reference: build a ws=1 MoEFFN with the same full weights.
    from mini_infer.models.blocks.mixtral_moe import MoEFFN

    reference = MoEFFN(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=num_experts,
        top_k=top_k,
    )
    with torch.no_grad():
        reference.gate.weight.copy_(gate_weight)
        for j in range(num_experts):
            reference.experts[j].w1.weight.copy_(per_expert_w1[j])
            reference.experts[j].w2.weight.copy_(per_expert_w2[j])
            reference.experts[j].w3.weight.copy_(per_expert_w3[j])
    expected = reference(x).detach()

    per_rank_outputs = run_multi_process(
        2,
        _moe_ffn_worker,
        hidden,
        intermediate,
        num_experts,
        top_k,
        gate_weight,
        per_expert_w1,
        per_expert_w2,
        per_expert_w3,
        x,
    )
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            expected,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda m, r=rank: f"rank {r} MoEFFN mismatch: {m}",
        )


# ----------------------------- HashRoutedMoEFFN world_size=2 -----------------------------


def _hash_routed_moe_ffn_worker(
    rank: int,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_routed_experts: int,
    num_activated_experts: int,
    gate_weight: torch.Tensor,
    per_expert_w1: list[torch.Tensor],
    per_expert_w2: list[torch.Tensor],
    per_expert_w3: list[torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN

    moe = HashRoutedMoEFFN(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_routed_experts=num_routed_experts,
        num_activated_experts=num_activated_experts,
        routing_mode="score_topk",
        score_func="softmax",
        route_scale=1.0,
    )
    with torch.no_grad():
        moe.gate.weight.copy_(gate_weight)
        for local_idx in range(moe.num_experts_per_rank):
            global_idx = moe.local_expert_start + local_idx
            moe.experts[local_idx].w1.weight.copy_(per_expert_w1[global_idx])
            moe.experts[local_idx].w2.weight.copy_(per_expert_w2[global_idx])
            moe.experts[local_idx].w3.weight.copy_(per_expert_w3[global_idx])
    return moe(x).detach().cpu()


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_hash_routed_moe_ffn_world_size_2_matches_single_device() -> None:
    torch.manual_seed(0)
    hidden = 16
    intermediate = 24
    num_routed_experts = 4
    num_activated_experts = 2

    gate_weight = torch.randn(num_routed_experts, hidden)
    per_expert_w1 = [torch.randn(intermediate, hidden) for _ in range(num_routed_experts)]
    per_expert_w2 = [torch.randn(hidden, intermediate) for _ in range(num_routed_experts)]
    per_expert_w3 = [torch.randn(intermediate, hidden) for _ in range(num_routed_experts)]
    x = torch.randn(1, 6, hidden)

    from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN

    reference = HashRoutedMoEFFN(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_routed_experts=num_routed_experts,
        num_activated_experts=num_activated_experts,
        routing_mode="score_topk",
        score_func="softmax",
        route_scale=1.0,
    )
    with torch.no_grad():
        reference.gate.weight.copy_(gate_weight)
        for j in range(num_routed_experts):
            reference.experts[j].w1.weight.copy_(per_expert_w1[j])
            reference.experts[j].w2.weight.copy_(per_expert_w2[j])
            reference.experts[j].w3.weight.copy_(per_expert_w3[j])
    expected = reference(x).detach()

    per_rank_outputs = run_multi_process(
        2,
        _hash_routed_moe_ffn_worker,
        hidden,
        intermediate,
        num_routed_experts,
        num_activated_experts,
        gate_weight,
        per_expert_w1,
        per_expert_w2,
        per_expert_w3,
        x,
    )
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            expected,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda m, r=rank: f"rank {r} HashRoutedMoEFFN mismatch: {m}",
        )
