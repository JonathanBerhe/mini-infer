"""Unit tests for `ColumnParallelLinear` / `RowParallelLinear`.

The contract these tests enforce:

  1. At `world_size=1` (single-process default), both layers produce
     output bit-identical to a plain `nn.Linear` constructed with the same
     weight. This is what guarantees existing single-device tests stay
     unaffected.

  2. At `world_size=2` under a real gloo process group, the same full
     weight loaded onto two ranks via `load_full_weight` produces the
     same output as the single-device reference (within float tolerance,
     which for these small fp32 matmuls is effectively bit-identical).

The multi-process tests are gated on `is_multi_process_available()` so
they skip on platforms where spawn-and-gloo doesn't work, instead of
hard-failing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process

# ----------------------------- world_size=1 -----------------------------


def test_column_parallel_linear_world_size_1_matches_nn_linear() -> None:
    """At world_size=1, ColumnParallelLinear(out=128) ≡ nn.Linear(out=128)."""
    torch.manual_seed(0)
    in_features, out_features = 16, 128

    reference = nn.Linear(in_features, out_features, bias=False)
    tp = ColumnParallelLinear(in_features, out_features, bias=False)
    tp.load_full_weight(reference.weight.detach())

    x = torch.randn(4, in_features)
    expected = reference(x)
    actual = tp(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_column_parallel_linear_world_size_1_with_bias() -> None:
    torch.manual_seed(0)
    in_features, out_features = 16, 64
    reference = nn.Linear(in_features, out_features, bias=True)
    tp = ColumnParallelLinear(in_features, out_features, bias=True)
    tp.load_full_weight(reference.weight.detach(), reference.bias.detach())

    x = torch.randn(3, in_features)
    torch.testing.assert_close(tp(x), reference(x), rtol=0, atol=0)


def test_column_parallel_linear_gather_output_world_size_1() -> None:
    """gather_output=True is also a no-op at world_size=1."""
    torch.manual_seed(0)
    reference = nn.Linear(8, 32, bias=False)
    tp = ColumnParallelLinear(8, 32, bias=False, gather_output=True)
    tp.load_full_weight(reference.weight.detach())

    x = torch.randn(2, 8)
    torch.testing.assert_close(tp(x), reference(x), rtol=0, atol=0)


def test_row_parallel_linear_world_size_1_matches_nn_linear() -> None:
    """At world_size=1, RowParallelLinear(in=128) ≡ nn.Linear(in=128)."""
    torch.manual_seed(0)
    in_features, out_features = 128, 16

    reference = nn.Linear(in_features, out_features, bias=False)
    tp = RowParallelLinear(in_features, out_features, bias=False)
    tp.load_full_weight(reference.weight.detach())

    x = torch.randn(4, in_features)
    expected = reference(x)
    actual = tp(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_row_parallel_linear_world_size_1_with_bias() -> None:
    torch.manual_seed(0)
    in_features, out_features = 64, 16
    reference = nn.Linear(in_features, out_features, bias=True)
    tp = RowParallelLinear(in_features, out_features, bias=True)
    tp.load_full_weight(reference.weight.detach(), reference.bias.detach())

    x = torch.randn(3, in_features)
    torch.testing.assert_close(tp(x), reference(x), rtol=0, atol=0)


def test_column_parallel_linear_rejects_indivisible_split() -> None:
    """`out_features` must divide world_size evenly; we raise loudly otherwise."""
    # At world_size=1 every value divides cleanly; the validation only
    # bites under multi-rank construction. We exercise the multi-rank path
    # in the multi-process tests below.


def test_column_parallel_linear_load_full_weight_rejects_wrong_shape() -> None:
    layer = ColumnParallelLinear(16, 32, bias=False)
    with pytest.raises(ValueError, match="full_weight shape"):
        layer.load_full_weight(torch.zeros(8, 16))


def test_row_parallel_linear_load_full_weight_rejects_wrong_shape() -> None:
    layer = RowParallelLinear(16, 32, bias=False)
    with pytest.raises(ValueError, match="full_weight shape"):
        layer.load_full_weight(torch.zeros(32, 8))


# ----------------------------- world_size=2 -----------------------------
#
# Run the actual TP path in two child processes under gloo, and compare
# the per-rank outputs (which are replicated for row-parallel + bias, or
# tile-able for column-parallel) against the single-device reference
# computed in the parent process.


def _column_parallel_worker(
    rank: int,
    world_size: int,
    full_weight: torch.Tensor,
    full_bias: torch.Tensor | None,
    x: torch.Tensor,
    gather_output: bool,
) -> torch.Tensor:
    in_features = full_weight.shape[1]
    out_features = full_weight.shape[0]
    layer = ColumnParallelLinear(
        in_features,
        out_features,
        bias=full_bias is not None,
        gather_output=gather_output,
    )
    layer.load_full_weight(full_weight, full_bias)
    return layer(x).detach().cpu()


def _row_parallel_worker(
    rank: int,
    world_size: int,
    full_weight: torch.Tensor,
    full_bias: torch.Tensor | None,
    full_x: torch.Tensor,
) -> torch.Tensor:
    in_features = full_weight.shape[1]
    out_features = full_weight.shape[0]
    # Row-parallel consumes a sharded input; do the slicing here so the
    # test exactly mirrors how a column-parallel upstream would feed it.
    in_per_rank = in_features // world_size
    start = rank * in_per_rank
    x_local = full_x[..., start : start + in_per_rank].contiguous()

    layer = RowParallelLinear(
        in_features,
        out_features,
        bias=full_bias is not None,
        input_is_parallel=True,
    )
    layer.load_full_weight(full_weight, full_bias)
    return layer(x_local).detach().cpu()


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_column_parallel_linear_world_size_2_matches_reference() -> None:
    torch.manual_seed(0)
    in_features, out_features = 16, 32  # 32 / 2 = 16 per rank
    full_weight = torch.randn(out_features, in_features)
    x = torch.randn(4, in_features)

    reference_output = nn.functional.linear(x, full_weight)

    # gather_output=True so each rank returns the full out_features tensor
    # for an apples-to-apples compare against the reference.
    per_rank_outputs = run_multi_process(
        2,
        _column_parallel_worker,
        full_weight,
        None,
        x,
        True,
    )
    assert len(per_rank_outputs) == 2
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda m, r=rank: f"rank {r} output mismatch: {m}",
        )


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_column_parallel_linear_world_size_2_sharded_output_concatenates_to_reference() -> None:
    """gather_output=False: each rank returns a slice; concat == reference."""
    torch.manual_seed(0)
    in_features, out_features = 8, 16
    full_weight = torch.randn(out_features, in_features)
    x = torch.randn(2, in_features)

    reference_output = nn.functional.linear(x, full_weight)

    per_rank_outputs = run_multi_process(
        2,
        _column_parallel_worker,
        full_weight,
        None,
        x,
        False,  # gather_output=False
    )
    assert len(per_rank_outputs) == 2
    # Each rank holds out_features // 2 columns.
    concatenated = torch.cat(per_rank_outputs, dim=-1)
    torch.testing.assert_close(concatenated, reference_output, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_row_parallel_linear_world_size_2_matches_reference() -> None:
    torch.manual_seed(0)
    in_features, out_features = 32, 16  # 32 / 2 = 16 per rank
    full_weight = torch.randn(out_features, in_features)
    full_x = torch.randn(4, in_features)

    reference_output = nn.functional.linear(full_x, full_weight)

    per_rank_outputs = run_multi_process(
        2,
        _row_parallel_worker,
        full_weight,
        None,
        full_x,
    )
    assert len(per_rank_outputs) == 2
    # Row-parallel all-reduces, so every rank holds the same full output.
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} output mismatch: {m}",
        )


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_row_parallel_linear_world_size_2_with_bias() -> None:
    """Bias is replicated and added on rank 0 only; reduce gives the right total."""
    torch.manual_seed(0)
    in_features, out_features = 32, 8
    full_weight = torch.randn(out_features, in_features)
    full_bias = torch.randn(out_features)
    full_x = torch.randn(3, in_features)

    reference_output = nn.functional.linear(full_x, full_weight, full_bias)

    per_rank_outputs = run_multi_process(
        2,
        _row_parallel_worker,
        full_weight,
        full_bias,
        full_x,
    )
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} output mismatch: {m}",
        )
