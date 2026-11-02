"""`QKVParallelLinear`: one column-parallel layer for the q, k and v projections.

The multi-process TP parity test (`test_attention_tp_parity.py`) covers the
sharded forward; these tests pin the layout of the merged weight, the two
projection paths of `project` and the per-rank slicing without processes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import mini_infer.distributed.linear as linear_module
from mini_infer.distributed.linear import QKVParallelLinear


def _reference_linears(layer: QKVParallelLinear) -> list[nn.Linear]:
    """One plain `nn.Linear` per projection, with the rows of the merged weight."""
    refs = []
    start = 0
    for size in layer.split_sizes:
        ref = nn.Linear(layer.in_features, size, bias=layer.bias is not None)
        with torch.no_grad():
            ref.weight.copy_(layer.weight[start : start + size])
            if layer.bias is not None:
                ref.bias.copy_(layer.bias[start : start + size])
        refs.append(ref)
        start += size
    return refs


def test_project_off_cuda_equals_three_separate_layers_bit_for_bit() -> None:
    torch.manual_seed(0)
    layer = QKVParallelLinear(32, num_q_heads=4, num_kv_heads=2, head_dim=8, bias=True)
    assert layer.split_sizes == (32, 16, 16)
    assert layer.weight.shape == (64, 32)
    refs = _reference_linears(layer)
    x = torch.randn(1, 5, 32)

    q, k, v = layer.project(x)
    assert v is not None
    for got, ref in zip((q, k, v), refs, strict=True):
        assert torch.equal(got, ref(x))
    # Each output is its own contiguous tensor, as the separate layers gave.
    assert q.stride(-2) == 32 and k.stride(-2) == 16


def test_project_on_cuda_runs_one_gemm_and_splits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the device seen as CUDA, `project` splits the merged output into views;
    the values agree with the three-GEMM path to rounding, and the bench hook
    brings the three-GEMM path back."""
    torch.manual_seed(0)
    layer = QKVParallelLinear(32, num_q_heads=4, num_kv_heads=2, head_dim=8, bias=True)
    x = torch.randn(1, 5, 32)
    q3, k3, v3 = layer.project(x)

    monkeypatch.setattr(linear_module, "is_cuda_device", lambda device: True)
    q1, k1, v1 = layer.project(x)
    assert v1 is not None and v3 is not None
    # Views into the (1, 5, 64) merged output: the row stride is the merged width.
    assert q1.stride(-2) == 64 and k1.stride(-2) == 64 and v1.stride(-2) == 64
    assert q1.stride(-1) == 1
    # The packed `(1, tokens, heads, head_dim)` view needs no copy.
    assert q1.view(1, 5, 4, 8).shape == (1, 5, 4, 8)
    for one, three in ((q1, q3), (k1, k3), (v1, v3)):
        torch.testing.assert_close(one, three)

    monkeypatch.setattr(linear_module, "_SEPARATE_PROJECTIONS_FOR_BENCH", True)
    q_bench, _, _ = layer.project(x)
    assert q_bench.stride(-2) == 32
    assert torch.equal(q_bench, q3)


def test_without_v_the_layer_has_q_and_k_rows_only() -> None:
    layer = QKVParallelLinear(32, num_q_heads=4, num_kv_heads=2, head_dim=8, with_v=False)
    assert layer.split_sizes == (32, 16)
    assert layer.weight.shape == (48, 32)
    q, k, v = layer.project(torch.randn(2, 32))
    assert v is None
    assert (q.shape, k.shape) == ((2, 32), (2, 16))


def test_load_full_weight_keeps_this_ranks_heads_of_each_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 1 of 2 holds the second half of the q rows, then the second half of
    the k rows, then the second half of the v rows, for the weight and the bias."""
    monkeypatch.setattr(linear_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(linear_module, "get_rank", lambda: 1)
    layer = QKVParallelLinear(32, num_q_heads=4, num_kv_heads=2, head_dim=8, bias=True)
    assert layer.weight.shape == (32, 32)
    assert layer.split_sizes == (16, 8, 8)

    torch.manual_seed(0)
    q_w, k_w, v_w = torch.randn(32, 32), torch.randn(16, 32), torch.randn(16, 32)
    q_b, k_b, v_b = torch.randn(32), torch.randn(16), torch.randn(16)
    layer.load_full_weight(torch.cat([q_w, k_w, v_w]), torch.cat([q_b, k_b, v_b]))
    assert torch.equal(layer.weight, torch.cat([q_w[16:], k_w[8:], v_w[8:]]))
    assert torch.equal(layer.bias, torch.cat([q_b[16:], k_b[8:], v_b[8:]]))

    with pytest.raises(ValueError, match="num_kv_heads=1"):
        QKVParallelLinear(32, num_q_heads=4, num_kv_heads=1, head_dim=8)
