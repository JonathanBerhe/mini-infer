"""Parity tests for the fused HC Sinkhorn Triton kernel.

All tests in this file run only when ``supports_hc_kernel(device, hc)``
reports True (CUDA + Triton + power-of-2 hc). On CPU / MPS / non-CUDA
runners they're skipped.

The PyTorch reference (``_hc_split_sinkhorn_torch``) is the numerical
oracle. The Triton output is checked for:

- Shape parity.
- Cosine similarity > 0.999 vs the reference on FP32 inputs.
- Absolute tolerance on ``pre`` / ``post`` (small values; cos-sim alone
  is too loose).
- Doubly-stochastic invariant on ``comb`` (row + column sums ~= 1
  modulo the additive ``eps``).
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.blocks.hc_sinkhorn_kernel import (
    hc_split_sinkhorn_triton,
    supports_hc_kernel,
)
from mini_infer.models.blocks.hyper_connections import _hc_split_sinkhorn_torch


def _device_for_test() -> torch.device | None:
    """Return a CUDA device if the Triton kernel can run; else None."""
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda", 0)
    # hc=4 (V4's value) is the canonical "is the kernel even runnable here"
    # check; if it fails, every test in this module should skip.
    if not supports_hc_kernel(device, hc_mult=4):
        return None
    return device


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).to(torch.float64)
    b_flat = b.reshape(-1).to(torch.float64)
    return float(
        torch.dot(a_flat, b_flat)
        / (torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat))
    )


@pytest.mark.gpu
@pytest.mark.parametrize("hc_mult", [2, 4, 8])
@pytest.mark.parametrize("seqlen", [1, 16, 256])
def test_kernel_matches_torch_reference(hc_mult: int, seqlen: int) -> None:
    device = _device_for_test()
    if device is None:
        pytest.skip("CUDA + Triton + power-of-2 hc required for the HC kernel")

    torch.manual_seed(0)
    batch = 2
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(batch, seqlen, mix_hc, dtype=torch.float32, device=device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device=device)

    pre_t, post_t, comb_t = hc_split_sinkhorn_triton(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=20, eps=1e-6
    )
    pre_ref, post_ref, comb_ref = _hc_split_sinkhorn_torch(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=20, eps=1e-6
    )

    assert pre_t.shape == pre_ref.shape
    assert post_t.shape == post_ref.shape
    assert comb_t.shape == comb_ref.shape

    # Cosine sim > 0.999 across each output. Same bar as paged_attention /
    # int8_kernel parity tests in this repo.
    assert _cos_sim(pre_t, pre_ref) > 0.999
    assert _cos_sim(post_t, post_ref) > 0.999
    assert _cos_sim(comb_t, comb_ref) > 0.999

    # `pre` and `post` are small in absolute value (sigmoid outputs in
    # [0, 1] / [0, 2] range). Cosine sim alone can hide a ~1e-3 systematic
    # shift, so tighten with an elementwise tolerance.
    torch.testing.assert_close(pre_t, pre_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(post_t, post_ref, rtol=1e-4, atol=1e-5)

    # `comb` is more sensitive (20 alternating normalizations compound
    # any reduction-order divergence). Allow a slightly looser bound.
    torch.testing.assert_close(comb_t, comb_ref, rtol=1e-3, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("hc_mult", [2, 4, 8])
def test_kernel_comb_is_doubly_stochastic(hc_mult: int) -> None:
    """Same invariant + same input tempering + same tolerance as the CPU
    oracle test (test_hyper_connections.py): raw randn scales produce
    near-degenerate softmax rows that converge slowly in 20 iterations,
    so the inputs are scaled down and the tolerance is 5e-3. Kernel
    correctness against the oracle is covered elementwise by
    test_kernel_matches_torch_reference on harsh inputs."""
    device = _device_for_test()
    if device is None:
        pytest.skip("CUDA + Triton + power-of-2 hc required for the HC kernel")

    torch.manual_seed(1)
    batch, seqlen = 2, 16
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(batch, seqlen, mix_hc, dtype=torch.float32, device=device) * 0.5
    hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device=device) * 0.1

    _, _, comb = hc_split_sinkhorn_triton(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=20, eps=1e-6
    )

    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)
    tolerance = 5e-3
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=tolerance), (
        f"row sums diverge from 1 by max {(row_sums - 1).abs().max().item():.4e}"
    )
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=tolerance), (
        f"col sums diverge from 1 by max {(col_sums - 1).abs().max().item():.4e}"
    )


@pytest.mark.gpu
def test_kernel_with_sinkhorn_iters_1() -> None:
    """Single iteration: kernel still produces the same numerics as the
    reference (the loop-after-first-iter body just never runs)."""
    device = _device_for_test()
    if device is None:
        pytest.skip("CUDA + Triton + power-of-2 hc required for the HC kernel")

    torch.manual_seed(2)
    hc_mult = 4
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(1, 8, mix_hc, dtype=torch.float32, device=device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device=device)

    pre_t, post_t, comb_t = hc_split_sinkhorn_triton(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=1, eps=1e-6
    )
    pre_ref, post_ref, comb_ref = _hc_split_sinkhorn_torch(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=1, eps=1e-6
    )

    torch.testing.assert_close(pre_t, pre_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(post_t, post_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb_t, comb_ref, rtol=1e-3, atol=1e-4)


@pytest.mark.gpu
def test_kernel_rejects_non_power_of_2_hc() -> None:
    """hc_mult=3 doesn't fit the power-of-2 contract; kernel must error
    rather than producing wrong output."""
    device = _device_for_test()
    if device is None:
        pytest.skip("CUDA + Triton required for the HC kernel")

    hc_mult = 3
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(1, 4, mix_hc, dtype=torch.float32, device=device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="power of 2"):
        hc_split_sinkhorn_triton(
            mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=20, eps=1e-6
        )


def test_supports_hc_kernel_predicate() -> None:
    """The dispatch predicate behaves correctly across device + shape combos."""
    cpu = torch.device("cpu")
    # CPU never supports the kernel (no Triton on CPU).
    assert not supports_hc_kernel(cpu, hc_mult=4)
    # Negative / zero hc is invalid regardless of device.
    assert not supports_hc_kernel(cpu, hc_mult=0)
    assert not supports_hc_kernel(cpu, hc_mult=-1)
    # Non-power-of-2 hc on any device falls back to PyTorch.
    if torch.cuda.is_available():
        cuda = torch.device("cuda", 0)
        assert not supports_hc_kernel(cuda, hc_mult=3)
        assert not supports_hc_kernel(cuda, hc_mult=5)
