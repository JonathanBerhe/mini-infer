"""The fused RMS-norm kernel against the PyTorch path of `RMSNorm` and `head_rms_norm`.

The kernel sums the squares in another order than PyTorch, so it is not
bit-identical: a few outputs differ in the last bit. The tests bound that
difference. Under the Triton interpreter on the CPU the bound is looser,
because the interpreter truncates fp32 to bf16 instead of rounding it.
"""

from __future__ import annotations

import os

import pytest
import torch

from mini_infer.models.blocks import rms_norm_kernel, rmsnorm
from mini_infer.models.blocks.rms_norm_kernel import fused_rms_norm, rms_norm_rows
from mini_infer.models.blocks.rmsnorm import RMSNorm, head_rms_norm

needs_triton = pytest.mark.skipif(not rms_norm_kernel._TRITON_AVAILABLE, reason="needs triton")


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _assert_near(got: torch.Tensor, want: torch.Tensor, x: torch.Tensor, device: str) -> None:
    """Within a step or two of x's dtype; on CUDA, mostly equal for 16-bit x.

    Both paths round `x * r` to x's dtype before the weight, so x's dtype sets
    the step. With fp32 x a last-bit change of the scale moves the whole row,
    so only 16-bit inputs count their unequal values.
    """
    assert got.dtype == want.dtype and got.shape == want.shape
    low = x.dtype in (torch.bfloat16, torch.float16)
    step = 2**-6 if device == "cpu" and low else 2**-7 if low else 2**-20
    torch.testing.assert_close(got.float(), want.float(), rtol=step, atol=step)
    if device == "cuda" and low:
        assert (got != want).float().mean() < 0.02


# (x shape, x dtype, weight dtype or None)
NORM_CASES = [
    ((3, 4096), torch.bfloat16, torch.bfloat16),  # a decoder norm over the hidden size
    ((16, 1, 1024), torch.bfloat16, torch.bfloat16),  # q_a_layernorm at batch 16
    ((2, 1, 512), torch.bfloat16, None),  # no learnable weight
    ((5, 512), torch.float32, torch.float32),
    ((4, 512), torch.bfloat16, torch.float32),  # an fp32 weight promotes the output
    ((7, 96), torch.bfloat16, torch.bfloat16),  # a width that is not a power of two
]


@needs_triton
@pytest.mark.parametrize(("shape", "x_dtype", "w_dtype"), NORM_CASES)
def test_kernel_stays_near_rmsnorm(
    shape: tuple[int, ...], x_dtype: torch.dtype, w_dtype: torch.dtype | None
) -> None:
    device = _device()
    generator = torch.Generator().manual_seed(0)
    x = (torch.randn(*shape, generator=generator) * 3).to(device=device, dtype=x_dtype)
    norm = RMSNorm(shape[-1], eps=1e-6, with_scale=w_dtype is not None)
    if w_dtype is not None:
        with torch.no_grad():
            norm.weight.copy_(torch.rand(shape[-1], generator=generator) + 0.5)
        norm.to(device=device, dtype=w_dtype)
    with torch.no_grad():
        want = norm(x)
        weight = norm.weight if w_dtype is not None else None
        got = fused_rms_norm(x.view(-1, shape[-1]), x.shape, weight, 1e-6)
    _assert_near(got, want, x, device)


@needs_triton
@pytest.mark.parametrize("shape", [(2, 1, 32, 512), (16, 1, 8, 128)])
def test_kernel_stays_near_the_head_norm(shape: tuple[int, ...]) -> None:
    """The q-norm casts the scale to bf16 before the multiply."""
    device = _device()
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(1)).to(device).bfloat16()
    want = head_rms_norm(x, 1e-6)
    got = fused_rms_norm(x.view(-1, shape[-1]), x.shape, None, 1e-6, scale_first=True)
    _assert_near(got, want, x, device)


@needs_triton
def test_kernel_reads_strided_rows() -> None:
    device = _device()
    wide = torch.randn(6, 576, generator=torch.Generator().manual_seed(2)).to(device).bfloat16()
    x = wide[:, :512]
    assert not x.is_contiguous()
    got = fused_rms_norm(x.view(-1, 512), x.shape, None, 1e-6)
    _assert_near(got, RMSNorm(512, with_scale=False)(x), x, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_rmsnorm_takes_the_kernel_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def spy(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return fused_rms_norm(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(rmsnorm, "fused_rms_norm", spy)
    norm = RMSNorm(4096, fused=True).cuda().bfloat16()
    x = torch.randn(2, 4096, device="cuda").bfloat16()
    with torch.no_grad():
        _assert_near(norm(x), RMSNorm(4096).cuda().bfloat16()(x), x, "cuda")
        _assert_near(head_rms_norm(x, 1e-6, fused=True), head_rms_norm(x, 1e-6), x, "cuda")
    assert calls == [1, 1]


def test_rows_need_a_cuda_tensor_the_kernel_can_index(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.zeros(4, 3, 64)
    assert rms_norm_rows(x) is None  # CPU
    monkeypatch.setattr(rms_norm_kernel, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(rms_norm_kernel, "is_cuda_device", lambda device: True)
    rows = rms_norm_rows(x)
    assert rows is not None and rows.shape == (12, 64)
    assert rms_norm_rows(x, torch.ones(64)) is not None
    assert rms_norm_rows(x, torch.ones(32)) is None  # weight of another width
    assert rms_norm_rows(x.double()) is None
    assert rms_norm_rows(x.transpose(1, 2)) is None  # features not contiguous
    assert rms_norm_rows(torch.zeros(4, 6, 64)[:, :3]) is None  # rows no 2D view describes
    assert rms_norm_rows(torch.zeros(2, 20000)) is None  # wider than a row in registers
    needs_grad = torch.zeros(2, 64, requires_grad=True)
    assert rms_norm_rows(needs_grad) is None
    with torch.no_grad():
        assert rms_norm_rows(needs_grad) is not None


def test_fused_rmsnorm_keeps_the_torch_path_off_cuda() -> None:
    x = torch.randn(3, 64)
    plain, fused = RMSNorm(64), RMSNorm(64, fused=True)
    assert not plain.fused
    assert torch.equal(fused(x), plain(x))
    assert torch.equal(head_rms_norm(x, 1e-6, fused=True), head_rms_norm(x, 1e-6))
