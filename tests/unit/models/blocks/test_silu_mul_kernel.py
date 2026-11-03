"""The fused `silu(gate) * up` kernel against the two PyTorch ops.

On CUDA the kernel repeats PyTorch's arithmetic and the outputs are equal.
Under the Triton interpreter on the CPU the kernel uses `tl.exp` and the
interpreter truncates fp32 to bf16, so the tests there bound the difference.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.nn import functional

from mini_infer.models.blocks import activations, silu_mul_kernel
from mini_infer.models.blocks.activations import swiglu
from mini_infer.models.blocks.silu_mul_kernel import fused_silu_mul, silu_mul_rows

needs_triton = pytest.mark.skipif(not silu_mul_kernel._TRITON_AVAILABLE, reason="needs triton")


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _check(got: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, device: str) -> None:
    want = functional.silu(gate) * up
    assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
    if device == "cuda":
        assert torch.equal(got, want)
    else:
        step = 2**-6 if gate.dtype in (torch.bfloat16, torch.float16) else 2**-18
        torch.testing.assert_close(got.float(), want.float(), rtol=step, atol=step)


@needs_triton
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((8, 18944), torch.bfloat16),  # the MLP of Qwen2.5-7B at batch 8
        ((3, 1, 1000), torch.bfloat16),  # a width that is not a multiple of the block
        ((4, 2048), torch.float16),
        ((5, 768), torch.float32),
    ],
)
def test_kernel_matches_silu_times_up(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    device = _device()
    generator = torch.Generator().manual_seed(0)
    gate = (torch.randn(*shape, generator=generator) * 4).to(device=device, dtype=dtype)
    up = (torch.randn(*shape, generator=generator) * 4).to(device=device, dtype=dtype)
    got = fused_silu_mul(gate.view(-1, shape[-1]), up.view(-1, shape[-1]), gate.shape)
    _check(got, gate, up, device)


@needs_triton
def test_kernel_reads_the_halves_of_one_tensor() -> None:
    """`chunk(2, -1)` of a merged gate-up output gives strided rows."""
    device = _device()
    merged = torch.randn(6, 2 * 384, generator=torch.Generator().manual_seed(1)).to(device)
    gate, up = merged.bfloat16().chunk(2, dim=-1)
    assert not gate.is_contiguous()
    _check(fused_silu_mul(gate.view(-1, 384), up.view(-1, 384), gate.shape), gate, up, device)


@needs_triton
def test_kernel_keeps_large_and_tiny_gates_finite() -> None:
    """exp(-x) overflows for large negative x; SiLU is then 0, not NaN."""
    device = _device()
    gate = torch.tensor([[-200.0, -90.0, -1e-30, 0.0, 1e-30, 90.0, 200.0, 3e38]]).to(device)
    up = torch.ones_like(gate)
    _check(fused_silu_mul(gate, up, gate.shape), gate, up, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_swiglu_takes_the_kernel_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def spy(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return fused_silu_mul(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(activations, "fused_silu_mul", spy)
    gate = torch.randn(2, 4096, device="cuda").bfloat16()
    up = torch.randn(2, 4096, device="cuda").bfloat16()
    assert torch.equal(swiglu(gate, up), functional.silu(gate) * up)
    swiglu(gate, up.float())  # dtypes differ: the PyTorch ops
    assert calls == [1]


def test_rows_need_cuda_tensors_the_kernel_can_index(monkeypatch: pytest.MonkeyPatch) -> None:
    gate, up = torch.zeros(4, 3, 64), torch.zeros(4, 3, 64)
    assert silu_mul_rows(gate, up) is None  # CPU
    monkeypatch.setattr(silu_mul_kernel, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(silu_mul_kernel, "libdevice", object())
    monkeypatch.setattr(silu_mul_kernel, "is_cuda_device", lambda device: True)
    rows = silu_mul_rows(gate, up)
    assert rows is not None and rows[0].shape == (12, 64) and rows[1].shape == (12, 64)
    assert silu_mul_rows(gate, torch.zeros(4, 3, 32)) is None  # shapes differ
    assert silu_mul_rows(gate, up.bfloat16()) is None  # dtypes differ
    assert silu_mul_rows(gate.double(), up.double()) is None
    assert silu_mul_rows(gate.transpose(1, 2), up.transpose(1, 2)) is None  # features strided
    assert silu_mul_rows(torch.zeros(4, 6, 64)[:, :3], up) is None  # rows no 2D view describes
    needs_grad = torch.zeros(4, 3, 64, requires_grad=True)
    assert silu_mul_rows(needs_grad, up) is None
    with torch.no_grad():
        assert silu_mul_rows(needs_grad, up) is not None


def test_swiglu_keeps_the_torch_path_off_cuda() -> None:
    gate, up = torch.randn(3, 64), torch.randn(3, 64)
    assert torch.equal(swiglu(gate, up), functional.silu(gate) * up)
