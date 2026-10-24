"""The fused partial-RoPE kernel against the PyTorch path of `apply_partial_rope_last_n_dims`.

On CUDA the two agree bit for bit. Under the Triton interpreter on the CPU,
fp32 outputs agree bit for bit too, but bf16 outputs differ by a few bf16
steps: the interpreter truncates fp32 to bf16 instead of rounding it.
"""

from __future__ import annotations

import os

import pytest
import torch

from mini_infer.models.blocks import partial_rope_kernel, rope
from mini_infer.models.blocks.partial_rope_kernel import (
    broadcast_rope_tables,
    partial_rope_triton,
    supports_partial_rope_kernel,
)
from mini_infer.models.blocks.rope import _partial_rope_torch, apply_partial_rope_last_n_dims

needs_triton = pytest.mark.skipif(not partial_rope_kernel._TRITON_AVAILABLE, reason="needs triton")


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _tables(
    batch: int, tokens: int, rope_dim: int, dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)`, `(batch, tokens, rope_dim)`, halves repeated as in `RotaryEmbedding`."""
    angles = torch.rand(batch, tokens, rope_dim // 2, generator=torch.Generator().manual_seed(1))
    emb = torch.cat([angles, angles], dim=-1) * 6.0
    return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)


def _assert_same(got: torch.Tensor, want: torch.Tensor, device: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape
    if device == "cuda" or want.dtype == torch.float32:
        assert torch.equal(got, want)
    else:
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-6, atol=2**-5)


# (x shape, batch of the tables, rope dim)
CASES = [
    ((2, 3, 64), 2, 16),  # (B, T, D), like a KV row
    ((2, 1, 4, 64), 2, 16),  # (B, T, H, D), like q: the tables broadcast over heads
    ((3, 5, 2, 64), 1, 16),  # one table for the whole batch
    ((2, 3, 16), 2, 16),  # every feature rotates
    ((2, 2, 3, 96), 2, 32),  # a width that is not a power of two
]


@needs_triton
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    ("x_dtype", "table_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(("shape", "table_batch", "rope_dim"), CASES)
def test_kernel_matches_the_torch_path(
    shape: tuple[int, ...],
    table_batch: int,
    rope_dim: int,
    x_dtype: torch.dtype,
    table_dtype: torch.dtype,
    inverse: bool,
) -> None:
    device = _device()
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    x = x.to(device=device, dtype=x_dtype)
    cos, sin = _tables(table_batch, shape[1], rope_dim, table_dtype, device)
    tables = broadcast_rope_tables(x, cos, sin, rope_dim)
    assert tables is not None
    got = partial_rope_triton(x, *tables, rope_dim, inverse=inverse)
    _assert_same(got, _partial_rope_torch(x, cos, sin, rope_dim, inverse), device)


@needs_triton
def test_kernel_reads_strided_inputs() -> None:
    """A slice of wider rows and a transposed tensor, as attention outputs can be."""
    device = _device()
    generator = torch.Generator().manual_seed(2)
    wide = torch.randn(2, 3, 4, 80, generator=generator).to(device)
    transposed = torch.randn(2, 4, 3, 64, generator=generator).to(device).transpose(1, 2)
    cos, sin = _tables(2, 3, 16, torch.float32, device)
    for x in (wide[..., 8:72], transposed):
        assert not x.is_contiguous()
        tables = broadcast_rope_tables(x, cos, sin, 16)
        assert tables is not None
        got = partial_rope_triton(x, *tables, 16, inverse=False)
        _assert_same(got, _partial_rope_torch(x, cos, sin, 16, False), device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_apply_partial_rope_uses_the_kernel_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def spy(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return partial_rope_triton(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(rope, "partial_rope_triton", spy)
    x = torch.randn(2, 1, 4, 512, device="cuda").bfloat16()
    cos, sin = _tables(2, 1, 64, torch.bfloat16, "cuda")
    got = apply_partial_rope_last_n_dims(x, cos, sin, 64, inverse=True)
    assert calls == [1]
    assert torch.equal(got, _partial_rope_torch(x, cos, sin, 64, True))


def test_tables_broadcast_like_the_torch_path() -> None:
    x = torch.zeros(2, 3, 4, 64)
    cos, sin = _tables(2, 3, 16, torch.float32, "cpu")
    tables = broadcast_rope_tables(x, cos, sin, 16)
    assert tables is not None
    cos_half, _ = tables
    assert cos_half.shape == (2, 3, 4, 8)
    assert cos_half.stride(2) == 0  # one table row serves every head
    torch.testing.assert_close(cos_half[:, :, 2], cos[..., :8])
    # A batch of 2 tables does not broadcast to a batch of 3 rows.
    assert broadcast_rope_tables(torch.zeros(3, 3, 4, 64), cos, sin, 16) is None
    # Tables with more dims than x do not either.
    assert broadcast_rope_tables(torch.zeros(3, 64), cos, sin, 16) is None


def test_kernel_takes_only_cuda_tensors_without_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.zeros(2, 3, 64)
    cos, sin = _tables(2, 3, 16, torch.float32, "cpu")
    assert not supports_partial_rope_kernel(x, cos, sin)
    monkeypatch.setattr(partial_rope_kernel, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(partial_rope_kernel, "is_cuda_device", lambda device: True)
    assert supports_partial_rope_kernel(x, cos, sin)
    assert not supports_partial_rope_kernel(x.clone().requires_grad_(), cos, sin)
    assert not supports_partial_rope_kernel(x.double(), cos, sin)
    assert not supports_partial_rope_kernel(x, cos, sin.double())
    assert not supports_partial_rope_kernel(torch.zeros(1, 2, 3, 4, 64), cos, sin)
    with torch.no_grad():
        assert supports_partial_rope_kernel(x.clone().requires_grad_(), cos, sin)
    monkeypatch.setattr(partial_rope_kernel, "_DISABLED_FOR_BENCH", True)
    assert not supports_partial_rope_kernel(x, cos, sin)
