"""Tests for `FP4Expert`: FP4-resident MoE expert with dequant-per-call.

The parity contract: `FP4Expert.forward(x)` must equal running the SwiGLU
on the *dequantized* weights. We fill the packed buffers with synthetic
NVFP4 + scale, then compare against an explicit dequant-then-SwiGLU
reference over the same buffers; the two share the exact dequant, so they
should match to floating-point exactness.

These are model-free + CPU-only (fast CI lane). The end-to-end V4-Flash
load + forward is the GPU follow-up.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from mini_infer.models.blocks.fp4_expert import FP4Expert
from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16

_SHAPES = {
    "w1": ("intermediate", "hidden"),
    "w2": ("hidden", "intermediate"),
    "w3": ("intermediate", "hidden"),
}


def _fill_synthetic_fp4(expert: FP4Expert, *, seed: int) -> None:
    """Populate the expert's packed/scale buffers with arbitrary FP4 data."""
    gen = torch.Generator().manual_seed(seed)
    dims = {"hidden": expert.hidden_size, "intermediate": expert.intermediate_size}
    for name, (out_key, in_key) in _SHAPES.items():
        out_dim, in_dim = dims[out_key], dims[in_key]
        packed = torch.randint(-128, 128, (out_dim, in_dim // 2), dtype=torch.int8, generator=gen)
        # Positive scales in a reasonable range (e8m0 stores powers of two;
        # any positive fp32 exercises the dequant arithmetic).
        scale = torch.rand(out_dim, in_dim // expert.block_size, generator=gen) + 0.5
        getattr(expert, f"{name}_packed").copy_(packed)
        getattr(expert, f"{name}_scale").copy_(scale)


def test_fp4_expert_matches_explicit_dequant_swiglu() -> None:
    torch.manual_seed(0)
    hidden, intermediate = 64, 128
    expert = FP4Expert(hidden, intermediate)
    _fill_synthetic_fp4(expert, seed=1)

    x = torch.randn(4, hidden, dtype=torch.float32)
    out = expert(x)

    # Reference: dequant the same buffers, run the SwiGLU explicitly.
    w1 = dequantize_nvfp4_to_bf16(expert.w1_packed, expert.w1_scale).float()
    w2 = dequantize_nvfp4_to_bf16(expert.w2_packed, expert.w2_scale).float()
    w3 = dequantize_nvfp4_to_bf16(expert.w3_packed, expert.w3_scale).float()
    ref = functional.linear(
        functional.silu(functional.linear(x, w1)) * functional.linear(x, w3), w2
    )

    assert out.shape == (4, hidden)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_fp4_expert_buffers_are_packed_int8() -> None:
    """Storage is the packed FP4 form (int8, half-width), not BF16."""
    hidden, intermediate = 64, 128
    expert = FP4Expert(hidden, intermediate)

    assert expert.w1_packed.dtype == torch.int8
    assert expert.w1_packed.shape == (intermediate, hidden // 2)  # half width on in dim
    assert expert.w1_scale.shape == (intermediate, hidden // expert.block_size)
    assert expert.w2_packed.shape == (hidden, intermediate // 2)
    assert expert.w3_packed.shape == (intermediate, hidden // 2)

    # No BF16 weight Parameters hiding here: an FP4 expert has zero params.
    assert sum(p.numel() for p in expert.parameters()) == 0


def test_fp4_expert_rejects_in_dim_not_block_aligned() -> None:
    # hidden=48 is not divisible by block_size=32 -> w1/w3 in_dim invalid.
    with pytest.raises(ValueError, match="divisible by block_size"):
        FP4Expert(hidden_size=48, intermediate_size=128)


def test_fp4_expert_dtype_follows_activation() -> None:
    """Dequant casts to the activation dtype so the matmul dtypes match."""
    expert = FP4Expert(64, 128)
    _fill_synthetic_fp4(expert, seed=2)
    out16 = expert(torch.randn(2, 64, dtype=torch.float16))
    assert out16.dtype == torch.float16
