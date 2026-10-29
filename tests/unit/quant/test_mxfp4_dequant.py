"""Tests for the MXFP4 (compressed-tensors `mxfp4-pack-quantized`) -> BF16 dequant.

Kimi K3's expert weights ship in this format. The byte layout (two E2M1
nibbles per byte, low first) is the NVFP4 one already pinned in
`test_nvfp4_dequant.py`; what is new is the scale: a `uint8` E8M0 exponent
with bias 127 per block of 32, so `scale = 2 ** (byte - 127)`.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.quant.mxfp4 import MXFP4_BLOCK_SIZE, dequantize_mxfp4_to_bf16, mx_scale_to_fp32


def _packed_row(nibbles: list[int]) -> torch.Tensor:
    """One row of E2M1 nibble indices -> packed uint8 bytes (low nibble first)."""
    assert len(nibbles) % 2 == 0
    out = [nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2)]
    return torch.tensor([out], dtype=torch.uint8)


def test_mx_scale_bias_127_is_one() -> None:
    scale = torch.tensor([[125, 126, 127, 128, 129]], dtype=torch.uint8)
    assert torch.equal(mx_scale_to_fp32(scale), torch.tensor([[0.25, 0.5, 1.0, 2.0, 4.0]]))


def test_mx_scale_rejects_float_input() -> None:
    with pytest.raises(ValueError, match="E8M0"):
        mx_scale_to_fp32(torch.ones(1, 1))


def test_mx_scale_rejects_e8m0_nan_byte() -> None:
    """Byte 255 is E8M0's NaN; it must fail at load, not become inf weights."""
    scale = torch.tensor([[127, 255]], dtype=torch.uint8)
    with pytest.raises(ValueError, match="NaN"):
        mx_scale_to_fp32(scale)
    with pytest.raises(ValueError, match="NaN"):
        dequantize_mxfp4_to_bf16(torch.zeros(1, 32, dtype=torch.uint8), scale)


def test_mx_scale_rejects_native_e8m0_nan() -> None:
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        pytest.skip("this torch build has no float8_e8m0fnu")
    scale = torch.tensor([[127, 255]], dtype=torch.uint8).view(e8m0)
    with pytest.raises(ValueError, match="NaN"):
        mx_scale_to_fp32(scale)


def test_mx_scale_largest_finite_byte_is_accepted() -> None:
    """Byte 254 (2**127) is the largest finite E8M0 value and stays finite."""
    scale = torch.tensor([[254]], dtype=torch.uint8)
    assert torch.isfinite(mx_scale_to_fp32(scale)).all()


def test_dequant_applies_power_of_two_block_scales() -> None:
    """Two blocks of 32 with scales 2^1 and 2^-1: every nibble value appears,
    scaled by its block's exponent."""
    nibbles = list(range(16)) * 4  # 64 values = 2 blocks of 32
    packed = _packed_row(nibbles)
    scale = torch.tensor([[128, 126]], dtype=torch.uint8)
    out = dequantize_mxfp4_to_bf16(packed, scale)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 64)
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    )
    expected = torch.cat([table.repeat(2) * 2.0, table.repeat(2) * 0.5]).to(torch.bfloat16)
    assert torch.equal(out[0], expected)


def test_dequant_block_size_is_32() -> None:
    assert MXFP4_BLOCK_SIZE == 32
    packed = torch.zeros(3, 16, dtype=torch.uint8)  # 32 values per row
    scale = torch.full((3, 1), 127, dtype=torch.uint8)
    assert dequantize_mxfp4_to_bf16(packed, scale).shape == (3, 32)
    with pytest.raises(ValueError, match="scale shape"):
        dequantize_mxfp4_to_bf16(packed, torch.full((3, 2), 127, dtype=torch.uint8))


def test_dequant_rejects_non_byte_packed_weight() -> None:
    with pytest.raises(ValueError, match="uint8"):
        dequantize_mxfp4_to_bf16(torch.zeros(1, 16), torch.full((1, 1), 127, dtype=torch.uint8))
