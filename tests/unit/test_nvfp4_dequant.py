"""Tests for the NVFP4 (`float4_e2m1fn_x2`) -> BF16 dequant path.

We can't validate against a real DeepSeek-V4-Flash checkpoint locally —
that needs the Modal smoke. Instead we exercise the dequant against
hand-computed expected outputs from the E2M1 spec, and verify that the
shape/scale arithmetic matches the V4 reference's `cast_e2m1fn_to_e4m3fn`
byte layout.

The tests cover:
  1. The E2M1 lookup table reproduces every representable value when
     each nibble is fed through.
  2. Packing two nibbles into one byte and dequantizing reproduces both
     values at the right positions (low nibble first, high second).
  3. Per-block scaling multiplies the dequantized values by the matching
     scale, with the block size matching V4's `fp4_block_size = 32`.
  4. The result is BF16 with the right shape.
"""

from __future__ import annotations

import math

import pytest
import torch

from mini_infer.quant.nvfp4 import (
    dequantize_block_fp8_to_bf16,
    dequantize_nvfp4_to_bf16,
    is_packed_nvfp4,
)


def _make_scale(out_dim: int, in_dim: int, block_size: int, value: float) -> torch.Tensor:
    """Build a per-block FP32 scale tensor full of `value`.

    The V4 production format stores these as `float8_e8m0fnu`; for parity
    tests we use FP32 so the multiplication is exact and we don't have to
    reason about e8m0's power-of-2 quantization on top of FP4's spec.
    """
    return torch.full((out_dim, in_dim // block_size), value, dtype=torch.float32)


def test_dequant_recovers_every_e2m1_value() -> None:
    """Packing nibble `n` then dequantizing gives `FP4_TABLE[n]` (scale=1)."""
    expected_values = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    # Pack all 16 nibbles into 8 bytes (low nibble = even index, high = odd).
    # Byte i carries nibble (2i) as low and nibble (2i+1) as high.
    packed = torch.tensor(
        [(b * 2 + 1) << 4 | (b * 2) for b in range(8)],
        dtype=torch.uint8,
    ).view(1, 8)
    in_dim = 16
    # Need a scale block size that divides in_dim. The smallest scale tensor
    # is one block per row -> block_size = 16.
    scale = _make_scale(out_dim=1, in_dim=in_dim, block_size=16, value=1.0)
    out = dequantize_nvfp4_to_bf16(packed, scale, block_size=16)
    assert out.shape == (1, in_dim)
    assert out.dtype == torch.bfloat16
    # BF16 represents each of these values exactly (they're all fractions
    # of powers of 2 with small magnitudes).
    expected_tensor = torch.tensor([expected_values], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected_tensor, rtol=0, atol=0)


def test_dequant_low_then_high_nibble_order() -> None:
    """Low nibble of byte i is element 2i; high nibble is element 2i+1."""
    # Byte 0xC3: low = 0x3 (1.5), high = 0xC (-2.0).
    # Byte 0x71: low = 0x1 (0.5), high = 0x7 (6.0).
    packed = torch.tensor([[0xC3, 0x71]], dtype=torch.uint8)
    scale = _make_scale(out_dim=1, in_dim=4, block_size=4, value=1.0)
    out = dequantize_nvfp4_to_bf16(packed, scale, block_size=4).float()
    torch.testing.assert_close(out, torch.tensor([[1.5, -2.0, 0.5, 6.0]]), rtol=0, atol=0)


def test_dequant_applies_per_block_scale() -> None:
    """A non-unit scale multiplies every value in its block; blocks are independent."""
    # 4 nibbles per row, two blocks of 2 nibbles. Scales: [2.0, 0.5].
    # Byte 0x21: low 0x1 (0.5), high 0x2 (1.0).
    # Byte 0x65: low 0x5 (3.0), high 0x6 (4.0).
    # Block 0 (elements 0..1) * 2.0 -> [1.0, 2.0]
    # Block 1 (elements 2..3) * 0.5 -> [1.5, 2.0]
    packed = torch.tensor([[0x21, 0x65]], dtype=torch.uint8)
    scale = torch.tensor([[2.0, 0.5]], dtype=torch.float32)
    out = dequantize_nvfp4_to_bf16(packed, scale, block_size=2).float()
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0, 1.5, 2.0]]), rtol=0, atol=0)


def test_dequant_full_block_size_32() -> None:
    """Production block size is 32: per-row in_dim must divide 32."""
    out_dim, in_dim = 2, 64  # 64 / 2 = 32 bytes per row; 64 / 32 = 2 blocks per row.
    packed = torch.zeros((out_dim, in_dim // 2), dtype=torch.uint8)
    # Sprinkle a few non-zero nibbles so the result isn't all zero.
    packed[0, 0] = 0x12  # row 0, bytes start with low=0x2 (1.0), high=0x1 (0.5)
    packed[1, -1] = 0x60  # row 1, last byte: low=0x0 (0.0), high=0x6 (4.0)
    scale = torch.tensor([[2.0, 1.0], [0.5, 3.0]], dtype=torch.float32)

    out = dequantize_nvfp4_to_bf16(packed, scale, block_size=32).float()
    assert out.shape == (out_dim, in_dim)

    # Row 0, element 0: low(0x2)=1.0, in block 0 (scale=2.0) -> 2.0.
    # Row 0, element 1: high(0x1)=0.5, in block 0 -> 1.0.
    assert out[0, 0].item() == 2.0
    assert out[0, 1].item() == 1.0
    # Row 1, element 62: low(0x0)=0.0, in block 1 (scale=3.0) -> 0.0.
    # Row 1, element 63: high(0x6)=4.0, in block 1 -> 12.0.
    assert out[1, 62].item() == 0.0
    assert out[1, 63].item() == 12.0


def test_dequant_rejects_wrong_rank() -> None:
    packed_3d = torch.zeros((1, 1, 4), dtype=torch.uint8)
    scale = torch.ones((1, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="packed_weight must be 2-D"):
        dequantize_nvfp4_to_bf16(packed_3d, scale)


def test_dequant_rejects_mismatched_scale_shape() -> None:
    packed = torch.zeros((1, 16), dtype=torch.uint8)  # in_dim=32
    wrong_scale = torch.ones((1, 2), dtype=torch.float32)  # expected (1, 1) for block=32
    with pytest.raises(ValueError, match="fp4_scale shape"):
        dequantize_nvfp4_to_bf16(packed, wrong_scale, block_size=32)


def test_is_packed_nvfp4_recognises_int_byte_storage() -> None:
    """`is_packed_nvfp4` accepts int8/uint8 because V4-Flash's published
    safetensors store packed FP4 weights with that dtype (the safetensors
    metadata doesn't carry the FP4 type marker).

    Note: the caller must independently verify a sibling `.scale` companion
    exists before treating an int8 tensor as packed FP4; this predicate
    is intentionally cheap and over-inclusive.
    """
    uint8_tensor = torch.zeros(4, dtype=torch.uint8)
    int8_tensor = torch.zeros(4, dtype=torch.int8)
    assert is_packed_nvfp4(uint8_tensor) is True
    assert is_packed_nvfp4(int8_tensor) is True
    # Non-byte dtypes are not packed FP4.
    bf16_tensor = torch.zeros(4, dtype=torch.bfloat16)
    assert is_packed_nvfp4(bf16_tensor) is False


def test_dequant_handles_negative_scale_block_with_negative_nibble() -> None:
    """Two negatives produce a positive — guards against accidental abs()."""
    # Byte 0x88: low 0x8 (-0.0), high 0x8 (-0.0). Trivial case.
    # Use 0xAA: low 0xA (-1.0), high 0xA (-1.0); scale = -2.0 -> +2.0 each.
    packed = torch.tensor([[0xAA]], dtype=torch.uint8)
    scale = torch.tensor([[-2.0]], dtype=torch.float32)
    out = dequantize_nvfp4_to_bf16(packed, scale, block_size=2).float()
    torch.testing.assert_close(out, torch.tensor([[2.0, 2.0]]), rtol=0, atol=0)


def test_dequant_zero_packed_is_zero_output() -> None:
    """Zero packed bytes with any scale must produce a zero output."""
    packed = torch.zeros((3, 4), dtype=torch.uint8)  # in_dim = 8
    scale = torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=torch.float32)
    out = dequantize_nvfp4_to_bf16(packed, scale, block_size=4)
    assert torch.all(out == 0.0).item()
    assert out.shape == (3, 8)
    assert out.dtype == torch.bfloat16


def test_dequant_matches_v4_reference_byte_layout() -> None:
    """The convert.py reference computes:
        x = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(...)
    Our dequant must produce the same byte ordering."""
    # Reference values reproduced inline to avoid importing `third_party`.
    fp4_table = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    # Random-looking bytes that exercise every nibble at least once.
    bytes_pattern = [0x07, 0x18, 0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E]
    packed = torch.tensor([bytes_pattern], dtype=torch.uint8)
    block_size = 16
    scale = torch.tensor([[1.0]], dtype=torch.float32)

    # Compute expected per the reference recipe.
    expected: list[float] = []
    for byte in bytes_pattern:
        low = byte & 0x0F
        high = (byte >> 4) & 0x0F
        expected.append(fp4_table[low])
        expected.append(fp4_table[high])
    expected_tensor = torch.tensor([expected], dtype=torch.bfloat16)

    out = dequantize_nvfp4_to_bf16(packed, scale, block_size=block_size)
    torch.testing.assert_close(out, expected_tensor, rtol=0, atol=0)
    # Sanity: BF16 represented these exactly (small fractions of 2^k).
    assert not math.isnan(out.sum().item())


# ----------------------------- Block-FP8 dequant -----------------------------


def test_block_fp8_dequant_recovers_scaled_weight_per_block() -> None:
    """A (128, 128)-block FP8 dequant multiplies each block's values by its scale."""
    # Build a small (M=4, N=4) weight + (2x2) scale so the block math is 2x2.
    weight_fp32 = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [4.0, 4.0, 8.0, 8.0],
            [4.0, 4.0, 8.0, 8.0],
        ],
        dtype=torch.float32,
    )
    # FP8 e4m3fn quantization of the above (representable exactly).
    weight_fp8 = weight_fp32.to(torch.float8_e4m3fn)
    # Per-block scale: 4 distinct values for the four 2x2 blocks.
    scale_fp32 = torch.tensor([[0.5, 1.5], [2.0, 0.25]], dtype=torch.float32)

    out = dequantize_block_fp8_to_bf16(weight_fp8, scale_fp32, block_size=(2, 2)).float()
    expected = torch.tensor(
        [
            [0.5, 0.5, 3.0, 3.0],  # block 0,0 * 0.5  |  block 0,1 * 1.5
            [0.5, 0.5, 3.0, 3.0],
            [8.0, 8.0, 2.0, 2.0],  # block 1,0 * 2.0  |  block 1,1 * 0.25
            [8.0, 8.0, 2.0, 2.0],
        ]
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_block_fp8_dequant_rejects_wrong_scale_shape() -> None:
    weight = torch.zeros((4, 4), dtype=torch.float8_e4m3fn)
    wrong_scale = torch.ones((4, 4), dtype=torch.float32)  # expected (2, 2) for block=(2,2)
    with pytest.raises(ValueError, match="fp8_scale shape"):
        dequantize_block_fp8_to_bf16(weight, wrong_scale, block_size=(2, 2))


def test_block_fp8_dequant_rejects_wrong_rank() -> None:
    weight_3d = torch.zeros((2, 4, 4), dtype=torch.float8_e4m3fn)
    scale = torch.ones((2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="fp8_weight must be 2-D"):
        dequantize_block_fp8_to_bf16(weight_3d, scale)


def test_block_fp8_dequant_returns_bf16() -> None:
    """Result dtype should always be BF16 regardless of input scale dtype."""
    weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones((1, 1), dtype=torch.float32)
    out = dequantize_block_fp8_to_bf16(weight, scale, block_size=(128, 128))
    assert out.dtype == torch.bfloat16
    assert out.shape == (128, 128)
