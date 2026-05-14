"""Block-quantized FP4 / FP8 dequantization helpers for V4-Flash.

V4-Flash ships weights in two block-quantized formats:

  - **NVFP4** (`float4_e2m1fn_x2`): MoE expert weights. Two FP4 values share
    a byte (packed); a per-block FP8 `e8m0fnu` scale (block size 32 along
    the inner dim) multiplies each block to its full magnitude.
  - **Block-FP8** (`float8_e4m3fn` weight + `float8_e8m0fnu` scale): every
    other quantized weight (attention projections, MoE gate, compressor /
    indexer projections). The block is `[128, 128]` per `config.json`'s
    `quantization_config.weight_block_size` — one scale per 128x128 tile.

E2M1 lookup (used by NVFP4; 1 sign bit, 2 exponent, 1 mantissa, NaN-free):

    bits 0xxx ->  0.0,  0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    bits 1xxx ->  0.0, -0.5, ... -6.0   (sign-flipped)

Both functions output BF16. They run on any device (CPU smoke tests work);
a Triton kernel that fuses dequant + GEMM is the optimization path.
"""

from __future__ import annotations

import torch

# E2M1 (no inf, no NaN; symmetric around zero). See class docstring.
_FP4_LOOKUP_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

# Block size matches the V4 reference: 32 FP4 values share one scale.
_FP4_BLOCK_SIZE = 32


def _fp4_table(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build (or fetch) the 16-entry E2M1 lookup table on the right device."""
    return torch.tensor(_FP4_LOOKUP_VALUES, dtype=dtype, device=device)


def is_packed_nvfp4(tensor: torch.Tensor) -> bool:
    """True iff this tensor is stored as packed NVFP4.

    The packed-FP4 dtype `torch.float4_e2m1fn_x2` only exists in PyTorch
    2.6+; on older builds the safetensors loader hands us `uint8` and we
    detect that path by callers passing the name explicitly.
    """
    packed_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return packed_dtype is not None and tensor.dtype == packed_dtype


def dequantize_nvfp4_to_bf16(
    packed_weight: torch.Tensor,
    fp4_scale: torch.Tensor,
    *,
    block_size: int = _FP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize packed NVFP4 + per-block FP8 scale to a BF16 weight tensor.

    Args:
        packed_weight: 2-D tensor of shape `(out_dim, in_dim // 2)`. Either
            dtype `torch.float4_e2m1fn_x2`, or any 1-byte integer dtype
            (`uint8` / `int8`) — we re-view to `uint8` for nibble extraction.
        fp4_scale: 2-D tensor of shape `(out_dim, in_dim // block_size)`,
            typically dtype `torch.float8_e8m0fnu`. Cast to FP32 via
            PyTorch's standard `.float()` (which handles the e8m0 path).
        block_size: how many FP4 values share each scale entry. V4 uses 32.

    Returns:
        BF16 tensor of shape `(out_dim, in_dim)` containing the dequantized weight.
    """
    if packed_weight.ndim != 2:
        raise ValueError(
            f"packed_weight must be 2-D (out_dim, in_dim/2); got shape {tuple(packed_weight.shape)}"
        )
    if fp4_scale.ndim != 2:
        raise ValueError(f"fp4_scale must be 2-D; got shape {tuple(fp4_scale.shape)}")

    # Bytes view — works for the native packed dtype, uint8, and int8 alike.
    bytes_view = packed_weight.view(torch.uint8)
    out_dim, half_in = bytes_view.shape
    in_dim = half_in * 2
    expected_scale_shape = (out_dim, in_dim // block_size)
    if fp4_scale.shape != expected_scale_shape:
        raise ValueError(
            f"fp4_scale shape {tuple(fp4_scale.shape)} does not match expected "
            f"{expected_scale_shape} (block_size={block_size}, in_dim={in_dim})"
        )

    # Nibble extract. Low nibble = first element of the pair, high = second.
    # Matches the reference's `cast_e2m1fn_to_e4m3fn` byte layout.
    low_nibbles = (bytes_view & 0x0F).long()
    high_nibbles = ((bytes_view >> 4) & 0x0F).long()
    table = _fp4_table(packed_weight.device, dtype=torch.float32)
    # Stack low/high so the result's last axis becomes
    # `(byte_idx, pair_position)` -> reshape to `(out_dim, in_dim)`.
    fp4_fp32 = torch.stack([table[low_nibbles], table[high_nibbles]], dim=-1).reshape(
        out_dim, in_dim
    )

    # Cast scale to fp32. PyTorch's `.float()` handles `float8_e8m0fnu` natively
    # (the e8m0 format is "powers of 2"; the cast just exponentiates).
    scale_fp32 = fp4_scale.float()
    # Expand the per-block scale to per-element by repeating each scale
    # `block_size` times along the inner dim.
    scale_expanded = scale_fp32.repeat_interleave(block_size, dim=-1)

    dequantized = fp4_fp32 * scale_expanded
    return dequantized.to(torch.bfloat16)


def dequantize_block_fp8_to_bf16(
    fp8_weight: torch.Tensor,
    fp8_scale: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Dequantize a block-quantized FP8 e4m3 weight to BF16.

    V4-Flash stores most non-MoE weights this way: shape `(M, N)` in
    `torch.float8_e4m3fn`, paired with a `(M / block_M, N / block_N)`
    scale tensor in `torch.float8_e8m0fnu`. Dequant is element-wise
    multiplication after upcasting both sides to FP32 and broadcasting
    the per-block scale over `(block_M, block_N)`.

    Args:
        fp8_weight: 2-D tensor of shape `(M, N)`, dtype `torch.float8_e4m3fn`
            (or any 1-byte type if storage was reinterpreted).
        fp8_scale: 2-D tensor of shape `(M // block_M, N // block_N)`,
            typically dtype `torch.float8_e8m0fnu`.
        block_size: per-block dims `(block_M, block_N)`. V4-Flash uses (128, 128).

    Returns:
        BF16 tensor of shape `(M, N)`.
    """
    if fp8_weight.ndim != 2:
        raise ValueError(
            f"fp8_weight must be 2-D; got shape {tuple(fp8_weight.shape)}"
        )
    if fp8_scale.ndim != 2:
        raise ValueError(f"fp8_scale must be 2-D; got shape {tuple(fp8_scale.shape)}")
    block_m, block_n = block_size
    m, n = fp8_weight.shape
    expected_scale = (m // block_m, n // block_n)
    if fp8_scale.shape != expected_scale:
        raise ValueError(
            f"fp8_scale shape {tuple(fp8_scale.shape)} does not match expected "
            f"{expected_scale} (block_size={block_size}, weight_shape={(m, n)})"
        )
    # Upcast both sides; the FP8 -> FP32 cast handles e4m3fn / e8m0fnu natively.
    weight_fp32 = fp8_weight.float()
    scale_fp32 = fp8_scale.float()
    # Expand scale to per-element via repeat_interleave along both dims.
    scale_expanded = (
        scale_fp32.repeat_interleave(block_m, dim=0).repeat_interleave(block_n, dim=1)
    )
    return (weight_fp32 * scale_expanded).to(torch.bfloat16)
