"""MXFP4 (compressed-tensors `mxfp4-pack-quantized`) -> BF16 dequantization.

Kimi K3's routed expert weights (`experts.N.w1/w2/w3`) ship in this format:

  - `weight_packed`: `uint8` of shape `(out_dim, in_dim // 2)`, two E2M1
    values per byte, LOW nibble first (the same byte layout as NVFP4 and the
    DeepSeek-V4 packed experts, see `quant/nvfp4.py`);
  - `weight_scale`: `uint8` of shape `(out_dim, in_dim // 32)`, one E8M0
    exponent per block of 32 along `in_dim`, stored with bias 127:
    `scale = 2 ** (byte - 127)`.

Unlike NVFP4 there is no global scale (group strategy, not tensor-group), so
dequantization is `value * 2 ** (byte - 127)` per block. The scale is a power
of two, so the product is exact in BF16 (E2M1 has a 1-bit mantissa).
"""

from __future__ import annotations

import torch

from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16

MXFP4_BLOCK_SIZE = 32
_E8M0_BIAS = 127.0
_E8M0_NAN_BYTE = 255


def mx_scale_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 block scales -> fp32 powers of two.

    Accepts the `uint8` storage compressed-tensors writes (biased exponent)
    or a native `float8_e8m0fnu` tensor (the same bits, read as bytes).

    Byte 255 is E8M0's NaN encoding. It would turn a whole 32-value block
    into non-finite weights, so a corrupt scale fails here at load time.
    The check reads the raw bytes, before the fp32 decode.
    """
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype == torch.uint8:
        raw = scale
    elif e8m0 is not None and scale.dtype == e8m0:
        raw = scale.view(torch.uint8)
    else:
        raise ValueError(
            f"MXFP4 scale must be uint8 (E8M0 bits) or float8_e8m0fnu, got {scale.dtype}"
        )
    if bool((raw == _E8M0_NAN_BYTE).any()):
        raise ValueError("MXFP4 scale holds the E8M0 NaN encoding (byte 255); the shard is corrupt")
    return torch.exp2(raw.to(torch.float32) - _E8M0_BIAS)


def dequantize_mxfp4_to_bf16(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize a compressed-tensors MXFP4 pair to a BF16 `(out_dim, in_dim)` weight.

    Args:
        weight_packed: `uint8` (or any 1-byte integer dtype) `(out_dim, in_dim // 2)`.
        weight_scale: `uint8` E8M0 `(out_dim, in_dim // block_size)`.
        block_size: values per scale; compressed-tensors' MXFP4 scheme uses 32.
    """
    if weight_packed.dtype not in (torch.uint8, torch.int8):
        raise ValueError(f"weight_packed must be uint8/int8 bytes, got {weight_packed.dtype}")
    return dequantize_nvfp4_to_bf16(
        weight_packed, mx_scale_to_fp32(weight_scale), block_size=block_size
    )
