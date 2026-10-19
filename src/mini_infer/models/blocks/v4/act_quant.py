"""Activation quantize-dequantize round trips of DeepSeek-V4's reference inference.

DeepSeek-V4 was trained with quantization-aware training (QAT) on its
attention activations, and the reference inference code
(`deepseek_v4_reference/model.py`) reproduces that at inference time:
it quantizes some activations to a low-precision format and immediately
dequantizes them back, so the values that reach attention carry the
same rounding the model saw during training. Three tensors get this:

  - Window KV and the main compressor's KV: FP8 e4m3 on the non-RoPE
    dims, one scale per block of 64 values (`act_quant(kv[..., :-rd],
    64, scale_fmt, scale_dtype, True)`). The RoPE dims stay unquantized.
  - The Lightning Indexer's compressed keys and its queries: a Hadamard
    rotation, then FP4 e2m1 on all dims, one scale per block of 32
    values (`rotate_activation` + `fp4_act_quant(x, 32, True)`).

The functions here are PyTorch transcriptions of the reference's
tilelang kernels (`act_quant_kernel` / `fp4_quant_kernel` with
`inplace=True`, and `fast_hadamard_transform.hadamard_transform`),
computed in FP32 and cast back to the input dtype exactly like the
kernels. With `scale_fmt="ue8m0"` (every released V4 checkpoint) and
for FP4, every scale is a power of two, so the dequantized values are
exactly representable in BF16 and the result is the same at FP32 and
BF16 working precision.

`SimulatedActQuant` is the switch the V4 blocks take: `None` (the
default everywhere) skips all of this and keeps the port's full
precision; an instance applies the round trips at the reference's
call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ScaleFormat = Literal["ue8m0"] | None

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
# Block sizes the reference passes at its call sites.
KV_FP8_BLOCK_SIZE = 64
FP4_BLOCK_SIZE = 32
# Floors on a block's absolute max, copied from the kernels: they keep the
# scale a normal FP32 number when a block is all zeros.
_FP8_AMAX_FLOOR = 1e-4
_FP4_AMAX_FLOOR = FP4_E2M1_MAX * 2.0**-126
# The kernels multiply by the FP32 reciprocal instead of dividing by the max.
# `1 / 448` and `1 / 6` are not exact in FP32, so the product can differ from
# `amax / max` in the last bit; keep the multiply to match.
_FP8_MAX_INV = torch.tensor(1.0 / FP8_E4M3_MAX, dtype=torch.float32)
_FP4_MAX_INV = torch.tensor(1.0 / FP4_E2M1_MAX, dtype=torch.float32)

_FP32_EXPONENT_BIAS = 127
_FP32_MANTISSA_BITS = 23


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """`2 ** ceil(log2(x))` for positive normal FP32 `x`, exact.

    Same bit trick as the kernels' `fast_round_scale`: read the exponent
    field, add one if any mantissa bit is set, rebuild the power of two.
    """
    bits = x.view(torch.int32)
    biased_exponent = (bits >> _FP32_MANTISSA_BITS) & 0xFF
    has_mantissa = (bits & ((1 << _FP32_MANTISSA_BITS) - 1)) != 0
    biased_exponent = biased_exponent + has_mantissa.to(torch.int32)
    return (biased_exponent << _FP32_MANTISSA_BITS).view(torch.float32)


def _round_to_minifloat(y: torch.Tensor, *, mantissa_bits: int, min_exponent: int) -> torch.Tensor:
    """Round FP32 `y` to the nearest value of a small float format, ties to even.

    The format has `mantissa_bits` stored mantissa bits and smallest normal
    exponent `min_exponent` (below it, subnormals keep that spacing). `y`
    must already be clamped to the format's finite range, so no value
    overflows. Matches CUDA's round-to-nearest-even float-to-FP8/FP4 casts.
    """
    # Unbiased exponent `e` of |y| (|y| in [2^e, 2^(e+1))); zeros and
    # subnormal-range values use the smallest normal exponent's spacing.
    biased_exponent = (y.view(torch.int32) >> _FP32_MANTISSA_BITS) & 0xFF
    exponent = torch.clamp(biased_exponent - _FP32_EXPONENT_BIAS, min=min_exponent)
    # Grid spacing 2^(e - mantissa_bits), built from bits so it is exact.
    step = ((exponent - mantissa_bits + _FP32_EXPONENT_BIAS) << _FP32_MANTISSA_BITS).view(
        torch.float32
    )
    # Division and multiplication by a power of two are exact; `torch.round`
    # rounds half to even.
    return torch.round(y / step) * step


def _split_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """`(..., n)` -> FP32 `(..., n // block_size, block_size)`."""
    n = x.shape[-1]
    if n % block_size != 0:
        raise ValueError(f"last dim {n} must be a multiple of block_size={block_size}")
    return x.float().reshape(*x.shape[:-1], n // block_size, block_size)


def fp8_e4m3_roundtrip(x: torch.Tensor, *, block_size: int, scale_fmt: ScaleFormat) -> torch.Tensor:
    """Quantize `x` to FP8 e4m3 per block along the last dim, then dequantize.

    Transcription of the reference `act_quant(x, block_size, scale_fmt,
    scale_dtype, inplace=True)`. Per block of `block_size` values:
        amax  = max(max|x|, 1e-4)
        scale = amax * fp32(1/448)          (scale_fmt=None)
              = 2^ceil(log2(that))          (scale_fmt="ue8m0")
        out   = e4m3(clamp(x / scale, -448, 448)) * scale
    `scale_dtype` only changes how the kernel stores the scale it returns;
    the round trip uses the FP32 scale, so it has no parameter here.

    Args:
        x: `(..., n)` with `n % block_size == 0`, any float dtype.
        block_size: Values per scale (64 at the V4 KV call sites).
        scale_fmt: `"ue8m0"` for power-of-two scales, `None` for plain FP32.

    Returns:
        `(..., n)` in `x.dtype`.
    """
    blocks = _split_blocks(x, block_size)  # (..., n_blocks, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=_FP8_AMAX_FLOOR)
    scale = amax * _FP8_MAX_INV.to(amax.device)  # (..., n_blocks, 1)
    if scale_fmt == "ue8m0":
        scale = _pow2_ceil(scale)
    elif scale_fmt is not None:
        raise ValueError(f"scale_fmt must be 'ue8m0' or None, got {scale_fmt!r}")
    codes = (blocks / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    # e4m3 (fn variant): 3 mantissa bits, smallest normal exponent -6.
    codes = _round_to_minifloat(codes, mantissa_bits=3, min_exponent=-6)
    return (codes * scale).flatten(-2).to(x.dtype)  # (..., n)


def fp4_e2m1_roundtrip(x: torch.Tensor, *, block_size: int = FP4_BLOCK_SIZE) -> torch.Tensor:
    """Quantize `x` to FP4 e2m1 per block along the last dim, then dequantize.

    Transcription of the reference `fp4_act_quant(x, block_size,
    inplace=True)`. Per block of `block_size` values:
        amax  = max(max|x|, 6 * 2^-126)
        scale = 2^ceil(log2(amax * fp32(1/6)))     (always a power of two)
        out   = e2m1(clamp(x / scale, -6, 6)) * scale
    The e2m1 grid is {0, 0.5, 1, 1.5, 2, 3, 4, 6} and its negatives.

    Args:
        x: `(..., n)` with `n % block_size == 0`, any float dtype.
        block_size: Values per scale (32 at every V4 call site).

    Returns:
        `(..., n)` in `x.dtype`.
    """
    blocks = _split_blocks(x, block_size)  # (..., n_blocks, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=_FP4_AMAX_FLOOR)
    scale = _pow2_ceil(amax * _FP4_MAX_INV.to(amax.device))  # (..., n_blocks, 1)
    codes = (blocks / scale).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    # e2m1: 1 mantissa bit, smallest normal exponent 0 (subnormal 0.5).
    codes = _round_to_minifloat(codes, mantissa_bits=1, min_exponent=0)
    return (codes * scale).flatten(-2).to(x.dtype)  # (..., n)


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Hadamard transform along the last dim: `x @ H_d / sqrt(d)`.

    `H_d` is the Sylvester-ordered Hadamard matrix (symmetric, entries
    +-1), the same matrix as `fast_hadamard_transform.hadamard_transform`
    with `scale=d ** -0.5`, which the reference's `rotate_activation`
    calls. Computed in FP32 with the fast Walsh-Hadamard butterfly,
    scaled at the end, cast back to `x.dtype`.

    Because `H_d / sqrt(d)` is orthogonal, rotating both sides of a dot
    product leaves it unchanged. The rotation only matters when a
    quantizer runs in between: it spreads outliers across the block so
    one large value does not set a coarse scale for all of its neighbors.

    Args:
        x: `(..., d)` with `d` a power of two.

    Returns:
        `(..., d)` in `x.dtype`.
    """
    d = x.shape[-1]
    if d <= 0 or d & (d - 1) != 0:
        raise ValueError(f"Hadamard rotation needs a power-of-two last dim, got {d}")
    leading = x.shape[:-1]
    y = x.float()
    half = 1
    while half < d:
        # Pair entry i with entry i + half inside each run of 2 * half entries.
        pairs = y.reshape(*leading, d // (2 * half), 2, half)  # (..., d / 2h, 2, h)
        first, second = pairs.unbind(dim=-2)  # each (..., d / 2h, h)
        y = torch.stack((first + second, first - second), dim=-2).reshape(*leading, d)
        half *= 2
    # Multiply by `d ** -0.5` (not divide by sqrt(d)), as the kernel's `scale` does.
    inv_sqrt_d: float = d**-0.5
    return (y * inv_sqrt_d).to(x.dtype)


@dataclass(frozen=True)
class SimulatedActQuant:
    """Apply the reference's QAT round trips at its call sites.

    Blocks take `act_quant: SimulatedActQuant | None`; `None` skips every
    round trip. `fp8_scale_fmt` mirrors the reference's module-level
    `scale_fmt`, which every released V4 config sets to `"ue8m0"`.
    """

    fp8_scale_fmt: ScaleFormat = "ue8m0"

    def __post_init__(self) -> None:
        if self.fp8_scale_fmt not in ("ue8m0", None):
            raise ValueError(f"fp8_scale_fmt must be 'ue8m0' or None, got {self.fp8_scale_fmt!r}")

    @staticmethod
    def check_kv_dims(kv_head_dim: int, rope_head_dim: int) -> None:
        """Fail early when the non-RoPE KV width does not split into FP8 blocks."""
        nope_dim = kv_head_dim - rope_head_dim
        if nope_dim % KV_FP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"simulated activation quant needs kv_head_dim - rope_head_dim "
                f"({kv_head_dim} - {rope_head_dim} = {nope_dim}) to be a multiple of "
                f"{KV_FP8_BLOCK_SIZE}"
            )

    @staticmethod
    def check_indexer_dim(head_dim: int) -> None:
        """Fail early when the indexer head dim cannot be rotated and FP4-quantized."""
        if head_dim % FP4_BLOCK_SIZE != 0 or head_dim & (head_dim - 1) != 0:
            raise ValueError(
                "simulated activation quant needs the indexer head_dim to be a power of "
                f"two and a multiple of {FP4_BLOCK_SIZE}, got {head_dim}"
            )

    def kv(self, kv: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
        """FP8 round trip on the non-RoPE dims of window or compressed KV.

        `kv`: `(..., kv_head_dim)`, RoPE on the last `rope_head_dim` dims.
        Returns the same shape and dtype; the RoPE dims pass through.
        """
        nope_dim = kv.shape[-1] - rope_head_dim
        nope = fp8_e4m3_roundtrip(
            kv[..., :nope_dim], block_size=KV_FP8_BLOCK_SIZE, scale_fmt=self.fp8_scale_fmt
        )  # (..., nope_dim)
        return torch.cat([nope, kv[..., nope_dim:]], dim=-1)

    def indexer(self, x: torch.Tensor) -> torch.Tensor:
        """Hadamard rotation then FP4 round trip, for indexer queries and keys.

        `x`: `(..., index_head_dim)`. Returns the same shape and dtype. The
        output is in the rotated basis; queries and keys must both go
        through this so their dot products stay comparable.
        """
        return fp4_e2m1_roundtrip(hadamard_rotate(x), block_size=FP4_BLOCK_SIZE)
