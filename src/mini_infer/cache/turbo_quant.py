"""TurboQuant-style KV cache compression: rotation precondition + 4-bit quant.

A V1 of Google's TurboQuant (ICLR 2026) restricted to:

1. **Random orthogonal rotation** of K/V vectors before storage. The rotation
   "Gaussianizes" the post-rotation distribution: in any orthonormal basis,
   any single coordinate of a rotated random vector is approximately
   Gaussian, so per-channel uniform quantization sees a tight, well-behaved
   distribution to discretize.
2. **Per-channel asymmetric 4-bit quantization** of each
   `(block_size, num_kv_heads, head_dim)` block, with a `(low, scale)` pair
   per channel. ~4x compression vs bf16 (scales add ~2% overhead).

Components from the full TurboQuant recipe NOT in V1: PolarQuant
(Cartesian->polar), Quantized Johnson-Lindenstrauss (QJL) residual sign
bits, Lloyd-Max optimal codebooks, asymmetric K vs V bits. See ADR-013 for
why each is deferred and the path back to them.

Rotation is mathematically benign for attention: `(Q @ R) @ (K @ R)^T == Q
@ K^T` because `R` is orthogonal. So Q is rotated at attention time, K is
rotated on cache write and inverse-rotated on cache read, and the QK^T
output is identical to the un-rotated path within numerical precision.
For V: rotation on write, inverse-rotation on read; the `softmax(QK^T) @ V`
step then sees un-rotated values.
"""

from __future__ import annotations

import torch


def generate_rotation_matrices(
    num_layers: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """Sample one random orthogonal matrix per layer via QR on a Gaussian.

    Returns shape `(num_layers, head_dim, head_dim)`. The matrix is
    orthogonal in fp32 and then cast to `dtype`; small departure from
    perfect orthogonality at bf16/fp16 is acceptable (the rotation is
    deterministic noise that cancels via R @ R^T at attention time within
    bf16 precision).

    The seed makes the rotation deterministic across runs of the same
    config — important for reproducibility and for unit tests asserting
    cosine similarity.
    """
    if num_layers <= 0 or head_dim <= 0:
        raise ValueError(
            f"num_layers and head_dim must be positive; got {num_layers} and {head_dim}"
        )
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Sample on CPU in fp32 for deterministic QR; move + cast at the end.
    matrices = torch.empty(num_layers, head_dim, head_dim, dtype=torch.float32)
    for layer_idx in range(num_layers):
        random_matrix = torch.randn(head_dim, head_dim, generator=g)
        q, _ = torch.linalg.qr(random_matrix)
        matrices[layer_idx] = q
    return matrices.to(device=device, dtype=dtype)


def rotate(
    x: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply rotation to the head_dim of `x`.

    `x` shape: `(..., head_dim)`. `rotation` shape: `(head_dim, head_dim)`.
    Returns `x @ rotation`, same shape as `x`. Caller passes the layer's
    rotation matrix; this module doesn't track which layer is which.
    """
    if rotation.shape[-2] != rotation.shape[-1] or rotation.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"rotation must be (head_dim, head_dim) matching x.shape[-1]; "
            f"got rotation {tuple(rotation.shape)} vs x last dim {x.shape[-1]}"
        )
    return x @ rotation


def inverse_rotate(
    x_rotated: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `rotate`. Equivalent to `x_rotated @ rotation.T`."""
    return x_rotated @ rotation.transpose(-1, -2)


def quantize_kv_block(
    block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel asymmetric 4-bit quantization of a single rotated KV block.

    Args:
        block: `(block_size, num_kv_heads, head_dim)` in bf16/fp16/fp32.
            Expected to be rotation-preconditioned for V1.

    Returns:
        packed: int8 of shape `(block_size * num_kv_heads * head_dim // 2,)`.
            Each byte stores two consecutive 4-bit values: low nibble = even
            index, high nibble = odd index.
        low:    same float dtype as `block`, shape `(num_kv_heads, head_dim)`.
        scale:  same float dtype as `block`, shape `(num_kv_heads, head_dim)`;
            equals `(high - low) / 15`. Constant blocks (high == low) get a
            small epsilon in `scale` to avoid divide-by-zero on dequant.

    The 4-bit values q ∈ [0, 15] reconstruct as `low + q * scale`. Per-channel
    means scales are per `(kv_head, dim)`; `block_size` tokens in a block
    share scales for that channel.
    """
    if block.ndim != 3:
        raise ValueError(
            f"quantize_kv_block expects (block_size, num_kv_heads, head_dim); "
            f"got shape {tuple(block.shape)}"
        )
    block_size, num_kv_heads, head_dim = block.shape
    if (block_size * num_kv_heads * head_dim) % 2 != 0:
        raise ValueError(
            f"total elements ({block_size * num_kv_heads * head_dim}) must be even "
            "to pack two 4-bit values per byte"
        )

    block_fp32 = block.float()
    # Per-channel low/high computed across the block-size dim.
    low = block_fp32.amin(dim=0)
    high = block_fp32.amax(dim=0)
    spread = (high - low).clamp_min(torch.finfo(torch.float32).tiny)
    scale_fp32 = spread / 15.0

    # Quantize to [0, 15] uint range; clamp guards against fp rounding.
    q_fp = ((block_fp32 - low.unsqueeze(0)) / scale_fp32.unsqueeze(0)).round().clamp_(0, 15)
    q_uint = q_fp.to(torch.uint8)  # (block_size, num_kv_heads, head_dim)

    # Pack pairs of 4-bit values into bytes. Reshape to a flat list of values
    # and combine consecutive pairs.
    flat = q_uint.reshape(-1)
    if flat.shape[0] % 2 != 0:
        raise ValueError("internal: total elements not divisible by 2 after reshape")
    low_nibbles = flat[0::2]
    high_nibbles = flat[1::2]
    packed = (low_nibbles | (high_nibbles << 4)).to(torch.int8)

    return packed, low.to(block.dtype), scale_fp32.to(block.dtype)


def dequantize_kv_block(
    packed: torch.Tensor,
    low: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inverse of `quantize_kv_block`. Returns the rotated bf16/fp16/fp32 block.

    Output shape: `(block_size, num_kv_heads, head_dim)`. Numerical contract:
    the per-channel low + q * scale reconstruction is bit-identical to the
    quantize step's intent; the only error is the per-element 4-bit
    round-to-nearest (max abs error per element <= scale / 2).
    """
    expected_packed_len = (block_size * num_kv_heads * head_dim) // 2
    if packed.shape != (expected_packed_len,):
        raise ValueError(f"packed shape {tuple(packed.shape)} != expected ({expected_packed_len},)")
    if low.shape != (num_kv_heads, head_dim) or scale.shape != (num_kv_heads, head_dim):
        raise ValueError(
            f"low/scale shapes {tuple(low.shape)}/{tuple(scale.shape)} != "
            f"({num_kv_heads}, {head_dim})"
        )

    # Unpack: each int8 byte holds (high nibble, low nibble); a uint8 view
    # makes the bit fiddling unambiguous (signed-shift hazards aside).
    packed_uint = packed.view(torch.uint8)
    low_nibbles = packed_uint & 0x0F
    high_nibbles = (packed_uint >> 4) & 0x0F

    # Re-interleave back into the original element order.
    flat = torch.empty(packed_uint.shape[0] * 2, dtype=torch.uint8, device=packed.device)
    flat[0::2] = low_nibbles
    flat[1::2] = high_nibbles

    q_uint = flat.reshape(block_size, num_kv_heads, head_dim)
    q_fp = q_uint.to(dtype)
    return low.to(dtype).unsqueeze(0) + q_fp * scale.to(dtype).unsqueeze(0)
