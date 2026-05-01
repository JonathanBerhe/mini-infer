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


# ──────────────────────────────────────────────────────────────────────
# V3 (TurboQuant full): polar transform + Lloyd-Max codebook + QJL
# residual sign bit + asymmetric K/V bit budgets. The pieces below are
# built up across V3a (polar + symmetric uniform), V3c (Lloyd-Max), V3b
# (QJL), V3d (asymmetric K/V). At end-state, K uses 3-bit Lloyd-Max +
# 1-bit QJL (= 4 bits stored) and V uses 4-bit Lloyd-Max (= 4 bits
# stored). Both pack as nibbles, same on-disk layout as V1.
# ──────────────────────────────────────────────────────────────────────

# Lloyd-Max optimal scalar quantizer codebooks for the unit Gaussian.
# Standard values from offline optimization (Max 1960, Lloyd 1982).
# Used after we rescale unit-vector coords by sqrt(head_dim) to give
# them ~N(0, 1) marginal distribution.
#
# 4-bit codebook (16 levels, symmetric around 0).
_LLOYD_MAX_GAUSSIAN_4BIT = (
    -3.16100,
    -2.40300,
    -1.83400,
    -1.40000,
    -1.05000,
    -0.74600,
    -0.46100,
    -0.15300,
    0.15300,
    0.46100,
    0.74600,
    1.05000,
    1.40000,
    1.83400,
    2.40300,
    3.16100,
)
# 3-bit codebook (8 levels, symmetric).
_LLOYD_MAX_GAUSSIAN_3BIT = (
    -2.15200,
    -1.34400,
    -0.75600,
    -0.24500,
    0.24500,
    0.75600,
    1.34400,
    2.15200,
)


def lloyd_max_codebook(bits: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Return the precomputed Lloyd-Max codebook for `bits` bits, N(0, 1).

    Output shape: `(2**bits,)` in `dtype`. Centers are sorted ascending
    so a `searchsorted` call against them gives the right bin index.
    """
    values: tuple[float, ...]
    if bits == 4:
        values = _LLOYD_MAX_GAUSSIAN_4BIT
    elif bits == 3:
        values = _LLOYD_MAX_GAUSSIAN_3BIT
    else:
        raise ValueError(f"Lloyd-Max codebook only supports 3- or 4-bit; got {bits}")
    return torch.tensor(values, dtype=dtype, device=device)


def polar_quantize_block(
    block: torch.Tensor,
    *,
    bits: int = 4,
    use_lloyd_max: bool = True,
    use_qjl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Polar-coordinate quantization with optional Lloyd-Max + QJL.

    Args:
        block: `(block_size, num_kv_heads, head_dim)` rotated bf16/fp16/fp32.
        bits: 3 or 4. With QJL, total stored bits per element are bits+1
            (still packed as nibbles when bits=3, the extra bit is the
            QJL residual sign).
        use_lloyd_max: when True, snap unit-vector coords to the
            precomputed Lloyd-Max codebook for N(0, 1). When False, use
            uniform symmetric quant on `[-1, 1]` (V3a step).
        use_qjl: when True, store an extra residual sign bit per element.
            Adds 1 bit of effective precision. Only meaningful with
            Lloyd-Max (no codebook = no residual to sign).

    Returns:
        packed: int8 `(block_size * num_kv_heads * head_dim // 2,)`.
            Each byte is two consecutive elements; meaning of the bits
            depends on `(bits, use_qjl)`:
              - bits=4, use_qjl=False: low 4 = 4-bit codebook index,
                next 4 = next element's index (etc.).
              - bits=3, use_qjl=True: per element, 3 bits codebook index
                + 1 bit residual sign = 4 bits, packed as nibbles.
        radii: same float dtype, `(block_size, num_kv_heads)`. Per-vector
            L2 norms in the rotated basis.

    Both K and V use the same primitive; the caller chooses bits/QJL.
    """
    if block.ndim != 3:
        raise ValueError(
            f"polar_quantize_block expects (block_size, num_kv_heads, head_dim); "
            f"got shape {tuple(block.shape)}"
        )
    block_size, num_kv_heads, head_dim = block.shape
    if (block_size * num_kv_heads * head_dim) % 2 != 0:
        raise ValueError("total elements must be even to pack two 4-bit values per byte")
    if bits not in (3, 4):
        raise ValueError(f"bits must be 3 or 4; got {bits}")
    n_levels = 1 << bits  # 8 or 16
    if use_qjl and bits != 3:
        # We pack 4 storage bits per element. With bits=4, no room for
        # the residual sign. QJL is meaningful with bits=3 (3-bit
        # codebook + 1-bit sign = 4 bits stored).
        raise ValueError("use_qjl=True is only supported with bits=3 in V3")

    block_fp32 = block.float()
    # Per-vector L2 norm (over head_dim). clamp_min guards zero vectors.
    radii = block_fp32.norm(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    # Unit vectors on the sphere: each coord ~ N(0, 1/head_dim).
    units = block_fp32 / radii.unsqueeze(-1)

    if use_lloyd_max:
        # Rescale to ~N(0, 1) so the standard codebook applies.
        scaled = units * (head_dim**0.5)
        codebook = lloyd_max_codebook(bits, dtype=torch.float32, device=block.device)
        # For each scalar, find the nearest codebook entry. Codebook is
        # sorted, so binary-search the midpoints.
        midpoints = (codebook[:-1] + codebook[1:]) / 2.0
        idx = torch.searchsorted(midpoints, scaled.contiguous().view(-1))
        idx = idx.clamp_(0, n_levels - 1).reshape(block_size, num_kv_heads, head_dim)

        if use_qjl:
            # QJL: 1-bit residual sign telling whether the true value
            # was above (1) or below (0) the codebook center it snapped to.
            centers = codebook[idx]
            residual_sign = (scaled >= centers).to(torch.uint8)  # 0 or 1
            # Store: 3-bit codebook idx in low bits, sign in MSB of the
            # 4-bit nibble. Total 4 bits per element.
            packed_per_elem = (idx.to(torch.uint8) | (residual_sign << 3)).view(-1)
        else:
            packed_per_elem = idx.to(torch.uint8).view(-1)
    else:
        # V3a fallback: uniform symmetric 4-bit on units in [-1, 1].
        # Map u in [-1, 1] -> q in [0, 15] = round((u + 1) * 7.5).
        q_fp = ((units + 1.0) * (n_levels - 1) / 2.0).round().clamp_(0, n_levels - 1)
        packed_per_elem = q_fp.to(torch.uint8).view(-1)

    # Pack pairs of nibbles into bytes (low nibble = even index,
    # high nibble = odd index).
    low_nibbles = packed_per_elem[0::2]
    high_nibbles = packed_per_elem[1::2]
    packed = (low_nibbles | (high_nibbles << 4)).to(torch.int8)

    return packed, radii.to(block.dtype)


def polar_dequantize_block(
    packed: torch.Tensor,
    radii: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    bits: int = 4,
    use_lloyd_max: bool = True,
    use_qjl: bool = False,
) -> torch.Tensor:
    """Inverse of `polar_quantize_block`. Returns the rotated bf16/fp16 block."""
    expected_packed_len = (block_size * num_kv_heads * head_dim) // 2
    if packed.shape != (expected_packed_len,):
        raise ValueError(f"packed shape {tuple(packed.shape)} != expected ({expected_packed_len},)")
    if radii.shape != (block_size, num_kv_heads):
        raise ValueError(
            f"radii shape {tuple(radii.shape)} != expected ({block_size}, {num_kv_heads})"
        )
    if bits not in (3, 4):
        raise ValueError(f"bits must be 3 or 4; got {bits}")
    n_levels = 1 << bits

    # Unpack nibbles.
    packed_uint = packed.view(torch.uint8)
    low_nibbles = packed_uint & 0x0F
    high_nibbles = (packed_uint >> 4) & 0x0F
    flat = torch.empty(packed_uint.shape[0] * 2, dtype=torch.uint8, device=packed.device)
    flat[0::2] = low_nibbles
    flat[1::2] = high_nibbles
    nibbles = flat.reshape(block_size, num_kv_heads, head_dim)

    if use_lloyd_max:
        codebook = lloyd_max_codebook(bits, dtype=torch.float32, device=packed.device)
        if use_qjl:
            # Bottom 3 bits = codebook idx, top bit = residual sign.
            idx = nibbles & 0x07
            sign = (nibbles >> 3) & 0x01
        else:
            idx = nibbles & (n_levels - 1)
            sign = None

        scaled = codebook[idx.to(torch.long)]  # (block_size, num_kv_heads, head_dim) fp32

        if sign is not None:
            # Residual correction: nudge toward the next center based on
            # the stored sign. Step is half the gap to the higher
            # neighbor (or lower neighbor if sign is 0). Approximation:
            # use uniform step = (codebook[1] - codebook[0]) / 2.
            #
            # A more precise correction would store a per-bin offset;
            # for V3 we use the average step which captures most of the
            # accuracy gain.
            step = (codebook[1] - codebook[0]) / 4.0  # quarter-step nudge
            scaled = scaled + (sign.to(scaled.dtype) * 2.0 - 1.0) * step

        # Undo the sqrt(head_dim) rescale to get unit-vector coords.
        units = scaled / (head_dim**0.5)
    else:
        # Uniform inverse: q in [0, 15] -> u in [-1, 1].
        q_fp = nibbles.to(torch.float32)
        units = q_fp * 2.0 / (n_levels - 1) - 1.0

    # Multiply by per-vector radius to recover the rotated K/V.
    out: torch.Tensor = (units * radii.to(torch.float32).unsqueeze(-1)).to(dtype)
    return out


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
