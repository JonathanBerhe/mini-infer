"""Fused Triton kernel for `apply_partial_rope_last_n_dims` (DeepSeek-V4 partial RoPE).

The PyTorch path rotates the last `rope_dim` features of each row with
separate ops: four multiplies, a subtract, an add, a stack, a cat and, for
the inverse rotation, a negation. A V4-Flash decode step calls it 212 times
on two GPUs, which is about 1,900 kernels and 3.6 to 5.2 ms of GPU time per
step on a B200. This kernel does one call in one launch: it copies the
features that do not rotate and rotates the interleaved pairs of the tail.

It keeps the rounding of the PyTorch path. Each multiply, subtract and add
rounds to the output dtype, as the separate ops do, and the launch turns
FMA contraction off. So the output equals the PyTorch path bit for bit on
CUDA.

CUDA-only. `rope.apply_partial_rope_last_n_dims` dispatches here when
`supports_partial_rope_kernel` and `broadcast_rope_tables` accept the
inputs, and keeps the PyTorch path otherwise.
"""

from __future__ import annotations

import torch

from mini_infer.device import is_cuda_device

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# Leading dims the kernel indexes (x is at most `(d0, d1, d2, features)`).
_MAX_LEADING_DIMS = 3
# Bench hook, not a public API: True keeps the PyTorch path everywhere.
_DISABLED_FOR_BENCH = False


def supports_partial_rope_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
    """Whether the kernel takes these tensors: CUDA, Triton, bf16, fp16 or fp32,
    at most three leading dims, and no tensor that needs a gradient.
    """
    if _DISABLED_FOR_BENCH or not _TRITON_AVAILABLE or not is_cuda_device(x.device):
        return False
    if x.dtype not in _DTYPES or cos.dtype not in _DTYPES or sin.dtype != cos.dtype:
        return False
    if not 1 <= x.ndim <= _MAX_LEADING_DIMS + 1:
        return False
    return not (
        torch.is_grad_enabled() and (x.requires_grad or cos.requires_grad or sin.requires_grad)
    )


def broadcast_rope_tables(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, last_n_rotated: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The first halves of `cos` and `sin` broadcast to `x`'s leading dims, as views.

    None if they do not broadcast the way the PyTorch path broadcasts them.
    """
    if cos.ndim > x.ndim or sin.ndim != cos.ndim:
        return None
    half = last_n_rotated // 2
    if cos.shape[-1] < half or sin.shape[-1] < half:
        return None
    # The PyTorch path inserts dims before the last one until the ranks match.
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    lead = x.shape[:-1]
    for table in (cos, sin):
        if any(size not in (1, n) for size, n in zip(table.shape[:-1], lead, strict=True)):
            return None
    return cos[..., :half].expand(*lead, half), sin[..., :half].expand(*lead, half)


def _strides(t: torch.Tensor) -> list[int]:
    """`t`'s strides with the leading dims padded to `_MAX_LEADING_DIMS`."""
    pad = _MAX_LEADING_DIMS + 1 - t.ndim
    return [0] * pad + list(t.stride())


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _partial_rope_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_rows,
        dim1,
        dim2,
        stride_x0,
        stride_x1,
        stride_x2,
        stride_xd,
        stride_c0,
        stride_c1,
        stride_c2,
        stride_cd,
        stride_s0,
        stride_s1,
        stride_s2,
        stride_sd,
        NOPE: tl.constexpr,
        HALF: tl.constexpr,
        NOPE_BLOCK: tl.constexpr,
        HALF_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        INVERSE: tl.constexpr,
    ) -> None:
        """`BLOCK_ROWS` rows per program. A row is one `(d0, d1, d2)` index of x.

        The output is contiguous, `(n_rows, NOPE + 2 * HALF)`. The first `NOPE`
        features are a cast copy of x. Pair `j` of the tail is
        `(x[NOPE + 2j], x[NOPE + 2j + 1])`, rotated by `cos[j]` and `sin[j]`.
        """
        out_ty = out_ptr.dtype.element_ty
        rows = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
        row_ok = rows < n_rows
        i2 = rows % dim2
        i1 = (rows // dim2) % dim1
        i0 = rows // (dim1 * dim2)
        x_row = i0 * stride_x0 + i1 * stride_x1 + i2 * stride_x2
        out_row = rows * (NOPE + 2 * HALF)

        if NOPE > 0:
            offs = tl.arange(0, NOPE_BLOCK)
            ok = row_ok[:, None] & (offs < NOPE)[None, :]
            rest = tl.load(x_ptr + x_row[:, None] + offs[None, :] * stride_xd, mask=ok)
            tl.store(out_ptr + out_row[:, None] + offs[None, :], rest.to(out_ty), mask=ok)

        pair = tl.arange(0, HALF_BLOCK)
        ok = row_ok[:, None] & (pair < HALF)[None, :]
        real_at = x_ptr + x_row[:, None] + (NOPE + 2 * pair[None, :]) * stride_xd
        real = tl.load(real_at, mask=ok).to(tl.float32)
        imag = tl.load(real_at + stride_xd, mask=ok).to(tl.float32)
        c_row = i0 * stride_c0 + i1 * stride_c1 + i2 * stride_c2
        s_row = i0 * stride_s0 + i1 * stride_s1 + i2 * stride_s2
        cos = tl.load(cos_ptr + c_row[:, None] + pair[None, :] * stride_cd, mask=ok)
        sin = tl.load(sin_ptr + s_row[:, None] + pair[None, :] * stride_sd, mask=ok)
        cos = cos.to(tl.float32)
        sin = sin.to(tl.float32)
        if INVERSE:
            sin = -sin
        # Each product, difference and sum rounds to the output dtype, as the
        # separate PyTorch ops do (they compute in fp32 and round).
        rot_real = (real * cos).to(out_ty).to(tl.float32) - (imag * sin).to(out_ty).to(tl.float32)
        rot_imag = (real * sin).to(out_ty).to(tl.float32) + (imag * cos).to(out_ty).to(tl.float32)
        pair_at = out_ptr + out_row[:, None] + NOPE + 2 * pair[None, :]
        tl.store(pair_at, rot_real.to(out_ty), mask=ok)
        tl.store(pair_at + 1, rot_imag.to(out_ty), mask=ok)


def partial_rope_triton(
    x: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    last_n_rotated: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """`apply_partial_rope_last_n_dims` in one launch.

    `cos_half` and `sin_half` come from `broadcast_rope_tables`: `(*x.shape[:-1],
    last_n_rotated // 2)`, any strides. Returns a contiguous tensor of `x`'s
    shape in the dtype the PyTorch path gives, `promote_types(x, cos)`. Runs
    on CPU tensors only under `TRITON_INTERPRET=1`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("partial_rope_triton needs Triton")
    features = x.shape[-1]
    half = last_n_rotated // 2
    nope = features - last_n_rotated
    out = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, cos_half.dtype), device=x.device)
    n_rows = out.numel() // features if features else 0
    if n_rows == 0:
        return out
    lead = [1] * (_MAX_LEADING_DIMS + 1 - x.ndim) + list(x.shape[:-1])
    # About 2,048 features per program: 4 rows of a V4-Flash KV head (512).
    block_rows = max(1, 2048 // triton.next_power_of_2(features))
    _partial_rope_kernel[(triton.cdiv(n_rows, block_rows),)](
        x,
        cos_half,
        sin_half,
        out,
        n_rows,
        lead[1],
        lead[2],
        *_strides(x),
        *_strides(cos_half),
        *_strides(sin_half),
        NOPE=nope,
        HALF=half,
        NOPE_BLOCK=triton.next_power_of_2(max(nope, 1)),
        HALF_BLOCK=triton.next_power_of_2(half),
        BLOCK_ROWS=block_rows,
        INVERSE=inverse,
        enable_fp_fusion=False,
    )
    return out
