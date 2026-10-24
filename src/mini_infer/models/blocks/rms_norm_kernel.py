"""Fused Triton kernel for the RMS norms of DeepSeek-V4.

`RMSNorm` and the per-head q-norm run as separate PyTorch ops: a cast to fp32,
a square, a mean, an add, an rsqrt, one or two multiplies and a cast back. A
V4-Flash decode step runs about 280 of these norms on each of two GPUs,
about 2,200 small kernels and 4 to 6 ms of GPU time on a B200. This kernel
does one norm in one launch: one program per row, the whole row in
registers.

It keeps the PyTorch path's casts and their order, but its sum of squares
runs in another order than PyTorch's reduction. So a few outputs differ
from the PyTorch path in the last bit (ADR-045).

CUDA-only. `rmsnorm.RMSNorm(fused=True)` and `rmsnorm.head_rms_norm(fused=True)`
dispatch here when `rms_norm_rows` accepts the input, and keep the PyTorch
path otherwise.
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
# A row stays in registers; wider rows keep the PyTorch path.
_MAX_FEATURES = 16384


def rms_norm_rows(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor | None:
    """`x` as a `(rows, features)` view the kernel takes, or None.

    None means the PyTorch path: not CUDA, no Triton, another dtype, rows
    wider than `_MAX_FEATURES`, a last dim that is not contiguous, rows that
    no 2D view describes, or a tensor that needs a gradient.
    """
    if not _TRITON_AVAILABLE or not is_cuda_device(x.device) or x.ndim == 0:
        return None
    if x.dtype not in _DTYPES or (weight is not None and weight.dtype not in _DTYPES):
        return None
    features = x.shape[-1]
    if not 0 < features <= _MAX_FEATURES or x.stride(-1) != 1:
        return None
    if weight is not None and (weight.shape != (features,) or not weight.is_contiguous()):
        return None
    tensors = (x,) if weight is None else (x, weight)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return None
    try:
        return x.view(-1, features)
    except RuntimeError:
        return None


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _rms_norm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        w_ptr,
        out_ptr,
        n_cols,
        stride_x,
        eps,
        inv_n,
        BLOCK: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        SCALE_FIRST: tl.constexpr,
    ) -> None:
        """One program per row; the row is `BLOCK` wide (masked to `n_cols`).

        `SCALE_FIRST=False` is `RMSNorm`'s order: `(x * r)` is cast to x's
        dtype, then multiplied by the weight. `SCALE_FIRST=True` is the V4
        q-norm's: `r` is cast to x's dtype first, then `x * r`. Each cast and
        each multiply rounds as the PyTorch op does.
        """
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < n_cols
        in_ty = x_ptr.dtype.element_ty
        out_ty = out_ptr.dtype.element_ty
        x = tl.load(x_ptr + row * stride_x + cols, mask=ok, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) * inv_n
        r = tl.math.rsqrt(variance + eps)
        if SCALE_FIRST:
            y = x * r.to(in_ty).to(tl.float32)
        else:
            y = (x * r).to(in_ty).to(tl.float32)
            if HAS_WEIGHT:
                y = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32) * y
        tl.store(out_ptr + row * n_cols + cols, y.to(out_ty), mask=ok)


def fused_rms_norm(
    rows: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor | None,
    eps: float,
    *,
    scale_first: bool = False,
) -> torch.Tensor:
    """One norm in one launch. `rows` comes from `rms_norm_rows`; returns `shape`.

    The output dtype is the PyTorch path's: `promote_types(weight, x)` with a
    weight, else x's. Runs on CPU tensors only under `TRITON_INTERPRET=1`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("fused_rms_norm needs Triton")
    n_rows, n_cols = rows.shape
    dtype = rows.dtype if weight is None else torch.promote_types(weight.dtype, rows.dtype)
    out = torch.empty(shape, dtype=dtype, device=rows.device)
    if n_rows == 0:
        return out
    block = triton.next_power_of_2(n_cols)
    _rms_norm_kernel[(n_rows,)](
        rows,
        rows if weight is None else weight,
        out,
        n_cols,
        rows.stride(0),
        eps,
        1.0 / n_cols,
        BLOCK=block,
        HAS_WEIGHT=weight is not None,
        SCALE_FIRST=scale_first,
        num_warps=min(max(block // 512, 1), 8),
    )
    return out
