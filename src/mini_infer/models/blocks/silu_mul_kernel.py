"""Fused Triton kernel for `silu(gate) * up`, the gate of a SwiGLU FFN.

The PyTorch path runs two kernels, the SiLU and the product, and writes the
SiLU output to memory in between. In a decode step of Qwen2.5-7B the pair is
56 of about 460 kernels. This kernel does both in one launch.

It repeats PyTorch's arithmetic on CUDA: the SiLU is `x / (1 + exp(-x))` in
fp32 with the accurate libdevice exp and an IEEE division, rounded to the
input dtype, and the product of that rounded value and `up` runs in fp32 and
rounds again. The Triton interpreter has no libdevice, so under
`TRITON_INTERPRET=1` the kernel uses `tl.exp`, which is close but not exact.

CUDA-only. `activations.swiglu` dispatches here when `silu_mul_rows` accepts
the inputs, and keeps the PyTorch path otherwise.
"""

from __future__ import annotations

import os

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

try:
    from triton.language.extra.cuda import libdevice
except ImportError:  # a Triton build without the CUDA extras
    libdevice = None

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_BLOCK = 1024


def silu_mul_rows(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """`gate` and `up` as `(rows, features)` views the kernel takes, or None.

    None means the PyTorch path: not CUDA, no Triton, shapes or dtypes that
    differ, another dtype, a last dim that is not contiguous, rows that no 2D
    view describes, or a tensor that needs a gradient.
    """
    if not _TRITON_AVAILABLE or libdevice is None:
        return None
    if not is_cuda_device(gate.device) or gate.device != up.device:
        return None
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.dtype not in _DTYPES:
        return None
    if gate.ndim == 0 or gate.shape[-1] == 0 or gate.stride(-1) != 1 or up.stride(-1) != 1:
        return None
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        return None
    features = gate.shape[-1]
    try:
        return gate.view(-1, features), up.view(-1, features)
    except RuntimeError:
        return None


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _silu_mul_kernel(  # type: ignore[no-untyped-def]
        gate_ptr,
        up_ptr,
        out_ptr,
        n_cols,
        stride_gate,
        stride_up,
        BLOCK: tl.constexpr,
        ACCURATE_EXP: tl.constexpr,
    ) -> None:
        """One program per (row, block of `BLOCK` columns)."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = cols < n_cols
        out_ty = out_ptr.dtype.element_ty
        gate = tl.load(gate_ptr + row * stride_gate + cols, mask=ok, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + row * stride_up + cols, mask=ok, other=0.0).to(tl.float32)
        exp_neg = libdevice.exp(-gate) if ACCURATE_EXP else tl.exp(-gate)
        silu = tl.div_rn(gate, 1.0 + exp_neg).to(out_ty).to(tl.float32)
        tl.store(out_ptr + row * n_cols + cols, (silu * up).to(out_ty), mask=ok)


def fused_silu_mul(
    gate_rows: torch.Tensor, up_rows: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    """`silu(gate) * up` in one launch, in `shape` and the dtype of `gate`.

    The rows come from `silu_mul_rows`. Runs on CPU tensors only under
    `TRITON_INTERPRET=1`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("fused_silu_mul needs Triton")
    n_rows, n_cols = gate_rows.shape
    out = torch.empty(shape, dtype=gate_rows.dtype, device=gate_rows.device)
    if n_rows == 0:
        return out
    _silu_mul_kernel[(n_rows, triton.cdiv(n_cols, _BLOCK))](
        gate_rows,
        up_rows,
        out,
        n_cols,
        gate_rows.stride(0),
        up_rows.stride(0),
        BLOCK=_BLOCK,
        ACCURATE_EXP=os.environ.get("TRITON_INTERPRET") != "1",
        num_warps=4,
    )
    return out
