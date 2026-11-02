"""Fused Triton kernel: Llama-style RoPE on q and k, and the paged K/V write.

The attention block of the Llama-shaped models ran the rotation as a chain of
small ops on q and on k (two multiplies, a negation, a concatenation and an
add each), moved q, k and v to the packed layout with three copies, and wrote
k and v into the paged cache with two indexed stores. Under the decode graph
each of those kernels costs GPU time and launch time. This kernel does all of
it in one launch: one program per (token, head) rotates a q head, or rotates
a k head and stores it with its v head straight into the pool pages.

It keeps the rounding of the PyTorch path. `x * cos`, `rotate_half(x) * sin`
and their sum each round to the tensor dtype, as the separate ops do, and
the launch turns FMA contraction off, so q and the cached k equal the
PyTorch path bit for bit on CUDA.

CUDA-only. `GroupedQueryAttention` dispatches here when
`supports_rope_write_kernel` accepts the tensors and the pool is
uncompressed, and keeps the PyTorch path otherwise.
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
# Half a head stays in registers; wider heads keep the PyTorch path.
_MAX_HALF = 512
# Bench hook, not a public API: True keeps the PyTorch path everywhere.
_DISABLED_FOR_BENCH = False


def supports_rope_write_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> bool:
    """Whether the kernel takes these tensors.

    `q`, `k` and `v` are `(tokens, heads, head_dim)` with a unit stride on the
    last dim, `cos` and `sin` are `(tokens, head_dim)` in the dtype of `q`, as
    `RotaryEmbedding` gives them, the head dim is even and at most
    `2 * _MAX_HALF`, and no tensor needs a gradient.
    """
    if _DISABLED_FOR_BENCH or not _TRITON_AVAILABLE or not is_cuda_device(q.device):
        return False
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or cos.ndim != 2 or sin.ndim != 2:
        return False
    tokens, _, head_dim = q.shape
    if head_dim % 2 or not 0 < head_dim // 2 <= _MAX_HALF:
        return False
    if k.shape[0] != tokens or k.shape[-1] != head_dim or v.shape != k.shape:
        return False
    if cos.shape != (tokens, head_dim) or sin.shape != (tokens, head_dim):
        return False
    tensors = (q, k, v, cos, sin)
    if q.dtype not in _DTYPES or any(t.dtype != q.dtype for t in tensors):
        return False
    if any(t.stride(-1) != 1 for t in tensors):
        return False
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _rope_write_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        k_ptr,
        v_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_pool_ptr,
        v_pool_ptr,
        block_ids_ptr,
        slots_ptr,
        num_q_heads,
        num_kv_heads,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        stride_vt,
        stride_vh,
        stride_ct,
        stride_st,
        stride_ot,
        stride_oh,
        stride_pb,
        stride_ps,
        stride_ph,
        HALF: tl.constexpr,
        HALF_BLOCK: tl.constexpr,
    ) -> None:
        """One program per (token, head). Heads `[0, num_q_heads)` are q heads, the
        rest are kv heads. Every tensor has stride 1 on its last dim.

        `rotate_half(x)` is `(-x2, x1)` for the halves `x1`, `x2` of a head, and
        the tables repeat their first half, so `out1 = x1 * cos + (-x2) * sin` and
        `out2 = x2 * cos + x1 * sin` with `cos`, `sin` the first half. Each
        product and each sum rounds to the tensor dtype, as the PyTorch ops do.
        """
        pid = tl.program_id(0).to(tl.int64)
        heads = num_q_heads + num_kv_heads
        t = pid // heads
        h = pid % heads
        cols = tl.arange(0, HALF_BLOCK)
        ok = cols < HALF
        cos = tl.load(cos_ptr + t * stride_ct + cols, mask=ok, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + t * stride_st + cols, mask=ok, other=0.0).to(tl.float32)
        out_ty = q_out_ptr.dtype.element_ty
        if h < num_q_heads:
            row = q_ptr + t * stride_qt + h * stride_qh
            x1 = tl.load(row + cols, mask=ok, other=0.0).to(tl.float32)
            x2 = tl.load(row + HALF + cols, mask=ok, other=0.0).to(tl.float32)
            a1 = (x1 * cos).to(out_ty).to(tl.float32)
            b1 = ((-x2) * sin).to(out_ty).to(tl.float32)
            a2 = (x2 * cos).to(out_ty).to(tl.float32)
            b2 = (x1 * sin).to(out_ty).to(tl.float32)
            out_row = q_out_ptr + t * stride_ot + h * stride_oh
            tl.store(out_row + cols, (a1 + b1).to(out_ty), mask=ok)
            tl.store(out_row + HALF + cols, (a2 + b2).to(out_ty), mask=ok)
        else:
            kh = h - num_q_heads
            pool_ty = k_pool_ptr.dtype.element_ty
            block = tl.load(block_ids_ptr + t).to(tl.int64)
            slot = tl.load(slots_ptr + t).to(tl.int64)
            pool_row = block * stride_pb + slot * stride_ps + kh * stride_ph
            k_row = k_ptr + t * stride_kt + kh * stride_kh
            x1 = tl.load(k_row + cols, mask=ok, other=0.0).to(tl.float32)
            x2 = tl.load(k_row + HALF + cols, mask=ok, other=0.0).to(tl.float32)
            a1 = (x1 * cos).to(out_ty).to(tl.float32)
            b1 = ((-x2) * sin).to(out_ty).to(tl.float32)
            a2 = (x2 * cos).to(out_ty).to(tl.float32)
            b2 = (x1 * sin).to(out_ty).to(tl.float32)
            tl.store(k_pool_ptr + pool_row + cols, (a1 + b1).to(out_ty).to(pool_ty), mask=ok)
            tl.store(k_pool_ptr + pool_row + HALF + cols, (a2 + b2).to(out_ty).to(pool_ty), mask=ok)
            v_row = v_ptr + t * stride_vt + kh * stride_vh
            v1 = tl.load(v_row + cols, mask=ok, other=0.0)
            v2 = tl.load(v_row + HALF + cols, mask=ok, other=0.0)
            tl.store(v_pool_ptr + pool_row + cols, v1.to(pool_ty), mask=ok)
            tl.store(v_pool_ptr + pool_row + HALF + cols, v2.to(pool_ty), mask=ok)


def rope_and_write_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_ids: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Rotate `q` and `k`, store the rotated `k` and `v` at `(block_ids[t], slots[t])`
    of the pools, and return the rotated `q`, contiguous `(tokens, heads, head_dim)`.

    `supports_rope_write_kernel` must have accepted the tensors. The pools are
    `(num_blocks, block_size, num_kv_heads, head_dim)` with the same strides,
    as `BlockPool.storage_for_layer` gives them. `block_ids` and `slots` hold
    one entry per token, int32 or int64. Runs on CPU tensors only under
    `TRITON_INTERPRET=1`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("rope_and_write_kv needs Triton")
    tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if k_pool.stride() != v_pool.stride() or k_pool.stride(-1) != 1:
        raise ValueError("the K and V pools must share a layout with a unit last stride")
    if block_ids.shape[0] < tokens or slots.shape[0] < tokens:
        raise ValueError(f"{tokens} tokens but {block_ids.shape[0]} pool positions")
    q_out = torch.empty((tokens, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    if tokens == 0:
        return q_out
    half = head_dim // 2
    half_block = triton.next_power_of_2(half)
    _rope_write_kernel[(tokens * (num_q_heads + num_kv_heads),)](
        q,
        k,
        v,
        cos,
        sin,
        q_out,
        k_pool,
        v_pool,
        block_ids,
        slots,
        num_q_heads,
        num_kv_heads,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos.stride(0),
        sin.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        HALF=half,
        HALF_BLOCK=half_block,
        num_warps=min(max(half_block // 128, 1), 4),
        enable_fp_fusion=False,
    )
    return q_out
