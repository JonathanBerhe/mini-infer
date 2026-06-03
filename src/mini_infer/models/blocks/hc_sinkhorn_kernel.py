"""Fused Triton kernel for `hc_split_sinkhorn` (DeepSeek-V4 Hyper-Connections).

The pure-PyTorch transcription in ``hyper_connections._hc_split_sinkhorn_torch``
expresses each step of the V4 reference's `hc_split_sinkhorn` tilelang
kernel as a separate PyTorch op (sigmoid, softmax, alternating row/col
normalizations, ...). For V4-Flash (61 layers, 2 HC sublayers per layer)
that's roughly 50 elementwise + reduction ops per call, 122 calls per
token. The compute per call is trivial (a small ``(hc, hc)`` matrix
iterated 20 times) but the kernel-launch count is what dominates.

This module fuses the whole sequence into one Triton kernel:

    Input:  mixes (n, mix_hc), hc_scale (3,), hc_base (mix_hc,)
    Output: pre (n, hc), post (n, hc), comb (n, hc, hc)

where ``n = B*T`` and ``mix_hc = (2 + hc) * hc``.

One program per row. The ``(hc, hc)`` comb matrix stays in registers
for the entire Sinkhorn iteration; no HBM round-trip between iterations.

CUDA-only. Non-CUDA callers fall back via the dispatch in
``hyper_connections.hc_split_sinkhorn``.
"""

from __future__ import annotations

import torch

from mini_infer.device import is_cuda_device, require_cuda_device

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


def supports_hc_kernel(device: torch.device | str, hc_mult: int) -> bool:
    """Whether the fused HC Sinkhorn kernel can run on this device + shape.

    Today: CUDA + Triton, with ``hc_mult`` a power of 2 (so ``tl.arange``
    on the inner tile is well-formed without masking). V4 uses hc=4,
    which fits. Non-power-of-2 hc values (e.g. hc=3 in shape-contract
    tests) fall back to the PyTorch path.
    """
    if not _TRITON_AVAILABLE:
        return False
    if not is_cuda_device(device):
        return False
    if hc_mult <= 0:
        return False
    return (hc_mult & (hc_mult - 1)) == 0


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _hc_split_sinkhorn_kernel(  # type: ignore[no-untyped-def]
        mixes_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        pre_ptr,
        post_ptr,
        comb_ptr,
        eps,
        HC: tl.constexpr,
        SINKHORN_ITERS: tl.constexpr,
    ) -> None:
        """One program per row of ``(n, mix_hc)``. Grid is exactly ``(n,)``
        so no bounds guard is needed.

        Matches the V4 reference's tilelang kernel line-by-line:

        - ``pre[j]  = sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + eps``
        - ``post[j] = 2 * sigmoid(mixes[j+hc] * hc_scale[1] + hc_base[j+hc])``
        - ``comb[j,k] = mixes[2*hc + j*hc + k] * hc_scale[2] + hc_base[2*hc + j*hc + k]``
        - First Sinkhorn iter: row-softmax + eps, then column normalize.
        - Next ``SINKHORN_ITERS - 1`` iters: row normalize, then column
          normalize, each with an additive eps in the denominator.

        ``HC`` is a constexpr so the compiler unrolls the ``(HC, HC)``
        tile arithmetic. ``eps`` stays a runtime scalar so it doesn't
        enter the JIT specialization key.

        The Sinkhorn loop is the SCALE-VECTOR formulation: the loop
        carries two 1D vectors (``row_scale``, ``col_scale``) and the
        matrix stays loop-invariant. The mathematically direct version
        (divide the carried matrix by its own row/col sums each
        iteration) segfaults Triton 3.1.0's compiler: any reduction
        whose operand chains back to a loop-carried 2D tile crashes
        codegen, regardless of reduction axis. Bisected empirically on
        L40S; see ``scripts/modal_hc_kernel_probe.py`` for the variant
        matrix. The scale-vector shape (carried 1D stats, reductions on
        body-local 2D values) is the same loop structure FlashAttention
        kernels use and compiles reliably on this stack.
        """
        row = tl.program_id(0)

        # Per-chunk scalars (read once per row). The hc_scale tensor has
        # exactly 3 entries; load each by direct index rather than via
        # tl.arange (which would over-read).
        scale_pre = tl.load(hc_scale_ptr + 0)
        scale_post = tl.load(hc_scale_ptr + 1)
        scale_comb = tl.load(hc_scale_ptr + 2)

        offs_hc = tl.arange(0, HC)
        mix_hc = (2 + HC) * HC
        row_mix_ptr = mixes_ptr + row * mix_hc

        # --- pre: sigmoid + eps, indices [0..HC) ---
        pre_features = tl.load(row_mix_ptr + offs_hc)
        pre_base = tl.load(hc_base_ptr + offs_hc)
        pre = tl.sigmoid(pre_features * scale_pre + pre_base) + eps
        tl.store(pre_ptr + row * HC + offs_hc, pre)

        # --- post: 2 * sigmoid, indices [HC..2*HC) ---
        post_features = tl.load(row_mix_ptr + HC + offs_hc)
        post_base = tl.load(hc_base_ptr + HC + offs_hc)
        post = 2.0 * tl.sigmoid(post_features * scale_post + post_base)
        tl.store(post_ptr + row * HC + offs_hc, post)

        # --- comb: (HC, HC) tile from indices [2*HC..2*HC + HC*HC) ---
        # 2D offsets into the per-row mixes / base for the comb chunk.
        offs_j = tl.arange(0, HC)
        offs_k = tl.arange(0, HC)
        comb_offs = 2 * HC + offs_j[:, None] * HC + offs_k[None, :]
        comb_features = tl.load(row_mix_ptr + comb_offs)
        comb_base = tl.load(hc_base_ptr + comb_offs)
        comb = comb_features * scale_comb + comb_base

        # --- First iteration step 1: row-softmax-with-eps. ---
        # tl.max / tl.sum collapse the named axis; broadcasting back with
        # [:, None] reproduces the row vector across columns.
        row_max = tl.max(comb, axis=1)
        comb = tl.exp(comb - row_max[:, None])
        row_sum = tl.sum(comb, axis=1)
        comb = comb / row_sum[:, None] + eps

        # --- First iteration step 2: column-normalize with eps. ---
        # Expressed as a scale vector instead of dividing the matrix:
        # dividing every column k by (col_sum[k] + eps) is the same as
        # multiplying the matrix by col_scale broadcast along rows.
        col_sum0 = tl.sum(comb, axis=0)
        col_scale = 1.0 / (col_sum0 + eps)
        row_scale = tl.full((HC,), 1.0, dtype=tl.float32)

        # --- Remaining iterations: row-then-column normalize. ---
        # The loop carries ONLY the 1D scale vectors. `comb` stays
        # loop-invariant; each reduction operates on `scaled`, a fresh
        # body-local value. The current matrix at any step is exactly
        # comb * row_scale[:, None] * col_scale[None, :], so the row /
        # col sums equal the reference's sums over its sequentially
        # divided matrix. FP reassociation drift (multiplicative
        # composition vs sequential division) measured at ~4e-7
        # max-abs on FP32, far inside the parity tolerance.
        for _ in range(SINKHORN_ITERS - 1):
            scaled = comb * row_scale[:, None] * col_scale[None, :]
            row_sum = tl.sum(scaled, axis=1)
            row_scale = row_scale / (row_sum + eps)
            scaled2 = comb * row_scale[:, None] * col_scale[None, :]
            col_sum = tl.sum(scaled2, axis=0)
            col_scale = col_scale / (col_sum + eps)

        comb = comb * row_scale[:, None] * col_scale[None, :]

        tl.store(
            comb_ptr + row * HC * HC + offs_j[:, None] * HC + offs_k[None, :],
            comb,
        )


def hc_split_sinkhorn_triton(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Triton entry point for ``hc_split_sinkhorn``.

    Same input / output contract as the PyTorch reference in
    ``hyper_connections._hc_split_sinkhorn_torch``. Caller is responsible
    for ensuring ``supports_hc_kernel(device, hc_mult)`` is True;
    otherwise dispatch to the PyTorch path.

    Shapes:
        mixes:    ``(..., (2 + hc_mult) * hc_mult)`` FP32 on CUDA.
        hc_scale: ``(3,)`` FP32 on CUDA.
        hc_base:  ``((2 + hc_mult) * hc_mult,)`` FP32 on CUDA.

    Returns:
        ``(pre, post, comb)`` with shapes
        ``(..., hc_mult)``, ``(..., hc_mult)``, ``(..., hc_mult, hc_mult)``.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available; cannot run hc_split_sinkhorn kernel")
    require_cuda_device(mixes.device, "hc_split_sinkhorn kernel")
    if (hc_mult & (hc_mult - 1)) != 0 or hc_mult <= 0:
        raise ValueError(
            f"hc_mult must be a positive power of 2 for the Triton kernel, got {hc_mult}"
        )
    if hc_scale.shape != (3,):
        raise ValueError(f"hc_scale must have shape (3,); got {tuple(hc_scale.shape)}")
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != mix_hc:
        raise ValueError(
            f"mixes.shape[-1]={mixes.shape[-1]} must equal (2 + hc_mult) * hc_mult = {mix_hc}"
        )
    if hc_base.shape != (mix_hc,):
        raise ValueError(f"hc_base must have shape ({mix_hc},); got {tuple(hc_base.shape)}")

    # The reference flattens leading dims; we do the same so the kernel
    # sees a 2D (n_rows, mix_hc) view. Output shapes carry the original
    # leading dims back.
    leading_shape = mixes.shape[:-1]
    n_rows = 1
    for d in leading_shape:
        n_rows *= d

    mixes_2d = mixes.reshape(n_rows, mix_hc).contiguous()
    hc_scale_c = hc_scale.contiguous()
    hc_base_c = hc_base.contiguous()

    pre = torch.empty((n_rows, hc_mult), dtype=mixes.dtype, device=mixes.device)
    post = torch.empty((n_rows, hc_mult), dtype=mixes.dtype, device=mixes.device)
    comb = torch.empty((n_rows, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device)

    # One program per row. The (hc_mult, hc_mult) tile fits in registers
    # for HC up to ~32. Default num_warps (4): the tile is tiny but
    # Triton's reduction lowering is best-exercised at the default; the
    # extra warps idle harmlessly. (num_warps=1 is an untested corner of
    # the reduction codegen and not worth the risk for zero measurable
    # gain on a launch-overhead-bound kernel.)
    grid = (n_rows,)
    _hc_split_sinkhorn_kernel[grid](
        mixes_2d,
        hc_scale_c,
        hc_base_c,
        pre,
        post,
        comb,
        eps,
        HC=hc_mult,
        SINKHORN_ITERS=sinkhorn_iters,
    )

    return (
        pre.reshape(*leading_shape, hc_mult),
        post.reshape(*leading_shape, hc_mult),
        comb.reshape(*leading_shape, hc_mult, hc_mult),
    )
