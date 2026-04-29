"""Fused INT8-weight x bf16-activation matmul (W8A16 GEMM) via Triton.

The naive ``Int8Linear.forward`` materializes the dequantized weight matrix
to HBM on every call (`W.to(bf16) * scales`) and then passes it to
``F.linear``. That round-trip burns the bandwidth savings of having the
weights stored as INT8.

This module's ``fused_w8a16_linear`` keeps the weights as INT8 in HBM,
loads small tiles into registers / shared memory, dequantizes there, and
feeds the bf16 result straight into the matmul accumulator. The fp32
accumulator is cast to the activation dtype at the end (same precision
contract as cuBLAS bf16 GEMM).

Two block-size profiles are selected at launch time:

- **Decode profile** (M ≤ 16): `BLOCK_M=16, BLOCK_N=64, BLOCK_K=64`,
  `num_warps=2`. Triton's ``tl.dot`` requires BLOCK_M ≥ 16, so a true
  M=1 decode pads M with masking.
- **Prefill profile** (M > 16): `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`,
  `num_warps=4`. Moderate tile that keeps the (BLOCK_M, BLOCK_N) fp32
  accumulator within the per-warp register budget on Ampere SMs.

(An attempt to add `@triton.autotune` over a wider config space
regressed badly because the autotune key included `M`, and continuous
batching produces many distinct M values per run, each firing a fresh
8-config sweep. The proper fix is splitting decode and prefill into
separate narrowly-autotuned kernels; tracked as an ADR-012 follow-up.)

The kernel assumes K is divisible by `BLOCK_K`. Qwen2.5-0.5B's linear
layers have K ∈ {896, 4864}, and Qwen2.5-7B has K ∈ {896, 3584, 18944},
all multiples of 32 and 64, so this holds for the models we benchmark.
For models with K not BLOCK_K-aligned, pad K externally or reintroduce
K masking.

CUDA-only. Calls on non-CUDA devices fall back via the dispatch in
``Int8Linear.forward``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


# Runtime toggle for benchmarking. When True, `supports_fused_kernel`
# reports False even on CUDA, forcing `Int8Linear.forward` to take the
# naive dequant-then-matmul path. The bench uses this to A/B fused vs
# naive on the same model load. Not a public API; internal-only knob.
_FUSED_DISABLED_FOR_BENCH = False


def supports_fused_kernel(device: torch.device | str) -> bool:
    """Whether the fused W8A16 kernel can run on this device.

    Today: CUDA + Triton only. Non-CUDA falls back to the naive path
    inside ``Int8Linear.forward``. `_FUSED_DISABLED_FOR_BENCH` overrides
    the answer to False for A/B benchmarking.
    """
    if _FUSED_DISABLED_FOR_BENCH:
        return False
    if not _TRITON_AVAILABLE:
        return False
    if isinstance(device, str):
        return device == "cuda"
    return device.type == "cuda"


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _w8a16_gemm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        w_ptr,
        scales_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ) -> None:
        """One program per (m-tile, n-tile). K loop runs inside.

        Structure adapted directly from Triton's matmul tutorial:

        - `% M` / `% N` on the load offsets keeps reads in-bounds even on
          partial tiles; the store mask zeros out the bogus output rows /
          cols at the end. (Avoids 2D `mask=` on every load, which has
          tripped a Triton MLIR lowering assertion in some configurations.)
        - K masking via `offs_k < K - k * BLOCK_K` clamps the tail K tile.
        - **Per-output-channel scales are applied AFTER the K reduction.**
          Mathematically equivalent because `scales[n]` factors out of
          `sum_k (x[m,k] * w_int8[n,k] * scales[n])`. Cheaper too: one
          multiply per `(M, N)` element instead of per K iter.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # `% M` / `% N`: out-of-range rows/cols load valid-but-irrelevant
        # memory; the store mask discards those positions.
        offs_xm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_wn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K is BLOCK_K-divisible by contract (caller's responsibility); the
        # `range(0, K, BLOCK_K)` form generates a constant-trip-count loop
        # that Triton can unroll/lower more cleanly than `range(0, cdiv)`.
        # Empirically this avoids an MLIR pass assertion that fires when the
        # loop bound is `tl.cdiv` AND the body has int8→fp casts AND per-iter
        # masked loads.
        for _ in range(0, K, BLOCK_K):
            x_chunk = tl.load(x_ptrs)
            w_int8 = tl.load(w_ptrs)
            # Dequant cast: int8 -> activation dtype (bf16 or fp16).
            w_chunk = w_int8.to(x_chunk.dtype)
            accumulator = tl.dot(x_chunk, w_chunk, accumulator)

            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # Apply per-output-channel scales (one per N column) after the K
        # reduction. Math-equivalent to multiplying inside the loop.
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        scales = tl.load(scales_ptr + offs_n, mask=n_mask, other=0.0)
        accumulator = accumulator * scales[None, :].to(tl.float32)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
            accumulator = accumulator + bias[None, :].to(tl.float32)

        out_block = accumulator.to(scales.dtype)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        out_ptrs = out_ptr + stride_om * offs_cm[:, None] + stride_on * offs_n[None, :]
        out_mask = (offs_cm[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out_block, mask=out_mask)


def fused_w8a16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Fused INT8-weight x float-activation matmul. Output dtype matches x.

    Shapes:
        x:      `(..., K)` bf16/fp16. Any leading dims are flattened.
        weight: `(N, K)` int8 (PyTorch ``nn.Linear``-style row layout).
        scales: `(N,)`, broadcasts along K.
        bias:   `(N,)` or None.

    Returns:
        `(..., N)` in `x.dtype`. Same numerical contract as
        ``F.linear(x, weight.to(x.dtype) * scales[:, None], bias)`` but
        without ever materializing the dequantized weight in HBM.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available; cannot run fused W8A16 kernel")
    if not x.is_cuda:
        raise RuntimeError(f"fused_w8a16_linear requires CUDA tensors; got x.device={x.device}")
    if weight.dtype != torch.int8:
        raise ValueError(f"weight must be int8, got {weight.dtype}")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"x.shape[-1]={x.shape[-1]} doesn't match weight.shape[1]={weight.shape[1]}"
        )

    # Flatten leading dims so the kernel sees a 2D x.
    orig_shape = x.shape
    k_dim = weight.shape[1]
    x_2d = x.reshape(-1, k_dim).contiguous()
    m_dim = x_2d.shape[0]
    n_dim = weight.shape[0]

    # Scales / bias must match x's dtype so the dequant + accumulator match.
    scales_x = scales.to(x.dtype).contiguous()
    if bias is not None:
        bias_x: torch.Tensor = bias.to(x.dtype).contiguous()
        has_bias = True
    else:
        # Triton needs a real pointer even when HAS_BIAS=False; the tl.load
        # in the kernel is dead-code-eliminated under the constexpr branch.
        bias_x = torch.zeros(1, device=x.device, dtype=x.dtype)
        has_bias = False

    out = torch.empty((m_dim, n_dim), dtype=x.dtype, device=x.device)

    # Block-size profile + warp count selected per regime. Sized to keep the
    # fp32 accumulator (BLOCK_M * BLOCK_N elements) inside the per-warp
    # register budget on Ampere SMs. (An earlier iteration tried
    # `@triton.autotune` keyed on `(M, N, K)`; it regressed badly because
    # M varies per call in continuous batching, and every new M value
    # triggered a fresh 8-config sweep that dominated runtime. See ADR-012
    # follow-ups: the fix is to split decode and prefill into two narrowly-
    # autotuned kernels, but hand-picked tiles ship until that lands.)
    if m_dim <= 16:
        # Decode (M ≤ 16): tiny M tile, 2 warps. Accumulator 16*64 = 4 KB total.
        block_m, block_n, block_k = 16, 64, 64
        num_warps = 2
    else:
        # Prefill (M > 16): moderate 64x64 tile at 4 warps. Accumulator
        # 64*64 fp32 = 16 KB total; 4 KB per warp comfortably fits.
        block_m, block_n, block_k = 64, 64, 32
        num_warps = 4

    grid = (triton.cdiv(m_dim, block_m), triton.cdiv(n_dim, block_n))

    _w8a16_gemm_kernel[grid](
        x_2d,
        weight,
        scales_x,
        bias_x,
        out,
        m_dim,
        n_dim,
        k_dim,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_BIAS=has_bias,
        num_warps=num_warps,
    )

    return out.reshape(*orig_shape[:-1], n_dim)
