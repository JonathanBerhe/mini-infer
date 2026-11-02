"""INT8-weight x bf16-activation matmul (W8A16) for the decode step, via Triton.

The naive ``Int8Linear.forward`` materializes the dequantized weight matrix
to HBM on every call (`W.to(bf16) * scales`) and then passes it to
``F.linear``. That round-trip burns the bandwidth savings of having the
weights stored as INT8.

``fused_w8a16_linear`` keeps the weights as INT8 in HBM for the shapes where
the weight bytes set the time: a decode step of up to `_DECODE_ROWS` rows.
Two kernels serve it. `_w8a16_splitk_kernel` runs one program per (n-tile,
K split), loads int8 tiles, dequantizes them in registers and accumulates in
fp32; the K split gives enough programs in flight to keep the memory system
busy on the narrow shapes (`down_proj` of Qwen2.5-7B has N=3584 and
K=18944). `_w8a16_reduce_kernel` sums the splits in a fixed order, applies
the per-channel scale and the bias, and casts, so the result is
deterministic. On one H100 the pair reads the int8 weights of the 7B MLP
shapes at 2.1 to 2.25 TB/s, 1.6 to 1.7 times faster than cuBLAS bf16.

Above `_DECODE_ROWS` rows (a prefill chunk) the GEMM is compute-bound and
the byte saving does not pay, so the weight is dequantized to the activation
dtype and cuBLAS runs it. A K that the tile does not divide takes the same
path.

Runs on CUDA; on CPU tensors only under `TRITON_INTERPRET=1`. Calls on
other devices fall back via the dispatch in ``Int8Linear.forward``.
"""

from __future__ import annotations

import dataclasses
import os

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


# Runtime toggle for benchmarking. When True, `supports_fused_kernel`
# reports False even on CUDA, forcing `Int8Linear.forward` to take the
# naive dequant-then-matmul path. The bench uses this to A/B fused vs
# naive on the same model load. Not a public API; internal-only knob.
_FUSED_DISABLED_FOR_BENCH = False

# The split-K kernel serves up to this many rows; `tl.dot` pads them to 16, 32 or 64.
_DECODE_ROWS = 64
# The split-K path aims at this many programs, about four per SM of an H100.
_DECODE_TARGET_PROGRAMS = 512
_REDUCE_BLOCK_N = 1024


@dataclasses.dataclass(frozen=True)
class DecodeTile:
    """The tile of the split-K decode kernel. `split_k=0` picks the split from the shape."""

    block_n: int = 128
    block_k: int = 128
    split_k: int = 0
    num_warps: int = 8
    num_stages: int = 3


# The tiles come from sweeps over the Qwen2.5-7B shapes at 1 and 8 rows, with
# the calls captured in a CUDA graph. On one H100, 128 by 128 tiles with 8
# warps sat within a few percent of the best tile of every shape, at 2.1 to
# 2.25 TB/s of int8 reads on the MLP shapes. On one B200 the best tiles use 4
# warps, and the long-K shape (`down_proj`, K=18944) wants 64-wide K tiles
# with an 8-way split and one more pipeline stage.
_HOPPER_TILE = DecodeTile(block_n=128, block_k=128, split_k=0, num_warps=8, num_stages=3)
_BLACKWELL_TILE = DecodeTile(block_n=128, block_k=128, split_k=0, num_warps=4, num_stages=3)
_BLACKWELL_LONG_K_TILE = DecodeTile(block_n=128, block_k=64, split_k=0, num_warps=4, num_stages=4)
_LONG_K = 8192

# A tile set here wins over `decode_tile_for`; a bench swaps it to sweep the configurations.
_DECODE_TILE: DecodeTile | None = None


def decode_tile_for(device: torch.device | str, k_dim: int) -> DecodeTile:
    """The tile for this GPU and this K: Blackwell gets its own, every other device the H100's."""
    if _DECODE_TILE is not None:
        return _DECODE_TILE
    if is_cuda_device(device) and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 10:
            return _BLACKWELL_LONG_K_TILE if k_dim >= _LONG_K else _BLACKWELL_TILE
    return _HOPPER_TILE


def split_k_for(k_dim: int, block_k: int, n_tiles: int) -> int:
    """The largest power-of-two split that lifts the program count toward the target.

    K must stay divisible by `split * block_k`, so every program sees whole
    tiles and the kernel needs no K mask.
    """
    split = 1
    while n_tiles * split < _DECODE_TARGET_PROGRAMS and k_dim % (split * 2 * block_k) == 0:
        split *= 2
    return split


def block_m_for(m_dim: int) -> int:
    """The smallest of 16, 32 and 64 that holds `m_dim` rows."""
    for block_m in (16, 32, 64):
        if m_dim <= block_m:
            return block_m
    raise ValueError(f"{m_dim} rows exceed the {_DECODE_ROWS} rows of the decode kernel")


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
    return is_cuda_device(device)


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _w8a16_splitk_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        w_ptr,
        part_ptr,
        M,
        N,
        K_PER_SPLIT,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ps,
        stride_pm,
        stride_pn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        """One program per (n-tile, K split), partial sums in fp32.

        Each program keeps one `(BLOCK_M, BLOCK_N)` accumulator over its
        slice of K and stores it to `part[split, m, n]`. Rows at or past `M`
        load a real row again (`% M`) and are dropped at the store, as are
        columns past `N`.
        """
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_m = tl.arange(0, BLOCK_M)
        offs_xm = offs_m % M
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_wn = offs_n % N
        offs_k = pid_s * K_PER_SPLIT + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K_PER_SPLIT, BLOCK_K):
            x_chunk = tl.load(x_ptrs)
            w_int8 = tl.load(w_ptrs)
            accumulator = tl.dot(x_chunk, w_int8.to(x_chunk.dtype), accumulator)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        part_ptrs = (
            part_ptr + pid_s * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        )
        part_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(part_ptrs, accumulator, mask=part_mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _w8a16_reduce_kernel(  # type: ignore[no-untyped-def]
        part_ptr,
        scales_ptr,
        bias_ptr,
        out_ptr,
        N,
        SPLIT_K,
        stride_ps,
        stride_pm,
        stride_pn,
        stride_om,
        stride_on,
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ) -> None:
        """Sum the K splits of one row, apply the per-channel scale and the bias, cast."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        total = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for split in range(0, SPLIT_K):
            total += tl.load(
                part_ptr + split * stride_ps + pid_m * stride_pm + offs_n * stride_pn,
                mask=n_mask,
                other=0.0,
            )
        scales = tl.load(scales_ptr + offs_n, mask=n_mask, other=0.0)
        total = total * scales.to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
            total = total + bias.to(tl.float32)
        tl.store(
            out_ptr + pid_m * stride_om + offs_n * stride_on,
            total.to(out_ptr.dtype.element_ty),
            mask=n_mask,
        )


def fused_w8a16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """INT8-weight x float-activation matmul. Output dtype matches x.

    Shapes:
        x:      `(..., K)` bf16/fp16. Any leading dims are flattened.
        weight: `(N, K)` int8 (PyTorch ``nn.Linear``-style row layout).
        scales: `(N,)`, broadcasts along K.
        bias:   `(N,)` or None.

    Returns:
        `(..., N)` in `x.dtype`. Same numerical contract as
        ``F.linear(x, weight.to(x.dtype) * scales[:, None], bias)``. Up to
        `_DECODE_ROWS` rows the split-K kernels read the int8 weight as is;
        more rows, or a K that the tile does not divide, dequantize it and
        run cuBLAS.

    Runs on CPU tensors only under `TRITON_INTERPRET=1`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available; cannot run fused W8A16 kernel")
    if os.environ.get("TRITON_INTERPRET") != "1":
        require_cuda_device(x.device, "fused_w8a16_linear")
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

    tile = decode_tile_for(x.device, k_dim)
    if m_dim > _DECODE_ROWS or k_dim % tile.block_k != 0:
        w_dq = weight.to(x.dtype) * scales.to(x.dtype).unsqueeze(1)
        bias_dq = bias.to(x.dtype) if bias is not None else None
        return torch.nn.functional.linear(x, w_dq, bias_dq)

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
    _decode_split_k(x_2d, weight, scales_x, bias_x, has_bias, out, tile)
    return out.reshape(*orig_shape[:-1], n_dim)


def _decode_split_k(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
    out: torch.Tensor,
    tile: DecodeTile,
) -> None:
    """Two launches: partial sums per K split into fp32, then the reduce into `out`."""
    m_dim, k_dim = x_2d.shape
    n_dim = weight.shape[0]
    n_tiles = triton.cdiv(n_dim, tile.block_n)
    split = tile.split_k if tile.split_k > 0 else split_k_for(k_dim, tile.block_k, n_tiles)
    if k_dim % (split * tile.block_k) != 0:
        raise ValueError(f"K={k_dim} is not divisible by split {split} x block {tile.block_k}")
    part = torch.empty((split, m_dim, n_dim), dtype=torch.float32, device=x_2d.device)
    _w8a16_splitk_kernel[(n_tiles, split)](
        x_2d,
        weight,
        part,
        m_dim,
        n_dim,
        k_dim // split,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        part.stride(0),
        part.stride(1),
        part.stride(2),
        BLOCK_M=block_m_for(m_dim),
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
    )
    _w8a16_reduce_kernel[(m_dim, triton.cdiv(n_dim, _REDUCE_BLOCK_N))](
        part,
        scales,
        bias,
        out,
        n_dim,
        split,
        part.stride(0),
        part.stride(1),
        part.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=_REDUCE_BLOCK_N,
        HAS_BIAS=has_bias,
        num_warps=4,
    )
