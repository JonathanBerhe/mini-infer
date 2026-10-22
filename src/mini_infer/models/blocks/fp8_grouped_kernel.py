"""Grouped matmul over MoE experts with block-FP8 weights, in Triton.

`grouped_expert_mlp` sorts the (token, expert) pairs by expert. With bf16
weights each projection is one `functional.grouped_mm`. PyTorch's FP8
grouped matmul takes only row-wise or 1x32 / 1x16 block scales, and the
GLM-5 and MiniMax-M3 checkpoints ship e4m3 weights with one fp32 scale per
128x128 block. Using it would mean quantizing the weights again and the
activations too. This kernel keeps the checkpoint's weights and the
activations as they are.

A program loads one expert's weight tiles, dequantizes them in registers
(`e4m3 * scale` in fp32, rounded to bf16, as `Fp8Expert` does), and
multiplies them with that expert's rows on the bf16 tensor cores (W8A16).
The weights are the left operand of the dot (`out.T = W @ rows.T`):
Hopper's `wgmma` takes its left operand from registers, where the
dequantized tile already is, and only its right operand from shared memory.

Two ways to split the work, both without a host sync:

- Per segment, for few rows (decode). The grid has one program per row and
  per column tile. A program runs only where a segment starts: the first
  row of an expert's group, then every `BLOCK_M` rows after it. Each
  segment gets its own program, so the experts of a batch load in parallel.
- Per row tile, for many rows (prefill). One program computes a
  `(BLOCK_M, BLOCK_N)` tile. Its rows can belong to several experts, so it
  walks them in order and masks the rows of the other experts. A grid of one
  program per row would be mostly idle here.

Rows routed to another rank's experts (expert parallelism) sort after the
local groups and get zeros.

On Hopper, the per-row-tile split goes to the Gluon kernel of
`fp8_grouped_gluon.py` when it takes the operands.
"""

from __future__ import annotations

import functools
import os

import torch

from mini_infer.device import is_blackwell_device, is_cuda_device
from mini_infer.models.blocks import fp8_grouped_gluon

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


Tiles = tuple[int, int, int, int, int]

# Tiles as (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for three batch
# sizes: per segment up to `_SMALL_ROWS` rows, per row tile up to
# `_MEDIUM_ROWS`, and per row tile above. Picked from tile sweeps of the
# GLM-5.3 layer on an H100 and a B200
# (`docs/benchmarks/2026-10-21-moe-grouped-gemm.md`). The two GPUs agree on
# the largest batches only.
_HOPPER_TILES: tuple[Tiles, Tiles, Tiles] = (
    (16, 64, 128, 4, 3),
    (64, 128, 128, 4, 3),
    (128, 256, 64, 8, 3),
)
_BLACKWELL_TILES: tuple[Tiles, Tiles, Tiles] = (
    (16, 128, 64, 4, 3),
    (64, 128, 64, 4, 4),
    (128, 256, 64, 8, 3),
)
_SMALL_ROWS = 256
_MEDIUM_ROWS = 8192
# Bench hooks, not a public API: force a tile, the operand order, the split,
# or the Gluon kernel on Hopper (True: also for few rows, False: never).
_TILES_FOR_BENCH: Tiles | None = None
_SWAP_AB_FOR_BENCH: bool | None = None
_PER_SEGMENT_FOR_BENCH: bool | None = None
_GLUON_FOR_BENCH: bool | None = None


def supports_block_fp8_grouped_mm(device: torch.device | str) -> bool:
    """Whether `block_fp8_grouped_mm` can run on this device."""
    return _TRITON_AVAILABLE and is_cuda_device(device)


@functools.cache
def _tile_table(device: torch.device) -> tuple[Tiles, Tiles, Tiles]:
    """The tiles for this GPU family; Hopper's off CUDA (the interpreter)."""
    return _BLACKWELL_TILES if is_blackwell_device(device) else _HOPPER_TILES


def tiles_for(device: torch.device, num_rows: int) -> Tiles:
    """The tile `block_fp8_grouped_mm` uses for `num_rows` rows on `device`."""
    small, medium, large = _tile_table(device)
    if num_rows <= _SMALL_ROWS:
        return small
    return medium if num_rows <= _MEDIUM_ROWS else large


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _dot(a, b, acc, DOT_IN_FP32: tl.constexpr):  # type: ignore[no-untyped-def]
        if DOT_IN_FP32:
            out = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
        else:
            out = tl.dot(a, b, acc)
        return out

    @triton.jit  # type: ignore[untyped-decorator]
    def _expert_rows_dot(  # type: ignore[no-untyped-def]
        acc,
        a_ptr,
        w_base,
        s_base,
        offs_m,
        row_start,
        row_end,
        offs_n,
        n_ok,
        k_dim,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sk,
        QBLOCK: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SWAP_AB: tl.constexpr,
        DOT_IN_FP32: tl.constexpr,
    ):
        """Add rows `[row_start, row_end)` times one expert's weights to `acc`."""
        in_rows = (offs_m >= row_start) & (offs_m < row_end)
        for k_start in range(0, k_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_ok = offs_k < k_dim
            # BLOCK_K divides QBLOCK, so one scale per output column.
            scale = tl.load(
                s_base + (offs_n // QBLOCK) * stride_sn + (k_start // QBLOCK) * stride_sk,
                mask=n_ok,
                other=0.0,
            )
            if SWAP_AB:
                w = tl.load(
                    w_base + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_ok[:, None] & k_ok[None, :],
                    other=0,
                ).to(tl.float8e4nv, bitcast=True)
                a_t = tl.load(
                    a_ptr + offs_m[None, :] * stride_am + offs_k[:, None] * stride_ak,
                    mask=k_ok[:, None] & in_rows[None, :],
                    other=0.0,
                )
                w_deq = (w.to(tl.float32) * scale[:, None]).to(tl.bfloat16).to(a_t.dtype)
                acc = _dot(w_deq, a_t, acc, DOT_IN_FP32)
            else:
                a = tl.load(
                    a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=in_rows[:, None] & k_ok[None, :],
                    other=0.0,
                )
                w = tl.load(
                    w_base + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                    mask=k_ok[:, None] & n_ok[None, :],
                    other=0,
                ).to(tl.float8e4nv, bitcast=True)
                w_deq = (w.to(tl.float32) * scale[None, :]).to(tl.bfloat16).to(a.dtype)
                acc = _dot(a, w_deq, acc, DOT_IN_FP32)
        return acc

    @triton.jit  # type: ignore[untyped-decorator]
    def _block_fp8_grouped_mm_kernel(  # type: ignore[no-untyped-def]
        a_ptr,
        w_ptr,
        s_ptr,
        row_expert_ptr,
        group_end_ptr,
        out_ptr,
        rows,
        n_dim,
        k_dim,
        num_groups,
        stride_am,
        stride_ak,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_sk,
        stride_om,
        stride_on,
        QBLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PER_SEGMENT: tl.constexpr,
        SWAP_AB: tl.constexpr,
        DOT_IN_FP32: tl.constexpr,
    ):
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < n_dim
        if SWAP_AB:
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if PER_SEGMENT:
            start = tl.program_id(0)
            expert = tl.load(row_expert_ptr + start).to(tl.int32)
            on_rank = expert < num_groups
            safe = tl.minimum(expert, num_groups - 1)
            previous_end = tl.load(group_end_ptr + tl.maximum(safe - 1, 0))
            own_start = tl.where(safe > 0, previous_end, 0)
            # Off-rank rows form one last group, from the end of the local ones.
            group_start = tl.where(on_rank, own_start, tl.load(group_end_ptr + num_groups - 1))
            group_end = tl.where(on_rank, tl.load(group_end_ptr + safe), rows)
            leads = (start - group_start) % BLOCK_M == 0
            row_end = tl.minimum(start + BLOCK_M, group_end)
            offs_m = start + tl.arange(0, BLOCK_M)
            if leads & on_rank:
                acc = _expert_rows_dot(
                    acc,
                    a_ptr,
                    w_ptr + safe.to(tl.int64) * stride_we,
                    s_ptr + safe.to(tl.int64) * stride_se,
                    offs_m,
                    start,
                    row_end,
                    offs_n,
                    n_ok,
                    k_dim,
                    stride_am,
                    stride_ak,
                    stride_wn,
                    stride_wk,
                    stride_sn,
                    stride_sk,
                    QBLOCK,
                    BLOCK_K,
                    SWAP_AB,
                    DOT_IN_FP32,
                )
            row_mask = leads & (offs_m < row_end)
        else:
            tile_start = tl.program_id(0) * BLOCK_M
            tile_end = tl.minimum(tile_start + BLOCK_M, rows)
            offs_m = tile_start + tl.arange(0, BLOCK_M)
            row = tile_start
            while row < tile_end:
                expert = tl.load(row_expert_ptr + row).to(tl.int32)
                # Off-rank rows sort last: the rest of the tile stays zero.
                on_rank = expert < num_groups
                safe = tl.minimum(expert, num_groups - 1)
                group_end = tl.minimum(tl.load(group_end_ptr + safe), tile_end)
                segment_end = tl.where(on_rank, group_end, tile_end)
                if on_rank:
                    acc = _expert_rows_dot(
                        acc,
                        a_ptr,
                        w_ptr + safe.to(tl.int64) * stride_we,
                        s_ptr + safe.to(tl.int64) * stride_se,
                        offs_m,
                        row,
                        segment_end,
                        offs_n,
                        n_ok,
                        k_dim,
                        stride_am,
                        stride_ak,
                        stride_wn,
                        stride_wk,
                        stride_sn,
                        stride_sk,
                        QBLOCK,
                        BLOCK_K,
                        SWAP_AB,
                        DOT_IN_FP32,
                    )
                row = segment_end
            row_mask = offs_m < rows

        if SWAP_AB:
            tl.store(
                out_ptr + offs_m[None, :] * stride_om + offs_n[:, None] * stride_on,
                acc.to(out_ptr.dtype.element_ty),
                mask=row_mask[None, :] & n_ok[:, None],
            )
        else:
            tl.store(
                out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                acc.to(out_ptr.dtype.element_ty),
                mask=row_mask[:, None] & n_ok[None, :],
            )


def block_fp8_grouped_mm(
    rows: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block: int,
    row_experts: torch.Tensor,
    group_ends: torch.Tensor,
) -> torch.Tensor:
    """`rows @ dequant(weight[e]).T` for each row's expert `e`, `(M, K)` -> `(M, N)`.

    `rows` are sorted by expert: `row_experts` `(M,)` gives each row's local
    expert, and `group_ends` `(E,)` the end row of each expert's group. A
    row whose expert is `E` or more (another rank's expert) gives zeros.
    `weight` is `(E, N, K)` `float8_e4m3fn` and `scale` its fp32 block scale
    `(E, ceil(N / block), ceil(K / block))`. The result has `rows.dtype`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("block_fp8_grouped_mm needs Triton")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    num_rows, k_dim = rows.shape
    num_groups, n_dim, weight_k = weight.shape
    if weight_k != k_dim:
        raise ValueError(f"rows have K={k_dim} but weight has K={weight_k}")
    if block < 16 or block & (block - 1):
        raise ValueError(f"block={block} must be a power of two of at least 16")
    small = num_rows <= _SMALL_ROWS
    per_segment = small if _PER_SEGMENT_FOR_BENCH is None else _PER_SEGMENT_FOR_BENCH
    gluon = not per_segment if _GLUON_FOR_BENCH is None else _GLUON_FOR_BENCH
    if gluon and group_ends.dtype == torch.int32:
        gluon_tiles = fp8_grouped_gluon.tiles_for(num_rows, num_groups)
        if fp8_grouped_gluon.supports_gluon_block_fp8_grouped_mm(rows, weight, block, gluon_tiles):
            return fp8_grouped_gluon.gluon_block_fp8_grouped_mm(
                rows, weight, scale, block=block, group_ends=group_ends, tiles=gluon_tiles
            )
    out = torch.empty(num_rows, n_dim, dtype=rows.dtype, device=rows.device)
    if num_rows == 0:
        return out
    swap_ab = True if _SWAP_AB_FOR_BENCH is None else _SWAP_AB_FOR_BENCH
    tiles = _TILES_FOR_BENCH or tiles_for(rows.device, num_rows)
    block_m, block_n, tile_k, warps, stages = tiles
    rows_grid = num_rows if per_segment else triton.cdiv(num_rows, block_m)
    grid = (rows_grid, triton.cdiv(n_dim, block_n))
    weight_bytes = weight.view(torch.uint8)
    _block_fp8_grouped_mm_kernel[grid](
        rows,
        weight_bytes,
        scale,
        row_experts,
        group_ends,
        out,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
        rows.stride(0),
        rows.stride(1),
        weight_bytes.stride(0),
        weight_bytes.stride(1),
        weight_bytes.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        QBLOCK=block,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        # A power of two no larger than the scale block, so it divides it.
        BLOCK_K=min(tile_k, block),
        PER_SEGMENT=per_segment,
        SWAP_AB=swap_ab,
        # Triton's interpreter computes a bf16 `tl.dot` wrongly. bf16 products
        # are exact in fp32, so an fp32 dot gives the same result there.
        DOT_IN_FP32=os.environ.get("TRITON_INTERPRET") == "1",
        num_warps=warps,
        num_stages=stages,
    )
    return out
