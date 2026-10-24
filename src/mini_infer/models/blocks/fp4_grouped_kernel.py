"""Grouped matmul over MoE experts with packed FP4 weights, in Triton.

DeepSeek-V4's routed experts ship as e2m1 values, two per byte (the low
nibble first), with one scale per row and per 32 input columns (`FP4Expert`).
This kernel runs the experts of a layer as one grouped matmul over the
(token, expert) pairs that `grouped_expert_mlp` sorted by expert, without
dequantizing a whole expert in memory.

A program loads one expert's packed weight tile and one scale per row and
32 columns. It unpacks the nibbles and moves each e2m1 code's bits into an
fp32 (its value times 2^-126). Then it multiplies by 2^126 and the scale in
fp32 and rounds to bf16, as `dequantize_nvfp4_to_bf16` does. The tile is the
left operand of a bf16 tensor-core dot (`out.T = W @ rows.T`), as in
`fp8_grouped_kernel.py`. The decode, not the dot, sets the kernel's speed.

The work splits as in the FP8 kernel: one program per expert segment up to
`_SMALL_ROWS` rows (decode), one program per tile of one expert's rows above
(prefill). A third split, per row tile across experts, stays for benchmarks.
Rows routed to another rank's experts sort after the local groups and get
zeros.

On sm90 and sm100, many rows go to the Gluon kernels of
`fp4_grouped_gluon.py` when one takes the operands. They decode a weight
tile while the tensor cores multiply the previous one.
"""

from __future__ import annotations

import functools
import os

import torch

from mini_infer.device import is_blackwell_device, is_cuda_device
from mini_infer.models.blocks import fp4_grouped_gluon

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
# sizes: per segment up to `_SMALL_ROWS` rows, per expert tile up to
# `_MEDIUM_ROWS`, and per expert tile above. BLOCK_K and the scale block nest.
# Sweeps of the DeepSeek-V4-Flash MoE layer on an H100 and a B200 set them.
_HOPPER_TILES: tuple[Tiles, Tiles, Tiles] = (
    (16, 64, 128, 4, 3),
    (64, 128, 64, 4, 4),
    (128, 256, 64, 8, 3),
)
_BLACKWELL_TILES: tuple[Tiles, Tiles, Tiles] = (
    (16, 64, 64, 4, 3),
    (64, 128, 64, 4, 4),
    (128, 256, 64, 8, 3),
)
_SMALL_ROWS = 256
_MEDIUM_ROWS = 8192
# Bench hooks, not a public API: force a tile or the split.
_TILES_FOR_BENCH: Tiles | None = None
_PER_SEGMENT_FOR_BENCH: bool | None = None
# Bench hook: False splits many rows into row tiles that span experts, not
# into tiles of one expert each (the default).
_EXPERT_TILES_FOR_BENCH: bool | None = None
# Bench hook: True forces a Gluon kernel (and fails when none takes the
# operands), False keeps the Triton kernel, None picks by default.
_GLUON_FOR_BENCH: bool | None = None


def supports_block_fp4_grouped_mm(device: torch.device | str) -> bool:
    """Whether `block_fp4_grouped_mm` can run on this device."""
    return _TRITON_AVAILABLE and is_cuda_device(device)


@functools.cache
def _tile_table(device: torch.device) -> tuple[Tiles, Tiles, Tiles]:
    """The tiles for this GPU family; Hopper's off CUDA (the interpreter)."""
    return _BLACKWELL_TILES if is_blackwell_device(device) else _HOPPER_TILES


def tiles_for(device: torch.device, num_rows: int) -> Tiles:
    """The tile `block_fp4_grouped_mm` uses for `num_rows` rows on `device`."""
    small, medium, large = _tile_table(device)
    if num_rows <= _SMALL_ROWS:
        return small
    return medium if num_rows <= _MEDIUM_ROWS else large


if _TRITON_AVAILABLE:

    @triton.jit
    def _e2m1_times_tiny(code):  # type: ignore[no-untyped-def]
        """4-bit e2m1 codes (in int32) to fp32, each its exact value times 2^-126.

        e2m1 has a sign bit, 2 exponent bits and 1 mantissa bit:
        0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives. Its exponent and
        mantissa bits go to the low end of the fp32 exponent field and the top
        of the mantissa. A normal code then reads as `2^(e - 127) (1 + m / 2)`
        and a subnormal one as `m 2^-127`, both 2^-126 times the e2m1 value.
        """
        bits = ((code & 7) << 22) | ((code & 8) << 28)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit  # type: ignore[untyped-decorator]
    def _fp4_weight_tile(  # type: ignore[no-untyped-def]
        w_base,
        s_base,
        offs_n,
        n_ok,
        k_start,
        k_dim,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sk,
        SBLOCK: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """`(BLOCK_N, BLOCK_K)` dequantized weight tile, starting at column `k_start`.

        The tile is `(BLOCK_N, groups, cols)`, with `cols` columns that share
        one scale per row. Each scale loads once and broadcasts.
        """
        cols: tl.constexpr = BLOCK_K if BLOCK_K < SBLOCK else SBLOCK
        groups: tl.constexpr = BLOCK_K // cols
        offs_group = k_start + tl.arange(0, groups) * cols
        group_ok = offs_group < k_dim
        offs_byte = offs_group[:, None] // 2 + tl.arange(0, cols // 2)[None, :]
        packed = tl.load(
            w_base + offs_n[:, None, None] * stride_wn + offs_byte[None, :, :] * stride_wk,
            mask=n_ok[:, None, None] & group_ok[None, :, None],
            other=0,
        ).to(tl.int32)
        scale = tl.load(
            s_base + offs_n[:, None] * stride_sn + (offs_group[None, :] // SBLOCK) * stride_sk,
            mask=n_ok[:, None] & group_ok[None, :],
            other=0.0,
        )
        # Element 2j is the low nibble of byte j, element 2j + 1 the high one.
        codes = tl.join(packed & 0xF, packed >> 4)
        codes = tl.reshape(codes, (codes.shape[0], groups, cols))
        # Multiplying by 2^126 is exact, so the product rounds once, as in
        # `dequantize_nvfp4_to_bf16`.
        values = _e2m1_times_tiny(codes) * 8.507059173023462e37 * scale[:, :, None]
        return tl.reshape(values, (values.shape[0], BLOCK_K)).to(tl.bfloat16)

    @triton.jit  # type: ignore[untyped-decorator]
    def _fp4_rows_dot(  # type: ignore[no-untyped-def]
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
        SBLOCK: tl.constexpr,
        BLOCK_K: tl.constexpr,
        DOT_IN_FP32: tl.constexpr,
    ):
        """Add rows `[row_start, row_end)` times one expert's weights to `acc` (transposed)."""
        in_rows = (offs_m >= row_start) & (offs_m < row_end)
        for k_start in range(0, k_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            w = _fp4_weight_tile(
                w_base,
                s_base,
                offs_n,
                n_ok,
                k_start,
                k_dim,
                stride_wn,
                stride_wk,
                stride_sn,
                stride_sk,
                SBLOCK,
                BLOCK_K,
            )
            a_t = tl.load(
                a_ptr + offs_m[None, :] * stride_am + offs_k[:, None] * stride_ak,
                mask=(offs_k < k_dim)[:, None] & in_rows[None, :],
                other=0.0,
            )
            if DOT_IN_FP32:
                acc = tl.dot(w.to(tl.float32), a_t.to(tl.float32), acc, input_precision="ieee")
            else:
                acc = tl.dot(w.to(a_t.dtype), a_t, acc)
        return acc

    @triton.jit  # type: ignore[untyped-decorator]
    def _block_fp4_grouped_mm_kernel(  # type: ignore[no-untyped-def]
        a_ptr,
        w_ptr,
        s_ptr,
        row_expert_ptr,
        group_end_ptr,
        tile_end_ptr,
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
        SBLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PER_SEGMENT: tl.constexpr,
        EXPERT_TILES: tl.constexpr,
        DOT_IN_FP32: tl.constexpr,
    ):
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < n_dim
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

        if EXPERT_TILES:
            # `tile_end_ptr[g]` is the end of group g's row tiles; group
            # `num_groups` holds the off-rank rows, whose tiles store zeros.
            tile = tl.program_id(0)
            in_range = tile < tl.load(tile_end_ptr + num_groups)
            # The first group whose tiles end after this one.
            group = 0
            high = num_groups
            while group < high:
                middle = (group + high) // 2
                before = tl.load(tile_end_ptr + middle) <= tile
                group = tl.where(before, middle + 1, group)
                high = tl.where(before, high, middle)
            previous = tl.maximum(group - 1, 0)
            first_tile = tl.where(group > 0, tl.load(tile_end_ptr + previous), 0)
            group_start = tl.where(group > 0, tl.load(group_end_ptr + previous), 0)
            own_end = tl.load(group_end_ptr + tl.minimum(group, num_groups - 1))
            group_end = tl.where(group < num_groups, own_end, rows)
            start = group_start + (tile - first_tile) * BLOCK_M
            row_end = tl.minimum(start + BLOCK_M, group_end)
            offs_m = start + tl.arange(0, BLOCK_M)
            if in_range & (group < num_groups):
                acc = _fp4_rows_dot(
                    acc,
                    a_ptr,
                    w_ptr + tl.cast(group, tl.int64) * stride_we,
                    s_ptr + tl.cast(group, tl.int64) * stride_se,
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
                    SBLOCK,
                    BLOCK_K,
                    DOT_IN_FP32,
                )
            row_mask = in_range & (offs_m < row_end)
        elif PER_SEGMENT:
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
                acc = _fp4_rows_dot(
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
                    SBLOCK,
                    BLOCK_K,
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
                    acc = _fp4_rows_dot(
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
                        SBLOCK,
                        BLOCK_K,
                        DOT_IN_FP32,
                    )
                row = segment_end
            row_mask = offs_m < rows

        tl.store(
            out_ptr + offs_m[None, :] * stride_om + offs_n[:, None] * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=row_mask[None, :] & n_ok[:, None],
        )


def block_fp4_grouped_mm(
    rows: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    block: int,
    row_experts: torch.Tensor,
    group_ends: torch.Tensor,
    num_experts: int | None = None,
) -> torch.Tensor:
    """`rows @ dequant(packed[e]).T` for each row's expert `e`, `(M, K)` -> `(M, N)`.

    `rows` are sorted by expert: `row_experts` `(M,)` gives each row's local
    expert, and `group_ends` `(E,)` the end row of each expert's group. A
    row whose expert is `E` or more (another rank's expert) gives zeros.
    `packed` is `(E, N, K / 2)` bytes of e2m1 pairs (low nibble first) and
    `scale` its fp32 scale per row and `block` columns, `(E, N, K / block)`.
    The result has `rows.dtype`.

    `num_experts` counts the experts of all ranks (default `E`). The Gluon
    kernels pick their tile from the mean rows per expert over all of them.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("block_fp4_grouped_mm needs Triton")
    if packed.element_size() != 1:
        raise ValueError(f"packed must hold bytes, got {packed.dtype}")
    num_rows, k_dim = rows.shape
    num_groups, n_dim, half_k = packed.shape
    if half_k * 2 != k_dim:
        raise ValueError(f"rows have K={k_dim} but packed weights hold K={half_k * 2}")
    if block < 2 or block & (block - 1) or k_dim % block:
        raise ValueError(f"block={block} must be a power of two dividing K={k_dim}")
    out = torch.empty(num_rows, n_dim, dtype=rows.dtype, device=rows.device)
    if num_rows == 0:
        return out
    small = num_rows <= _SMALL_ROWS
    per_segment = small if _PER_SEGMENT_FOR_BENCH is None else _PER_SEGMENT_FOR_BENCH
    force_gluon = _GLUON_FOR_BENCH
    if force_gluon or (force_gluon is None and not per_segment):
        gluon_tiles = fp4_grouped_gluon.gluon_tiles(
            rows,
            packed,
            scale,
            block,
            group_ends,
            num_experts=num_experts or num_groups,
            by_default=force_gluon is None,
        )
        if gluon_tiles is not None:
            return fp4_grouped_gluon.gluon_block_fp4_grouped_mm(
                rows, packed, scale, group_ends=group_ends, tiles=gluon_tiles
            )
        if force_gluon:
            raise ValueError("_GLUON_FOR_BENCH is set, but no Gluon kernel takes these operands")
    tiles = _TILES_FOR_BENCH or tiles_for(rows.device, num_rows)
    block_m, block_n, block_k, warps, stages = tiles
    if block_k % block and block % block_k:
        raise ValueError(f"BLOCK_K={block_k} and the scale block {block} must nest")
    expert_tiles = not per_segment and _EXPERT_TILES_FOR_BENCH is not False
    tile_ends = group_ends
    if expert_tiles:
        # Row tiles per group, the off-rank tail last, counted on the GPU.
        ends = torch.cat([group_ends, group_ends.new_full((1,), num_rows)])
        counts = torch.diff(ends, prepend=ends.new_zeros(1))
        tile_ends = torch.cumsum((counts + block_m - 1) // block_m, 0).to(torch.int32)
        # At most one partial tile per group and one for the tail.
        rows_grid = triton.cdiv(num_rows, block_m) + num_groups + 1
    else:
        rows_grid = num_rows if per_segment else triton.cdiv(num_rows, block_m)
    grid = (rows_grid, triton.cdiv(n_dim, block_n))
    packed_bytes = packed.view(torch.uint8)
    _block_fp4_grouped_mm_kernel[grid](
        rows,
        packed_bytes,
        scale,
        row_experts,
        group_ends,
        tile_ends,
        out,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
        rows.stride(0),
        rows.stride(1),
        packed_bytes.stride(0),
        packed_bytes.stride(1),
        packed_bytes.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        SBLOCK=block,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        PER_SEGMENT=per_segment,
        EXPERT_TILES=expert_tiles,
        # Triton's interpreter computes a bf16 `tl.dot` wrongly. bf16 products
        # are exact in fp32, so an fp32 dot gives the same result there.
        DOT_IN_FP32=os.environ.get("TRITON_INTERPRET") == "1",
        num_warps=warps,
        num_stages=stages,
    )
    return out
