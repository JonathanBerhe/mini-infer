"""Grouped matmul over MoE experts with packed FP4 weights, in Gluon.

The same product as `fp4_grouped_kernel.block_fp4_grouped_mm` with the same
numerics: each e2m1 weight times its fp32 scale (one per row and 32
columns), rounded to bf16, multiplied with the bf16 rows on the tensor cores
with an fp32 accumulator. These kernels are for many rows (prefill), one
for sm90 (`wgmma`) and one for sm100 (`tcgen05`). The Triton kernel keeps
the other cases.

The kernels keep the structure of the block-FP8 kernels of
`fp8_grouped_gluon.py` and reuse its parts: work items of one expert each,
the persistent walk, the copy warp, and on sm100 the MMA and store warps.
The weight tile differs. TMA copies its packed bytes, two e2m1 values each.
The warps that dequantize load the bytes in the layout that `gl.fp4_to_fp`
turns into the operand layout of the multiply, so no layout conversion
runs. They load one scale per row and 32 columns and multiply in fp32.

On sm90 the multiply warps decode the tile into one of two shared-memory
buffers, and an asynchronous `wgmma` reads it from there while they decode
the next one. On sm100 one or two warpgroups of dequantization warps take
turns and write the tile into tensor memory.

Rows routed to another rank's experts (expert parallelism) sort after the
local groups. Their items store zeros.
"""

from __future__ import annotations

from typing import Any

import torch

from mini_infer.device import is_datacenter_blackwell_device, is_hopper_device
from mini_infer.models.blocks import fp8_grouped_gluon
from mini_infer.models.blocks.fp8_grouped_gluon import _GLUON_AVAILABLE, GluonTiles

if _GLUON_AVAILABLE:
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout,
        allocate_tensor_memory,
    )
    from triton.experimental.gluon.language.nvidia.hopper import (
        fence_async_shared,
        mbarrier,
        tma,
        warpgroup_mma,
        warpgroup_mma_wait,
    )
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    from mini_infer.models.blocks.fp8_grouped_gluon import (
        _allocate_barriers,
        _bw_mma_partition,
        _bw_store_partition,
        _ring_slot,
        _store_tile,
        _Walk,
        _wgmma_layouts,
    )

# Tiles as in `fp8_grouped_gluon`: (BLOCK_W, BLOCK_T, BLOCK_K, num_warps,
# num_stages, warp specialization), by the mean rows per expert. Both kernels
# are warp-specialized. On sm100 `num_warps` counts the dequantization warps.
_HOPPER_TILES: tuple[tuple[float, GluonTiles], ...] = (
    (64, (128, 64, 64, 4, 4, True)),
    (192, (128, 128, 64, 8, 4, True)),
    (float("inf"), (128, 256, 64, 8, 3, True)),
)
_BLACKWELL_TILES: tuple[tuple[float, GluonTiles], ...] = (
    (64, (128, 64, 64, 8, 4, True)),
    (192, (128, 128, 64, 8, 4, True)),
    (float("inf"), (128, 256, 64, 8, 3, True)),
)
# Up to this many mean rows per expert, the default path keeps the Triton
# kernel. On the V4-Flash layer, the H100 Triton kernel is faster at 24 mean
# rows (1024 tokens) and slower at 48.
_HOPPER_MIN_MEAN_ROWS = 32
_BLACKWELL_MIN_MEAN_ROWS = _BLACKWELL_TILES[0][0]
# The columns that share one scale. The decode loads one scale per row and group.
_SCALE_BLOCK = 32
# Bench hook, not a public API: force a tile.
_TILES_FOR_BENCH: GluonTiles | None = None
# Test hook, not a public API: the number of programs.
_PROGRAMS_FOR_TESTS: int | None = None


def default_tiles(mean_rows: float, device: torch.device) -> GluonTiles:
    """The tile for this many sorted rows per expert, from the table of `device`."""
    table = _BLACKWELL_TILES if is_datacenter_blackwell_device(device) else _HOPPER_TILES
    return next(tiles for limit, tiles in table if mean_rows <= limit)


def gluon_tiles(
    rows: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    block: int,
    group_ends: torch.Tensor,
    *,
    num_experts: int,
    by_default: bool,
) -> GluonTiles | None:
    """The tile of a Gluon kernel for these operands, or None for the Triton kernel.

    The kernels need Triton 3.8 and an sm90 or sm100 GPU. They also need
    contiguous bf16 rows, contiguous packed weights and scales, int32
    `group_ends`, and one scale per 32 columns. TMA needs 16-byte aligned
    operands. The sm100 kernel dequantizes with one or two warpgroups (4 or
    8 warps).

    `num_experts` counts the experts of all ranks, so the rows over all
    experts give the mean rows of a local expert. With `by_default`, the
    kernels run only above `_HOPPER_MIN_MEAN_ROWS` (sm90) or
    `_BLACKWELL_MIN_MEAN_ROWS` (sm100) mean rows.
    """
    if not _GLUON_AVAILABLE:
        return None
    hopper = is_hopper_device(rows.device)
    if not hopper and not is_datacenter_blackwell_device(rows.device):
        return None
    num_rows, k_dim = rows.shape
    mean_rows = num_rows / max(num_experts, 1)
    min_mean_rows = _HOPPER_MIN_MEAN_ROWS if hopper else _BLACKWELL_MIN_MEAN_ROWS
    if by_default and mean_rows <= min_mean_rows:
        return None
    tiles = _TILES_FOR_BENCH or default_tiles(mean_rows, rows.device)
    block_k, warps = tiles[2:4]
    fits = (
        rows.dtype == torch.bfloat16
        and group_ends.dtype == torch.int32
        and block == _SCALE_BLOCK
        and rows.is_contiguous()
        and packed.is_contiguous()
        and scale.is_contiguous()
        and rows.data_ptr() % 16 == 0
        and packed.data_ptr() % 16 == 0
        and k_dim % _SCALE_BLOCK == 0
        and block_k % _SCALE_BLOCK == 0
        and (hopper or warps in (4, 8))
    )
    return tiles if fits else None


if _GLUON_AVAILABLE:

    @gluon.jit  # type: ignore[untyped-decorator]
    def _dequantize(  # type: ignore[no-untyped-def]
        w_buf,
        s_base,
        k0,
        n_left,
        k_dim,
        stride_sn,
        stride_sk,
        bytes_layout: gl.constexpr,
        layout: gl.constexpr,
    ):
        """The weight tile in `w_buf` times its scales in fp32, rounded to bf16.

        The bytes load in `bytes_layout`, which `gl.fp4_to_fp` unpacks into
        `layout`, the layout of the bf16 tile. For a blocked or dot-operand
        layout it doubles the values per thread along K, so each thread holds
        both values of its bytes. `s_base` points at the scale of the tile's
        first row, and `k0` is its first column. Rows from `n_left` on and
        groups past `k_dim` read no scale. Their weights are outside the
        expert, and the store drops their outputs.
        """
        BLOCK_W: gl.constexpr = w_buf.shape[0]
        BLOCK_K: gl.constexpr = w_buf.shape[1] * 2
        GROUPS: gl.constexpr = BLOCK_K // 32
        w = gl.convert_layout(gl.fp4_to_fp(w_buf.load(bytes_layout), gl.bfloat16, axis=1), layout)
        # A group of 32 columns shares one scale per row: load it once and
        # broadcast it over the group.
        w3 = gl.reshape(w, [BLOCK_W, GROUPS, 32])
        scale_layout: gl.constexpr = gl.SliceLayout(2, w3.type.layout)
        rows = gl.arange(0, BLOCK_W, layout=gl.SliceLayout(1, scale_layout))
        groups = k0 // 32 + gl.arange(0, GROUPS, layout=gl.SliceLayout(0, scale_layout))
        scale = gl.load(
            s_base + rows[:, None] * stride_sn + groups[None, :] * stride_sk,
            mask=(rows[:, None] < n_left) & (groups[None, :] < k_dim // 32),
            other=0.0,
        )
        values = (w3.to(gl.float32) * gl.expand_dims(scale, 2)).to(gl.bfloat16)
        return gl.convert_layout(gl.reshape(values, [BLOCK_W, BLOCK_K]), layout)

    @gluon.jit
    def _load_partition(  # type: ignore[no-untyped-def]
        x_desc,
        w_desc,
        x_bufs,
        w_bufs,
        ready,
        empty,
        group_ends_ptr,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
    ):
        """The copy warp: TMA copies of a row tile and its packed weight tile.

        A K step of `BLOCK_K` columns is `BLOCK_K / 2` bytes of weights.
        """
        BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
        BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
        BLOCK_K: gl.constexpr = x_desc.block_type.shape[1]
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        produced = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                col, row = walk.tile(item, BLOCK_W)
                weight_row = walk.group * n_dim + col
                for ki in range(walk.k_iters(k_steps)):
                    stage, phase = _ring_slot(produced, NUM_STAGES)
                    # A new barrier counts as released once, so the first
                    # round of stages does not wait.
                    mbarrier.wait(empty.index(stage), phase ^ 1)
                    bar = ready.index(stage)
                    mbarrier.expect(bar, x_desc.block_type.nbytes + w_desc.block_type.nbytes)
                    tma.async_load(x_desc, [row, ki * BLOCK_K], bar, x_bufs.index(stage))
                    tma.async_load(
                        w_desc, [weight_row, ki * (BLOCK_K // 2)], bar, w_bufs.index(stage)
                    )
                    produced += 1
                item += gl.num_programs(0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _allocate_ring(x_desc, w_desc, NUM_STAGES: gl.constexpr):  # type: ignore[no-untyped-def]
        """The shared-memory stages of the row tiles and the packed weight tiles."""
        x_bufs = gl.allocate_shared_memory(
            x_desc.dtype, [NUM_STAGES] + x_desc.block_type.shape, x_desc.layout
        )
        w_bufs = gl.allocate_shared_memory(
            w_desc.dtype, [NUM_STAGES] + w_desc.block_type.shape, w_desc.layout
        )
        return x_bufs, w_bufs

    @gluon.jit  # type: ignore[untyped-decorator]
    def _mma_step(  # type: ignore[no-untyped-def]
        acc,
        a_bufs,
        x_bufs,
        w_bufs,
        ready,
        consumed,
        s_base,
        ki,
        n_left,
        k_dim,
        stride_sn,
        stride_sk,
        num_warps: gl.constexpr,
    ):
        """One sm90 K step: decode the stage's weight tile into shared memory, then its wgmma.

        Each thread reads 16 bytes of one row, which is one scale group. In
        the register operand layout of a wgmma, a thread's bytes are
        scattered, so they load one byte at a time. The bf16 tile goes into
        one of two shared-memory buffers, and the async wgmma reads it from
        there. The previous step's wgmma reads the other buffer while this
        step decodes. The wait at the end lets that wgmma finish, so its
        buffer is free for the next step.
        """
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        BLOCK_K: gl.constexpr = x_bufs.type.shape[2]
        THREADS_K: gl.constexpr = BLOCK_K // 32
        bytes_layout: gl.constexpr = gl.BlockedLayout(
            [1, 16], [32 // THREADS_K, THREADS_K], [num_warps, 1], [1, 0]
        )
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 32], [32 // THREADS_K, THREADS_K], [num_warps, 1], [1, 0]
        )
        stage, phase = _ring_slot(consumed, NUM_STAGES)
        mbarrier.wait(ready.index(stage), phase)
        a_buf = a_bufs.index(consumed % 2)
        a_buf.store(
            _dequantize(
                w_bufs.index(stage),
                s_base,
                ki * BLOCK_K,
                n_left,
                k_dim,
                stride_sn,
                stride_sk,
                bytes_layout,
                layout,
            )
        )
        # Each thread makes its stores visible to the wgmma, then all warps
        # are done before it starts.
        fence_async_shared()
        gl.barrier()
        acc = warpgroup_mma(a_buf, x_bufs.index(stage).permute((1, 0)), acc, is_async=True)
        return warpgroup_mma_wait(num_outstanding=1, deps=(acc,))

    @gluon.jit  # type: ignore[untyped-decorator]
    def _hopper_mma_partition(  # type: ignore[no-untyped-def]
        a_bufs,
        x_bufs,
        w_bufs,
        ready,
        empty,
        s_ptr,
        group_ends_ptr,
        out_ptr,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
        stride_se,
        stride_sn,
        stride_sk,
        stride_om,
        num_warps: gl.constexpr,
    ):
        """The sm90 multiply warps: dequantize, `wgmma`, free the stage, store the tile.

        Before each release of a stage, a fence orders the shared-memory read
        of its weight tile before the next TMA write into it.
        """
        BLOCK_W: gl.constexpr = w_bufs.type.shape[1]
        BLOCK_K: gl.constexpr = x_bufs.type.shape[2]
        BLOCK_T: gl.constexpr = x_bufs.type.shape[1]
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        mma_layout: gl.constexpr = _wgmma_layouts(num_warps, BLOCK_T)[0]
        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                col, row = walk.tile(item, BLOCK_W)
                k_iters = walk.k_iters(k_steps)
                s_base = s_ptr + walk.group * stride_se + col * stride_sn
                n_left = n_dim - col
                acc = gl.zeros((BLOCK_W, BLOCK_T), dtype=gl.float32, layout=mma_layout)
                for ki in range(k_iters):
                    acc = _mma_step(
                        acc,
                        a_bufs,
                        x_bufs,
                        w_bufs,
                        ready,
                        consumed,
                        s_base,
                        ki,
                        n_left,
                        k_dim,
                        stride_sn,
                        stride_sk,
                        num_warps,
                    )
                    # The previous step's wgmma is done: free its stage. The
                    # arrive waits for all warps of this partition first.
                    fence_async_shared()
                    mbarrier.arrive(
                        empty.index((consumed + NUM_STAGES - 1) % NUM_STAGES), pred=ki > 0
                    )
                    consumed += 1
                acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
                fence_async_shared()
                mbarrier.arrive(
                    empty.index((consumed + NUM_STAGES - 1) % NUM_STAGES), pred=k_iters > 0
                )
                _store_tile(
                    acc,
                    out_ptr,
                    col,
                    row,
                    gl.minimum(row + BLOCK_T, walk.end),
                    n_dim,
                    stride_om,
                    num_warps,
                )
                item += gl.num_programs(0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _fp4_grouped_mm_hopper(  # type: ignore[no-untyped-def]
        x_desc,
        w_desc,
        s_ptr,
        group_ends_ptr,
        out_ptr,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
        stride_se,
        stride_sn,
        stride_sk,
        stride_om,
        NUM_STAGES: gl.constexpr,
        LOAD_REGS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """sm90 version: one warp copies, `num_warps` warps dequantize and multiply."""
        BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
        BLOCK_K: gl.constexpr = x_desc.block_type.shape[1]
        x_bufs, w_bufs = _allocate_ring(x_desc, w_desc, NUM_STAGES)
        # Two buffers for the bf16 weight tiles: one for the running wgmma,
        # one for the next decode.
        a_layout: gl.constexpr = gl.NVMMASharedLayout(
            swizzle_byte_width=min(128, BLOCK_K * 2), element_bitwidth=16, rank=2
        )
        a_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_W, BLOCK_K], a_layout)
        ready = _allocate_barriers(NUM_STAGES)
        empty = _allocate_barriers(NUM_STAGES)
        gl.warp_specialize(
            [
                (
                    _hopper_mma_partition,
                    (
                        a_bufs,
                        x_bufs,
                        w_bufs,
                        ready,
                        empty,
                        s_ptr,
                        group_ends_ptr,
                        out_ptr,
                        num_rows,
                        n_dim,
                        k_dim,
                        num_groups,
                        stride_se,
                        stride_sn,
                        stride_sk,
                        stride_om,
                        num_warps,
                    ),
                ),
                (
                    _load_partition,
                    (
                        x_desc,
                        w_desc,
                        x_bufs,
                        w_bufs,
                        ready,
                        empty,
                        group_ends_ptr,
                        num_rows,
                        n_dim,
                        k_dim,
                        num_groups,
                    ),
                ),
            ],
            [1],
            [LOAD_REGS],
        )

    @gluon.jit  # type: ignore[untyped-decorator]
    def _bw_dequant_partition(  # type: ignore[no-untyped-def]
        w_bufs,
        ready,
        a_tmem,
        a_ready,
        a_empty,
        s_ptr,
        group_ends_ptr,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
        stride_se,
        stride_sn,
        stride_sk,
        BLOCK_T: gl.constexpr,
        PART: gl.constexpr,
        PARTS: gl.constexpr,
    ):
        """sm100 dequantization warps: packed tile in shared memory to bf16 in tensor memory.

        A partition is one warpgroup: in the register layout of a
        tensor-memory store, each thread then holds one weight row. Its bytes
        load as one row per thread too, which `gl.fp4_to_fp` unpacks into that
        layout, so neither the unpack nor the store converts a layout. With
        `PARTS` partitions, this one takes the K steps whose count is `PART`
        modulo `PARTS`, so the decode of the next step overlaps this one.
        """
        gl.static_assert(gl.num_warps() == 4, "a dequantization partition is one warpgroup")
        NUM_STAGES: gl.constexpr = w_bufs.type.shape[0]
        BLOCK_W: gl.constexpr = w_bufs.type.shape[1]
        BLOCK_K: gl.constexpr = a_tmem.shape[2]
        WEIGHT_BUFFERS: gl.constexpr = a_tmem.shape[0]
        layout: gl.constexpr = a_tmem.index(0).get_reg_layout()
        bytes_layout: gl.constexpr = gl.BlockedLayout([1, BLOCK_K // 2], [32, 1], [4, 1], [1, 0])
        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                col, _row = walk.tile(item, BLOCK_W)
                s_base = s_ptr + walk.group * stride_se + col * stride_sn
                for ki in range(walk.k_iters(k_steps)):
                    if consumed % PARTS == PART:
                        stage, phase = _ring_slot(consumed, NUM_STAGES)
                        buffer, buffer_phase = _ring_slot(consumed, WEIGHT_BUFFERS)
                        mbarrier.wait(ready.index(stage), phase)
                        mbarrier.wait(a_empty.index(buffer), buffer_phase ^ 1)
                        w = _dequantize(
                            w_bufs.index(stage),
                            s_base,
                            ki * BLOCK_K,
                            n_dim - col,
                            k_dim,
                            stride_sn,
                            stride_sk,
                            bytes_layout,
                            layout,
                        )
                        a_tmem.index(buffer).store(w)
                        # The completion of the multiply frees the stage for
                        # the next TMA write. The fence orders this
                        # shared-memory read before that write.
                        fence_async_shared()
                        mbarrier.arrive(a_ready.index(buffer))
                    consumed += 1
                item += gl.num_programs(0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _fp4_grouped_mm_blackwell(  # type: ignore[no-untyped-def]
        x_desc,
        w_desc,
        s_ptr,
        group_ends_ptr,
        out_ptr,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
        stride_se,
        stride_sn,
        stride_sk,
        stride_om,
        NUM_STAGES: gl.constexpr,
        WEIGHT_BUFFERS: gl.constexpr,
        ACC_BUFFERS: gl.constexpr,
        STORE_COLUMNS: gl.constexpr,
        DEQUANT_PARTS: gl.constexpr,
        DEQUANT_REGS: gl.constexpr,
        MMA_REGS: gl.constexpr,
        LOAD_REGS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """sm100 version: copy, dequantization, MMA and store warps.

        `DEQUANT_PARTS` dequantization warpgroups (1 or 2) take turns on the K
        steps. One warpgroup puts one dequantization warp on each scheduler of
        the SM, which issues too slowly to keep the tensor cores busy.
        """
        BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
        BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
        BLOCK_K: gl.constexpr = x_desc.block_type.shape[1]
        x_bufs, w_bufs = _allocate_ring(x_desc, w_desc, NUM_STAGES)
        a_tmem = allocate_tensor_memory(
            gl.bfloat16,
            [WEIGHT_BUFFERS, BLOCK_W, BLOCK_K],
            TensorMemoryLayout((BLOCK_W, BLOCK_K), col_stride=1),
        )
        acc_tmem = allocate_tensor_memory(
            gl.float32,
            [ACC_BUFFERS, BLOCK_W, BLOCK_T],
            TensorMemoryLayout((BLOCK_W, BLOCK_T), col_stride=1),
        )
        ready = _allocate_barriers(NUM_STAGES)
        empty = _allocate_barriers(NUM_STAGES)
        a_ready = _allocate_barriers(WEIGHT_BUFFERS)
        a_empty = _allocate_barriers(WEIGHT_BUFFERS)
        acc_ready = _allocate_barriers(ACC_BUFFERS)
        acc_empty = _allocate_barriers(ACC_BUFFERS)
        gl.static_assert(DEQUANT_PARTS == 1 or DEQUANT_PARTS == 2, "1 or 2 dequant warpgroups")
        if DEQUANT_PARTS == 1:
            gl.warp_specialize(
                [
                    (
                        _bw_store_partition,
                        (
                            acc_tmem,
                            acc_ready,
                            acc_empty,
                            out_ptr,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                            stride_om,
                            BLOCK_K,
                            STORE_COLUMNS,
                        ),
                    ),
                    (
                        _bw_dequant_partition,
                        (
                            w_bufs,
                            ready,
                            a_tmem,
                            a_ready,
                            a_empty,
                            s_ptr,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                            stride_se,
                            stride_sn,
                            stride_sk,
                            BLOCK_T,
                            0,
                            1,
                        ),
                    ),
                    (
                        _bw_mma_partition,
                        (
                            x_bufs,
                            ready,
                            empty,
                            a_tmem,
                            a_ready,
                            a_empty,
                            acc_tmem,
                            acc_ready,
                            acc_empty,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                        ),
                    ),
                    (
                        _load_partition,
                        (
                            x_desc,
                            w_desc,
                            x_bufs,
                            w_bufs,
                            ready,
                            empty,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                        ),
                    ),
                ],
                [4, 1, 1],
                [DEQUANT_REGS, MMA_REGS, LOAD_REGS],
            )
        else:
            gl.warp_specialize(
                [
                    (
                        _bw_store_partition,
                        (
                            acc_tmem,
                            acc_ready,
                            acc_empty,
                            out_ptr,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                            stride_om,
                            BLOCK_K,
                            STORE_COLUMNS,
                        ),
                    ),
                    (
                        _bw_dequant_partition,
                        (
                            w_bufs,
                            ready,
                            a_tmem,
                            a_ready,
                            a_empty,
                            s_ptr,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                            stride_se,
                            stride_sn,
                            stride_sk,
                            BLOCK_T,
                            0,
                            2,
                        ),
                    ),
                    (
                        _bw_dequant_partition,
                        (
                            w_bufs,
                            ready,
                            a_tmem,
                            a_ready,
                            a_empty,
                            s_ptr,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                            stride_se,
                            stride_sn,
                            stride_sk,
                            BLOCK_T,
                            1,
                            2,
                        ),
                    ),
                    (
                        _bw_mma_partition,
                        (
                            x_bufs,
                            ready,
                            empty,
                            a_tmem,
                            a_ready,
                            a_empty,
                            acc_tmem,
                            acc_ready,
                            acc_empty,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                        ),
                    ),
                    (
                        _load_partition,
                        (
                            x_desc,
                            w_desc,
                            x_bufs,
                            w_bufs,
                            ready,
                            empty,
                            group_ends_ptr,
                            num_rows,
                            n_dim,
                            k_dim,
                            num_groups,
                        ),
                    ),
                ],
                [4, 4, 1, 1],
                [DEQUANT_REGS, DEQUANT_REGS, MMA_REGS, LOAD_REGS],
            )


def _kernel_and_arguments(
    rows: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    *,
    group_ends: torch.Tensor,
    tiles: GluonTiles,
    blackwell: bool,
) -> tuple[Any, tuple[object, ...], dict[str, Any]]:
    """The kernel for these tiles, its arguments, and its constexprs and launch options.

    The launcher and the compile tests both build the kernel call here.
    """
    num_rows, k_dim = rows.shape
    num_groups, n_dim, half_k = packed.shape
    block_w, block_t, block_k, warps, stages, _ = tiles
    x_layout = gl.NVMMASharedLayout.get_default_for([block_t, block_k], gl.bfloat16)
    w_layout = gl.NVMMASharedLayout.get_default_for([block_w, block_k // 2], gl.uint8)
    x_desc = TensorDescriptor.from_tensor(rows, [block_t, block_k], x_layout)
    w_desc = TensorDescriptor.from_tensor(
        packed.view(torch.uint8).view(num_groups * n_dim, half_k), [block_w, block_k // 2], w_layout
    )
    args = (
        x_desc,
        w_desc,
        scale,
        group_ends,
        out,
        num_rows,
        n_dim,
        k_dim,
        num_groups,
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
    )
    if blackwell:
        # The FP8 launch options, with the dequantization warps in warpgroups.
        options = fp8_grouped_gluon.blackwell_launch_options(tiles)
        options["DEQUANT_PARTS"] = options.pop("DEQUANT_WARPS") // 4
        return _fp4_grouped_mm_blackwell, args, options
    return (
        _fp4_grouped_mm_hopper,
        args,
        {
            "NUM_STAGES": stages,
            "LOAD_REGS": fp8_grouped_gluon._LOAD_REGS,
            "num_warps": warps,
            "maxnreg": fp8_grouped_gluon._max_registers(warps, True),
        },
    )


def gluon_block_fp4_grouped_mm(
    rows: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_ends: torch.Tensor,
    tiles: GluonTiles,
) -> torch.Tensor:
    """`rows @ dequant(packed[e]).T` for each row's expert `e`, on sm90 or sm100.

    Arguments as `fp4_grouped_kernel.block_fp4_grouped_mm`, which checks them
    and calls this function when `gluon_tiles` gives a tile.
    """
    num_rows, _ = rows.shape
    num_groups, n_dim, _ = packed.shape
    out = torch.empty(num_rows, n_dim, dtype=rows.dtype, device=rows.device)
    if num_rows == 0:
        return out
    kernel, args, options = _kernel_and_arguments(
        rows,
        packed,
        scale,
        out,
        group_ends=group_ends,
        tiles=tiles,
        blackwell=is_datacenter_blackwell_device(rows.device),
    )
    block_w, block_t = tiles[:2]
    # At most one partial row tile per group, plus the off-rank tail.
    items = (triton.cdiv(num_rows, block_t) + num_groups + 1) * triton.cdiv(n_dim, block_w)
    programs = _PROGRAMS_FOR_TESTS or fp8_grouped_gluon._num_sms(rows.device.index or 0)
    kernel[(min(programs, items),)](*args, **options)
    return out
