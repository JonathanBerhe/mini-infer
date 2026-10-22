"""Grouped matmul over MoE experts with block-FP8 weights, in Gluon.

The same product as `fp8_grouped_kernel.block_fp8_grouped_mm` with the same
numerics: each e4m3 weight times its fp32 128x128 block scale, rounded to
bf16, multiplied with the bf16 rows on the tensor cores with an fp32
accumulator. These kernels are for many rows (prefill), one for sm90
(`wgmma`) and one for sm100 (`tcgen05`). The Triton kernel keeps the other
cases.

Differences from the Triton kernel:

- Work items follow the experts. An item is one expert, one tile of that
  expert's rows and one tile of output columns, so a tile never mixes rows
  of two experts. The Triton row tiles span several experts and run the
  whole K loop once for each of them.
- The grid is persistent: one program per SM walks the items with a stride.
  The items of an expert are ordered by column tile, then by row tile, so
  programs that run at the same time read the same weight tile from L2.
- TMA copies the e4m3 weight tile and the bf16 row tile into a ring of
  shared-memory stages. The weight tile is dequantized in registers and is
  the register operand of an asynchronous `wgmma`: `acc[cols, rows] =
  W @ rows.T`. The dequantization of one K step runs while the tensor cores
  multiply the previous one. A `wgmma` reads its register operand while it
  runs, so the K loop takes two steps at a time with two register sets.
- With warp specialization, one extra warp issues the TMA copies, so the
  copies for the next item start while the other warps store a finished
  tile. Without it, the same warps copy and multiply.

On sm100 the kernel keeps the work items, the persistent walk and the copy
warp, and splits the rest over more warps. `tcgen05` takes no register
operand, so dequantization warps write the bf16 weight tile into tensor
memory, a single MMA warp multiplies it with the row tile from shared
memory into an accumulator in tensor memory, and the default warps store
finished accumulators while the next tile is multiplied.

Rows routed to another rank's experts (expert parallelism) sort after the
local groups. Their items store zeros.
"""

from __future__ import annotations

import functools

import torch

from mini_infer.device import is_datacenter_blackwell_device, is_hopper_device

try:
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout,
        allocate_tensor_memory,
        tcgen05_commit,
        tcgen05_mma,
    )
    from triton.experimental.gluon.language.nvidia.hopper import (
        fence_async_shared,
        mbarrier,
        tma,
        warpgroup_mma,
        warpgroup_mma_wait,
    )
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    # Gluon is experimental and its API changes between releases. The
    # kernels are written against Triton 3.8.
    _GLUON_AVAILABLE = tuple(int(part) for part in triton.__version__.split(".")[:2]) >= (3, 8)
except ImportError:  # no Triton, or a Triton without this Gluon API
    _GLUON_AVAILABLE = False


Tiles = tuple[int, int, int, int, int, bool]

# Tiles as (BLOCK_W, BLOCK_T, BLOCK_K, num_warps, num_stages, warp
# specialization): weight rows (output columns), rows of tokens, K, warps of
# the multiply, ring stages, and one extra warp for the copies. `BLOCK_T` is
# picked from the mean rows per expert, so that few rows of a tile are
# padding. Picked from a sweep of the GLM-5.3 layer on an H100
# (`docs/benchmarks/2026-10-21-moe-grouped-gemm.md`).
_HOPPER_TILES: tuple[tuple[float, Tiles], ...] = (
    (64, (128, 64, 64, 4, 4, True)),
    (192, (128, 128, 64, 8, 4, True)),
    (float("inf"), (128, 256, 64, 8, 3, True)),
)
# On sm100, `num_warps` counts the dequantization warps; the kernel always
# has a copy warp, an MMA warp and 4 store warps. Picked from a sweep on a
# B200. Up to 64 mean rows per expert the Triton kernel is as fast, so the
# default path keeps it there (`on_by_default`).
_BLACKWELL_TILES: tuple[tuple[float, Tiles], ...] = (
    (64, (128, 64, 64, 8, 4, True)),
    (float("inf"), (128, 128, 128, 4, 3, True)),
)
_BLACKWELL_MIN_MEAN_ROWS = 64
# Bench hook, not a public API: force a tile.
_TILES_FOR_BENCH: Tiles | None = None
# Registers of the copy warp. It runs the item walk and issues TMA copies.
_LOAD_REGS = 40
# sm100: registers of the dequantization warps and of the MMA warp, the
# weight tiles in tensor memory, and the accumulator columns that the store
# warps read at a time.
_DEQUANT_REGS = 192
_MMA_REGS = 40
_WEIGHT_BUFFERS = 4
_STORE_COLUMNS = 32


def tiles_for(num_rows: int, num_groups: int, device: torch.device) -> Tiles:
    """The tile for `num_rows` sorted rows over `num_groups` experts on `device`."""
    if _TILES_FOR_BENCH is not None:
        return _TILES_FOR_BENCH
    table = _BLACKWELL_TILES if is_datacenter_blackwell_device(device) else _HOPPER_TILES
    mean_rows = num_rows / max(num_groups, 1)
    return next(tiles for limit, tiles in table if mean_rows <= limit)


def on_by_default(num_rows: int, num_groups: int, device: torch.device) -> bool:
    """Whether the default path takes a Gluon kernel for this many sorted rows.

    On sm90 it always does. On sm100 the Triton kernel's expert tiles are as
    fast up to 64 mean rows per expert, so the Gluon kernel runs above.
    """
    if not is_datacenter_blackwell_device(device):
        return True
    return num_rows / max(num_groups, 1) > _BLACKWELL_MIN_MEAN_ROWS


def supports_gluon_block_fp8_grouped_mm(
    rows: torch.Tensor, weight: torch.Tensor, block: int, tiles: Tiles
) -> bool:
    """Whether a Gluon kernel takes these operands on this device.

    Both need Gluon, contiguous bf16 rows and weights, a K that TMA can copy
    (16-byte rows), and a scale block that each weight tile fits in, so that
    one scale covers a tile at each K step. The sm90 kernel also needs an
    even number of K steps.
    """
    if not _GLUON_AVAILABLE:
        return False
    hopper = is_hopper_device(rows.device)
    if not hopper and not is_datacenter_blackwell_device(rows.device):
        return False
    block_w, _block_t, block_k = tiles[:3]
    k_dim = rows.shape[1]
    return (
        rows.dtype == torch.bfloat16
        and rows.is_contiguous()
        and weight.is_contiguous()
        and k_dim % 16 == 0
        and (not hopper or -(-k_dim // block_k) % 2 == 0)
        and block % block_w == 0
        and block % block_k == 0
    )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


if _GLUON_AVAILABLE:

    @gluon.jit
    def _group_end(group_ends_ptr, group, num_groups, num_rows):  # type: ignore[no-untyped-def]
        """End row of group `group`; group `num_groups` is the off-rank tail."""
        own_end = gl.load(group_ends_ptr + gl.minimum(group, num_groups - 1))
        return gl.where(group < num_groups, own_end, num_rows)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _walk_to(  # type: ignore[no-untyped-def]
        item,
        group,
        first,
        items,
        tiles,
        start,
        end,
        group_ends_ptr,
        num_groups,
        num_rows,
        n_tiles,
        BLOCK_T: gl.constexpr,
    ):
        """Move the group cursor to the group of `item`.

        The cursor is (group, its first item, its item count, its row tiles,
        its start row, its end row). A program's items only grow, so it
        walks the groups once. Past the last group, `group` is
        `num_groups + 1`.
        """
        while (item >= first + items) & (group <= num_groups):
            first += items
            group += 1
            start = end
            end = _group_end(group_ends_ptr, group, num_groups, num_rows)
            tiles = gl.cdiv(end - start, BLOCK_T)
            items = tiles * n_tiles
        return group, first, items, tiles, start, end

    @gluon.jit
    def _dequantize(w, scale):  # type: ignore[no-untyped-def]
        """e4m3 times the block scale in fp32, rounded to bf16 (as `Fp8Expert`)."""
        return (w.to(gl.float32) * scale).to(gl.bfloat16)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _store_tile(  # type: ignore[no-untyped-def]
        acc, out_ptr, col, row, row_end, n_dim, stride_om, num_warps: gl.constexpr
    ):
        """Store `acc[cols, rows]` into `out[rows, cols]`, rows below `row_end`."""
        BLOCK_W: gl.constexpr = acc.shape[0]
        BLOCK_T: gl.constexpr = acc.shape[1]
        # Output columns (the weight rows) are contiguous in `out`.
        threads_w: gl.constexpr = 32 if BLOCK_W >= 256 else BLOCK_W // 8
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[8, 1],
            threads_per_warp=[threads_w, 32 // threads_w],
            warps_per_cta=[1, num_warps],
            order=[0, 1],
        )
        out = gl.convert_layout(acc.to(gl.bfloat16), layout)
        cols = col + gl.arange(0, BLOCK_W, layout=gl.SliceLayout(1, layout))
        rows = row + gl.arange(0, BLOCK_T, layout=gl.SliceLayout(0, layout))
        mask = (cols[:, None] < n_dim) & (rows[None, :] < row_end)
        gl.store(out_ptr + rows[None, :].to(gl.int64) * stride_om + cols[:, None], out, mask=mask)

    @gluon.jit
    def _issue_load(  # type: ignore[no-untyped-def]
        produced, x_desc, w_desc, x_bufs, w_bufs, ready, row, weight_row, k, pred
    ):
        """Copy one row tile and one weight tile into the next ring stage."""
        num_stages: gl.constexpr = x_bufs.type.shape[0]
        stage = produced % num_stages
        bar = ready.index(stage)
        mbarrier.expect(bar, x_desc.block_type.nbytes + w_desc.block_type.nbytes, pred=pred)
        tma.async_load(x_desc, [row, k], bar, x_bufs.index(stage), pred)
        tma.async_load(w_desc, [weight_row, k], bar, w_bufs.index(stage), pred)
        return produced + gl.where(pred, 1, 0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _mma_step(  # type: ignore[no-untyped-def]
        acc,
        w_prev,
        x_bufs,
        w_bufs,
        ready,
        consumed,
        s_base,
        ki,
        stride_sk,
        QBLOCK: gl.constexpr,
        w_layout: gl.constexpr,
    ):
        """One K step: dequantize the stage's weight tile and issue its async wgmma.

        A wgmma reads its register operand while it runs. The wait at the end
        retires the previous step's wgmma, which read `w_prev`, so `w_prev`
        stays live up to that wait and the new tile goes into other
        registers. The new operand comes back, to pass as `w_prev` to the
        next step.
        """
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        BLOCK_K: gl.constexpr = w_bufs.type.shape[2]
        stage = consumed % NUM_STAGES
        mbarrier.wait(ready.index(stage), (consumed // NUM_STAGES) & 1)
        scale = gl.load(s_base + ((ki * BLOCK_K) // QBLOCK) * stride_sk)
        w = _dequantize(w_bufs.index(stage).load(w_layout), scale)
        acc = warpgroup_mma(w, x_bufs.index(stage).permute((1, 0)), acc, is_async=True)
        acc, w_prev = warpgroup_mma_wait(num_outstanding=1, deps=(acc, w_prev))
        return acc, w

    @gluon.jit  # type: ignore[untyped-decorator]
    def _block_fp8_grouped_mm_hopper(  # type: ignore[no-untyped-def]
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
        QBLOCK: gl.constexpr,
        NUM_STAGES: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """Software-pipelined version: the same warps copy and multiply."""
        BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
        BLOCK_K: gl.constexpr = w_desc.block_type.shape[1]
        BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
        gl.static_assert(QBLOCK % BLOCK_W == 0, "a weight tile must fit in one scale block")
        gl.static_assert(QBLOCK % BLOCK_K == 0, "a K step must fit in one scale block")
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BLOCK_T, 16]
        )
        w_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)

        x_bufs = gl.allocate_shared_memory(
            x_desc.dtype, [NUM_STAGES] + x_desc.block_type.shape, x_desc.layout
        )
        w_bufs = gl.allocate_shared_memory(
            w_desc.dtype, [NUM_STAGES] + w_desc.block_type.shape, w_desc.layout
        )
        ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NUM_STAGES):
            mbarrier.init(ready.index(i), count=1)

        n_tiles = gl.cdiv(n_dim, BLOCK_W)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        produced = 0
        consumed = 0
        group = 0
        first = 0
        start = 0
        end = _group_end(group_ends_ptr, 0, num_groups, num_rows)
        tiles = gl.cdiv(end, BLOCK_T)
        items = tiles * n_tiles
        item = gl.program_id(0)
        while group <= num_groups:
            group, first, items, tiles, start, end = _walk_to(
                item,
                group,
                first,
                items,
                tiles,
                start,
                end,
                group_ends_ptr,
                num_groups,
                num_rows,
                n_tiles,
                BLOCK_T,
            )
            if group <= num_groups:
                local = item - first
                col = (local // tiles) * BLOCK_W
                row = start + (local % tiles) * BLOCK_T
                weight_row = group * n_dim + col
                # The off-rank tail runs no K step and stores zeros.
                k_iters = gl.where(group < num_groups, k_steps, 0)
                s_base = s_ptr + group * stride_se + (col // QBLOCK) * stride_sn

                # Every warp is done with the stages of the previous item.
                gl.barrier()
                fence_async_shared()
                for i in gl.static_range(NUM_STAGES - 1):
                    produced = _issue_load(
                        produced,
                        x_desc,
                        w_desc,
                        x_bufs,
                        w_bufs,
                        ready,
                        row,
                        weight_row,
                        i * BLOCK_K,
                        i < k_iters,
                    )
                acc = gl.zeros((BLOCK_W, BLOCK_T), dtype=gl.float32, layout=mma_layout)
                # Two K steps per iteration (`k_iters` is even), so the two
                # register operands of the wgmmas in flight alternate.
                w_odd = gl.zeros((BLOCK_W, BLOCK_K), dtype=gl.bfloat16, layout=w_layout)
                for ki in range(0, k_iters, 2):
                    acc, w_even = _mma_step(
                        acc,
                        w_odd,
                        x_bufs,
                        w_bufs,
                        ready,
                        consumed,
                        s_base,
                        ki,
                        stride_sk,
                        QBLOCK,
                        w_layout,
                    )
                    # Every warpgroup's wgmma of the previous step is done, so
                    # its stage takes the next copy.
                    gl.barrier()
                    fence_async_shared()
                    produced = _issue_load(
                        produced,
                        x_desc,
                        w_desc,
                        x_bufs,
                        w_bufs,
                        ready,
                        row,
                        weight_row,
                        (ki + NUM_STAGES - 1) * BLOCK_K,
                        ki + NUM_STAGES - 1 < k_iters,
                    )
                    acc, w_odd = _mma_step(
                        acc,
                        w_even,
                        x_bufs,
                        w_bufs,
                        ready,
                        consumed + 1,
                        s_base,
                        ki + 1,
                        stride_sk,
                        QBLOCK,
                        w_layout,
                    )
                    gl.barrier()
                    fence_async_shared()
                    produced = _issue_load(
                        produced,
                        x_desc,
                        w_desc,
                        x_bufs,
                        w_bufs,
                        ready,
                        row,
                        weight_row,
                        (ki + NUM_STAGES) * BLOCK_K,
                        ki + NUM_STAGES < k_iters,
                    )
                    consumed += 2
                acc, w_odd = warpgroup_mma_wait(num_outstanding=0, deps=(acc, w_odd))
                _store_tile(
                    acc,
                    out_ptr,
                    col,
                    row,
                    gl.minimum(row + BLOCK_T, end),
                    n_dim,
                    stride_om,
                    num_warps,
                )
                item += gl.num_programs(0)

    @gluon.jit
    def _ws_load_partition(  # type: ignore[no-untyped-def]
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
        """The copy warp: TMA copies into each stage that the multiply freed."""
        BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
        BLOCK_K: gl.constexpr = w_desc.block_type.shape[1]
        BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        nbytes: gl.constexpr = x_desc.block_type.nbytes + w_desc.block_type.nbytes
        n_tiles = gl.cdiv(n_dim, BLOCK_W)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        produced = 0
        group = 0
        first = 0
        start = 0
        end = _group_end(group_ends_ptr, 0, num_groups, num_rows)
        tiles = gl.cdiv(end, BLOCK_T)
        items = tiles * n_tiles
        item = gl.program_id(0)
        while group <= num_groups:
            group, first, items, tiles, start, end = _walk_to(
                item,
                group,
                first,
                items,
                tiles,
                start,
                end,
                group_ends_ptr,
                num_groups,
                num_rows,
                n_tiles,
                BLOCK_T,
            )
            if group <= num_groups:
                local = item - first
                col = (local // tiles) * BLOCK_W
                row = start + (local % tiles) * BLOCK_T
                weight_row = group * n_dim + col
                k_iters = gl.where(group < num_groups, k_steps, 0)
                for ki in range(k_iters):
                    stage = produced % NUM_STAGES
                    # A new barrier counts as released once, so the first
                    # round of stages does not wait.
                    mbarrier.wait(empty.index(stage), ((produced // NUM_STAGES) & 1) ^ 1)
                    bar = ready.index(stage)
                    mbarrier.expect(bar, nbytes)
                    tma.async_load(x_desc, [row, ki * BLOCK_K], bar, x_bufs.index(stage))
                    tma.async_load(w_desc, [weight_row, ki * BLOCK_K], bar, w_bufs.index(stage))
                    produced += 1
                item += gl.num_programs(0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _ws_mma_partition(  # type: ignore[no-untyped-def]
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
        QBLOCK: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """The multiply warps: dequantize, `wgmma`, free the stage, store the tile."""
        BLOCK_W: gl.constexpr = w_bufs.type.shape[1]
        BLOCK_K: gl.constexpr = w_bufs.type.shape[2]
        BLOCK_T: gl.constexpr = x_bufs.type.shape[1]
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BLOCK_T, 16]
        )
        w_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
        n_tiles = gl.cdiv(n_dim, BLOCK_W)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        group = 0
        first = 0
        start = 0
        end = _group_end(group_ends_ptr, 0, num_groups, num_rows)
        tiles = gl.cdiv(end, BLOCK_T)
        items = tiles * n_tiles
        item = gl.program_id(0)
        while group <= num_groups:
            group, first, items, tiles, start, end = _walk_to(
                item,
                group,
                first,
                items,
                tiles,
                start,
                end,
                group_ends_ptr,
                num_groups,
                num_rows,
                n_tiles,
                BLOCK_T,
            )
            if group <= num_groups:
                local = item - first
                col = (local // tiles) * BLOCK_W
                row = start + (local % tiles) * BLOCK_T
                k_iters = gl.where(group < num_groups, k_steps, 0)
                s_base = s_ptr + group * stride_se + (col // QBLOCK) * stride_sn
                acc = gl.zeros((BLOCK_W, BLOCK_T), dtype=gl.float32, layout=mma_layout)
                # Two K steps per iteration (`k_iters` is even), so the two
                # register operands of the wgmmas in flight alternate.
                w_odd = gl.zeros((BLOCK_W, BLOCK_K), dtype=gl.bfloat16, layout=w_layout)
                for ki in range(0, k_iters, 2):
                    acc, w_even = _mma_step(
                        acc,
                        w_odd,
                        x_bufs,
                        w_bufs,
                        ready,
                        consumed,
                        s_base,
                        ki,
                        stride_sk,
                        QBLOCK,
                        w_layout,
                    )
                    # The previous step's wgmma is done: free its stage. The
                    # arrive waits for all warps of this partition first.
                    fence_async_shared()
                    mbarrier.arrive(
                        empty.index((consumed + NUM_STAGES - 1) % NUM_STAGES), pred=ki > 0
                    )
                    acc, w_odd = _mma_step(
                        acc,
                        w_even,
                        x_bufs,
                        w_bufs,
                        ready,
                        consumed + 1,
                        s_base,
                        ki + 1,
                        stride_sk,
                        QBLOCK,
                        w_layout,
                    )
                    fence_async_shared()
                    mbarrier.arrive(empty.index(consumed % NUM_STAGES))
                    consumed += 2
                acc, w_odd = warpgroup_mma_wait(num_outstanding=0, deps=(acc, w_odd))
                mbarrier.arrive(
                    empty.index((consumed + NUM_STAGES - 1) % NUM_STAGES), pred=k_iters > 0
                )
                _store_tile(
                    acc,
                    out_ptr,
                    col,
                    row,
                    gl.minimum(row + BLOCK_T, end),
                    n_dim,
                    stride_om,
                    num_warps,
                )
                item += gl.num_programs(0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _block_fp8_grouped_mm_hopper_ws(  # type: ignore[no-untyped-def]
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
        QBLOCK: gl.constexpr,
        NUM_STAGES: gl.constexpr,
        LOAD_REGS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """Warp-specialized version: one warp copies, `num_warps` warps multiply."""
        BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
        BLOCK_K: gl.constexpr = w_desc.block_type.shape[1]
        gl.static_assert(QBLOCK % BLOCK_W == 0, "a weight tile must fit in one scale block")
        gl.static_assert(QBLOCK % BLOCK_K == 0, "a K step must fit in one scale block")
        x_bufs = gl.allocate_shared_memory(
            x_desc.dtype, [NUM_STAGES] + x_desc.block_type.shape, x_desc.layout
        )
        w_bufs = gl.allocate_shared_memory(
            w_desc.dtype, [NUM_STAGES] + w_desc.block_type.shape, w_desc.layout
        )
        ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NUM_STAGES):
            mbarrier.init(ready.index(i), count=1)
            mbarrier.init(empty.index(i), count=1)
        gl.warp_specialize(
            [
                (
                    _ws_mma_partition,
                    (
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
                        QBLOCK,
                        num_warps,
                    ),
                ),
                (
                    _ws_load_partition,
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
        QBLOCK: gl.constexpr,
    ):
        """sm100 dequantization warps: e4m3 tile in shared memory to bf16 in tensor memory."""
        NUM_STAGES: gl.constexpr = w_bufs.type.shape[0]
        BLOCK_W: gl.constexpr = w_bufs.type.shape[1]
        BLOCK_K: gl.constexpr = w_bufs.type.shape[2]
        WEIGHT_BUFFERS: gl.constexpr = a_tmem.shape[0]
        layout: gl.constexpr = a_tmem.index(0).get_reg_layout()
        n_tiles = gl.cdiv(n_dim, BLOCK_W)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        group = 0
        first = 0
        start = 0
        end = _group_end(group_ends_ptr, 0, num_groups, num_rows)
        tiles = gl.cdiv(end, BLOCK_T)
        items = tiles * n_tiles
        item = gl.program_id(0)
        while group <= num_groups:
            group, first, items, tiles, start, end = _walk_to(
                item,
                group,
                first,
                items,
                tiles,
                start,
                end,
                group_ends_ptr,
                num_groups,
                num_rows,
                n_tiles,
                BLOCK_T,
            )
            if group <= num_groups:
                col = ((item - first) // tiles) * BLOCK_W
                k_iters = gl.where(group < num_groups, k_steps, 0)
                s_base = s_ptr + group * stride_se + (col // QBLOCK) * stride_sn
                for ki in range(k_iters):
                    stage = consumed % NUM_STAGES
                    buffer = consumed % WEIGHT_BUFFERS
                    mbarrier.wait(ready.index(stage), (consumed // NUM_STAGES) & 1)
                    mbarrier.wait(a_empty.index(buffer), ((consumed // WEIGHT_BUFFERS) & 1) ^ 1)
                    scale = gl.load(s_base + ((ki * BLOCK_K) // QBLOCK) * stride_sk)
                    w = _dequantize(w_bufs.index(stage).load(layout), scale)
                    a_tmem.index(buffer).store(w)
                    mbarrier.arrive(a_ready.index(buffer))
                    consumed += 1
                item += gl.num_programs(0)

    @gluon.jit
    def _bw_mma_partition(  # type: ignore[no-untyped-def]
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
    ):
        """sm100 MMA warp: `acc[cols, rows] += W @ rows.T` with W from tensor memory."""
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        BLOCK_T: gl.constexpr = x_bufs.type.shape[1]
        BLOCK_K: gl.constexpr = x_bufs.type.shape[2]
        BLOCK_W: gl.constexpr = a_tmem.shape[1]
        WEIGHT_BUFFERS: gl.constexpr = a_tmem.shape[0]
        ACC_BUFFERS: gl.constexpr = acc_tmem.shape[0]
        n_tiles = gl.cdiv(n_dim, BLOCK_W)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        done = 0
        group = 0
        first = 0
        start = 0
        end = _group_end(group_ends_ptr, 0, num_groups, num_rows)
        tiles = gl.cdiv(end, BLOCK_T)
        items = tiles * n_tiles
        item = gl.program_id(0)
        while group <= num_groups:
            group, first, items, tiles, start, end = _walk_to(
                item,
                group,
                first,
                items,
                tiles,
                start,
                end,
                group_ends_ptr,
                num_groups,
                num_rows,
                n_tiles,
                BLOCK_T,
            )
            if group <= num_groups:
                k_iters = gl.where(group < num_groups, k_steps, 0)
                acc_buffer = done % ACC_BUFFERS
                # A new barrier counts as released once, so the first round
                # of accumulators does not wait.
                mbarrier.wait(acc_empty.index(acc_buffer), ((done // ACC_BUFFERS) & 1) ^ 1)
                acc = acc_tmem.index(acc_buffer)
                for ki in range(k_iters):
                    stage = consumed % NUM_STAGES
                    buffer = consumed % WEIGHT_BUFFERS
                    mbarrier.wait(a_ready.index(buffer), (consumed // WEIGHT_BUFFERS) & 1)
                    mbarrier.wait(ready.index(stage), (consumed // NUM_STAGES) & 1)
                    # Its completion frees the stage and the weight buffer.
                    tcgen05_mma(
                        a_tmem.index(buffer),
                        x_bufs.index(stage).permute((1, 0)),
                        acc,
                        use_acc=ki > 0,
                        mbarriers=[empty.index(stage), a_empty.index(buffer)],
                    )
                    consumed += 1
                tcgen05_commit(acc_ready.index(acc_buffer))
                done += 1
                item += gl.num_programs(0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _bw_store_partition(  # type: ignore[no-untyped-def]
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
        BLOCK_K: gl.constexpr,
        STORE_COLUMNS: gl.constexpr,
    ):
        """sm100 store warps: finished accumulators from tensor memory to `out`."""
        ACC_BUFFERS: gl.constexpr = acc_tmem.shape[0]
        BLOCK_W: gl.constexpr = acc_tmem.shape[1]
        BLOCK_T: gl.constexpr = acc_tmem.shape[2]
        layout: gl.constexpr = acc_tmem.index(0).slice(0, STORE_COLUMNS).get_reg_layout()
        # In the tensor-memory layout each thread holds one output column, so
        # its stores are 2 bytes wide. This layout gives each thread 8
        # neighbouring columns: 16-byte stores, 8 times fewer of them.
        store_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[8, 1],
            threads_per_warp=[BLOCK_W // 8, 256 // BLOCK_W],
            warps_per_cta=[1, gl.num_warps()],
            order=[0, 1],
        )
        n_tiles = gl.cdiv(n_dim, BLOCK_W)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        done = 0
        group = 0
        first = 0
        start = 0
        end = _group_end(group_ends_ptr, 0, num_groups, num_rows)
        tiles = gl.cdiv(end, BLOCK_T)
        items = tiles * n_tiles
        item = gl.program_id(0)
        while group <= num_groups:
            group, first, items, tiles, start, end = _walk_to(
                item,
                group,
                first,
                items,
                tiles,
                start,
                end,
                group_ends_ptr,
                num_groups,
                num_rows,
                n_tiles,
                BLOCK_T,
            )
            if group <= num_groups:
                local = item - first
                col = (local // tiles) * BLOCK_W
                row = start + (local % tiles) * BLOCK_T
                row_end = gl.minimum(row + BLOCK_T, end)
                # The off-rank tail ran no multiply: its tile is zeros.
                computed = (group < num_groups) & (k_steps > 0)
                acc_buffer = done % ACC_BUFFERS
                mbarrier.wait(acc_ready.index(acc_buffer), (done // ACC_BUFFERS) & 1)
                acc = acc_tmem.index(acc_buffer)
                for part in gl.static_range(BLOCK_T // STORE_COLUMNS):
                    columns = acc.slice(part * STORE_COLUMNS, STORE_COLUMNS)
                    values = gl.where(computed, columns.load(layout), 0.0)
                    if part == BLOCK_T // STORE_COLUMNS - 1:
                        # Every column is in registers: the MMA warp may reuse it.
                        mbarrier.arrive(acc_empty.index(acc_buffer))
                    values = gl.convert_layout(values.to(gl.bfloat16), store_layout)
                    cols = col + gl.arange(0, BLOCK_W, layout=gl.SliceLayout(1, store_layout))
                    rows = (
                        row
                        + part * STORE_COLUMNS
                        + gl.arange(0, STORE_COLUMNS, layout=gl.SliceLayout(0, store_layout))
                    )
                    mask = (cols[:, None] < n_dim) & (rows[None, :] < row_end)
                    gl.store(
                        out_ptr + rows[None, :].to(gl.int64) * stride_om + cols[:, None],
                        values,
                        mask=mask,
                    )
                done += 1
                item += gl.num_programs(0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _block_fp8_grouped_mm_blackwell(  # type: ignore[no-untyped-def]
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
        QBLOCK: gl.constexpr,
        NUM_STAGES: gl.constexpr,
        WEIGHT_BUFFERS: gl.constexpr,
        ACC_BUFFERS: gl.constexpr,
        STORE_COLUMNS: gl.constexpr,
        DEQUANT_WARPS: gl.constexpr,
        DEQUANT_REGS: gl.constexpr,
        MMA_REGS: gl.constexpr,
        LOAD_REGS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """sm100 version: copy, dequantization, MMA and store warps."""
        BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
        BLOCK_K: gl.constexpr = w_desc.block_type.shape[1]
        BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
        gl.static_assert(QBLOCK % BLOCK_W == 0, "a weight tile must fit in one scale block")
        gl.static_assert(QBLOCK % BLOCK_K == 0, "a K step must fit in one scale block")
        x_bufs = gl.allocate_shared_memory(
            x_desc.dtype, [NUM_STAGES] + x_desc.block_type.shape, x_desc.layout
        )
        w_bufs = gl.allocate_shared_memory(
            w_desc.dtype, [NUM_STAGES] + w_desc.block_type.shape, w_desc.layout
        )
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
        ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
        a_ready = gl.allocate_shared_memory(
            gl.int64, [WEIGHT_BUFFERS, 1], mbarrier.MBarrierLayout()
        )
        a_empty = gl.allocate_shared_memory(
            gl.int64, [WEIGHT_BUFFERS, 1], mbarrier.MBarrierLayout()
        )
        acc_ready = gl.allocate_shared_memory(gl.int64, [ACC_BUFFERS, 1], mbarrier.MBarrierLayout())
        acc_empty = gl.allocate_shared_memory(gl.int64, [ACC_BUFFERS, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NUM_STAGES):
            mbarrier.init(ready.index(i), count=1)
            mbarrier.init(empty.index(i), count=1)
        for i in gl.static_range(WEIGHT_BUFFERS):
            mbarrier.init(a_ready.index(i), count=1)
            mbarrier.init(a_empty.index(i), count=1)
        for i in gl.static_range(ACC_BUFFERS):
            mbarrier.init(acc_ready.index(i), count=1)
            mbarrier.init(acc_empty.index(i), count=1)
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
                        QBLOCK,
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
                    _ws_load_partition,
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
            [DEQUANT_WARPS, 1, 1],
            [DEQUANT_REGS, MMA_REGS, LOAD_REGS],
        )


def _max_registers(num_warps: int, warp_specialize: bool) -> int | None:
    """Register limit per thread so that one program fits on an SM.

    Registers go to warpgroups of 4 warps. The copy warp adds a warpgroup.
    ptxas takes at most 255, and register reallocation moves multiples of 8.
    """
    if not warp_specialize:
        return None
    warpgroups = (num_warps + 3) // 4 + 1
    return min(255, 65536 // (warpgroups * 128)) // 8 * 8


def blackwell_launch_options(tiles: Tiles) -> dict[str, int]:
    """Constexprs and launch options of the sm100 kernel for these tiles.

    Tensor memory has 512 columns. Each bf16 weight tile takes `BLOCK_K / 2`
    of them and each fp32 accumulator `BLOCK_T`, so two accumulators fit up
    to 128 token rows and one above.
    """
    _block_w, block_t, _block_k, dequant_warps, stages, _ = tiles
    # The store warps (4), the dequantization warps, then one warpgroup that
    # holds the MMA and copy warps.
    warpgroups = 1 + dequant_warps // 4 + 1
    return {
        "NUM_STAGES": stages,
        "WEIGHT_BUFFERS": _WEIGHT_BUFFERS,
        "ACC_BUFFERS": 2 if block_t <= 128 else 1,
        "STORE_COLUMNS": _STORE_COLUMNS,
        "DEQUANT_WARPS": dequant_warps,
        "DEQUANT_REGS": _DEQUANT_REGS,
        "MMA_REGS": _MMA_REGS,
        "LOAD_REGS": _LOAD_REGS,
        "num_warps": 4,
        "maxnreg": min(255, 65536 // (warpgroups * 128)) // 8 * 8,
    }


def gluon_block_fp8_grouped_mm(
    rows: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block: int,
    group_ends: torch.Tensor,
    tiles: Tiles,
) -> torch.Tensor:
    """`rows @ dequant(weight[e]).T` for each row's expert `e`, on sm90 or sm100.

    Arguments as `fp8_grouped_kernel.block_fp8_grouped_mm`, which checks them
    and calls this function when `supports_gluon_block_fp8_grouped_mm` holds.
    `group_ends` must be int32.
    """
    num_rows, k_dim = rows.shape
    num_groups, n_dim, _ = weight.shape
    out = torch.empty(num_rows, n_dim, dtype=rows.dtype, device=rows.device)
    if num_rows == 0:
        return out
    block_w, block_t, block_k, warps, stages, warp_specialize = tiles
    x_layout = gl.NVMMASharedLayout.get_default_for([block_t, block_k], gl.bfloat16)
    w_layout = gl.NVMMASharedLayout.get_default_for([block_w, block_k], gl.float8e4nv)
    x_desc = TensorDescriptor.from_tensor(rows, [block_t, block_k], x_layout)
    w_desc = TensorDescriptor.from_tensor(
        weight.view(num_groups * n_dim, k_dim), [block_w, block_k], w_layout
    )
    # At most one partial row tile per group, plus the off-rank tail.
    items = (triton.cdiv(num_rows, block_t) + num_groups + 1) * triton.cdiv(n_dim, block_w)
    grid = (min(_num_sms(rows.device.index or 0), items),)
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
    if is_datacenter_blackwell_device(rows.device):
        _block_fp8_grouped_mm_blackwell[grid](
            *args, QBLOCK=block, **blackwell_launch_options(tiles)
        )
    elif warp_specialize:
        _block_fp8_grouped_mm_hopper_ws[grid](
            *args,
            QBLOCK=block,
            NUM_STAGES=stages,
            LOAD_REGS=_LOAD_REGS,
            num_warps=warps,
            maxnreg=_max_registers(warps, True),
        )
    else:
        _block_fp8_grouped_mm_hopper[grid](*args, QBLOCK=block, NUM_STAGES=stages, num_warps=warps)
    return out
