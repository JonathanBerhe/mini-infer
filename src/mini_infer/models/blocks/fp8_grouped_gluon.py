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
  The walk takes the items of an expert by column tile, then by row tile.
  So programs that run at the same time read the same weight tile from L2.
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
operand. So dequantization warps write the bf16 weight tile into tensor
memory, and one MMA warp multiplies it with the row tile from shared memory.
The accumulator is in tensor memory too. Store warps write a finished
accumulator to `out` while the MMA warp multiplies the next tile.

Rows routed to another rank's experts (expert parallelism) sort after the
local groups. Their items store zeros.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from mini_infer.device import is_datacenter_blackwell_device, is_hopper_device

try:
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import aggregate
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
    # kernels are tested with Triton 3.8, so a newer Triton keeps the Triton
    # kernel until they are tested with it.
    _GLUON_AVAILABLE = tuple(int(part) for part in triton.__version__.split(".")[:2]) == (3, 8)
except ImportError:  # no Triton, or a Triton without this Gluon API
    _GLUON_AVAILABLE = False


GluonTiles = tuple[int, int, int, int, int, bool]

# Tiles as (BLOCK_W, BLOCK_T, BLOCK_K, num_warps, num_stages, warp
# specialization): weight rows (output columns), rows of tokens, K, warps of
# the multiply, ring stages, and one extra warp for the copies. `BLOCK_T`
# comes from the mean rows per expert, so that few rows of a tile are
# padding. Picked from a sweep of the GLM-5.3 layer on an H100
# (`docs/benchmarks/2026-10-21-moe-grouped-gemm.md`).
_HOPPER_TILES: tuple[tuple[float, GluonTiles], ...] = (
    (64, (128, 64, 64, 4, 4, True)),
    (192, (128, 128, 64, 8, 4, True)),
    (float("inf"), (128, 256, 64, 8, 3, True)),
)
# On sm100, `num_warps` counts the dequantization warps. The kernel also has
# a copy warp, an MMA warp and 4 store warps, and ignores the last field.
# Picked from a sweep on a B200. A stage stays busy from its copy through the
# dequantization and the multiply, so the large tile takes a fourth stage
# (196 KB of shared memory): the tensor cores then wait less for the copies.
_BLACKWELL_TILES: tuple[tuple[float, GluonTiles], ...] = (
    (64, (128, 64, 64, 8, 4, True)),
    (float("inf"), (128, 128, 128, 4, 4, True)),
)
# Up to the first Blackwell tile's limit, the Triton kernel's expert tiles are
# as fast, so the default path keeps them there.
_BLACKWELL_MIN_MEAN_ROWS = _BLACKWELL_TILES[0][0]
# Bench hook, not a public API: force a tile.
_TILES_FOR_BENCH: GluonTiles | None = None
# Test hook, not a public API: the number of programs. With fewer programs
# than items, each program walks several items.
_PROGRAMS_FOR_TESTS: int | None = None
# Registers of the copy warp. It runs the item walk and issues TMA copies.
_LOAD_REGS = 40
# sm100: registers of the dequantization warps and of the MMA warp, the
# weight tiles in tensor memory, and the accumulator columns that the store
# warps read at a time.
_DEQUANT_REGS = 192
_MMA_REGS = 40
_WEIGHT_BUFFERS = 4
_STORE_COLUMNS = 32


def default_tiles(mean_rows: float, device: torch.device) -> GluonTiles:
    """The tile for this many sorted rows per expert, from the table of `device`.

    sm100 has its own table. Every other device gets the sm90 one.
    """
    table = _BLACKWELL_TILES if is_datacenter_blackwell_device(device) else _HOPPER_TILES
    return next(tiles for limit, tiles in table if mean_rows <= limit)


def gluon_tiles(
    rows: torch.Tensor,
    weight: torch.Tensor,
    block: int,
    group_ends: torch.Tensor,
    *,
    num_experts: int,
    by_default: bool,
) -> GluonTiles | None:
    """The tile of a Gluon kernel for these operands, or None for the Triton kernel.

    The kernels need Triton 3.8 and an sm90 or sm100 GPU. They also need
    contiguous bf16 rows, contiguous weights, int32 `group_ends`, and a K that
    is a multiple of 16. TMA needs 16-byte aligned operands. One scale block
    must cover a whole weight tile and a whole K step. The sm90 kernel also
    needs an even number of K steps.

    `num_experts` counts the experts of all ranks. With expert parallelism,
    `rows` holds the rows of every rank, so the rows over all experts give
    the mean rows of a local expert. With `by_default`, the sm100 kernel runs
    only above `_BLACKWELL_MIN_MEAN_ROWS` mean rows per expert.
    """
    if not _GLUON_AVAILABLE:
        return None
    hopper = is_hopper_device(rows.device)
    if not hopper and not is_datacenter_blackwell_device(rows.device):
        return None
    num_rows, k_dim = rows.shape
    mean_rows = num_rows / max(num_experts, 1)
    if by_default and not hopper and mean_rows <= _BLACKWELL_MIN_MEAN_ROWS:
        return None
    tiles = _TILES_FOR_BENCH or default_tiles(mean_rows, rows.device)
    block_w, _block_t, block_k = tiles[:3]
    fits = (
        rows.dtype == torch.bfloat16
        and group_ends.dtype == torch.int32
        and rows.is_contiguous()
        and weight.is_contiguous()
        and rows.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and k_dim % 16 == 0
        and (not hopper or -(-k_dim // block_k) % 2 == 0)
        and block % block_w == 0
        and block % block_k == 0
    )
    return tiles if fits else None


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


if _GLUON_AVAILABLE:

    @gluon.jit
    def _group_end(group_ends_ptr, group, num_groups, num_rows):  # type: ignore[no-untyped-def]
        """End row of group `group`. Group `num_groups` is the off-rank tail."""
        own_end = gl.load(group_ends_ptr + gl.minimum(group, num_groups - 1))
        return gl.where(group < num_groups, own_end, num_rows)

    @aggregate
    class _Walk:
        """A program's cursor over the work items, one group (expert) at a time.

        Inside a group, the items go by column tile, then by row tile. The
        cursor holds the group, its first item, its item count, its row
        tiles, and its start and end rows. Group `num_groups` is the
        off-rank tail. A program's items only grow, so it walks the groups
        once. Past the tail, `group` is `num_groups + 1`.
        """

        group_ends_ptr: gl.tensor
        num_groups: gl.tensor
        num_rows: gl.tensor
        col_tiles: gl.tensor
        group: gl.tensor
        first: gl.tensor
        items: gl.tensor
        row_tiles: gl.tensor
        start: gl.tensor
        end: gl.tensor
        BLOCK_T: gl.constexpr

        @gluon.jit  # type: ignore[untyped-decorator]
        def create(  # type: ignore[no-untyped-def]
            group_ends_ptr, num_groups, num_rows, col_tiles, BLOCK_T: gl.constexpr
        ):
            """A cursor on group 0."""
            end = _group_end(group_ends_ptr, 0, num_groups, num_rows)
            row_tiles = gl.cdiv(end, BLOCK_T)
            return _Walk(  # type: ignore[call-arg]
                group_ends_ptr,
                num_groups,
                num_rows,
                col_tiles,
                gl.to_tensor(0),
                gl.to_tensor(0),
                row_tiles * col_tiles,
                row_tiles,
                gl.to_tensor(0),
                end,
                BLOCK_T,
            )

        @gluon.jit
        def seek(self, item):  # type: ignore[no-untyped-def]
            """The cursor on the group of `item`."""
            group = self.group
            first = self.first
            items = self.items
            row_tiles = self.row_tiles
            start = self.start
            end = self.end
            while (item >= first + items) & (group <= self.num_groups):
                first += items
                group += 1
                start = end
                end = _group_end(self.group_ends_ptr, group, self.num_groups, self.num_rows)
                row_tiles = gl.cdiv(end - start, self.BLOCK_T)
                items = row_tiles * self.col_tiles
            return _Walk(  # type: ignore[call-arg]
                self.group_ends_ptr,
                self.num_groups,
                self.num_rows,
                self.col_tiles,
                group,
                first,
                items,
                row_tiles,
                start,
                end,
                self.BLOCK_T,
            )

        @gluon.jit  # type: ignore[untyped-decorator]
        def tile(self, item, BLOCK_W: gl.constexpr):  # type: ignore[no-untyped-def]
            """The first output column and the first row of `item`."""
            local = item - self.first
            col = (local // self.row_tiles) * BLOCK_W
            row = self.start + (local % self.row_tiles) * self.BLOCK_T
            return col, row

        @gluon.jit
        def k_iters(self, k_steps):  # type: ignore[no-untyped-def]
            """The K steps of the group: none for the off-rank tail, which stores zeros."""
            return gl.where(self.group < self.num_groups, k_steps, 0)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _ring_slot(count, SIZE: gl.constexpr):  # type: ignore[no-untyped-def]
        """The buffer of the `count`-th use of a ring and its barrier phase."""
        return count % SIZE, (count // SIZE) & 1

    @gluon.jit  # type: ignore[untyped-decorator]
    def _allocate_ring(  # type: ignore[no-untyped-def]
        x_desc, w_desc, QBLOCK: gl.constexpr, NUM_STAGES: gl.constexpr
    ):
        """The shared-memory stages of the row tiles and the weight tiles."""
        gl.static_assert(
            QBLOCK % w_desc.block_type.shape[0] == 0, "a weight tile must fit in one scale block"
        )
        gl.static_assert(
            QBLOCK % w_desc.block_type.shape[1] == 0, "a K step must fit in one scale block"
        )
        x_bufs = gl.allocate_shared_memory(
            x_desc.dtype, [NUM_STAGES] + x_desc.block_type.shape, x_desc.layout
        )
        w_bufs = gl.allocate_shared_memory(
            w_desc.dtype, [NUM_STAGES] + w_desc.block_type.shape, w_desc.layout
        )
        return x_bufs, w_bufs

    @gluon.jit  # type: ignore[untyped-decorator]
    def _allocate_barriers(COUNT: gl.constexpr):  # type: ignore[no-untyped-def]
        """`COUNT` mbarriers, each completed by one arrive."""
        bars = gl.allocate_shared_memory(gl.int64, [COUNT, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(COUNT):
            mbarrier.init(bars.index(i), count=1)
        return bars

    @gluon.constexpr_function
    def _wgmma_layouts(num_warps, BLOCK_T):  # type: ignore[no-untyped-def]
        """The sm90 accumulator layout and the layout of the weight-tile operand."""
        mma_layout = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BLOCK_T, 16]
        )
        return mma_layout, gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)

    @gluon.jit
    def _dequantize(w, scale):  # type: ignore[no-untyped-def]
        """e4m3 times the block scale in fp32, rounded to bf16 (as `Fp8Expert`)."""
        return (w.to(gl.float32) * scale).to(gl.bfloat16)

    @gluon.jit  # type: ignore[untyped-decorator]
    def _store_tile(  # type: ignore[no-untyped-def]
        acc, out_ptr, col, row, row_end, n_dim, stride_om, num_warps: gl.constexpr
    ):
        """Store `acc[cols, rows]` into `out[rows, cols]`, rows below `row_end`.

        Output columns (the weight rows) are contiguous in `out`. The layout
        gives each thread 8 neighboring columns, so it writes 16 bytes at a
        time.
        """
        BLOCK_W: gl.constexpr = acc.shape[0]
        BLOCK_T: gl.constexpr = acc.shape[1]
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
        w_reg_layout: gl.constexpr,
    ):
        """One K step: dequantize the stage's weight tile and issue its async wgmma.

        A wgmma reads its register operand while it runs. The wait at the end
        lets the previous step's wgmma finish, and that wgmma reads `w_prev`.
        So `w_prev` keeps its registers up to that wait, and the new tile goes
        into other registers. The function returns the new operand, which the
        next step takes as `w_prev`.
        """
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        BLOCK_K: gl.constexpr = w_bufs.type.shape[2]
        stage, phase = _ring_slot(consumed, NUM_STAGES)
        mbarrier.wait(ready.index(stage), phase)
        scale = gl.load(s_base + ((ki * BLOCK_K) // QBLOCK) * stride_sk)
        w = _dequantize(w_bufs.index(stage).load(w_reg_layout), scale)
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
        mma_layout: gl.constexpr = _wgmma_layouts(num_warps, BLOCK_T)[0]
        w_reg_layout: gl.constexpr = _wgmma_layouts(num_warps, BLOCK_T)[1]
        x_bufs, w_bufs = _allocate_ring(x_desc, w_desc, QBLOCK, NUM_STAGES)
        ready = _allocate_barriers(NUM_STAGES)

        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        produced = 0
        consumed = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                col, row = walk.tile(item, BLOCK_W)
                weight_row = walk.group * n_dim + col
                k_iters = walk.k_iters(k_steps)
                s_base = s_ptr + walk.group * stride_se + (col // QBLOCK) * stride_sn

                # Every warp is done with the stages of the previous item. The
                # fence orders their shared-memory reads before the next TMA
                # writes into the same stages.
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
                # register operands of the running wgmmas alternate.
                w_odd = gl.zeros((BLOCK_W, BLOCK_K), dtype=gl.bfloat16, layout=w_reg_layout)
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
                        w_reg_layout,
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
                        w_reg_layout,
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
                    gl.minimum(row + BLOCK_T, walk.end),
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
                    produced = _issue_load(
                        produced,
                        x_desc,
                        w_desc,
                        x_bufs,
                        w_bufs,
                        ready,
                        row,
                        weight_row,
                        ki * BLOCK_K,
                        True,
                    )
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
        """The multiply warps: dequantize, `wgmma`, free the stage, store the tile.

        Before each release of a stage, a fence orders the shared-memory read
        of its weight tile before the next TMA write into it.
        """
        BLOCK_W: gl.constexpr = w_bufs.type.shape[1]
        BLOCK_K: gl.constexpr = w_bufs.type.shape[2]
        BLOCK_T: gl.constexpr = x_bufs.type.shape[1]
        NUM_STAGES: gl.constexpr = x_bufs.type.shape[0]
        mma_layout: gl.constexpr = _wgmma_layouts(num_warps, BLOCK_T)[0]
        w_reg_layout: gl.constexpr = _wgmma_layouts(num_warps, BLOCK_T)[1]
        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                col, row = walk.tile(item, BLOCK_W)
                k_iters = walk.k_iters(k_steps)
                s_base = s_ptr + walk.group * stride_se + (col // QBLOCK) * stride_sn
                acc = gl.zeros((BLOCK_W, BLOCK_T), dtype=gl.float32, layout=mma_layout)
                # Two K steps per iteration (`k_iters` is even), so the two
                # register operands of the running wgmmas alternate.
                w_odd = gl.zeros((BLOCK_W, BLOCK_K), dtype=gl.bfloat16, layout=w_reg_layout)
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
                        w_reg_layout,
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
                        w_reg_layout,
                    )
                    fence_async_shared()
                    mbarrier.arrive(empty.index(consumed % NUM_STAGES))
                    consumed += 2
                acc, w_odd = warpgroup_mma_wait(num_outstanding=0, deps=(acc, w_odd))
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
        x_bufs, w_bufs = _allocate_ring(x_desc, w_desc, QBLOCK, NUM_STAGES)
        ready = _allocate_barriers(NUM_STAGES)
        empty = _allocate_barriers(NUM_STAGES)
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
        # The e4m3 tile loads straight into the register layout of a
        # tensor-memory store, so no layout conversion runs.
        layout: gl.constexpr = a_tmem.index(0).get_reg_layout()
        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                col, _row = walk.tile(item, BLOCK_W)
                s_base = s_ptr + walk.group * stride_se + (col // QBLOCK) * stride_sn
                for ki in range(walk.k_iters(k_steps)):
                    stage, phase = _ring_slot(consumed, NUM_STAGES)
                    buffer, buffer_phase = _ring_slot(consumed, WEIGHT_BUFFERS)
                    mbarrier.wait(ready.index(stage), phase)
                    mbarrier.wait(a_empty.index(buffer), buffer_phase ^ 1)
                    scale = gl.load(s_base + ((ki * BLOCK_K) // QBLOCK) * stride_sk)
                    w = _dequantize(w_bufs.index(stage).load(layout), scale)
                    a_tmem.index(buffer).store(w)
                    # The completion of the multiply frees the stage for the
                    # next TMA write. The fence orders this shared-memory read
                    # before that write.
                    fence_async_shared()
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
        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        consumed = 0
        done = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                acc_buffer, acc_phase = _ring_slot(done, ACC_BUFFERS)
                # A new barrier counts as released once, so the first round
                # of accumulators does not wait.
                mbarrier.wait(acc_empty.index(acc_buffer), acc_phase ^ 1)
                acc = acc_tmem.index(acc_buffer)
                for ki in range(walk.k_iters(k_steps)):
                    stage, phase = _ring_slot(consumed, NUM_STAGES)
                    buffer, buffer_phase = _ring_slot(consumed, WEIGHT_BUFFERS)
                    mbarrier.wait(a_ready.index(buffer), buffer_phase)
                    mbarrier.wait(ready.index(stage), phase)
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
        """sm100 store warps: finished accumulators from tensor memory to `out`.

        They read an accumulator in slices of `STORE_COLUMNS` token rows. In
        the tensor-memory layout each thread holds one output column, so
        `_store_tile` converts each slice before its 16-byte stores.
        """
        ACC_BUFFERS: gl.constexpr = acc_tmem.shape[0]
        BLOCK_W: gl.constexpr = acc_tmem.shape[1]
        BLOCK_T: gl.constexpr = acc_tmem.shape[2]
        layout: gl.constexpr = acc_tmem.index(0).slice(0, STORE_COLUMNS).get_reg_layout()
        walk = _Walk.create(group_ends_ptr, num_groups, num_rows, gl.cdiv(n_dim, BLOCK_W), BLOCK_T)
        k_steps = gl.cdiv(k_dim, BLOCK_K)
        done = 0
        item = gl.program_id(0)
        while walk.group <= num_groups:
            walk = walk.seek(item)
            if walk.group <= num_groups:
                col, row = walk.tile(item, BLOCK_W)
                row_end = gl.minimum(row + BLOCK_T, walk.end)
                # Without K steps the MMA warp wrote nothing: the tile is zeros.
                computed = walk.k_iters(k_steps) > 0
                acc_buffer, acc_phase = _ring_slot(done, ACC_BUFFERS)
                mbarrier.wait(acc_ready.index(acc_buffer), acc_phase)
                acc = acc_tmem.index(acc_buffer)
                for part in gl.static_range(BLOCK_T // STORE_COLUMNS):
                    columns = acc.slice(part * STORE_COLUMNS, STORE_COLUMNS)
                    values = gl.where(computed, columns.load(layout), 0.0)
                    if part == BLOCK_T // STORE_COLUMNS - 1:
                        # All slices are in registers, so the MMA warp can
                        # write the next tile into this accumulator.
                        mbarrier.arrive(acc_empty.index(acc_buffer))
                    _store_tile(
                        values,
                        out_ptr,
                        col,
                        row + part * STORE_COLUMNS,
                        row_end,
                        n_dim,
                        stride_om,
                        gl.num_warps(),
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
        x_bufs, w_bufs = _allocate_ring(x_desc, w_desc, QBLOCK, NUM_STAGES)
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

    @functools.cache
    def _shared_layouts(block_w: int, block_t: int, block_k: int) -> tuple[Any, Any]:
        """The shared-memory layouts of a row tile and of a weight tile."""
        return (
            gl.NVMMASharedLayout.get_default_for([block_t, block_k], gl.bfloat16),
            gl.NVMMASharedLayout.get_default_for([block_w, block_k], gl.float8e4nv),
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


def blackwell_launch_options(tiles: GluonTiles) -> dict[str, int]:
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


def _kernel_and_arguments(
    rows: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    *,
    block: int,
    group_ends: torch.Tensor,
    tiles: GluonTiles,
    blackwell: bool,
) -> tuple[Any, tuple[object, ...], dict[str, Any]]:
    """The kernel for these tiles, its arguments, and its constexprs and launch options.

    The launcher and the compile tests both build the kernel call here.
    """
    num_rows, k_dim = rows.shape
    num_groups, n_dim, _ = weight.shape
    block_w, block_t, block_k, warps, stages, warp_specialize = tiles
    x_layout, w_layout = _shared_layouts(block_w, block_t, block_k)
    x_desc = TensorDescriptor.from_tensor(rows, [block_t, block_k], x_layout)
    w_desc = TensorDescriptor.from_tensor(
        weight.view(num_groups * n_dim, k_dim), [block_w, block_k], w_layout
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
        return (
            _block_fp8_grouped_mm_blackwell,
            args,
            {"QBLOCK": block, **blackwell_launch_options(tiles)},
        )
    if warp_specialize:
        return (
            _block_fp8_grouped_mm_hopper_ws,
            args,
            {
                "QBLOCK": block,
                "NUM_STAGES": stages,
                "LOAD_REGS": _LOAD_REGS,
                "num_warps": warps,
                "maxnreg": _max_registers(warps, True),
            },
        )
    return (
        _block_fp8_grouped_mm_hopper,
        args,
        {"QBLOCK": block, "NUM_STAGES": stages, "num_warps": warps},
    )


def gluon_block_fp8_grouped_mm(
    rows: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block: int,
    group_ends: torch.Tensor,
    tiles: GluonTiles,
) -> torch.Tensor:
    """`rows @ dequant(weight[e]).T` for each row's expert `e`, on sm90 or sm100.

    Arguments as `fp8_grouped_kernel.block_fp8_grouped_mm`, which checks them
    and calls this function when `gluon_tiles` gives a tile.
    """
    num_rows, _ = rows.shape
    num_groups, n_dim, _ = weight.shape
    out = torch.empty(num_rows, n_dim, dtype=rows.dtype, device=rows.device)
    if num_rows == 0:
        return out
    kernel, args, options = _kernel_and_arguments(
        rows,
        weight,
        scale,
        out,
        block=block,
        group_ends=group_ends,
        tiles=tiles,
        blackwell=is_datacenter_blackwell_device(rows.device),
    )
    block_w, block_t = tiles[:2]
    # At most one partial row tile per group, plus the off-rank tail.
    items = (triton.cdiv(num_rows, block_t) + num_groups + 1) * triton.cdiv(n_dim, block_w)
    programs = _PROGRAMS_FOR_TESTS or _num_sms(rows.device.index or 0)
    kernel[(min(programs, items),)](*args, **options)
    return out
