# ADR-041: Gluon kernels for FP4 MoE experts on Hopper and Blackwell

Date: 2026-10-23
Status: Accepted

## Context

[ADR-040](ADR-040-fp4-grouped-kernel.md) runs the FP4 experts of
DeepSeek-V4-Flash through a Triton kernel. At 16384 tokens the V4-Flash
layer was still 1.28 (H100) and 1.55 (B200) times slower than the same
layer with bf16 experts on `grouped_mm`.

We profiled both projections at 16384 tokens with Nsight Compute
([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md), section "Gluon
kernels for FP4"). The kernels alone take 3.2 (B200) and 1.9 (H100) times
the cycles of `grouped_mm`. Their tensor cores are busy 23 and 38 percent of
the time, against 97 and 85 percent. The Triton code decodes a weight tile,
then multiplies it. With 141 to 219 registers per thread, one program fills
an SM. So no other program runs while the decode waits for the multiply,
or the multiply for the decode.

The block-FP8 kernels had the same problem. ADR-038 and ADR-039 solved it
with Gluon. Some warps copy, some dequantize and some multiply, so the
decode of one tile overlaps the multiply of the previous one.

## Decision

1. **Gluon kernels for FP4** (`models/blocks/fp4_grouped_gluon.py`), one
   for sm90 and one for sm100. They reuse parts of `fp8_grouped_gluon.py`:
   the work items of one expert each, the persistent walk and the copy
   warp. On sm100 they also reuse the MMA and store warps.
   `block_fp4_grouped_mm` calls them for many rows (decision 6).
2. **Packed bytes and their scales.** TMA copies the packed weight tile,
   two e2m1 values per byte. The decode loads one fp32 scale per row and 32
   columns, once, and broadcasts it over the 32 values.
3. **The compiler's unpack.** `gl.fp4_to_fp` turns the bytes into bf16 (on
   sm100 with the `cvt.rn.bf16x2.e2m1x2` instruction). For a blocked
   layout it doubles the values per thread. So the bytes load in a layout
   that unpacks straight into the layout the next step needs. For a linear
   layout, Triton evaluates the input at the doubled shape, which adds a
   register basis past the tile. So the kernels use blocked layouts.
4. **sm100: one or two dequantization warpgroups.** A warpgroup writes the
   bf16 tile into tensor memory. Each thread holds one weight row, the
   layout of a tensor-memory store. With one warpgroup, each scheduler of the SM has one
   dequantization warp, and the MMA warp waited for decoded tiles. With two
   warpgroups, they take turns on the K steps and use two of the four
   weight buffers each.
5. **sm90: decode into shared memory.** The multiply warps read 16 bytes
   per thread, which is one scale group. They decode and write the bf16
   tile into one of two shared-memory buffers. An asynchronous `wgmma` reads it from there
   while they decode the next tile. As the register operand of a `wgmma`, a
   thread's bytes are scattered, so the first version loaded them one byte
   at a time.
6. **When they run.** Above 256 rows, as the expert tiles of ADR-040. On
   sm90 only above 32 mean rows per expert, counted over the experts of
   all ranks: below, the Triton kernel is faster. On sm100 the Gluon kernel
   is faster from 6 mean rows (256 tokens). Tiles by mean rows per expert,
   from sweeps of the V4-Flash layer.
7. **Tests** (`tests/unit/test_fp4_grouped_gluon.py`). Without a GPU, the
   default tiles compile for their GPU family without spills, and the
   operand checks run. On a GPU, the tests compare the kernel with fp32
   math on the dequantized weights and with the Triton kernel. They also
   run it in a CUDA graph and under Triton's concurrency sanitizer.

## Alternatives Considered

- **A better Triton tile.** A tile of 256 token rows, 4 stages, made the
  B200 kernels 1.27 times faster. On the H100 no tile beat the default.
  Triton still decodes and multiplies in turn.
- **Two Triton programs per SM.** A limit of 128 registers lets two
  programs share an SM on sm100 without spills. On sm90 the accumulator is
  in registers, and the limit spills. The Gluon kernels overlap the work
  inside one program on both GPUs.
- **The `wgmma` register operand on sm90.** It was the first version
  (decision 5). It reached 61 to 64 percent tensor-core use, held back by
  the byte loads.
- **Eight dequantization warps in one sm100 partition.** The register
  layout of a tensor-memory store then splits columns over the warps in an
  order that no blocked layout gives. So the unpack needs a layout
  conversion. Two partitions of four warps avoid it.
- **Dequantize every expert to bf16, then `grouped_mm`.** From the memory
  bandwidth, we estimated it about 1.6 times faster than the Triton kernel
  at 16384 tokens on the B200. It gives no gain on the H100 and needs a
  temporary of 6 to 13 GB per layer. Not built.
- **FP8 activations with FP4 weights on the tensor cores.** This changes
  the numerics. For FP8 experts, W8A8 had 8 times the error of W8A16
  (ADR-037).

## Consequences

- The outputs are bit-identical to the Triton kernel of ADR-040.
- At 16384 tokens the V4-Flash layer takes 20.74 ms on the H100 (Triton
  25.78 ms) and 13.65 ms on the B200 (Triton 18.39 ms). The bf16 layer
  takes 19.34 and 12.04 ms. So the FP4 layer is 1.07 (H100) and 1.13
  (B200) times slower than the bf16 layer, against 1.32 and 1.57 times
  before.
- At 2048 and 4096 tokens the FP4 layer is faster than the bf16 layer. It
  is 1.24 and 1.22 times faster on the B200, and 1.16 and 1.09 times on the
  H100. At 8192 tokens it is 1.21 (B200) and 1.26 (H100) times slower.
- At 16384 tokens the work is compute-bound. Both layers multiply in bf16,
  and the FP4 layer also decodes. So with these numerics, FP4 experts can
  approach the bf16 layer at large batches but not beat it. FP4 saves
  memory and memory traffic, which wins at small batches.
- The kernels alone remain slower than `grouped_mm` at 16384 tokens: 1.23
  times on the H100 and 1.49 times on the B200. On the B200,
  `grouped_mm` keeps its tensor cores busy 98 percent of the time at a
  lower clock, because the GPU limits its power.
- Gluon needs Triton 3.8. With another Triton, the Triton kernel of ADR-040
  runs.
