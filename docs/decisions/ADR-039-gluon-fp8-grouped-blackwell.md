# ADR-039: Gluon tcgen05 kernel for block-FP8 MoE experts on the B200

Date: 2026-10-22
Status: Accepted

## Context

After the expert tiles of [ADR-037](ADR-037-block-fp8-grouped-kernel.md)
(update), the GLM-5.3 layer with block-FP8 experts on the B200 was faster
than the bf16 layer up to 1024 tokens, but 1.20 and 1.40 times slower at
4096 and 16384 tokens. Alone, the Triton kernel reached about 550 useful
TFLOPS at 16384 tokens.

In the sm100 code that Triton makes for this kernel, the dequantized
weight tile goes into one tensor-memory buffer, which `tcgen05` (the
Blackwell tensor-core instruction) reads. With one buffer, the
dequantization of a K step waits until the previous multiply is done. The
row tile goes through registers into shared memory without TMA, and no
warp is specialized.

The Hopper kernel of [ADR-038](ADR-038-gluon-fp8-grouped-hopper.md) does
not run on sm100: there is no `wgmma`, and `tcgen05` takes no operand from
registers. Its left operand comes from shared memory or tensor memory, a
128-row memory next to each SM's tensor cores.

## Decision

1. **A Gluon kernel for sm100** in `models/blocks/fp8_grouped_gluon.py`,
   next to the Hopper one, with the same numerics: each e4m3 weight times
   its fp32 block scale, rounded to bf16, bf16 tensor-core math with an
   fp32 accumulator.
2. **The schedule of ADR-038.** The same expert-aligned work items, the
   same persistent walk, and the same copy warp, which fills a ring of
   shared-memory stages with TMA.
3. **Dequantization warps.** Four warps read the e4m3 weight tile from
   shared memory with 16-byte loads, dequantize it, and write the bf16
   tile into one of four tensor-memory buffers. The compiler uses
   Blackwell's packed fp32 multiply for the scale.
4. **One MMA warp.** It issues `tcgen05_mma` with the weights from tensor
   memory, the row tile from shared memory, and the accumulator in tensor
   memory. The completion of each multiply frees its shared-memory stage
   and its weight buffer.
5. **Store warps.** The default partition reads a finished accumulator from
   tensor memory in slices of 32 columns, converts each slice so that a
   thread holds 8 neighbouring output columns, and writes them with
   16-byte stores. With up to 128 token rows there are two accumulators,
   so the store of one tile overlaps the multiply of the next. Two 128-column
   accumulators and four 64-column weight buffers fill the 512 columns of
   tensor memory.
6. **When it runs.** On GPUs with compute capability 10.x (B200, B300;
   the new `is_datacenter_blackwell_device`) above 64 mean rows per expert.
   Below, the Triton expert tiles are as fast, and they stay the default.
   The tile is 128 weight rows, 128 token rows, K steps of 128, and 3
   stages. The RTX 50 series (12.0) has no tensor memory and keeps the
   Triton kernel.
7. **Tests.** The compile tests build each sm100 tile on the CPU and fail
   on register spills or a missing `tcgen05` multiply or tensor-memory
   store. On a B200 the tests of ADR-038 run for this kernel too: against
   fp32 math and the Triton kernel, in a CUDA graph, and under Triton's
   concurrency sanitizer.

## Alternatives Considered

- **Dequantize into shared memory.** `tcgen05` then reads both operands
  from shared memory. Each K step writes and reads the bf16 tile through
  shared memory once more, which costs bandwidth that tensor memory does
  not use.
- **One accumulator with 256 token rows.** It halves the dequantization per
  multiply, and it was 1.5 percent faster at 16384 tokens. But the store
  of a tile then stalls the next multiply, and it was about 9 percent
  slower at 4096 tokens. The kernel keeps one tile.
- **The 2-CTA form of `tcgen05`.** Two SMs share one 256-row multiply and
  read half of the row tile each. It is more code (clusters, multicast),
  and it is not done.
- **On by default for every size.** From 64 to 2048 tokens the Gluon
  kernel was 0.96 to 1.03 times as fast as the Triton expert tiles.

## Consequences

- Measured on one B200 with the GLM-5.3 layer
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)). Eager
  milliseconds per forward, Triton expert tiles against the Gluon kernel:
  8.27 and 6.58 at 4096 tokens (1.26x), 14.33 and 11.76 at 8192 tokens
  (1.22x), 26.55 and 21.11 at 16384 tokens (1.26x).
- Against the bf16 layer on `grouped_mm` in the same run, the FP8 layer
  is now 1.04 times faster at 4096 tokens (6.58 against 6.87 ms) and 1.15
  times slower at 16384 tokens (21.11 against 18.41 ms).
- Alone, each projection at 16384 tokens reaches 759 to 769 useful TFLOPS,
  against 521 to 559 with the Triton kernel. That is still far from the
  B200's bf16 peak. The estimated limit is the issue rate of the
  dequantization on the CUDA cores: about 460 instructions per warp and K
  step, against about 512 cycles for the multiply of the same step. It was
  not profiled.
- The outputs of the two kernels are bit-identical at every measured size.
- The first version wrote 2 bytes per store. With the same tile, the
  16-byte stores made the layer 1 to 6 percent faster from 4096 tokens.
  Together with the tile from the first sweep, the layer at 4096 tokens
  went from 7.33 to 6.58 ms.
- Both B200 runs passed all GPU tests on the first try. `tcgen05` reads no
  registers, so the register fault of ADR-038 cannot occur here.
- There are three FP8 grouped kernels to maintain: Triton for all GPUs,
  Gluon for sm90, Gluon for sm100.
