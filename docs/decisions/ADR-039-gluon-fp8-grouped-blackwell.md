# ADR-039: Gluon tcgen05 kernel for block-FP8 MoE experts on the B200

Date: 2026-10-22
Status: Accepted

## Context

[ADR-037](ADR-037-block-fp8-grouped-kernel.md) (update) splits many rows
into expert tiles, tiles of one expert each. With these tiles, the GLM-5.3
layer with block-FP8 experts on the B200 was faster than the bf16 layer up
to 1024 tokens. But it was 1.20 and 1.40 times slower at 4096 and 16384
tokens. Alone, the Triton kernel reached about 550 useful TFLOPS at 16384
tokens.

The B200 is an sm100 GPU. Its tensor-core instruction is `tcgen05`, which
takes its left operand from shared memory or from tensor memory. Tensor
memory is a 128-row memory next to the tensor cores. Each SM (one of the
GPU's compute units) has its own.

In the sm100 code that Triton makes for this kernel, the dequantized
weight tile goes into one buffer in tensor memory. `tcgen05` reads it
there. A program walks the shared dimension K of the matmul one tile, or K
step, at a time. With one buffer, the dequantization of a K step waits
until the previous multiply is done. The tile of token rows goes through
registers into shared memory without TMA (the copy engine between global
and shared memory). Triton also gives all warps (groups of 32 threads) the
same work.

The Hopper kernel of [ADR-038](ADR-038-gluon-fp8-grouped-hopper.md) does
not run on sm100. It needs `wgmma`, which sm100 does not have. It also
keeps its weight operand in registers, and `tcgen05` takes no operand from
registers.

## Decision

1. **A Gluon kernel for sm100** (`models/blocks/fp8_grouped_gluon.py`,
   next to the Hopper one). It keeps the numerics of ADR-038. It multiplies
   each e4m3 weight by its fp32 block scale and rounds the product to bf16.
   The tensor cores then multiply in bf16 into an fp32 accumulator.
2. **The schedule of ADR-038.** The kernel keeps the work items and the
   persistent grid of ADR-038. An item is one expert, one tile of its token
   rows, and one tile of output columns. One program per SM walks the
   items. One copy warp fills a ring of shared-memory stages with TMA.
3. **Dequantization warps.** Four warps read the e4m3 weight tile from
   shared memory with 16-byte loads and dequantize it. They write the bf16
   tile into one of four buffers in tensor memory. The compiler uses the
   packed fp32 multiply of Blackwell for the scale.
4. **One MMA warp.** MMA means matrix multiply and accumulate. This warp
   issues `tcgen05_mma`. It takes the weights from tensor memory and the
   tile of token rows from shared memory. The accumulator stays in tensor
   memory. The completion of each multiply frees its shared-memory stage
   and its weight buffer.
5. **Store warps.** Four store warps read a finished accumulator from
   tensor memory in slices of 32 columns. They convert each slice so that a
   thread holds 8 neighboring output columns. Then they write these columns
   with 16-byte stores. With up to 128 token rows the kernel has two
   accumulators, so the store of one tile overlaps the multiply of the
   next. Two 128-column accumulators and four 64-column weight buffers fill
   the 512 columns of tensor memory.
6. **When it runs.** `block_fp8_grouped_mm` calls it above 64 mean rows per
   expert on GPUs with compute capability 10.x (B200, B300). The new
   `is_datacenter_blackwell_device` finds these GPUs. Up to 64 mean rows,
   the Triton expert tiles are as fast, and they stay the default. The RTX
   50 series (12.0) has no tensor memory and keeps the Triton kernel.

   As in ADR-038, the mean is the rows divided by the number of experts on
   all ranks. A rank is one GPU process under expert parallelism. If the
   mean counts only the local experts, expert parallelism makes it too high
   by the number of ranks. Then the Gluon kernel also runs
   below 64 real mean rows. The tile has 128 weight rows, 128 token rows, K
   steps of 128, and 3 stages. The conditions of ADR-038 on Triton and on
   the operands apply too, except the even number of K steps.
7. **Tests.** The compile tests build each sm100 tile on the CPU through
   the launcher's argument binding, so they compile the same binary that
   runs.
   They fail on register spills (values moved from registers to slower
   memory) and on code without a `tcgen05` multiply or a store to tensor
   memory. On a B200, the tests of ADR-038 also compare this kernel with
   fp32 math and with the Triton kernel. They capture it in a CUDA graph and
   run it under Triton's concurrency sanitizer, which stops on a
   shared-memory race or a deadlock. One shape has more work items than
   SMs. So each program walks several items, and its rings of buffers and
   barriers wrap around, also under the sanitizer.

## Alternatives Considered

- **Dequantize into shared memory.** `tcgen05` then reads both operands
  from shared memory. Each K step then also writes the bf16 tile to shared
  memory and reads it back. That costs shared-memory bandwidth, which the
  path through tensor memory does not use.
- **One accumulator with 256 token rows.** It halves the dequantization per
  multiply, and it was 1.5 percent faster at 16384 tokens. But the store of
  a tile then stalls the next multiply. At 4096 tokens it was about 9
  percent slower. The default keeps 128 token rows.
- **The 2-CTA form of `tcgen05`.** A CTA is one program on one SM. In this
  form, two SMs share one 256-row multiply, and each reads half of the tile
  of token rows. It needs a cluster, a group of SMs that can share memory.
  It also needs multicast, one TMA copy into several SMs. That is more
  code, and we did not write it.
- **On by default for every size.** From 64 to 2048 tokens the Gluon
  kernel was 0.96 to 1.03 times as fast as the Triton expert tiles.

## Consequences

- We measured the GLM-5.3 layer on one B200 without a CUDA graph
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)). One forward
  took 8.27 ms with the Triton expert tiles and 6.58 ms with the Gluon
  kernel at 4096 tokens (1.26x). At 8192 tokens it took 14.33 and 11.76 ms
  (1.22x), and at 16384 tokens 26.55 and 21.11 ms (1.26x).
- Against the bf16 layer on `grouped_mm` in the same run, the FP8 layer is
  now 1.04 times faster at 4096 tokens (6.58 against 6.87 ms). At 16384
  tokens it is 1.15 times slower (21.11 against 18.41 ms).
- Alone, each projection at 16384 tokens reaches 759 to 769 useful TFLOPS,
  against 521 to 559 with the Triton kernel. That is still far from the
  bf16 peak of the B200. The issue rate is the speed at which the GPU
  starts instructions. We estimate that the issue rate of the
  dequantization on the CUDA cores sets the limit. The dequantization needs
  about 460 instructions per warp and K step, against about 512 cycles for
  the multiply of the same step. We did not profile it.
- The outputs of the two kernels are bit-identical at every measured size.
- The first version wrote 2 bytes per store. With the same tile, the
  16-byte stores made the layer 1 to 6 percent faster from 4096 tokens.
  Together with the tile from the first sweep, the layer at 4096 tokens
  went from 7.33 to 6.58 ms.
- The dequantization warps run a fence (an instruction that orders memory
  accesses) before they give a weight tile to the MMA warp. Triton's Gluon
  guide asks for a fence between a read of shared memory and a later TMA
  write into it.
- Both B200 runs passed all GPU tests on the first try. `tcgen05` reads no
  registers, so the register fault of ADR-038 cannot occur here.
- There are three FP8 grouped kernels to maintain: Triton for all GPUs,
  Gluon for sm90, Gluon for sm100.
