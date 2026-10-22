# ADR-038: Gluon grouped kernel for block-FP8 MoE experts on Hopper

Date: 2026-10-22
Status: Accepted

## Context

[ADR-037](ADR-037-block-fp8-grouped-kernel.md) runs block-FP8 experts
(e4m3 weights, one fp32 scale per 128x128 block) through a Triton grouped
kernel. Up to 256 (token, expert) rows it is faster than the bf16 experts
on `grouped_mm`. From 4096 tokens it was 1.26 to 1.46 times slower on the
H100.

Two things made the Triton kernel slow at large batches:

- Its per-row-tile split gives each program a tile of sorted rows. The
  rows of one tile belong to several experts, so the program runs the
  whole K loop once for each expert in the tile and masks the other rows.
  At 4096 tokens (128 rows per expert, 128-row tiles), a tile spans about
  two experts. The kernel took 13.1 ms of the layer's 16.1 ms GPU time.
- Triton's compiler waits for each `wgmma` (Hopper's asynchronous
  tensor-core instruction) whose left operand is in registers before the
  next K step starts. So the dequantization of the next weight tile does
  not overlap with the tensor cores.

Gluon is the lower-level Triton dialect that exposes TMA (the copy engine
between global and shared memory), mbarriers, `wgmma` and warp
specialization directly.

## Decision

1. **A Gluon kernel for sm90** (`models/blocks/fp8_grouped_gluon.py`) with
   the numerics of ADR-037: each e4m3 weight times its fp32 block scale,
   rounded to bf16, bf16 tensor-core math with an fp32 accumulator.
2. **Work items follow the experts.** An item is one expert, one tile of
   that expert's rows, and one tile of output columns. A tile never mixes
   rows of two experts. Rows routed to another rank (expert parallelism)
   get items that only store zeros.
3. **Persistent grid.** One program per SM walks the items with a stride
   and finds each item's expert from `group_ends` on the GPU. Nothing reads
   back to the host, so the kernel runs in a CUDA graph. Within an expert,
   the items go by column tile, then by row tile, so programs that run at
   the same time read the same weight tile from L2.
4. **TMA ring and a register operand.** TMA copies the e4m3 weight tile and
   the bf16 row tile into a ring of shared-memory stages. The weight tile
   is dequantized in registers and is the left operand of an asynchronous
   `wgmma` (`acc[cols, rows] = W @ rows.T`, as in ADR-037). The
   dequantization of one K step runs while the tensor cores multiply the
   previous one.
5. **Two K steps per loop iteration.** A `wgmma` reads its register
   operand while it runs. The loop alternates two register sets, and each
   set stays live until the wait that retires its `wgmma`. So the kernel
   needs an even number of K steps.
6. **Warp specialization.** One extra warp issues the TMA copies and frees
   the multiply warps from it. The copies for the next item start while
   the multiply warps store the finished tile. The version without it
   stays in the module for comparison.
7. **Tiles by mean rows per expert.** 128 weight rows, K steps of 64, and
   64, 128 or 256 token rows for up to 64, up to 192, and more mean rows
   per expert. 4 multiply warps for 64 rows, 8 for the others.
8. **When it runs.** `block_fp8_grouped_mm` calls it above 256 rows, when
   all of these hold: an sm90 GPU, Triton 3.8 or newer (Gluon changes
   between releases), bf16 rows, a scale block that is a multiple of the
   tile, K a multiple of 16, and an even number of K steps. GLM-5 and
   MiniMax-M3 meet them. In every other case the Triton kernel runs. Up to
   256 rows the per-segment Triton path stays. The B200 keeps the Triton
   kernel.
9. **Tests without a GPU where possible.** Gluon does not run under
   Triton's interpreter. The compile tests build every default tile for
   sm90 on the CPU and fail on register spills and on any instruction that
   writes a register of a `wgmma` in flight (`tests/unit/_wgmma_sass.py`).
   On an H100, the tests compare the kernel with fp32 math and with the
   Triton kernel, capture it in a CUDA graph, and run it under Triton's
   concurrency sanitizer, which stops on a shared-memory race or a
   deadlock.

## Alternatives Considered

- **The same schedule in Triton.** Expert-aligned work items remove the
  repeated K loops, but Triton's compiler still waits for each
  register-operand `wgmma`, so the dequantization does not overlap. It was
  measured later as the Triton kernel's expert-tile split (ADR-037,
  update). On the H100 it gives 60 to 70 percent of the Gluon gain at 1024
  to 4096 tokens and 23 percent at 16384 tokens. The Gluon kernel stays
  1.16 to 1.30 times faster than it from 1024 tokens.
- **Wait for each `wgmma`, as Triton does.** It is safe without the
  two-step loop, but the dequantization then runs while the tensor cores
  are idle.
- **Dequantize into shared memory.** Both `wgmma` operands then come from
  shared memory and no register is at risk. Each K step then writes and
  reads the bf16 tile through shared memory once more, which costs
  shared-memory bandwidth that the register operand does not use.
- **A Blackwell (`tcgen05`) version.** It is a different kernel with tensor
  memory. The B200 numbers of ADR-037 show a smaller gap, and this change
  does not cover it.

## Consequences

- Measured on one H100 with the GLM-5.3 layer
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)). Eager
  milliseconds per forward, Triton kernel against Gluon kernel: 8.07 and
  5.39 at 1024 tokens (1.50x), 16.16 and 10.03 at 4096 tokens (1.61x),
  42.38 and 30.05 at 16384 tokens (1.41x). At 64 and 256 tokens the Gluon
  kernel is 1.07 and 1.06 times faster.
- Against the same layer with bf16 experts on `grouped_mm`, the FP8 layer
  is now 1.52 and 1.26 times faster at 1024 and 4096 tokens (5.39 against
  8.21 ms, 10.03 against 12.62 ms) and equal at 16384 tokens (30.05
  against 30.29 ms). It keeps half the expert memory.
- At 4096 tokens the kernel takes 7.0 ms of the layer's 10.2 ms GPU time,
  against 13.1 ms before. Alone, the gate-up projection at 16384 tokens
  reaches 495 useful TFLOPS, against 314 with the Triton kernel.
- The outputs of the two kernels are bit-identical at every measured size.
- Warp specialization makes the layer 3.5 to 8.5 percent faster than the
  version without it, at every measured size.
- The first warp-specialized version gave wrong results that changed from
  run to run. The next K step wrote the registers that the previous
  `wgmma` was still reading. ptxas gives no warning about it, and the
  concurrency sanitizer does not track registers. The SASS check of
  decision 9 reports the fault without a GPU: 42 and 66 writes in the two
  faulty builds, none after the fix.
- There are two FP8 grouped kernels to maintain for the H100. CI pins
  Triton 3.6, so its Gluon tests skip there. The `mini-infer-triton-interp`
  Docker image runs the compile tests, and the H100 runs the rest.
