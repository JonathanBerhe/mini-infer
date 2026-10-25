# ADR-038: Gluon grouped kernel for block-FP8 MoE experts on Hopper

Date: 2026-10-22
Status: Accepted

## Context

[ADR-037](ADR-037-block-fp8-grouped-kernel.md) runs block-FP8 experts (e4m3
weights, one fp32 scale per 128x128 block) through a Triton grouped kernel.
From 4096 tokens it was 1.26 to 1.39 times slower than bf16 on the H100,
for two reasons:

- A program gets sorted rows of several experts. So it runs its K loop (the
  loop over the shared dimension K of the matmul) once per expert.
- Triton waits for each `wgmma` (the asynchronous tensor-core instruction of
  Hopper) with an operand in registers before the next K step. So the
  dequantization of the next weight tile does not overlap the multiply.

Gluon is the lower-level Triton dialect. It gives direct access to TMA (the
copy engine to shared memory), `wgmma` and warp specialization (different
warps do different jobs).

## Decision

1. A Gluon kernel for sm90 (H100) in
   `models/blocks/moe/fp8_grouped_gluon.py`. It keeps the numerics of ADR-037:
   each e4m3 weight times its fp32 block scale, rounded to bf16, then a bf16
   multiply into an fp32 accumulator.
2. A work item is one expert, one tile of its token rows and one tile of
   output columns. Under expert parallelism, rows routed to another rank
   (GPU process) get items that only store zeros.
3. A persistent grid: one program per SM walks the items and finds the
   expert of each item from `group_ends` on the GPU. So the kernel runs in a
   CUDA graph.
4. TMA fills a ring of shared-memory stages. The kernel dequantizes the
   weight tile in registers, as the left operand of an asynchronous
   `wgmma`. So the dequantization overlaps the previous multiply.
5. A `wgmma` reads its register operand while it runs. So the loop
   alternates two register sets over two K steps per iteration, and the
   number of K steps must be even.
6. Warp specialization: one copy warp issues the TMA copies, and the
   multiply warps dequantize and multiply.
7. Tiles follow the mean rows per expert: the rows divided by the experts on
   all ranks, passed as `num_experts` to `block_fp8_grouped_mm`. A count of
   only the local experts makes the mean too high by the number of ranks.
   A tile has 64 token rows up to 64 mean rows, 128 up to 192, and 256
   above.
8. `block_fp8_grouped_mm` calls it above 256 rows on sm90, with Triton 3.8
   (any 3.8.x), because Gluon changes between releases. The rows must be
   bf16, rows and weights contiguous and 16-byte aligned, and `group_ends`
   int32. The scale block must be a multiple of the tile, K a multiple of
   16, and the number of K steps even. Otherwise the Triton kernel runs.
9. Gluon does not run under Triton's interpreter. The compile tests build
   every default sm90 tile on the CPU, the same binary that runs on the
   GPU. They fail on register spills and on a register write while a
   `wgmma` still reads it. The SASS check (`tests/unit/_wgmma_sass.py`)
   finds such writes in the machine code. On an H100, the tests compare the
   kernel with fp32 math and the Triton kernel, in a CUDA graph and under
   Triton's concurrency sanitizer.

## Alternatives Considered

- The same schedule in Triton, the expert-tile split of ADR-037. Triton
  still waits for each register-operand `wgmma`, so from 1024 tokens Gluon
  stays 1.16 to 1.30 times faster on the H100
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)).
- Dequantize into shared memory. That costs shared-memory bandwidth on
  every K step.
- A `tcgen05` version for Blackwell. It is a different kernel
  ([ADR-039](ADR-039-gluon-fp8-grouped-blackwell.md)).

## Consequences

- The GLM-5.3 layer on one H100, without a CUDA graph, in ms per forward
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)):

  | Tokens | Triton row tiles | Gluon | Speedup | bf16 `grouped_mm` |
  |---:|---:|---:|---:|---:|
  | 1024 | 8.07 | 5.39 | 1.50x | 8.21 |
  | 4096 | 16.16 | 10.03 | 1.61x | 12.62 |
  | 16384 | 42.38 | 30.05 | 1.41x | 30.29 |

  The FP8 layer now beats or equals bf16 with half the expert memory.
- At 4096 tokens the kernel takes 7.0 ms of the layer's 10.2 ms GPU time,
  against 13.1 ms before.
- The two kernels give bit-identical outputs. Warp specialization makes the
  layer 3.5 to 8.5 percent faster.
- The first warp-specialized version gave wrong results that varied by run.
  The next K step wrote registers while the previous `wgmma` still read
  them. ptxas gives no warning, and the sanitizer does not track registers.
  The SASS check found the fault without a GPU: 42 and 66 writes in the two
  faulty builds, none after the fix.
- The multiply warps run a fence (an instruction that orders memory
  accesses) before they free each stage. Triton's Gluon guide asks for one
  between a read of shared memory and a later TMA write into it.
- CI pins Triton 3.6, so the Gluon tests skip there. The compile tests run
  in the `mini-infer-triton-interp` Docker image, the rest on an H100.
