# ADR-038: Gluon grouped kernel for block-FP8 MoE experts on Hopper

Date: 2026-10-22
Status: Accepted

## Context

[ADR-037](ADR-037-block-fp8-grouped-kernel.md) runs block-FP8 experts
(e4m3 weights, one fp32 scale per 128x128 block) through a Triton grouped
kernel. Up to 256 tokens it is faster than the bf16 experts on
`grouped_mm`. From 4096 tokens it was 1.26 to 1.39 times slower on the
H100.

A program of the kernel walks the shared dimension K of the matmul in a
loop, the K loop. Each pass of the loop, a K step, handles one tile of K.
Two things made the Triton kernel slow at large batches:

- Its per-row-tile split gives each program a tile of sorted rows, and
  these rows belong to several experts. So the program runs the whole K
  loop once for each expert in the tile and masks the other rows. At 4096
  tokens (128 rows per expert, 128-row tiles), a tile spans about two
  experts. The kernel took 13.1 ms of the layer's 16.1 ms GPU time.
- Triton's compiler waits for each `wgmma` (Hopper's asynchronous
  tensor-core instruction) whose left operand is in registers before the
  next K step starts. So the dequantization of the next weight tile does
  not overlap with the tensor cores.

Gluon is the lower-level Triton dialect. It gives direct access to TMA,
mbarriers, `wgmma` and warp specialization. TMA is the copy engine between
global and shared memory. An mbarrier is a barrier in shared memory that
waits for threads and copies. A warp is a group of 32 threads that run
together. With warp specialization, different warps do different jobs.

## Decision

1. **A Gluon kernel for sm90** (`models/blocks/fp8_grouped_gluon.py`).
   sm90 is the Hopper architecture of the H100. The kernel keeps the
   numerics of ADR-037. It multiplies each e4m3 weight by its fp32 block
   scale and rounds the product to bf16. The tensor cores then multiply in
   bf16 into an fp32 accumulator.
2. **Work items follow the experts.** An item is one expert, one tile of
   the token rows of that expert, and one tile of output columns. A tile
   never mixes rows of two experts. With expert parallelism, each rank (one
   GPU process) holds some of the experts. Rows routed to another rank get
   items that only store zeros.
3. **Persistent grid.** The grid has one program per SM (one of the GPU's
   compute units). Each program walks the items with a stride and finds the
   expert of each item from `group_ends` on the GPU. Nothing reads back to
   the host, so the kernel runs in a CUDA graph. Within an expert, the
   items go by column tile first, then by tile of token rows. So programs
   that run at the same time read the same weight tile from L2 (the cache
   that all SMs share).
4. **TMA ring and a register operand.** TMA copies the e4m3 weight tile and
   the bf16 tile of token rows into a ring of stages. A stage is a buffer
   in shared memory, and the kernel uses the stages in turn. The kernel
   dequantizes the weight tile in registers. That tile is the left operand
   of an asynchronous `wgmma` (`acc[cols, rows] = W @ rows.T`, as in
   ADR-037). The dequantization of one K step runs while the tensor cores
   multiply the previous one.
5. **Two K steps per loop iteration.** A `wgmma` reads its register operand
   while it runs, so the kernel must not write these registers before the
   `wgmma` is done. The loop alternates two register sets. Each set stays
   live until the kernel waits for its `wgmma` to finish. So the kernel
   needs an even number of K steps.
6. **Warp specialization.** One extra warp, the copy warp, issues the TMA
   copies. The other warps, the multiply warps, dequantize and multiply. So
   the copies for the next item start while the multiply warps store the
   finished tile. The version without warp specialization stays in the
   module for comparison.
7. **Tiles by mean rows per expert.** The mean is the rows divided by the
   number of experts on all ranks. `block_fp8_grouped_mm` takes that number
   as `num_experts` (by default the experts in `weight`), and
   `grouped_expert_mlp` passes it. Under expert parallelism, a count of only
   the local experts makes the mean too high by the number of ranks. Every
   tile has 128 weight rows and K steps of 64. It has 64 token rows up to
   64 mean rows, 128 up to 192, and 256 above. It has 4 multiply warps with
   64 token rows and 8 with more.
8. **When it runs.** `block_fp8_grouped_mm` calls it above 256 rows on an
   sm90 GPU. Up to 256 rows the per-segment Triton path stays. The Gluon
   kernels need Triton 3.8 (any 3.8.x release), because Gluon changes
   between releases. A newer Triton keeps the Triton kernel until we test
   the Gluon kernels with it. The B200 has its own Gluon kernel
   ([ADR-039](ADR-039-gluon-fp8-grouped-blackwell.md)).

   The rows must be bf16. The rows and the weights must be contiguous and
   16-byte aligned, because TMA needs aligned base addresses. `group_ends`
   must be int32. The scale block must be a multiple of the tile, K a
   multiple of 16, and the number of K steps even. GLM-5 and MiniMax-M3
   meet these conditions. In every other case the Triton kernel runs.
9. **Tests without a GPU where possible.** Gluon does not run under
   Triton's interpreter. The compile tests build every default sm90 tile on
   the CPU through the launcher's argument binding. So they compile the
   same binary that runs on the GPU. They fail on register spills, which
   move values from registers to slower memory. They also fail on any
   instruction that writes a register while a `wgmma` still reads it.

   SASS is the machine code of the GPU. The SASS check
   (`tests/unit/_wgmma_sass.py`) finds these writes. It follows `BRA.U`
   branches back to the loop start and counts every register of a
   multi-register write. It also scans the end of the K loop up to the last
   `wgmma` wait.

   On an H100, the tests compare the kernel with fp32 math and with the
   Triton kernel. They also capture it in a CUDA graph and run it under
   Triton's concurrency sanitizer, which stops on a shared-memory race or a
   deadlock. One shape has more work items than SMs. So each program walks
   several items, and the rings of mbarriers wrap around, also under the
   sanitizer.

## Alternatives Considered

- **The same schedule in Triton.** Expert-aligned work items remove the
  repeated K loops, but Triton's compiler still waits for each
  register-operand `wgmma`. We measured this schedule later as the
  expert-tile split of the Triton kernel (ADR-037, update). On the H100 it
  gives 60 to 70 percent of the Gluon gain at 1024 to 4096 tokens and 23
  percent at 16384 tokens. From 1024 tokens the Gluon kernel stays 1.16 to
  1.30 times faster than it in the
  [`--fp8-expert-tiles` run](../benchmarks/data/moe-grouped-fp8-expert-tiles-h100.txt).
  A [later `--fp8-gluon` run](../benchmarks/data/moe-grouped-fp8-gluon-h100-expert-tiles.txt)
  gives 1.17 to 1.33.
- **Wait for each `wgmma`, as Triton does.** It is safe without the
  two-step loop, but the dequantization then runs while the tensor cores
  are idle.
- **Dequantize into shared memory.** Both `wgmma` operands then come from
  shared memory, and no register is at risk. Each K step then also writes
  the bf16 tile to shared memory and reads it back. That costs
  shared-memory bandwidth, which the register operand does not use.
- **A Blackwell (`tcgen05`) version.** `tcgen05` is the tensor-core
  instruction of Blackwell GPUs. A kernel for it uses tensor memory, a
  memory next to the tensor cores. It is a different kernel, and this
  change does not cover it.
  [ADR-039](ADR-039-gluon-fp8-grouped-blackwell.md) adds it for the B200.

## Consequences

- We measured the GLM-5.3 layer on one H100 without a CUDA graph
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)). The Triton
  kernel used its row tiles. One forward took 8.07 ms with the Triton
  kernel and 5.39 ms with the Gluon kernel at 1024 tokens (1.50x). It took
  16.16 and 10.03 ms at 4096 tokens (1.61x), and 42.38 and 30.05 ms at
  16384 tokens (1.41x). At 64 and 256 tokens the Gluon kernel is 1.07 and
  1.06 times faster.
- Against the same layer with bf16 experts on `grouped_mm`, the FP8 layer
  is now 1.52 times faster at 1024 tokens (5.39 against 8.21 ms). It is
  1.26 times faster at 4096 tokens (10.03 against 12.62 ms) and equal at
  16384 tokens (30.05 against 30.29 ms). It keeps half the expert memory.
- At 4096 tokens the kernel takes 7.0 ms of the layer's 10.2 ms GPU time,
  against 13.1 ms before. Alone, the gate-up projection at 16384 tokens
  reaches 495 useful TFLOPS, against 314 with the Triton kernel.
- The outputs of the two kernels are bit-identical at every measured size.
- Warp specialization makes the layer 3.5 to 8.5 percent faster than the
  version without it, at every measured size.
- The first warp-specialized version gave wrong results that changed from
  run to run. The next K step wrote registers while the previous `wgmma`
  still read them. ptxas (the NVIDIA assembler that makes SASS) gives no
  warning about it. The concurrency sanitizer does not track registers.
  The SASS check of decision 9 found the fault without a GPU. It reported
  42 and 66 writes in the two faulty builds, and none after the fix.
- The multiply warps run a fence (an instruction that orders memory
  accesses) before they free each stage, the last stage of an item too.
  Triton's Gluon guide asks for a fence between a read of shared memory and
  a later TMA write into it. A rerun with the fence passed all GPU tests,
  and the kernel alone was as fast as before
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)).
- There are two FP8 grouped kernels to maintain for the H100. CI pins
  Triton 3.6, so its Gluon tests skip there. The `mini-infer-triton-interp`
  Docker image runs the compile tests, and the H100 runs the rest.
