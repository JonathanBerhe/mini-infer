# ADR-037: Triton grouped kernel for block-FP8 MoE experts

Date: 2026-10-21
Status: Accepted

## Context

The GLM-5 FP8 and MiniMax-M3 checkpoints keep their routed experts in e4m3,
with one fp32 scale per 128x128 block. `GlmMoeFFN` holds them as
`Fp8Expert`s, so that they fit in GPU memory
([ADR-021](ADR-021-glm-moe-dsa-port.md)). After
[ADR-036](ADR-036-grouped-expert-matmuls.md), bf16 experts run as grouped
matmuls, but `Fp8Expert`s kept the per-expert loop. That loop is slower
than the bf16 one: each call dequantizes the expert's three weights through
full-size fp32 temporaries. One GLM-5.3 MoE layer with FP8 experts took 9
to 190 ms per forward on an H100 and a B200.

`torch.nn.functional.scaled_grouped_mm` takes FP8 inputs only with
row-wise scales, or with the 1x32 and 1x16 block scales of MXFP8, MXFP4 and
NVFP4 (the dispatch table in PyTorch's `GroupedBlas.cpp`). It does not take
128x128 weight blocks, and it needs quantized activations.

## Decision

1. **A W8A16 grouped kernel in Triton** (`models/blocks/fp8_grouped_kernel.py`).
   Activations stay bf16. Each weight tile is loaded as e4m3, multiplied by
   its block scale in fp32, rounded to bf16 (the rounding of
   `Fp8Expert.forward`), and multiplied on the bf16 tensor cores with an
   fp32 accumulator. Nothing reads back to the host, so the kernel runs in a
   CUDA graph. Rows routed to another rank give zeros.
2. **Two ways to split the work.** Up to 256 (token, expert) rows, the grid
   has one program per row and per column tile. A program runs only where
   an expert's segment starts (the first row of its group, then every
   `BLOCK_M` rows), so every expert of a decode batch loads in parallel.
   Above 256 rows, one program computes a row tile and walks the experts of
   its rows in order, masking the rows of the others. A grid with one
   program per row would be mostly idle there.
3. **Weights as the left operand.** The kernel computes `out.T = W @ rows.T`.
   Hopper's `wgmma` takes its left operand from registers, where the
   dequantized tile already is, and its right operand from shared memory.
4. **Same routing code as bf16.** `grouped_expert_mlp` takes a
   `BlockFp8Weight` (stacked e4m3 weights, their scales, the block size) in
   place of a bf16 tensor and calls the kernel for it.
5. **`PackedFp8Experts`** stacks the experts as `PackedSwigluExperts` does:
   `w1` over `w3` into one `(E, 2I, in)` tensor with their scales, `w2` into
   `(E, out, I)`, and the experts' buffers become slices of the stacks. `w1`
   and `w3` share scale blocks only when `I` is a multiple of 128, so other
   widths keep the loop. GLM-5.3 (2048) qualifies. The pack tests the
   storage of the first and last expert's weights and scales, because the
   meta-device loader registers new buffers instead of copying into them.
6. **When it runs.** `GlmMoeFFN` takes the path when the grouped path is on
   (Hopper or newer), Triton is installed, and the activations are bf16 or
   fp16. In every other case it keeps the loop, which stays the reference.
7. **Tiles for each GPU family.** Three tiers by row count (up to 256, up to
   8192, above) and one table for Hopper and one for Blackwell, picked with
   `is_blackwell_device`. The two GPUs agree only on the largest tier.
   Sweeps of 14 tiles on an H100 and a B200 picked them (see the benchmark).
8. **Tested without a GPU.** `tests/unit/test_fp8_grouped_kernel.py` runs on
   CUDA or under Triton's interpreter (the `mini-infer-triton-interp` Docker
   image), with both splits and both operand orders. Two interpreter faults
   shape the test. Its bf16 `tl.dot` gives wrong results, so under the
   interpreter the kernel multiplies in fp32 (bf16 products are exact in
   fp32). It also truncates fp32 to bf16, so the CPU tolerance allows one
   more bf16 step than the GPU one.

## Alternatives Considered

- **`scaled_grouped_mm` with row-wise or MXFP8 scales.** The weights would
  be quantized a second time, the activations too, and MXFP8 runs only on
  Blackwell. The results would no longer match the loop.
- **Split-K for decode.** Several programs would share one output tile and
  add their partial sums with atomics or a second pass. The per-segment
  split already fills the GPU at decode sizes, without either.
- **W8A8 with per-token, per-128 activation scales** (how DeepSeek, vLLM
  and transformers run these checkpoints). Tried and removed. A W8A8 mode
  of this kernel quantized the activations in PyTorch, multiplied e4m3 by
  e4m3 on the FP8 tensor cores, and applied both scales to each block's
  fp32 partial sum. From 4096 tokens it was slower than W8A16 on both GPUs,
  even with its own best tiles: at 16384 tokens 55 ms, against 43 ms (H100)
  and 28 ms (B200). Its error against fp32 math was 8 times larger (4.4e-2
  against 5.4e-3). The scaling of each partial sum needs a second fp32 tile
  in registers and does not fuse into the tensor-core accumulator. DeepGEMM
  hides that cost with warp specialization, which needs Gluon or CuTe DSL.
- **Dequantize the hit experts into a bf16 stack, then `grouped_mm`.** The
  list of hit experts lives on the GPU: reading it needs a host sync, and
  dequantizing all experts costs 32 times the memory traffic at decode.
- **Gluon or CuTe DSL.** TMA and warp specialization would close the
  remaining gap to the CUTLASS bf16 kernel at large batches (see below).
  That is two kernels to own (Hopper `wgmma`, Blackwell `tcgen05`), and
  Gluon does not run under Triton's interpreter, so every debugging step
  needs a GPU. It waits for a measured need.
  [ADR-038](ADR-038-gluon-fp8-grouped-hopper.md) adds a Gluon kernel for
  the H100 above 256 rows.

## Consequences

- Measured on an H100 and a B200 with the GLM-5.3 layer
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)). The first
  version of the kernel (one split, weights on the right, one tile table)
  made the layer 13 to 30 times faster than the loop up to 1024 tokens and
  2.7 to 3.2 times faster at 16384 tokens, and it runs in a CUDA graph,
  which the loop does not.
- The second version (decisions 2, 3 and 7) is another 1.1 to 2.4 times
  faster in a CUDA graph from 1 to 64 tokens, and 1.05 to 1.39 times faster
  from 256 to 16384 tokens. At one token in a CUDA graph the layer takes
  0.28 ms on the H100 and 0.24 ms on the B200. The decode gain comes from
  the per-segment split. The operand order saves 1 to 12 percent of the
  time for larger batches.
- At 2048 tokens the layer went from about 8400 GPU operations and 256
  host syncs to 63 to 65 operations and none.
- The FP8 experts keep half the memory of bf16 experts. Against the same
  layer with bf16 experts on `grouped_mm`, the FP8 layer is 1.05 to 2.0
  times faster up to 256 tokens and about equal at 1024. From 4096 tokens
  it is 1.26 to 1.46 times slower: the kernel dequantizes on the CUDA cores
  and multiplies in bf16, without TMA or warp specialization. On the H100,
  ADR-038 closes this gap.
- The outputs differ from the loop by a relative 3.8e-3, because the loop
  adds the 8 expert outputs into a bf16 tensor one at a time. bf16 experts
  show the same effect (3.3e-3). Against fp32 math with the same routing,
  the routed-expert output has a relative error of 5.4e-3 to 5.5e-3.
- No real FP8 checkpoint was measured end to end. The GLM-5.x and
  MiniMax-M3 FP8 checkpoints need 4 to 8 GPUs.

## Update 2026-10-22: tiles of one expert each

The per-row-tile split of decision 2 gives a program rows of several
experts, and the program runs the whole K loop once for each of them. The
default split for more than 256 rows is now per expert tile: the rows of
each expert are cut into tiles of `BLOCK_M`, and each program takes one
tile of one expert. A few small GPU operations count the tiles of each
expert, and a binary search over those counts finds a program's expert.
Nothing reads back to the host. The row tiles stay as a bench option.

Measured with the GLM-5.3 layer in one process per GPU
([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)):

- On the B200, the expert tiles are 1.08 to 1.42 times faster than the row
  tiles from 64 to 16384 tokens. The FP8 layer is now faster than the bf16
  layer at 1024 tokens (4.02 against 5.12 ms), and still 1.20 and 1.40
  times slower at 4096 and 16384 tokens.
- On the H100, the expert tiles are 1.06 to 1.31 times faster than the row
  tiles. There the Gluon kernel of ADR-038 takes many rows when it can, and
  the expert tiles run for the other shapes. On the B200 the Gluon kernel
  of [ADR-039](ADR-039-gluon-fp8-grouped-blackwell.md) takes over above 64
  mean rows per expert.
- The outputs of the two splits are bit-identical.
- The Blackwell tile for 257 to 8192 rows changes from (64, 128, 64, 4, 4)
  to (64, 256, 64, 8, 3), the fastest expert tile at 1024 tokens in the
  sweep.
