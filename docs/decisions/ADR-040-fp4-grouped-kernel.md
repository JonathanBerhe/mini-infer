# ADR-040: Triton grouped kernel for FP4 MoE experts

Date: 2026-10-23
Status: Accepted

## Context

DeepSeek-V4-Flash keeps its routed experts in FP4. A weight is an e2m1
value (a sign bit, 2 exponent bits and 1 mantissa bit), packed two per
byte. Each row has one scale per 32 columns, an e8m0 power of two in the
checkpoint. `HashRoutedMoEFFN` holds the experts as `FP4Expert`s, which
stay packed in GPU memory and keep their scales in fp32.

After [ADR-036](ADR-036-grouped-expert-matmuls.md) and
[ADR-037](ADR-037-block-fp8-grouped-kernel.md), bf16 and block-FP8 experts
run as grouped matmuls. The V4 block kept its per-expert loop for both
expert types. For each expert that got tokens, the loop dequantizes the
three weights to bf16 and calls `F.linear`. One V4-Flash MoE layer (256
experts, top-6) took 9.5 to 273 ms per forward on an H100 and a B200. At
2048 tokens it ran about 14,370 GPU operations and 1024 host syncs per
forward, so it does not fit in a CUDA graph.

`torch.nn.functional.scaled_grouped_mm` takes MXFP4 weights only with
MXFP4 activations, and only on Blackwell.

## Decision

1. **A W4A16 grouped kernel in Triton** (`models/blocks/fp4_grouped_kernel.py`).
   It has the structure of the FP8 kernel of ADR-037. Activations stay
   bf16, the weights are the left operand of the dot (`out.T = W @ rows.T`),
   and the accumulator is fp32. Up to 256 (token, expert) rows, a program
   takes one segment of one expert. Above 256 rows, a program takes one
   tile of one expert's rows (the update of ADR-037). Row tiles that span
   experts stay as a bench option. Nothing reads back to the host.
2. **A decode with few instructions.** For each K step, a program loads
   the packed bytes and one fp32 scale per row and 32 columns. It moves
   the 3 low bits of each e2m1 code to the low end of the fp32 exponent
   field and the top of the mantissa. The sign bit goes to the fp32 sign.
   That fp32 equals the e2m1 value times 2^-126, for small codes too, as
   fp32 subnormals. The program multiplies by 2^126, which is exact, and
   by the scale. Then it rounds to bf16 once, as
   `dequantize_nvfp4_to_bf16` does. The fp32 multiplies of the generated
   code have no flush-to-zero, so the subnormals stay exact.
3. **Same routing code.** `grouped_expert_mlp` takes a `BlockFp4Weight`
   (stacked packed weights, their scales, the scale block) in place of a
   bf16 tensor. A new `out_dtype` argument lets the V4 block keep the
   expert sum in fp32 until after the all-reduce, as its loop does.
4. **`PackedFp4Experts`** stacks `w1` over `w3` into `(E, 2I, H/2)` bytes
   with `(E, 2I, H/32)` scales. It stacks `w2` into `(E, H, I/2)` with
   `(E, H, I/32)` scales. A scale covers one row, so `w1` and `w3` always
   stack. The experts' buffers become slices of the stacks, so the pack
   needs no extra memory. As for FP8, the pack tests the storage of the
   first and last expert's buffers and stacks again after a reload.
5. **When it runs.** `HashRoutedMoEFFN` takes the grouped path when the
   grouped path is on (Hopper or newer). bf16 experts then use
   `PackedSwigluExperts`. FP4 experts use the new kernel when the
   activations are bf16 or fp16 and Triton runs on the device. The
   reference's FP8 GEMM-input mode keeps the loop, because it quantizes
   each expert's inputs. The loop stays the reference in every case.
6. **Tiles for each GPU family.** Three tiers by row count, as in ADR-037.
   Sweeps of the V4-Flash layer on an H100 and a B200 picked them. The two
   GPUs share the tiles above 256 rows.
7. **Tested without a GPU.** `tests/unit/test_fp4_grouped_kernel.py` runs
   on CUDA or under Triton's interpreter, against
   `dequantize_nvfp4_to_bf16`. It covers the three splits, off-rank rows
   and power-of-two scales. It also covers a scale block wider than the
   tile, and the V4 block on both paths with both routing modes.

## Alternatives Considered

- **The first decode.** It computed each bf16 value with a compare and
  select for the small codes, and it loaded the scale once per weight.
  The kernel then took 64 scale loads per thread and K step, and it
  spilled registers. The layer was 3 to 10 times slower than the bf16
  layer. The decode of decision 2 makes the layer 4.7 to 13 times faster
  in a CUDA graph from 1 to 64 tokens. From 256 tokens it makes it 6 to
  12.5 times faster.
- **Dequantize the hit experts into a bf16 stack, then `grouped_mm`.** The
  list of hit experts lives on the GPU, so reading it needs a host sync.
  All 256 experts in bf16 take 12.9 GB per layer.
- **`scaled_grouped_mm` with MXFP4.** It needs FP4 activations, so its
  results do not match the loop. It runs only on Blackwell.
- **`tl.dot_scaled`**, Triton's dot for MX formats. It reads e8m0 scales
  as bytes. `FP4Expert` keeps fp32 scales, so this needs a second copy of
  the scales (about 8.7 GB for the 43 layers) or a new stored format. Not
  tried.
- **Gluon.** As for FP8, it means two kernels to own, and Gluon does not
  run under Triton's interpreter. The FP4 layer is already faster than the
  bf16 layer up to 1024 tokens on both GPUs.

## Consequences

- Measured on an H100 and a B200 with the V4-Flash layer and random
  weights ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md),
  section "FP4 experts"). Against the loop, the grouped layer is 20 to 66
  times faster up to 1024 tokens on the H100. On the B200 it is 28 to 50
  times faster. At 16384 tokens it is 10.7 and 7.5 times faster.
- The grouped layer runs in a CUDA graph. At 2048 tokens it runs 84 or 85
  GPU operations and no host syncs per forward.
- The FP4 experts take 0.625 bytes per weight with their fp32 scales,
  against 2 bytes in bf16. We compared it with the same layer with bf16
  experts on `grouped_mm`. In a CUDA graph from 1 to 64 tokens, the FP4
  layer is 1.02 to 1.57 times faster. It is 1.15 to 1.19 times faster at 256 and 1024
  tokens. At 16384 tokens it is 1.28 (H100) and 1.55 (B200) times slower.
- The decode sets the speed. The kernel decodes each weight tile once per
  tile of rows. With tiles of 128 rows, the B200 layer takes 3.8 ms at 256
  tokens and 5.4 ms at 4096 tokens. The second has 16 times the rows. At
  16384 tokens an expert has about 3 tiles of rows, so each
  weight tile decodes 3 times.
- The outputs differ from the loop by a relative 2.6e-3 to 2.8e-3. Both
  paths dequantize to the same bf16 weights and add the experts in fp32.
  The difference comes from the order of the sums in the matmuls.
- No real V4-Flash checkpoint was measured with this path. It needs two
  B200 GPUs.
