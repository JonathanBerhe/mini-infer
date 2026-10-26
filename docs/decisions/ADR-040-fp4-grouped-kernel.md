# ADR-040: Triton grouped kernel for FP4 MoE experts

Date: 2026-10-23
Status: Accepted

## Context

DeepSeek-V4-Flash keeps its routed experts in FP4: e2m1 values (1 sign, 2
exponent, 1 mantissa bit), packed two per byte. Each row has one e8m0 scale
per 32 columns. `FP4Expert` keeps them packed in GPU memory, with fp32
scales.

After [ADR-036](ADR-036-grouped-expert-matmuls.md) and
[ADR-037](ADR-037-block-fp8-grouped-kernel.md), bf16 and block-FP8 experts
run as grouped matmuls. The V4 block kept a per-expert loop that dequantizes
to bf16. One V4-Flash MoE layer took 9.5 to 273 ms per forward on
an H100 and a B200. At 2048 tokens it ran 1024 host syncs, so it does not fit
in a CUDA graph.

## Decision

1. A W4A16 grouped kernel in Triton (`models/blocks/moe/fp4_grouped_kernel.py`),
   shaped like the FP8 kernel of ADR-037. Activations stay bf16, the weights
   are the left operand of the dot (`out.T = W @ rows.T`), and the
   accumulator is fp32. Up to 256 (token, expert) rows, a program takes
   one segment of one expert. Above 256 rows, it takes one tile of one
   expert's rows. Nothing reads back to the host.
2. A decode with few instructions. A program loads one fp32 scale per row and
   32 columns. Bit moves turn each e2m1 code into an fp32 that equals its
   value times 2^-126. It multiplies by 2^126 and the scale, then rounds to
   bf16 once, as `dequantize_nvfp4_to_bf16` does.
3. `grouped_expert_mlp` takes a `BlockFp4Weight` and a new `out_dtype`
   argument.
4. `PackedFp4Experts` stacks the weights and scales with no extra memory: the
   expert buffers become slices of the stacks.
5. FP4 experts use the kernel on Hopper or newer when the activations are
   bf16 or fp16. The loop stays the reference.
6. Three tile tiers by row count, tuned on an H100 and a B200.
7. `tests/unit/models/blocks/moe/test_fp4_grouped_kernel.py` runs on CUDA or under the Triton
   interpreter, against `dequantize_nvfp4_to_bf16`.

## Alternatives Considered

- The first decode loaded the scale once per weight and spilled registers.
  Its layer was 3 to 10 times slower than the bf16 layer. Decision 2 is 4.7
  to 13 times faster from 1 to 64 tokens.
- Dequantize the hit experts to bf16, then `grouped_mm`. This needs a host
  sync, and 256 bf16 experts take 12.9 GB per layer.
- `scaled_grouped_mm` with MXFP4. It needs FP4 activations and Blackwell, and
  its results differ from the loop.
- `tl.dot_scaled`. It reads e8m0 scales as bytes, so it needs a second copy
  of the scales (about 8.7 GB for the 43 layers). Not tried.
- Gluon. It does not run under the Triton interpreter.
  [ADR-041](ADR-041-gluon-fp4-grouped.md) adds Gluon kernels for many rows.

## Consequences

- Against the loop, the grouped layer is 20 to 66 times faster up to 1024
  tokens on the H100. The B200 gives 28 to 50 times
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md), section "FP4
  experts").
- It runs in a CUDA graph: 84 or 85 GPU operations and no host syncs per
  forward at 2048 tokens.
- FP4 experts take 0.625 bytes per weight with scales, against 2 in bf16.
  Against bf16 experts on `grouped_mm`, the FP4 layer is 1.02 to 1.57 times
  faster from 1 to 64 tokens. At 16384 tokens it is 1.28 (H100) and 1.55
  (B200) times slower, because each weight tile decodes once per tile of
  rows.
- Outputs differ from the loop by a relative 2.6e-3 to 2.8e-3, from the sum
  order.
- On the real V4-Flash with two B200 GPUs, prefill of 2021 tokens took 464
  ms, against 3287 ms with the loop. Greedy output matched for 20 tokens,
  then took a different fluent word.
