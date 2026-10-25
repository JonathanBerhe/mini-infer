# ADR-041: Gluon kernels for FP4 MoE experts on Hopper and Blackwell

Date: 2026-10-23
Status: Accepted

## Context

[ADR-040](ADR-040-fp4-grouped-kernel.md) runs the FP4 experts of
DeepSeek-V4-Flash through a Triton kernel. At 16384 tokens it was still
slower than bf16 experts on `grouped_mm`. A profile
([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md), section "Gluon
kernels for FP4") showed why. The tensor cores were busy 23 (B200) and 38
(H100) percent of the time, against 97 and 85 percent for `grouped_mm`.
The Triton code decodes a weight tile, then multiplies it. One program
fills an SM, so the two steps never overlap. ADR-038 and ADR-039 solved
this for block-FP8 with Gluon, where some warps copy, some dequantize and
some multiply.

## Decision

1. Gluon kernels for FP4 (`models/blocks/fp4_grouped_gluon.py`) for sm90
   and sm100, built on parts of `fp8_grouped_gluon.py`.
   `block_fp4_grouped_mm` calls them (decision 6).
2. TMA copies the packed bytes (two e2m1 values each). Each fp32 scale
   loads once.
3. `gl.fp4_to_fp` unpacks the bytes to bf16 in blocked layouts.
4. On sm100, up to two dequantization warpgroups write bf16 tiles into
   tensor memory in turns, so the MMA warp does not wait.
5. On sm90, the multiply warps decode into one of two shared-memory
   buffers. An asynchronous `wgmma` reads one buffer while they fill the
   other.
6. The kernels run above 256 rows. On sm90 they run only above 32 mean
   rows per expert, over the experts of all ranks.
7. `tests/unit/test_fp4_grouped_gluon.py` compiles the default tiles
   without spills, with no GPU. On a GPU, it compares the kernel with fp32
   math and the Triton kernel, also in a CUDA graph and under Triton's
   concurrency sanitizer.

## Alternatives Considered

- A better Triton tile. It made the B200 kernels 1.27 times faster, but the
  work still does not overlap.
- Two Triton programs per SM. On sm90 the limit of 128 registers spills.
- The `wgmma` register operand on sm90 (the first version of decision 5).
  Byte loads held it at 61 to 64 percent tensor-core use.
- Dequantize every expert to bf16, then `grouped_mm`. It gives no gain on
  the H100 and needs a temporary of 6 to 13 GB per layer.
- FP8 activations. For FP8 experts, W8A8 had 8 times the error of W8A16
  (ADR-037).

## Consequences

- The outputs are bit-identical to the Triton kernel of ADR-040.
- At 16384 tokens the layer takes 20.74 ms on the H100 and 13.65 ms on the
  B200. Triton took 25.78 and 18.39 ms, and the bf16 layer 19.34 and
  12.04 ms. So FP4 is 1.07 (H100) and 1.13 (B200) times slower than bf16,
  against 1.32 and 1.57 times before.
- At 2048 and 4096 tokens the FP4 layer is faster than bf16, up to 1.24
  times on the B200. At 8192 tokens it is 1.21 (B200) and 1.26 (H100)
  times slower.
- At 16384 tokens the work is compute-bound, and the FP4 layer also
  decodes. So FP4 experts can approach the bf16 layer at large batches but
  not beat it. They save memory traffic, which wins at small batches.
- Gluon needs Triton 3.8. Other Triton versions use the kernel of ADR-040.
