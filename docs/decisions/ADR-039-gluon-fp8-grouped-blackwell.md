# ADR-039: Gluon tcgen05 kernel for block-FP8 MoE experts on the B200

Date: 2026-10-22
Status: Accepted

## Context

[ADR-037](ADR-037-block-fp8-grouped-kernel.md) (update) runs block-FP8
experts in expert tiles. On the B200, the GLM-5.3 layer with these tiles was
1.20 and 1.40 times slower than the bf16 layer at 4096 and 16384 tokens.

On the B200 (sm100), the tensor-core instruction `tcgen05` reads its left
operand from shared memory or from tensor memory. Tensor memory is a 128-row
memory next to the tensor cores. The sm100 code of Triton puts the
dequantized weights into one tensor-memory buffer, so each dequantization
waits for the previous multiply. It also loads token rows without TMA (the
copy engine into shared memory), and all warps do the same work. The Hopper
kernel of [ADR-038](ADR-038-gluon-fp8-grouped-hopper.md) needs `wgmma`,
which sm100 does not have.

## Decision

1. A Gluon kernel for sm100 (`models/blocks/moe/fp8_grouped_gluon.py`) keeps the
   numerics of ADR-038. Each e4m3 weight times its fp32 block scale is
   rounded to bf16, and the multiply runs in bf16 into an fp32 accumulator.
2. It keeps the persistent schedule of ADR-038: one program per SM walks
   items of one expert, one token-row tile and one output-column tile.
3. The warps have separate jobs. One copy warp loads tiles with TMA, four
   dequantization warps fill four bf16 buffers in tensor memory, and one
   MMA (matrix multiply) warp issues `tcgen05_mma`. Four store warps write
   the output with 16-byte stores.
4. Two 128-column accumulators let the store of one tile overlap the next
   multiply. With four 64-column weight buffers, they fill the 512 columns
   of tensor memory.
5. `block_fp8_grouped_mm` calls the kernel above 64 mean rows per expert on
   compute capability 10.x (B200, B300), which the new
   `is_datacenter_blackwell_device` detects. Up to 64, the Triton expert
   tiles are as fast and stay the default. The mean counts the experts on
   all ranks, and the conditions of ADR-038 apply, except the even number of
   K steps. The tile has 128 weight rows, 128 token rows, K steps of 128 and
   3 stages (4 since the update below).
6. The compile tests build each sm100 tile on the CPU through the launcher,
   so they build the binary that runs. They fail on register spills and on
   code without a `tcgen05` multiply or a tensor-memory store. On a B200,
   the tests of ADR-038 compare the kernel with fp32 math and with Triton.
   They also run it in a CUDA graph and under the concurrency sanitizer of
   Triton.

## Alternatives Considered

- Dequantize into shared memory. Each step then writes and reads the bf16
  tile there, which costs shared-memory bandwidth.
- One accumulator with 256 token rows. It was 1.5 percent faster at 16384
  tokens, but about 9 percent slower at 4096 tokens. The store of a tile
  then stalls the next multiply.
- The 2-CTA form of `tcgen05`, where two SMs share one 256-row multiply. It
  needs a cluster and TMA multicast, which is more code.
- On by default for every size. From 64 to 2048 tokens the Gluon kernel was
  0.96 to 1.03 times as fast as the Triton expert tiles.

## Consequences

- Without a CUDA graph
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)), one forward
  took 8.27 ms with the Triton expert tiles and 6.58 ms with Gluon at 4096
  tokens (1.26x). It took 14.33 and 11.76 ms at 8192 tokens (1.22x), and
  26.55 and 21.11 ms at 16384 tokens (1.26x).
- Against bf16 `grouped_mm`, the FP8 layer is now 1.04 times faster at 4096
  tokens and 1.15 times slower at 16384 tokens.
- Alone, each projection at 16384 tokens reaches 759 to 769 useful TFLOPS,
  against 521 to 559 with Triton.
- The outputs of the two kernels are bit-identical at every measured size.
- Before the MMA warp gets a weight tile, the dequantization warps run a
  fence (an instruction that orders memory accesses). The Gluon guide of
  Triton asks for it. It cost the down projection 3 percent at 16384 tokens.
- `tcgen05` reads no registers, so the register fault of ADR-038 cannot
  occur.
- Three FP8 grouped kernels need maintenance.

## Update 2026-10-23: the profile and a fourth stage

Nsight Compute on one B200 at 16384 tokens
([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)) showed that the
issue rate of the dequantization, our first guess, was not the limit. The
tensor cores were busy in 49 percent of the cycles, against 96 percent in
bf16 `grouped_mm`. The copy warp waited for a free stage in about 90
percent of its time, so 3 stages were too few.

Cycles per projection at 16384 tokens (independent of the clock):

| stages | gate-up cycles | tensor cores busy | down cycles |
|---|---:|---:|---:|
| 3 (before) | 12.22 M | 50.4% | 6.39 M |
| 4 | 10.53 M | 58.4% | 5.59 M |

A fourth stage uses 196 KB of shared memory and saves 12 to 14 percent of
the cycles. Split rings (separate rings and copy warps for weights and
token rows) save less than 1 percent more, for much more code. So the
default sm100 tile now has 4 stages.

- With more tensor-core work, the clock fell from 1.66 to about 1.50 GHz to
  keep the power limit. So 12 to 14 percent fewer cycles gave only 4 to 5
  percent less time per projection.
- With the new tile (`--fp8-gluon` run), the layer took 6.60, 11.02 and
  20.66 ms at 4096, 8192 and 16384 tokens. That is 1.25, 1.30 and 1.29
  times faster than the Triton expert tiles.
- Against bf16, the FP8 layer is 1.05 times faster at 4096 tokens and 1.08
  times slower at 16384 tokens (1.15 before).
- bf16 `grouped_mm` uses the 2-CTA form and keeps the tensor cores busy in
  97 percent of the cycles, at 1.26 GHz. Both kernels multiply in bf16, so
  the power limit caps the FP8 layer near that speed. The 2-CTA form stays
  the next option.
- All 87 GPU tests passed with the new tile, and outputs stayed
  bit-identical.
