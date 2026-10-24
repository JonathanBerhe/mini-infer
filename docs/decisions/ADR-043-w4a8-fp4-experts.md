# ADR-043: W4A8 for the FP4 MoE experts on Blackwell

Date: 2026-10-24
Status: Accepted (on by default on sm100)

## Context

The batch-16 profile of the real DeepSeek-V4-Flash on 2x B200
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md#graph-step-at-batch-16))
showed that the FP4 grouped kernel of [ADR-040](ADR-040-fp4-grouped-kernel.md)
grows from 3.6 ms per step at batch 1 to 14.3 ms at batch 16. That is at
least 4.4 times slower than the limit that the memory bandwidth sets. The
kernel is the largest single item of a serving step.

Nsight Compute on a B200 gave the reason. At 16 tokens the gate-up kernel
keeps the instruction issue slots 71% busy, but reads DRAM at only 22% of
its peak. The kernel is bound by the instructions it issues, not by its
memory reads. The decode of ADR-040 turns each e2m1 weight into bf16 before
the `tl.dot`. In the compiled sm100 code, the K loop runs 7.9 to 9.3
instructions per weight, and most of them are the decode. At batch 1
the same kernel reaches only 10% occupancy, because a decode step gives it
few (token, expert) rows.

A probe on a B200 timed variants of the kernel on one layer's projections,
at the real shapes, on one rank of two. At 16 tokens, cheaper bf16 decodes
(an fp16 decode, other tiles, a split along K) gained 12% at most. Only one
variant removed the decode: `tl.dot_scaled`, which takes the packed e2m1
weights and their e8m0 scales directly. On Blackwell it runs on the tensor
cores as a native MX matmul. It needs the other operand in e4m3 (FP8) with
its own e8m0 scales per 32 columns.

| Projection | Tokens | bf16 inputs (ADR-040) | e4m3 inputs, `tl.dot_scaled` |
|---|---:|---:|---:|
| gate-up (N 4096, K 4096) | 1 | 43.6 µs | 23.0 µs |
| gate-up (N 4096, K 4096) | 16 | 230.6 µs | 135.3 µs |
| down (N 4096, K 2048) | 1 | 24.2 µs | 13.4 µs |
| down (N 4096, K 2048) | 16 | 132.9 µs | 70.9 µs |

The DeepSeek reference code runs the FP4 experts this way too. Its `linear`
quantizes the input to e4m3 with one power-of-two scale per 128 columns,
then calls its `fp4_gemm`. So the model was built to run with FP8 expert
inputs.

## Decision

1. **A W4A8 mode of the FP4 grouped kernel.** With the `W4A8` constexpr, the
   K loop loads the packed bytes, the e8m0 weight scales, the e4m3 inputs and
   their e8m0 scales, and calls
   `tl.dot_scaled(packed, w_scale, "e2m1", inputs, input_scale, "e4m3", acc)`.
   The weights stay the left operand, as in ADR-040. The grid, the two row
   splits and the output stay the same.
2. **Its own tiles.** The native Blackwell instruction needs BLOCK_N 128. On
   sm100 the tiles are `(16, 128, 256)`, `(64, 128, 256)` and
   `(128, 128, 256)` (BLOCK_M, N, K), 8 warps, 3 stages. Hopper has no MX
   tensor cores, so Triton emulates `tl.dot_scaled` there. Other GPUs get
   smaller tiles that compile for sm90 without register spills.
3. **A quantizer kernel.** `quantize_e4m3_rows` takes the bf16 rows of a
   projection and returns e4m3 values and one e8m0 scale per 32 columns. The
   scale is the smallest power of two that keeps the block maximum inside
   448, the e4m3 maximum, the same rounding as the reference's `ue8m0`
   scales. The block of 32 matches the weight scales and is finer than the
   reference's 128.
4. **e8m0 weight scales next to the fp32 ones.** `PackedFp4Experts` keeps a
   byte copy of each scale when the mode is on. The V4 checkpoints store
   e8m0 scales, so the copy is exact.
5. **On by default where it is native.** `DeepseekV4Config.fp4_fp8_activations`
   is None by default, which means auto. Auto turns the mode on where
   `supports_native_w4a8(device)` is true (sm100: B200, B300) and every
   expert scale is a power of two. Elsewhere, or with other scales, the
   experts keep bf16 inputs. True forces the mode and raises on a scale that
   is not a power of two. False keeps bf16 inputs. Only the grouped path
   changes: the per-expert loop keeps bf16 inputs. The pack checks the
   scales once, when it is built, so a forward does not read them back.

## Alternatives Considered

- **A cheaper bf16 decode.** An fp16 decode, other tiles and splits along K
  gained 12% at most at 16 tokens in the probe. The decode instructions
  stay.
- **Keep it opt-in.** The FP8 inputs change the numbers of every expert
  matmul. In the probe, the largest error of one projection against an fp32
  reference on the bf16 inputs grew from 0.3% to 2.6% of the largest
  output. But on the real model the passage perplexity and the greedy
  tokens did not get worse (below), so the default follows the speed.
- **On by default on every GPU.** Hopper emulates `tl.dot_scaled`, and the
  emulation costs more instructions per weight than the bf16 decode (below).
- **`torch.nn.functional.scaled_grouped_mm`.** It takes MXFP4 weights only
  with MXFP4 inputs, which lose more than e4m3 inputs, and it runs only on
  Blackwell.

## Consequences

Measured on 2x B200, real V4-Flash, both modes on one load
(`scripts/modal_v4_flash_generate.py --compare-fp8-activations`). The
passage is the first 2,048 tokens of `scripts/data/technical_passage.md`,
prefilled once per mode. A step time is the median of 16 CUDA-graph decode
steps after 8 warm-up steps. At batch 16, row `b` prefills its own slice of
the passage.

| Measure | bf16 inputs | e4m3 inputs (W4A8) |
|---|---:|---:|
| Passage perplexity | 7.925 | 7.845 |
| Passage top-1 accuracy | 52.9% | 53.5% |
| Graph step, batch 1 | 34.0 ms | 33.2 ms |
| Graph step, batch 16 | 51.7 ms | 45.8 ms |
| Tokens per second, batch 16 | 309 | 349 |

- The quality does not drop. The perplexity is 1.0% lower with e4m3 inputs.
  The two modes pick the same next token at 95.1% of the passage positions,
  and the cosine of their last logits is 0.9966. The 33 greedy tokens of
  "The capital of France is" are the same. One passage of 2,048 tokens
  cannot show whether the lower perplexity is a real gain or noise.
- At batch 16 the step is 5.9 ms faster (1.13x). In the batch-16 profile the
  FP4 kernel took 14.3 ms per step. The 1.7 to 1.9x of the probe predicts a
  gain of 5.9 to 6.8 ms. The quantizer adds two small kernels per MoE
  layer, 86 per step, and its time is inside these numbers.
- At batch 1 the step is 0.8 ms faster (1.02x). The FP4 kernel takes only
  about 3.6 ms of a batch-1 step.
- Every V4 run on a B200 with the released checkpoint now takes e4m3 expert
  inputs: the two-GPU server, the serving script and the profile modes.
  `scripts/modal_v4_flash_generate.py --compare-paths` sets bf16 inputs on
  both paths, so it still compares the grouped path with the loop alone.
- On Hopper the mode does more work per weight than the bf16 decode. The
  emulated `tl.dot_scaled` costs 9.9 to 15.3 instructions per weight in the
  sm90 K loop, against 8.3 to 8.5 for the bf16 decode. The H100 tests pass,
  but no H100 timing exists. Auto keeps the mode off there.
- Test models with random scales keep bf16 inputs under auto, because their
  scales are not powers of two.
- `tests/unit/test_fp4_grouped_kernel.py` checks the quantizer scales, the
  W4A8 kernel against fp32 math on its own e4m3 inputs for both row splits,
  and a full MoE layer against the bf16 loop.
- The Triton interpreter casts fp32 to e4m3 wrongly (it drops the carry at
  the edge of a power of two and rounds ties away from zero). The tests
  check the e4m3 values bit for bit only on CUDA.
