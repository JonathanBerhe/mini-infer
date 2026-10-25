# ADR-043: W4A8 for the FP4 MoE experts on Blackwell

Date: 2026-10-24
Status: Accepted (on by default on sm100)

## Context

The batch-16 profile of the real DeepSeek-V4-Flash on 2x B200
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md#graph-step-at-batch-16))
shows the largest item of a serving step. It is the FP4 grouped kernel of
[ADR-040](ADR-040-fp4-grouped-kernel.md), which grows from 3.6 ms per step
at batch 1 to 14.3 ms at batch 16.

Nsight Compute shows that instructions limit the kernel, not memory. At 16
tokens the gate-up kernel keeps the issue slots 71% busy but reads DRAM at
only 22% of peak. The K loop runs 7.9 to 9.3 instructions per weight,
mostly to decode each e2m1 weight into bf16.

A probe on a B200 timed kernel variants. Cheaper bf16 decodes gained 12% at
most at 16 tokens. Only `tl.dot_scaled` removed the decode. It takes the
packed e2m1 weights and their e8m0 (power-of-two) scales, and on Blackwell
it runs as a native MX matmul on the tensor cores. The other operand must
be e4m3 (FP8) with e8m0 scales per 32 columns. At 16 tokens it cut the
gate-up projection from 230.6 to 135.3 µs and the down projection from
132.9 to 70.9 µs.

The DeepSeek reference also gives its `fp4_gemm` e4m3 inputs with
power-of-two scales.

## Decision

1. W4A8 (4-bit weights, 8-bit inputs) is a mode of the FP4 grouped kernel.
   With the `W4A8` constexpr, the K loop calls
   `tl.dot_scaled(packed, w_scale, "e2m1", inputs, input_scale, "e4m3", acc)`.
2. `quantize_e4m3_rows` turns the bf16 inputs of a projection into e4m3
   values with one e8m0 scale per 32 columns. The scale is the smallest
   power of two that keeps the block maximum inside 448, the e4m3 maximum.
3. `PackedFp4Experts` keeps an e8m0 byte copy of each weight scale. The V4
   checkpoints store e8m0 scales, so the copy is exact.
4. `DeepseekV4Config.fp4_fp8_activations` is None (auto) by default. Auto
   turns the mode on where `supports_native_w4a8(device)` is true (sm100:
   B200, B300) and every expert scale is a power of two. True forces the
   mode, and False keeps bf16 inputs. Only the grouped path changes.

The default rests on quality on the real model: the passage perplexity
does not get worse, and the greedy tokens do not change.

## Alternatives Considered

- Keep it opt-in. In the probe, the largest error of one projection grew
  from 0.3% to 2.6% of the largest output. But the real model did not get
  worse, so the default follows the speed.
- On by default on every GPU. Hopper emulates `tl.dot_scaled` at 9.9 to
  15.3 instructions per weight, against 8.3 to 8.5 for the bf16 decode. No
  H100 timing exists.
- `torch.nn.functional.scaled_grouped_mm`. It takes MXFP4 weights only with
  MXFP4 inputs, which lose more than e4m3 inputs.

## Consequences

Measured on 2x B200, real V4-Flash, both modes on one load
(`scripts/modal_v4_flash_generate.py --compare-fp8-activations`), on the
first 2,048 tokens of `scripts/data/technical_passage.md`:

| Measure | bf16 inputs | e4m3 inputs (W4A8) |
|---|---:|---:|
| Passage perplexity | 7.925 | 7.845 |
| Graph step, batch 1 | 34.0 ms | 33.2 ms |
| Graph step, batch 16 | 51.7 ms | 45.8 ms |

- The perplexity is 1.0% lower. The modes pick the same next token at 95.1%
  of the passage positions, with a last-logit cosine of 0.9966. The 33
  greedy tokens of "The capital of France is" are the same. One passage
  cannot show whether the lower perplexity is a real gain or noise.
- At batch 16 the step is 5.9 ms faster (1.13x), quantizer included. At
  batch 1 it is 0.8 ms faster (1.02x).
- Every V4 run on a B200 with the released checkpoint now takes e4m3 expert
  inputs.
- `tests/unit/test_fp4_grouped_kernel.py` tests the quantizer scales, the
  W4A8 kernel against fp32 math, and a full MoE layer against the bf16
  loop. The Triton interpreter casts fp32 to e4m3 wrongly, so bit-exact
  e4m3 tests run only on CUDA.
