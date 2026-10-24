# ADR-045: Fused RMS norms for DeepSeek-V4

Date: 2026-10-24
Status: Accepted (on by default for DeepSeek-V4)

## Context

After the fused partial RoPE of [ADR-044](ADR-044-fused-partial-rope.md),
the DeepSeek-V4-Flash decode step on 2x B200 is still made mostly of small
kernels. The batch traces
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md#graph-step-at-batch-16))
show 1,916 copy kernels per step at batch 16, 5.3 ms of GPU time, almost
all of them dtype casts. About half of those casts belong to RMS norms.

A step runs 365 RMS norms on each rank:

| Norm | Per step | Kernels per step | GPU ms, batch 1 | GPU ms, batch 16 |
|---|---:|---:|---:|---:|
| `RMSNorm` modules | 235 | 1,880 | 3.52 | 5.16 |
| Per-head q-norm of the attention | 43 | 301 | 0.69 | 0.76 |
| Norm inside the hyper-connection mix | 87 | not measured | | |

`RMSNorm` follows the HF order: a cast to fp32, a square, a mean, an add,
an rsqrt, a multiply, a cast back and a multiply by the weight. That is 8
kernels for very little math. The q-norm is 7 kernels. Each kernel costs a
few microseconds to start and end, whatever it computes.

## Decision

1. **One Triton kernel per norm** (`models/blocks/rms_norm_kernel.py`). One
   program takes one row, keeps it in registers, sums the squares, and
   writes the normalized row. It handles rows up to 16,384 features.
2. **The casts of the PyTorch path.** For `RMSNorm`, `x * r` is cast to x's
   dtype before the weight multiply, as in HF. For the q-norm, `r` is cast
   to x's dtype before the multiply, as in the reference. The output dtype
   is the PyTorch path's.
3. **Not bit-identical.** The kernel sums the squares in another order than
   PyTorch's reduction. The fp32 variance can differ in its last bit, and
   then a few outputs differ in their last bf16 bit. The real model sets
   the bar (below), not bit equality.
4. **On for DeepSeek-V4 only.** `RMSNorm(fused=...)` is off by default, so
   Llama, Qwen, Mistral and Gemma keep the PyTorch path.
   `DeepseekV4Config.fused_rms_norm` (default True) turns it on for every
   `RMSNorm` of the model and, through `fused_q_norm`, for the q-norm of
   every CSA, HCA and SWA attention (`rmsnorm.head_rms_norm`). The CPU,
   other dtypes, wider rows, rows that no 2D view describes, and tensors that
   need a gradient keep the PyTorch path.

## Alternatives Considered

- **Fuse all but the mean.** PyTorch's own mean keeps the sum order, so the
  output stays bit-identical, but a norm still runs 4 kernels instead of
  8. It saves about half as much.
- **`torch.nn.functional.rms_norm`.** It multiplies by the weight in fp32
  before the cast, so it rounds differently from the HF order, and its
  reduction order differs too. It gives no bit equality either, and less
  control over the casts.
- **On for every model.** Each model's outputs would change in the last
  bit, so each needs its own check. This ADR checks DeepSeek-V4 only.

## Consequences

Measured on 2x B200, real V4-Flash, both modes on one load
(`scripts/modal_v4_flash_generate.py --compare-fused-norm`), with the FP4
experts on W4A8 ([ADR-043](ADR-043-w4a8-fp4-experts.md)) and the fused
RoPE (ADR-044). A step time is the median of 16 CUDA-graph decode steps
after 8 warm-up steps. Each mode captures its own graph.

| Measure | PyTorch norms | Fused kernel |
|---|---:|---:|
| Passage perplexity (2,048 tokens) | 7.845 | 7.874 |
| Passage top-1 accuracy | 53.5% | 53.4% |
| Graph step, batch 1 | 30.2 ms | 27.1 ms |
| Graph step, batch 16 | 41.9 ms | 37.0 ms |
| Tokens per second, batch 16 | 382 | 432 |

- The step is 3.1 ms faster at batch 1 and 4.9 ms faster at batch 16
  (1.11x and 1.13x).
- The perplexity is 0.37% higher. The bar of ADR-043, within 1% of the
  current path, holds. The 33 greedy tokens of "The capital of France is"
  are the same.
- The two modes pick the same next token at 95.0% of the passage positions,
  and the cosine of their last logits is 0.9973. The FP8 expert inputs of
  ADR-043 gave about the same figures (95.1%, 0.9966) and a 1.0% lower
  perplexity. A likely reason, not measured here, is the MoE routing: a
  last-bit change moves a token near a routing boundary to another expert,
  so small differences grow. One passage cannot separate a change of 0.37%
  from that noise.

- `tests/unit/test_rms_norm_kernel.py` bounds the difference: within one
  or two steps of the input dtype, and on CUDA fewer than 2% of the bf16
  outputs differ. It covers the decoder width (4,096), the q-lora width
  (1,024), the head width (512), no weight, an fp32 weight, a width that is
  not a power of two, and strided rows. The H100 and B200 runs pass it.
- The kernel compiles for sm90 and sm100 at the V4-Flash widths with 16 to
  42 registers per thread and no spills.
- The norm inside the hyper-connection mix (87 per step) and about 716
  other casts per step (2.4 ms at batch 16, spread over about 16 call
  sites) stay as they are.
