# ADR-045: Fused RMS norms for DeepSeek-V4

Date: 2026-10-24
Status: Accepted (on by default for DeepSeek-V4)

## Context

After the fused partial RoPE of [ADR-044](ADR-044-fused-partial-rope.md),
the DeepSeek-V4-Flash decode step on 2x B200 still has 1,916 copy kernels
per step at batch 16. Almost all are dtype casts
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md#graph-step-at-batch-16)),
and about half belong to RMS norms. Each rank runs 365 RMS norms per
step. In HF order, one `RMSNorm` is 8 small kernels, and each kernel costs
a few microseconds to start and end.

## Decision

1. One Triton kernel per norm (`models/blocks/rms_norm_kernel.py`) keeps a
   row of up to 16,384 features in registers.
2. The kernel keeps the casts of the PyTorch path, as in HF for `RMSNorm`
   and as in the reference for the q-norm.
3. It sums the squares in another order, so it is not bit-identical. A few
   bf16 outputs differ in the last bit. The real model sets the bar.
4. `DeepseekV4Config.fused_rms_norm` (default True) turns on `fused` for
   every `RMSNorm` and `fused_q_norm` for every attention q-norm.
   `RMSNorm(fused=...)` is off by default, so other models keep the
   PyTorch path. The CPU, other dtypes and tensors that need a gradient
   also keep it.

## Alternatives Considered

- Fuse all but the mean. This stays bit-identical, but a norm still runs 4
  kernels.
- `torch.nn.functional.rms_norm`. It is not bit-equal either and gives less
  control over the casts.
- On for every model. Each model then needs its own accuracy test.

## Consequences

Measured on 2x B200, real V4-Flash, both modes on one load
(`scripts/modal_v4_flash_generate.py --compare-fused-norm`), with W4A8 FP4
experts ([ADR-043](ADR-043-w4a8-fp4-experts.md)) and the fused RoPE. A
step time is the median of 16 CUDA-graph decode steps.

| Measure | PyTorch norms | Fused kernel |
|---|---:|---:|
| Passage perplexity | 7.845 | 7.874 |
| Graph step, batch 1 | 30.2 ms | 27.1 ms |
| Graph step, batch 16 | 41.9 ms | 37.0 ms |

- The step is 1.11x faster at batch 1 and 1.13x at batch 16.
- Perplexity is 0.37% higher, inside the 1% bar of ADR-043. The greedy
  tokens of "The capital of France is" do not change.
- The modes pick the same next token at 95.0% of positions. A likely
  cause, not measured, is MoE routing: a last-bit change can move a token
  to another expert.
- `tests/unit/test_rms_norm_kernel.py` bounds the difference. On CUDA,
  fewer than 2% of the bf16 outputs differ. It passes on H100 and B200.
- The hyper-connection norm (87 per step) and about 716 other casts per
  step stay as they are.
