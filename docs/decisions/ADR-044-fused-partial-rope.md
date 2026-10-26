# ADR-044: Fused partial RoPE for DeepSeek-V4

Date: 2026-10-24
Status: Accepted

## Context

[ADR-042](ADR-042-cuda-graph-v4-decode.md) replays the DeepSeek-V4-Flash
decode step from a CUDA graph, so small GPU kernels bound the step. ADR-042
named the full-width `torch.cat` of the sliding window and the compressed
history as the next lever, at about 2.5 ms per step. The traces on 2x B200
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md#graph-step-at-batch-16))
showed that this was a misread. At batch 16 with the 1,024-token cache, the
cat takes about 0.09 ms in the 21 CSA layers. It takes 0.06 ms in the 20
HCA layers.

The larger item is `apply_partial_rope_last_n_dims`. It rotates the last 64
features of each head with separate PyTorch ops. One decode step calls it
212 times on each rank, which is 1,908 small kernels. They take 3.6 ms of
GPU time per step at batch 1 and 5.2 ms at batch 16.

## Decision

1. One Triton kernel per call (`models/blocks/partial_rope_kernel.py`)
   copies the features that do not rotate and rotates the tail.
2. The kernel rounds after each multiply, subtract and add, as PyTorch
   does, with FMA contraction off (`enable_fp_fusion=False`). So the
   outputs are bit-identical.
3. The kernel reads `x` through its strides. `broadcast_rope_tables`
   builds the cos and sin tables as views, so no copy runs first.
4. `apply_partial_rope_last_n_dims` runs the kernel on CUDA with bf16, fp16
   or fp32 tensors. Other cases keep the PyTorch path
   (`_partial_rope_torch`).

## Alternatives Considered

- One buffer per layer for the window and the history. It removes the cat,
  which costs about 0.14 ms per step here. It pays only with much longer
  caches or larger batches.
- Round once in fp32. The last bits then differ, and greedy decode turns
  that into other tokens. For that reason
  [ADR-020](ADR-020-v4-decode-kernel-and-prefix-sharing.md) rejected a fused
  V4 decode-attention kernel.
- `torch.compile` of the helper. ADR-033 and ADR-042 keep it out of the
  decode step for its compile time.

## Consequences

Measured on 2x B200 with real V4-Flash
(`scripts/deepseek_v4/modal_v4_flash_generate.py --compare-fused-rope`), with W4A8 FP4
experts ([ADR-043](ADR-043-w4a8-fp4-experts.md)):

| Measure | PyTorch partial RoPE | Fused kernel |
|---|---:|---:|
| Passage perplexity (2,048 tokens) | 7.845 | 7.845 |
| Graph step, batch 1 | 33.5 ms | 30.5 ms |
| Graph step, batch 16 | 46.1 ms | 41.9 ms |

- The real-model outputs are bit-identical, down to the 33 greedy tokens.
- The step is 1.10x faster at both batch sizes. So the 212 fused launches
  cost about 0.6 to 1.0 ms per step.
- `tests/unit/models/blocks/test_partial_rope_kernel.py` compares the kernel with the
  PyTorch path. On an H100 and a B200 the outputs are equal bit for bit.
- All V4 callers on CUDA take the kernel: the CSA, HCA and SWA attention,
  the compressor and the Lightning Indexer.
