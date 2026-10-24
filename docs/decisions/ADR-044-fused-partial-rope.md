# ADR-044: Fused partial RoPE for DeepSeek-V4

Date: 2026-10-24
Status: Accepted

## Context

[ADR-042](ADR-042-cuda-graph-v4-decode.md) replays the DeepSeek-V4-Flash
decode step from a CUDA graph, so the step is bound by GPU work, and most of
that work is small kernels. ADR-042 named the full-width `torch.cat` of the
sliding window and the compressed history in each layer as the next lever,
at about 2.5 ms per step.

The batch-1 and batch-16 traces of the graph step on 2x B200
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md#graph-step-at-batch-16))
did not confirm that. Grouped by launch size, the window-plus-history cat
is 41 launches per step: about 0.09 ms in the 21 CSA layers and 0.06 ms in
the 20 HCA layers at batch 16, with the 1,024-token cache that serving uses.
Its cost grows with the batch and the cache length, but it is small at
these sizes. The 2.5 ms in ADR-042 was a misread of the profile.

The traces showed a larger item. `apply_partial_rope_last_n_dims` rotates
the last 64 features of each head with separate PyTorch ops: four
multiplies, a subtract, an add, a stack, a cat and, for the inverse
rotation, a negation. One decode step calls it 212 times on each rank: on
q, on the new KV row, on the attention output (inverse), in the compressor
and in the indexer. That is 1,908 kernels per step. They take 3.6 ms of GPU
time per step at batch 1 and 5.2 ms at batch 16. Each kernel moves little
data, so the start and the end of each kernel set the cost.

## Decision

1. **One Triton kernel per call** (`models/blocks/partial_rope_kernel.py`).
   A program takes a few rows of `x` (a row is one head of one token). It
   copies the features that do not rotate and rotates the interleaved pairs
   of the tail with the first half of the cos and sin tables. The output is
   contiguous, in the dtype the PyTorch path gives (`promote_types(x, cos)`).
2. **The same rounding as the PyTorch path.** PyTorch computes each
   multiply, subtract and add in fp32 and rounds it to the output dtype. The
   kernel rounds after each of those operations too, and the launch turns
   FMA contraction off (`enable_fp_fusion=False`), so an fp32 output is not
   computed with a fused multiply-add either. The outputs are bit-identical,
   so the decode tokens do not change.
3. **Any strides and the PyTorch broadcast.** The kernel reads `x` through
   its strides (up to three leading dims). The tables broadcast as in the
   PyTorch path: dims inserted before the last one until the ranks match,
   size-1 dims stretched. `broadcast_rope_tables` builds them as views, so
   no copy runs before the kernel.
4. **Dispatch in `apply_partial_rope_last_n_dims`.** On CUDA with bf16, fp16
   or fp32 tensors, the kernel runs. The CPU, other dtypes, more than three
   leading dims, tables that do not broadcast, and tensors that need a
   gradient keep the PyTorch path (`_partial_rope_torch`).

## Alternatives Considered

- **One buffer per layer for the window and the history.** It removes the
  cat, which costs about 0.14 ms per step here. It becomes worth doing only
  with much longer caches or larger batches.
- **Round once in fp32.** Fewer instructions, but the output differs from
  the PyTorch path in the last bits, and greedy decode amplifies such
  differences into other tokens. That is why
  [ADR-020](ADR-020-v4-decode-kernel-and-prefix-sharing.md) rejected a fused
  V4 decode-attention kernel. The bit-identical kernel costs the same one
  launch per call.
- **`torch.compile` of the helper.** It compiles each shape of the 212
  calls. ADR-033 and ADR-042 keep `torch.compile` out of the decode step for
  its compile time.

## Consequences

Measured on 2x B200, real V4-Flash, both modes on one load
(`scripts/modal_v4_flash_generate.py --compare-fused-rope`), with the FP4
experts on their default W4A8 path ([ADR-043](ADR-043-w4a8-fp4-experts.md)).
A step time is the median of 16 CUDA-graph decode steps after 8 warm-up
steps. Each mode captures its own graph.

| Measure | PyTorch partial RoPE | Fused kernel |
|---|---:|---:|
| Passage perplexity (2,048 tokens) | 7.845 | 7.845 |
| Graph step, batch 1 | 33.5 ms | 30.5 ms |
| Graph step, batch 16 | 46.1 ms | 41.9 ms |
| Tokens per second, batch 16 | 347 | 382 |

- The outputs are bit-identical on the real model too: the same passage
  NLL, the same next token at every passage position, bit-identical last
  logits, and the same 33 greedy tokens.
- The step is 3.0 ms faster at batch 1 and 4.2 ms faster at batch 16
  (1.10x at both). The PyTorch path took 3.6 and 5.2 ms of GPU time in the
  traces, so the 212 fused launches cost about 0.6 to 1.0 ms per step.

- `tests/unit/test_partial_rope_kernel.py` compares the kernel with the
  PyTorch path for bf16, fp32 and mixed dtypes, both directions, heads that
  broadcast the tables, one table for a batch, no features that stay, a
  width that is not a power of two, and strided inputs. On CUDA the outputs
  must be equal bit for bit. On an H100 and a B200 they are.
- The Triton interpreter truncates fp32 to bf16 instead of rounding it, so
  on the CPU the bf16 cases compare with a tolerance.
- The kernel compiles for sm90 and sm100 with 44 to 112 registers per
  thread and no spills. With fusion off, the code has no FFMA on either.
- Every model that calls `apply_partial_rope_last_n_dims` on CUDA takes the
  kernel: the DeepSeek-V4 CSA, HCA and SWA attention, the compressor and the
  Lightning Indexer.
