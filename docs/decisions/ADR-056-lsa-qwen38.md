# ADR-056: Latent Sparse Attention as an opt-in mode for Qwen3.8

Date: 2026-11-05
Status: Accepted

## Context

Latent Sparse Attention (LSA, arXiv 2610.42147) picks, for each query
token, which earlier keys it reads. Its authors publish the code as a plugin
for their own copy of SGLang, at `Prefrontal-AI/latent-sparse-attention`.
They also publish index maps for `Qwen/Qwen3.8-27B` at the revision that
this port already matches ([ADR-032](ADR-032-qwen38-hybrid-port.md)).

LSA reads dims 64 to 255 of the stored keys of the 4 KV heads. Rotary does
not touch these dims. A pair of `[768, 64]` maps per full-attention layer
projects them to 64 dims. Each closed block of 512 keys becomes 64 buckets
of at most 16 keys, by Ward clustering. A query scores the bucket centers,
keeps the best 1,024 buckets and takes their keys up to 4,096. It also reads
a 512-key window and 4 sink keys, so at most 4,612 keys. Below about 4.6k
tokens, LSA is equal to dense attention.

The dense decode of the port read the whole `max_seq_len` buffer through a
mask at each step. A comparison against that path overstates any sparse
gain, so the dense path needed a fair version first.

## Decision

1. `models/qwen/lsa.py` holds the selection. It is bit-equal to the
   reference at commit 906be19. `scripts/qwen/clone_lsa_reference.py` pins
   that clone, and the parity tests skip without it.
2. LSA is off by default. `load_index_maps` reads the published file with
   `weights_only=True`. `enable_lsa` rejects a wrong shape or a missing
   full-attention layer, and `disable_lsa` turns LSA off again.
3. The state cache keeps the bucket centers and members. Over the 16
   layers this is less than 1% of the K and V size.
4. LSA decode runs from the CUDA graph. `DecodeCudaGraph` calls the model's
   `after_decode_step` after each replay, and that hook clusters the blocks
   the step closed. Without LSA the hook does nothing.
5. On CUDA with `flash-attn`, dense attention calls
   `flash_attn_with_kvcache` on the contiguous buffers
   (`cache/contiguous_attention.py`). It reads each row only up to its
   length. Other devices keep the PyTorch path.

## Alternatives Considered

- A fused Triton kernel for selection, gather and attention. Not written:
  at 32k and 64k, attention is a small part of a decode step (see below).
- FlashInfer with page size 1 over the same buffer. Deferred for the same
  reason.
- GLM-5.3-Flash, the other target of the authors. mini-infer does not port
  that model.

## Consequences

All runs used one H100 and the real 27B in bf16. They ran before
[ADR-054](ADR-054-gdn-scalar-gate-prefill.md), so their prefill times use
the old Gated DeltaNet prefill.

- Accuracy (real prose with a needle at 25% and at 75%, 32k and 64k
  tokens): dense and LSA found the needle in all 4 cases. The
  last-position logits had a cosine of 0.988 to 0.997, with the same top-1
  token.
- Decode speed (torch 2.8 with `flash-attn` 2.8.3, ms per step):

  | Mode | Context | Batch 1, eager | Batch 1, graph | Batch 4, graph |
  |---|---|---:|---:|---:|
  | Dense | 32,768 | 36.6 | 24.7 | 27.8 |
  | LSA | 32,768 | 55.2 | 31.8 | 37.0 |
  | Dense | 65,536 | 38.8 | 25.5 | 30.5 |

- At 32k, LSA is 29% slower than dense in graph decode. With FlashAttention,
  dense reads about 0.6 ms of K and V per step, next to about 16 ms of
  weights. The PyTorch selection and gather of LSA add about 7 ms.
- The FlashAttention path alone took dense eager decode at 64k from 82 ms
  to 39 ms per step.
- Attention reads equal weight reads at about 820k tokens of total context
  (batch times context). So LSA can only help at 256k and more, or at large
  batches, and it needs the fused kernel first.
- LSA prefill was 34% slower than dense at 32k (123.5 s against 91.9 s).
  The extra time comes from the per-query gather, which ADR-054 did not
  change. Against the new dense prefill of 8.4 s, that gather is now the
  largest cost. It was not measured again.
- `flash-attn` 2 has no prebuilt wheel past torch 2.8. The main Modal
  images use torch 2.14.1, so they keep the PyTorch dense path, and
  `fast_path_report` prints `flash_attn=none`.
- LSA needs all 4 KV heads on one rank, and the port is single-rank.
  Contexts past 262,144 tokens need YaRN, which the port does not support.
