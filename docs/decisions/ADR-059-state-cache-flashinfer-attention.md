# ADR-059: Read the state-cache attention with FlashInfer up to each row's length

Date: 2026-11-06
Status: Accepted

## Context

Qwen3.8 and Nemotron-H keep the keys and values of each full-attention layer
in one buffer per request, `(B, max_seq_len, kv_heads, head_dim)`
([ADR-032](ADR-032-qwen38-hybrid-port.md), [ADR-035](ADR-035-nemotron-h-port.md)).
The decode step read the whole buffer and masked the empty positions, so
that it has fixed shapes and runs from a CUDA graph
([ADR-033](ADR-033-cuda-graph-state-cache-decode.md)). A step thus cost the
same at the first token as at `max_seq_len`. A server sizes the buffer for
its longest prompt, so every request paid for the full buffer.

[ADR-056](ADR-056-lsa-qwen38.md) added a `flash_attn_with_kvcache` path that
reads each row only up to its length. flash-attn 2 has no prebuilt wheel
past torch 2.8, so no main Modal image (torch 2.14.1) has it. FlashInfer is
in every CUDA image.

## Decision

1. `cache/contiguous_attention.py` views each buffer as FlashInfer pages of
   16 tokens. Row `b` owns pages `b * pages_per_row` and up. The state
   caches round their attention buffers up to whole pages. The usable
   length does not change.
2. `ContiguousFlashInferPlan` lists, once per step, the pages that hold each
   row's keys. Every attention layer of the step runs on that plan. The
   lengths come from the host, so a plan never waits for the GPU.
3. `DecodeCudaGraph` calls the model's new `before_decode_step` hook before
   each replay. The hook plans the step into FlashInfer's fixed buffers, so
   the captured `run()` reads the new plan.
4. One-token steps use FA2. Prefill spans keep FlashInfer's automatic choice.
5. `contiguous_backend` picks flash-attn when it is installed, else
   FlashInfer for head sizes 64, 128 and 256, else the masked PyTorch path.
   On Hopper FlashInfer has no kernel for smaller heads.

## Alternatives Considered

- One FlashInfer page per row, with the row length as the last page length.
  It needs no page list, but FlashInfer splits long rows by pages, so one
  page per row cannot be split across SMs.
- FlashInfer's automatic backend for decode. On Hopper it picks
  FlashAttention 3, which gives each query tile one CTA. A 32k-token
  Qwen3.8 step took 42.0 ms with it, against 25.6 ms with FA2.
- A Triton decode kernel that reads the lengths on the device, with no plan.
  FlashInfer is already tuned for these GPUs, and the plan costs no sync.

## Consequences

Measured on one H100 and one B200 with the real checkpoints, in bf16, from
the decode CUDA graph
([report](../benchmarks/2026-11-06-state-cache-flashinfer.md)):

| Case | Qwen3.8-27B H100 | Qwen3.8-27B B200 | Nemotron-H H100 | Nemotron-H B200 |
|---|---:|---:|---:|---:|
| 2k request, 2k buffer | 1.05x | 1.12x | 1.01x | 1.01x |
| 2k request, 64k buffer | 3.09x | 5.30x | 1.82x | 2.05x |
| 32k request | 2.00x | 3.07x | 1.39x | 1.51x |

- With FlashInfer, the step time no longer depends on the buffer size. On
  the B200 a 32k-token Qwen3.8 step takes 15.24 ms, against 14.88 ms at
  2k tokens.
- A 32k-token prefill is 1% to 8% faster. The recurrent layers set the
  prefill time.
- FlashInfer and PyTorch give last-position logits with a cosine of 0.99962
  to 0.99994 and the same top-1 token. Greedy text splits after some
  tokens, because the two paths sum in another order in bf16.
- The buffers grow by at most 15 positions per request.
- The FlashInfer workspace takes 256 MiB more GPU memory per process.
- The H200 and the B300 were not measured. They run the same kernels as the
  H100 and the B200.
