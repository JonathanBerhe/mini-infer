# ADR-050: Replay the paged decode step from a CUDA graph

Date: 2026-11-01
Status: Accepted

## Context

The paged path (`ContinuousScheduler` over `PagedKVCache`) ran every decode
step eagerly. The step profile (`scripts/bench/bench_all.py --profile`) of
Qwen2.5-7B on one H100 showed about 8 ms of GPU work inside a step of 17 to
20 ms at concurrency 8, with about 1250 kernel launches. The host was the
bottleneck. The state-cache models already replay their decode step from a
CUDA graph ([ADR-042](ADR-042-cuda-graph-v4-decode.md)). The paged path did
not qualify, because parts of the step ran on the host inside it: the cache counted
tokens, allocated blocks and built its write and page indices during layer
0, FlashInfer planned its kernel in every layer and copied index lists from
the host, and the scheduler read each sampled token with its own wait on the
GPU.

## Decision

1. The cache gets a static decode step. `begin_static_decode` does the host
   work of a one-token-per-slot step (counts, block allocation, indices)
   before the replay and copies the indices into fixed device buffers in one
   copy. Until `end_static_decode`, the K/V writes and the page index go
   through those buffers with no host read. Rows above the real batch are
   padding: they write into one scratch block and attend to it.
2. FlashInfer gets one wrapper with fixed buffers per batch-size bucket
   (`use_cuda_graph=True`). `plan_static_decode` plans once before each
   replay with the parameters recorded on the first eager run. A plan inside
   a capture raises.
3. `PagedDecodeGraph` captures one graph per bucket (1, 2, 4, 8, 16, 32) at
   load, after two eager warmup runs on a side stream, and replays the
   smallest bucket that holds the batch. Every plan and replay runs in
   inference mode, because the wrapper buffers are born inside an
   inference-mode forward.
4. `ModelRunner.forward_step_packed` sends a step with one new token per
   request to the graph and runs everything else eagerly. `cuda_graph=True`
   on `from_pretrained` turns it on. The default stays off.
5. The scheduler samples all decoders in one batch (`sample_batch`), one copy
   to the host per step.

The graph needs CUDA, the FlashInfer backend, a bf16 KV pool and full
attention with the same heads in every layer. Sliding-window layers (Gemma),
MoE routing loops, latent-attention models, quantized KV and the flash-attn
backend stay eager.

## Alternatives Considered

- Capture the whole step, bookkeeping included. The bookkeeping is host
  code, so a graph cannot hold it.
- One graph per exact batch size. More graphs and more capture time. Buckets
  cost one shared scratch block and a few padding rows instead.
- Plan inside the forward and skip only the captured plan. A plan during a
  capture raises, so the plan had to move before the replay.
- Turn the graph on by default. Not yet: the first H100 gate failed at load
  on the inference-mode mismatch, so the feature stays opt-in until a second
  model family is measured.

## Consequences

Measured on one H100, Qwen2.5-7B bf16, FlashInfer backend, 32 new tokens,
eager and graph in one container on the same prompts. The eager row at
concurrency 1 includes the FlashInfer kernel compile and is left out.

| Concurrency | Eager (tok/s) | Graph (tok/s) | Graph step (ms) |
|---:|---:|---:|---:|
| 1 | compile in the timed window | 97 | 10.3 |
| 3 | 192 | 317 | 9.4 |
| 8 | 414 | 812 | 9.8 |

- The tokens are identical to the eager FlashInfer path at every
  concurrency, including the padded bucket at 3.
- The step at concurrency 8 is now close to its GPU time (about 8 ms of
  kernels). The next levers are on the GPU side: a fused RMSNorm, merged
  projections, fewer weight bytes.
- The batched sampling alone took the eager path from 406 to 457 tok/s at
  concurrency 8 in an earlier run on the same hardware.
- Each bucket holds its own graph and output buffer. The largest bucket
  bounds the batch that the graph serves, and a larger batch runs eagerly.
- A padding row's logits are dropped, and its K/V lands in the scratch block
  that every bucket shares.
- The bench registry runs the graph as the `paged_graph` technique, marked
  not bit-equal to the SDPA baseline like the other FlashInfer techniques.
