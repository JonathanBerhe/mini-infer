# ADR-033: CUDA-graph decode for state-cache models

Date: 2026-10-20
Status: Accepted

## Context

The real Qwen3.8-27B gate ([ADR-032](ADR-032-qwen38-hybrid-port.md)) served
greedy output equal to transformers, but slowly: 7.2 tokens per second at
batch 1 on an A100, end to end. Reading the 50.1 GiB of bf16 weights once
per step bounds batch-1 decode at about 26 ms per step on an A100 SXM
(2.0 TB/s), about 38 tokens per second. The measured step was 107 ms.

A profile of one decode step with the 27B layer layout (64 layers, tiny
dimensions, CPU) counted about 7,900 PyTorch operations per step, most of
them small element-wise kernels. Each full-attention layer also read the
largest position back to the host (`positions.max().item()`) to slice its
K/V buffer: 16 host syncs per step. The step was bound by kernel launches,
not by memory bandwidth.

## Decision

1. **Make the decode step sync-free and static.** A one-token step with a
   cache now attends over the whole fixed-length K/V buffer and masks the
   slots past each row's position on the device. The query heads that share
   a KV head are stacked on the query axis, so grouped-query attention needs
   no per-head copy of K and V. Prefill spans pass their cache length as a
   host integer. The step has no host sync and no data-dependent shape.
2. **Capture the step in a CUDA graph.** `DecodeCudaGraph` warms the step up
   on a side stream, records it once, and replays it with one launch. A model
   opts in with `SUPPORTS_DECODE_CUDA_GRAPH` and `decode_step_static`;
   `supports_decode_cuda_graph(model, device)` is the single check.
   `StateCacheContinuousScheduler(cuda_graph=True)` captures in `start()`,
   right after it builds the batched cache and before any admission, because
   warmup and capture write into every row. Admission copies a full row in
   place, so the graph sees new requests without a new capture.
3. **On by default in the HTTP server** for models that support it on CUDA.
   Only the Qwen hybrid declares support today. DeepSeek-V4 and Kimi Linear
   keep the eager path.

## Alternatives Considered

- **`torch.compile(mode="reduce-overhead")`.** It would fuse the element-wise
  kernels as well as graph the step, but we expect a long compile for 64
  layers, and the old per-layer `.item()` would have split the graph anyway. The
  explicit graph is small and keeps the eager kernels, so its output is the
  eager output.
- **A fused Gated DeltaNet decode kernel.** It cuts the kernel count of 48 of
  the 64 layers, but [the fused V4 decode kernel](ADR-020-v4-decode-kernel-and-prefix-sharing.md)
  gave no end-to-end gain and broke bit parity. Removing launch overhead
  first shows how much a kernel could still add.
- **Slice the K/V buffer with a host-side length.** It keeps attention short,
  but the length changes every step, so a graph would need one capture per
  length. Masked full-buffer attention costs a few MB of extra reads per
  layer, small next to the weights.

## Consequences

Measured on one A100-SXM4-80GB, bf16, real Qwen3.8-27B
(`scripts/modal_qwen38_smoke.py --cuda-graph`), 64-token prompts, greedy,
one host sync per step for sampling:

| Batch | Eager ms/step | CUDA graph ms/step | Speedup |
|---|---|---|---|
| 1 | 107.3 | 46.2 | 2.32x |
| 4 | 109.4 | 51.8 | 2.11x |

- End to end through the scheduler, prefill included, batch 1 went from 7.4
  to 13.4 tokens per second in the same run.
- Tokens are identical to the eager step in every run, and greedy output
  still equals transformers for 32 of 32 tokens on all four gate prompts.
  `tests/unit/test_cuda_graph_decode.py` pins graph-equals-eager in fp32 and
  bf16 on CUDA.
- Removing the host syncs alone did not change eager speed measurably (7.2
  and 7.4 tokens per second, on PCIe and SXM A100s). It was needed for the
  graph, not for its own sake.
- At 46 ms the graph runs at about 57 percent of the bandwidth bound. The
  remaining gap is the GPU running thousands of small kernels back to back;
  fusing the Gated DeltaNet step and the norms is the next lever, and this
  benchmark is its baseline.
- Batch-1 and batch-4 steps cost almost the same, so a fuller batch gives
  close to linear throughput: about 77 tokens per second at batch 4.
- Graph capture adds a few seconds to scheduler start. A model whose decode
  step is not capture-safe must not set `SUPPORTS_DECODE_CUDA_GRAPH`.
