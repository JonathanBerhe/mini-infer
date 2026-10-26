# ADR-033: CUDA-graph decode for state-cache models

Date: 2026-10-20
Status: Accepted

## Context

The Qwen3.8-27B gate ([ADR-032](ADR-032-qwen38-hybrid-port.md)) ran at 7.2
tokens per second at batch 1 on an A100. Reading the 50.1 GiB of bf16
weights bounds a batch-1 step at about 26 ms. The measured step was 107 ms.

A profile counted about 7,900 PyTorch operations per decode step, mostly
small element-wise kernels. Each full-attention layer also read
`positions.max().item()` to slice its K/V buffer: 16 host syncs per step.
Kernel launches, not memory bandwidth, bound the step.

## Decision

1. Make the decode step sync-free and static. A one-token step attends over
   the whole fixed-length K/V buffer and masks the unwritten slots on the
   device. The mask adds a few MB of reads per layer.
2. Capture the step in a CUDA graph. `DecodeCudaGraph` records it once and
   replays it with one launch. A model opts in with
   `SUPPORTS_DECODE_CUDA_GRAPH` and `decode_step_static`.
   `supports_decode_cuda_graph(model, device)` is the single gate.
   `StateCacheContinuousScheduler(cuda_graph=True)` captures in `start()`,
   before any admission. Admission copies rows in place, so it needs no new
   capture.
3. On by default in the HTTP server on CUDA. DeepSeek-V4 declares support
   since [ADR-042](ADR-042-cuda-graph-v4-decode.md).

## Alternatives Considered

- `torch.compile(mode="reduce-overhead")`. The compile for 64 layers is
  long, and the per-layer `.item()` splits the graph.
- A fused Gated DeltaNet decode kernel.
  [The fused V4 decode kernel](ADR-020-v4-decode-kernel-and-prefix-sharing.md)
  gave no end-to-end gain and broke bit parity.
- Slice the K/V buffer with a host-side length. The length changes every
  step, so a graph needs one capture per length.

## Consequences

With `scripts/qwen/modal_qwen38_smoke.py --cuda-graph` on one A100-SXM4-80GB, a
batch-1 step went from 107.3 to 46.2 ms (2.32x). Batch 4 went from 109.4 to
51.8 ms (2.11x), about 77 tokens per second.

- End to end, batch 1 went from 7.4 to 13.4 tokens per second.
- Tokens equal the eager step and still match transformers.
  `tests/unit/engine/test_cuda_graph_decode.py` pins graph-equals-eager in fp32 and
  bf16 on CUDA.
- The graph reaches about 57 percent of the bandwidth bound. Fusing the
  Gated DeltaNet step and the norms is the next lever.
- Capture adds a few seconds to scheduler start. A model whose decode step
  is not capture-safe must not set `SUPPORTS_DECODE_CUDA_GRAPH`.
