# ADR-034: Fused Gated DeltaNet decode step in Triton

Date: 2026-10-20
Status: Accepted

## Context

With the decode step in a CUDA graph
([ADR-033](ADR-033-cuda-graph-state-cache-decode.md)), a batch-1 step of the
real Qwen3.8-27B took 46 ms on an A100. The weight-bandwidth bound is about
26 ms. Most remaining small kernels were in the 48 Gated DeltaNet layers,
about 50 element-wise kernels per layer.

[ADR-020](ADR-020-v4-decode-kernel-and-prefix-sharing.md) rejected a fused
DeepSeek-V4 decode kernel that gave no end-to-end gain. So this kernel had to
show an end-to-end gain, with greedy output compared against transformers.

## Decision

1. Two Triton launches per layer (`models/qwen/gdn_decode_kernel.py`).
   `_conv_step_kernel` updates the conv cache in place, with one owner per
   channel. `_recurrent_step_kernel` does the rest, with the `(128, 128)`
   fp32 state in registers.
2. The kernels round to the model dtype where `qwen/gdn.py` does, so the
   two paths differ only in fp32 summation order.
3. On by default on CUDA (`supports_gdn_decode_kernel(device)`), inside the
   CUDA graph. Other devices keep the eager path as the reference.
4. `tests/unit/models/qwen/test_gdn_decode_kernel.py` runs on CUDA or under
   `TRITON_INTERPRET=1`. Against the eager layer, fp32 is within 1e-5 and
   bf16 cosine is above 0.999.

## Alternatives Considered

- One kernel per value head, conv included. Three programs share the q/k
  conv tails, so this needs a grid-wide sync. A second launch is cheaper.
- Split the state tile over the value dimension. The gated RMSNorm then
  needs a second reduction pass, and the 128 x 128 tile fits.
- Keep the kernel opt-in. Rejected: the gain is real end to end, and fp32
  parity holds.

## Consequences

One A100-SXM4-80GB, bf16 (`scripts/qwen/modal_qwen38_smoke.py --cuda-graph`),
milliseconds per decode step:

| Batch | Eager, PyTorch | Graph, PyTorch | Graph, fused |
|---|---|---|---|
| 1 | 110.4 | 46.0 | 39.6 |
| 4 | 112.3 | 51.5 | 41.5 |

- Under the graph the fused step is 1.16x faster at batch 1 and 1.24x at
  batch 4. Graph and kernel together give 2.8x at batch 1.
- Greedy output matches transformers for 32 of 32 tokens on three of four
  gate prompts. The fourth differs at token 14, which batching alone also
  flips with the PyTorch step, so it is a bf16 near-tie.
- A batch-1 step runs at about 66 percent of the bandwidth bound.
