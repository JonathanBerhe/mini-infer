# ADR-034: Fused Gated DeltaNet decode step in Triton

Date: 2026-10-20
Status: Accepted

## Context

With the decode step replayed from a CUDA graph
([ADR-033](ADR-033-cuda-graph-state-cache-decode.md)), a batch-1 step of the
real Qwen3.8-27B took 46 ms on an A100, against a weight-bandwidth bound of
about 26 ms. The graph removes the host's launch cost, but the GPU still runs
thousands of small kernels back to back. The Gated DeltaNet layers, 48 of the
64, hold most of them: between the input projections and `out_proj`, one
layer ran about 50 element-wise kernels (the short-conv update, the q/k/v
split and head broadcast, two L2 norms, the gate, the delta-rule update and
readout, the gated RMSNorm), each on a few KB.

A fused DeepSeek-V4 decode kernel was rejected before
([ADR-020](ADR-020-v4-decode-kernel-and-prefix-sharing.md)): it gave no
end-to-end gain on a model dominated by its MoE, and it was not bit-exact.
So this kernel had to show an end-to-end gain on the real model, with greedy
output checked against transformers.

## Decision

1. **Two Triton launches per layer** (`models/blocks/gdn_decode_kernel.py`).
   `_conv_step_kernel` runs one program per row and channel block: it rolls
   the raw-input tail, applies the depthwise taps and SiLU, and updates the
   conv cache in place. `_recurrent_step_kernel` runs one program per row and
   value head: L2 norms, gate and step size, the decay and delta-rule update
   of the `(128, 128)` fp32 state in registers, the readout, and the
   SiLU-gated RMSNorm. The conv step is its own launch because the q and k
   channels are shared by the three value heads of a key head; one owner per
   channel is the only race-free way to update them in place.
2. **Same rounding points as the eager code.** The kernels round to the
   model dtype where `blocks/kda.py` and `blocks/gdn.py` do (the conv output,
   the step size, the readout, the normalized and the weighted value), so the
   two paths differ only in fp32 summation order.
3. **On by default on CUDA.** `supports_gdn_decode_kernel(device)` gates it
   (CUDA and Triton). Other devices keep the eager path, which stays the
   reference. Both launches are capturable, so the kernel runs inside the
   CUDA graph.
4. **Debugged without a GPU.** `tests/unit/test_gdn_decode_kernel.py` runs on
   CUDA or under Triton's interpreter (`TRITON_INTERPRET=1`, for example in a
   Linux container). It checks the fused step against the eager layer over
   several steps (fp32 within 1e-5, bf16 cosine above 0.999, the conv tail
   exactly) and the whole-model ragged decode with the kernel on and off.

## Alternatives Considered

- **One kernel per value head, conv included.** Three programs would read
  the shared q/k conv tails while one of them writes them. A second launch
  is cheaper than a grid-wide sync.
- **Split the state tile over the value dimension.** It eases register
  pressure, but the gated RMSNorm needs the whole readout of a head, so it
  needs a second reduction pass. At 128 x 128 fp32 with 8 warps the tile fits.
- **Keep the kernel opt-in so the default stays bit-exact with the eager
  step.** Rejected: the gain is real end to end, fp32 parity holds, and the
  one greedy change seen is a bf16 near-tie (see below).

## Consequences

Measured on one A100-SXM4-80GB, bf16, real Qwen3.8-27B
(`scripts/modal_qwen38_smoke.py --cuda-graph`), 64-token prompts, greedy,
milliseconds per decode step:

| Batch | Eager, PyTorch GDN | Eager, fused GDN | Graph, PyTorch GDN | Graph, fused GDN |
|---|---|---|---|---|
| 1 | 110.4 | 68.6 | 46.0 | 39.6 |
| 4 | 112.3 | 70.2 | 51.5 | 41.5 |

- Under the CUDA graph the fused step is 1.16x faster at batch 1 and 1.24x
  at batch 4. Without the graph it is 1.6x. Against the first eager step,
  graph and kernel together are 2.8x at batch 1.
- End to end through the scheduler, prefill included, batch 1 serves 14.3
  tokens per second (13.4 with the graph alone, 7.4 with neither).
- The graph still gives the same tokens as eager with the same step.
- Greedy output against transformers: 32 of 32 tokens on three of the four
  gate prompts. On the fourth, the fused step picks " wavelengths" where the
  reference picks "," at token 14. Both continuations are fluent, and the
  same position already flips between batch 1 and batch 2 with the PyTorch
  step, so it is a bf16 near-tie, not a defect. The prefill logits, which do
  not use the kernel, match as before.
- A batch-1 step now runs at about 66 percent of the bandwidth bound. The
  remaining small kernels are mostly the decoder RMSNorms, the attention
  layers and the MLP activations.
