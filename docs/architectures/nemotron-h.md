# Nemotron-H (Nemotron 3.5 Lightning, Nemotron 3 Nano)

## TL;DR

Nemotron-H is a hybrid where every layer is one RMSNorm and one mixer with a
residual add. Most mixers are Mamba-2 state-space layers, which keep a
fixed-size state per head instead of a growing KV cache. The rest are MoE
layers or dense MLPs, and a few attention layers with no positional
encoding. Nemotron 3.5 Lightning 30B-A3B has 23 Mamba-2, 23 MoE and 6
attention layers; Nemotron 3 Nano 4B has 21 Mamba-2, 17 MLP and 4 attention
layers.

Three details matter most:

- The MoE experts are non-gated: `down(relu(up(x)) ** 2)`, with no gate
  projection. The router is the DeepSeek-V3 sigmoid router with an aux-free
  selection bias, which mini-infer already had for GLM.
- The attention layers use no RoPE. Order comes from the Mamba layers.
- In the pinned transformers 5.14, the CPU fallback for Mamba-2 is wrong
  for spans longer than one chunk (fixed upstream in 5.15.0), so parity for
  long spans uses its token-by-token path
  ([ADR-035](../decisions/ADR-035-nemotron-h-port.md)).

## The map

| Primitive | Reference (`transformers/models/nemotron_h/modeling_nemotron_h.py`) | Our code |
| --- | --- | --- |
| Layer | `NemotronHBlock` | [nemotron_h.py](../../src/mini_infer/models/nemotron_h.py) `_NemotronBlock` |
| Mamba-2 mixer | `NemotronHMamba2Mixer.torch_forward` | `_NemotronMamba2Mixer` |
| Short conv | `conv1d` + SiLU | [mamba2.py](../../src/mini_infer/models/blocks/mamba2.py) `mamba_conv_prefill`, `mamba_conv_step` |
| SSD scan, prefill | chunked naive SSD | `ssd_chunked` |
| SSM step, decode | single-token update | `ssd_recurrent` |
| Gated norm | `Zamba2RMSNormGated` (gate before norm, groups) | `gated_group_rmsnorm` |
| Router | `NemotronHTopkRouter` | [glm_moe_gate.py](../../src/mini_infer/models/blocks/glm_moe_gate.py) `GlmNoAuxTcGate` |
| Experts | `NemotronHExperts` (3D tensors) | `_NemotronMoE.experts_up`, `experts_down` |
| Shared expert, dense MLP | `NemotronHMLP` | `_ReluSquaredMLP` |
| Attention | `NemotronHAttention` (no RoPE) | `_NemotronAttention` |
| Norms | `NemotronHRMSNorm` | [rmsnorm.py](../../src/mini_infer/models/blocks/rmsnorm.py) `RMSNorm` |

## Mamba-2

The input projection splits into a gate `z` (4096 wide on Lightning), the
conv input `xBC` (6144: x plus B and C for 8 groups of state size 128), and
one raw step size per head (64). The short conv with bias and SiLU runs
over `xBC`, which then splits into x, B and C. B and C are shared by the 8
heads of each group.

Per head, with `dt = softplus(dt_raw + dt_bias)` and `A = -exp(A_log)`:

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * x_t B_t^T
    y_t = h_t C_t + D * x_t

The output norm multiplies by `silu(z)` first, then RMS-normalizes each of
the 8 groups of 512 channels, then applies the weight. All SSM math runs in
fp32, and the state is cached in fp32.

`ssd_chunked` evaluates the same recurrence in chunks of 128 tokens: inside
a chunk the output is a masked product of C against B, and only one state per
chunk is carried forward. `ssd_recurrent` steps one token at a time and is
the decode path. `tests/unit/test_mamba2_block.py` pins them to each other,
including carried states and partial last chunks.

## MoE

The router computes sigmoid scores in fp32, adds the selection bias only to
choose the top 6 experts, and weights the experts with the unbiased scores,
normalized and scaled by 2.5. The routed experts are stored stacked. A
decode step gathers its chosen experts' weights, which needs no host sync
and so works in a CUDA graph. A long prefill loops over the experts that
received tokens. The shared expert runs on every token, and outputs add in
fp32.

## Cache structure

`NemotronHStateCache` holds, per layer:

- Mamba-2: an fp32 state `(64, 64, 128)` and a conv tail `(6144, 4)` with
  the last 4 raw inputs, newest last.
- Attention: append-only K and V buffers `(max_seq_len, 2, 128)`.
- MoE and MLP: nothing.

Only the 6 attention layers grow with context. The decode step attends over
the whole fixed-length buffer and masks the unwritten slots on the device,
so it has no host sync and runs under the CUDA graph of
[ADR-033](../decisions/ADR-033-cuda-graph-state-cache-decode.md).

## Loading

`from_checkpoint` builds the model in its final dtype on its device and
copies the checkpoint in about 1 GiB batches. The loader strips nothing from
the names: our modules mirror the checkpoint's `backbone.` layout. It
squeezes the conv taps from `(channels, 1, kernel)`, copies each routed
expert into its slot, accepts the older `backbone.embedding.weight`, and
drops the `mtp.` layer.

## Validation contract

- Synthetic parity against transformers 5.14 in fp32 on CPU, with
  checkpoint-shaped keys: prefill, greedy decode, chunked prefill, ragged
  decode, the dense-MLP and latent-MoE layer kinds, and the weight-drop path.
  Spans longer than one chunk are compared against the reference fed one
  token at a time (see the TL;DR).
- Real weights on a laptop: Nemotron 3 Nano 4B completes "Paris" on an M1
  Pro (MPS, fp16). Run it with
  `pytest tests/unit/test_nemotron_h_real_weights.py --run-large-models`.
- Real 30B-A3B on one A100 80 GB GPU, bf16, against transformers on the
  same weights: `scripts/modal_nemotron_h_smoke.py` passed. Prefill logit
  cosine was 0.9949 to 0.9989 with the same last-token top-1 on all four
  prompts, and greedy output matched for 31, 7, 2 and 32 of 32 tokens. The
  first prompt completes as " Paris.". ADR-035 discusses why the agreement
  is looser than for the dense Qwen3.8.
- Decode with the CUDA graph on that GPU: 13.6 ms per step at batch 1
  (42.3 eager) and 29.5 at batch 4 (43.5 eager).

## What is still open

- Tensor parallelism: the Mamba heads, the conv channels and the experts
  would need sharding.
- A fused Mamba-2 decode kernel, like the Gated DeltaNet one of ADR-034, and
  a grouped expert matmul in place of the per-step expert gather.
- A layer-by-layer MoE routing comparison against the reference on the
  real 30B-A3B weights.
- The vision towers of Bolt and Super-VL, and the multi-token-prediction
  layer.
