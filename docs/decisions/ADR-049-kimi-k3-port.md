# ADR-049: Kimi K3 port (SiTU, gated MLA, AttnRes, latent MoE, MXFP4 experts)

Date: 2026-10-27
Status: Accepted

## Context

Kimi K3 (`moonshotai/Kimi-K3`, weights out since 2026-07-27) is a 2.8T MoE
with 69 KDA layers, 24 NoPE-MLA layers and 896 experts. Kimi K3.1 Pro,
announced on 2026-10-23, is a post-trained K3 with the same architecture.
Moonshot promises its weights by 2026-11-04.

ADR-026 ported Kimi Linear, the stack that K3 builds on. K3's text model is
the same `KimiLinearForCausalLM` class (`text_config.model_type` is
`kimi_linear`). A diff of the two reference files shows six config-gated
additions and two changed conventions. This ADR records how the repo
supports that delta, so that K3.1 Pro loads on the day its weights land.

## Decision

1. K3 is the `kimi_linear` stack with config switches, not a new family.
   `KimiLinearConfig.from_hf` reads the K3 fields and each module turns its
   feature on only when the field is set. Every switch is off for Kimi
   Linear 48B, so the Stage A parity suite did not change.
2. `KimiK3ForConditionalGeneration` (`kimi_linear/kimi_k3.py`) is a thin
   wrapper over the text stack. Its config reads `text_config`. Its loader
   drops `vision_tower.` and `mm_projector.` and strips `language_model.`,
   as the Inkling port does. The text stack's own loader dequantizes the
   MXFP4 experts, so a text-only re-export loads the same way.
3. The math is pinned from the K3 reference code, not from the blog or the
   report. SiTU-GLU: `beta * tanh(g / beta) * sigmoid(g)` times
   `lb * tanh(u / lb)`, with `beta = 4` and `lb = 25`, in fp32. MLA: low-rank
   Q and a sigmoid output gate before `o_proj`. KDA: one full-rank output
   gate and the bounded decay gate `-5 * sigmoid(exp(A_log) * (raw + dt_bias))`,
   which is FLA's `lower_bound` kernel form. Latent MoE: one shared down
   projection before dispatch, experts in the latent width, an RMSNorm and
   one shared up projection after; the gate and the shared experts see the
   full width. AttnRes: a per-token softmax over residual-stream snapshots
   taken every 12 layers, applied before each attention, before each FFN and
   before the final norm (`kimi_linear/attn_res.py`).
4. The router convention differs between the two reference files. Kimi
   Linear adds the correction bias in place, so its expert weights are the
   biased scores. K3 adds it out of place, so its weights are the unbiased
   scores, and it masks dropped groups with `-inf`. One gate class serves
   both; `moe_weights_from_biased_scores` is false when the config carries
   any K3-only field.
5. MXFP4 experts dequantize to BF16 at load (`quant/mxfp4.py`). The byte
   layout is the NVFP4 one; the scale is `2 ** (byte - 127)` per block of
   32. Resident FP4 experts in the grouped GEMM are a follow-up.
6. Validation is tiny-random parity against the vendored reference at
   revision `f831ab66` with every K3 feature on: prefill, greedy decode,
   batched ragged decode, chunked prefill, and the gate. The test stub for
   the FLA kernels now also implements the in-kernel gate, the in-kernel
   `sigmoid(beta)`, the lower bound, and the `[V, K]` state layout.

## Consequences

- K3.1 Pro will load through this path if its config and tensor names match
  K3. Check both on the day the weights land before any GPU run.
- The port found three corrections. The reference builds `q_a_layernorm`
  and `kv_a_layernorm` with eps 1e-6, not `rms_norm_eps`; Stage A used 1e-5
  and is fixed. `A_log` is `(H,)` in K3 and `(1, 1, H, 1)` in Kimi Linear;
  the loader flattens both. The K3 reference no longer calls
  `fused_kda_gate`; the gate moved into the recurrence kernels.
- No real-weight run exists, the bar GLM-753B shipped with. The first real
  run needs a streaming loader and a multi-GPU script.
- Not in this change: tensor parallelism, resident MXFP4 experts, fused
  kernels, and prefix sharing for the KDA state.
