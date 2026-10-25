# ADR-025: Inkling port (relative-bias hybrid attention, SConv, shared-expert-sink MoE)

Date: 2026-07-16
Status: Accepted

## Context

Thinking Machines released Inkling (975B total / 41B active MoE, Apache 2.0,
`thinkingmachines/inkling`). It is the first prominent open-weights model
with no RoPE. The reference is transformers 5.14 (`InklingForCausalLM`), and
the port covers the text path only. The new parts are:

- A learned, query-conditioned relative position bias in place of RoPE.
- Sliding layers (window 512) and global layers at 5:1.
- Four short depthwise causal convs (kernel 4) per layer.
- A MoE that weights experts by a log-sigmoid softmax over the top-6 routed
  logits and the 2 shared experts.

## Decision

- Fold relative bias, causality and window into a per-request additive mask
  for the `block_mask` path of `packed_attention_torch`. Pin
  `required_attention_backend() = "torch"`. HF does the same, because no
  fused kernel takes a per-head additive bias.
- Store the pre-conv inputs of each conv as per-token streams in the
  `PagedKVCache`. Each step gathers the `kernel_size - 1` tail.
- Add `InklingGate`/`InklingMoE` rather than extend `GlmNoAuxTcGate`.
- Single rank. `load_weights` raises under TP.
- Bump the transformers pin to 5.14.x, the first release with Inkling, and
  realign the two families whose reference changed.

## Alternatives Considered

- Rolling conv states per request (ring buffers). They use less memory but
  break under `truncate_to` and prefix-cache rollback. Deferred until a
  benchmark needs them.
- A bias argument on `packed_attention_forward`. The `block_mask` hook
  already gives this per request.
- Pin transformers 5.12 and vendor the Inkling code. Rejected: HF is the
  parity reference, and the 5.14 changes are corrections.

## Consequences

- Inkling runs on the standard `PagedKVCache` path with continuous batching,
  chunked prefill and ragged decode. Tests match HF incremental decode token
  for token.
- The conv streams cost about `2*hidden + 2*kv_dim` extra pool bytes per
  token per layer, 4-7x the raw KV footprint. A rolling buffer must come
  before any long-context serving claim.
- All attention uses the materialized torch path. The next step is a paged
  decode kernel that applies the bias inline.
- The 5.14 bump fixed two references: the GLM-MoE-DSA indexer RoPE is
  interleaved, and MiniMax-M3 MSA selects blocks per index head. The M3
  Triton kernel change matches the torch reference on CPU only, and needs a
  GPU A/B before the next M3 real-model claim.
- The real 975B checkpoint needs `load_weights_streaming` and a multi-GPU
  staging script. Until then, tiny-model parity tests are the gate.

## Amendment (2026-10-19): released checkpoint layout

The released files use `model.llm.*` names and fuse gate and up into `w13`
tensors. The first loader read only the HF in-memory names
(`model.language_model.*`), so `ModelRunner.from_pretrained` failed on real
weights. `load_weights` now applies the transformers renames
(`conversion_mapping.py`, entry `inkling_mm_model`) first. A test loads a
`save_pretrained` checkpoint, which has the released layout. The parity
suite now uses init scale 0.1, because at 0.02 a swapped gate and up stayed
inside the logit tolerance.
