# ADR-021: MiniMax-M3 (MSA) text-only port

Date: 2026-06-26
Status: Accepted (Phase 0). Design fixed; implementation phased (see
[docs/plans/minimax-m3.md](../plans/minimax-m3.md)). Full implementable detail in
[docs/plans/minimax-m3-spec.md](../plans/minimax-m3-spec.md).

## Context

MiniMax-M3 (arXiv 2606.13392, June 2026) is a new multimodal model with
~428B parameters (23B active) and MSA sparse attention. We port the text
path only.

MSA keeps the full KV cache. Each query attends only to the top-16
128-token blocks plus the local block. MSA has no compression and no
sliding window, so M3 fits the existing `PagedKVCache` and serving stack.
V4 needed its own `StateCache`.

## Decision

Port M3 as `MiniMaxM3ForCausalLM` on the standard PagedKVCache path
(`USES_STATE_CACHE=False`) and reuse the GLM-MoE-DSA family:

- Attention: GQA (64q/4kv, head_dim 128), per-head Gemma QK-norm and
  partial RoPE (first rotary_dim=64 dims, theta 5e6). Layers 0-2 are dense,
  and layers 3-59 add the MSA indexer. Parity configurations must have the
  deployment shape, or HF silently uses full RoPE.
- MSA indexer (new): 4 index heads (one per GQA group) and one shared index
  key head, dim 128. It scores tokens in fp32, max-pools them into
  128-token blocks and builds a dense additive `[B,64,S,Sk]` mask for the
  top-16 plus the local block. We copy the HF dense-mask reference path,
  which is the bit-parity target.
- MoE: reuse `GlmNoAuxTcGate` and `GlmMoeFFN` with `n_group=topk_group=1`.
  128 experts, top-4, 1 shared.
- `swigluoai` (new), a clamped GLU for the dense MLP and every expert:
  `(clamp(up,-7,7)+1)*clamp(gate,max=7)*sigmoid(1.702*clamp(gate,max=7))`.
- Norms: `GemmaRMSNorm` `(1+w)` everywhere (`use_gemma_norm=true`).
- Serving: the scheduler, prefix cache and tensor parallelism do not
  change. Pin `required_attention_backend()="torch"`, because the additive
  mask needs materialized SDPA.
- Loader: map the on-disk `language_model.*` weights directly (expert
  `w1=gate, w3=up, w2=down`). Drop the vision and MTP keys.

## Parity strategy

The reference is the eager path of `transformers` `minimax_m3_vl`. A tiny
random text-only configuration on the CPU loads the same weights into both
models and compares each layer. Selections must be equal integer sets, and
tensors pass `allclose(rtol=1e-4, atol=1e-5)` in fp32. The real 428B model
also gets a GPU coherence gate. Pins: `MiniMaxAI/MiniMax-M3` @
`bfd6c97f0296da547f10ecb20102c5d51a5c462e` and `transformers==5.12.1`
(test-only).

## Alternatives Considered

- Gather the selected blocks as the reference. Rejected, because HF uses
  a dense mask. A later gather kernel is tested against the dense-mask
  path.
- Mixtral MoE as the base. Rejected, because it routes with softmax and has
  no selection bias.
- `swigluoai` through the existing SiLU SwiGLU. Rejected, because the
  clamps change the math.

## Consequences

- About 80% of the code is reuse.
- M3 runs only on the `torch` attention backend. Throughput is a non-goal.
- The MSA Triton kernel is deferred. As ADR-020 taught, it ships only
  after an end-to-end A/B test, and default-on only if token-identical.
- The parity harness settles the open items in Section 6 of the spec.
