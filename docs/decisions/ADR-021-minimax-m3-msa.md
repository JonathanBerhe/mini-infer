# ADR-021: MiniMax-M3 (MSA) text-only port

Date: 2026-06-26
Status: Accepted (Phase 0). Design fixed; implementation phased (see
[docs/plans/minimax-m3.md](../plans/minimax-m3.md)). Full implementable detail in
[docs/plans/minimax-m3-spec.md](../plans/minimax-m3-spec.md).

## Context

MiniMax-M3 (weights + the MSA report, arXiv 2606.13392, June 2026) is the next
from-scratch port, on the paper-watch list and squarely on-thesis (a freshly
published sparse-attention architecture; the time-from-paper primary metric). It
is multimodal and ~428B (23B active); we port the **text path only** (vision is a
hard non-goal) and the MSA attention mechanism.

The defining property, confirmed against the HF reference: **MSA keeps the full KV
cache and only selects which 128-token blocks each query attends to** (top-16 +
the local block). No compression, no sliding window, no accumulator state. So M3
fits the existing `PagedKVCache` + standard serving stack, unlike V4 which forced
a bespoke `StateCache`. Block selection is a per-query additive mask over the
block table, not a cache rewrite.

Phase 0 read the authoritative HF code (`transformers/models/minimax_m3_vl/`), the
MSA kernel repo, and the paper, and mapped it onto mini-infer's primitives.

## Decision

Port M3 as `MiniMaxM3ForCausalLM` on the **standard PagedKVCache path**
(`USES_STATE_CACHE=False`), reusing the GLM-MoE-DSA family the repo already has,
with a small set of new pieces. The design:

- **Attention:** GQA (64q/4kv, head_dim 128) with per-head Gemma QK-norm and
  partial RoPE (dim 64, theta 5e6). Layers 0-2 are dense full attention; layers
  3-59 add the MSA indexer.
- **MSA indexer (new, the heaviest piece):** 4 index heads (one per GQA group) +
  one shared index key head, dim 128. Score tokens (fp32, no `1/sqrt(d)`),
  causal-mask at token granularity, **max-pool into 128-blocks**, top-16 +
  forced local block, emit per-query block selection -> a dense additive
  `[B,64,S,Sk]` mask. We mirror the HF **dense-mask** reference path (it does not
  gather), which is the bit-parity target.
- **MoE:** reuse `GlmNoAuxTcGate` + `GlmMoeFFN` (already sigmoid scoring +
  `e_score_correction_bias` selection + unbiased weighting + `routed_scaling_factor`
  + shared expert; `n_group=topk_group=1`). 128 experts, top-4, 1 shared. Only
  change: experts use the new `swigluoai` activation.
- **`swigluoai` (new):** clamped GLU, `(clamp(up,-7,7)+1)*clamp(gate,max=7)*sigmoid(1.702*clamp(gate,max=7))`.
  Used by the dense MLP and every expert.
- **Norms:** `GemmaRMSNorm` `(1+w)` everywhere (`use_gemma_norm=true`), including
  the four per-head qk/index norms.
- **Serving:** route through `ModelRunner` + `ContinuousScheduler` + prefix cache
  + tensor parallelism unchanged; pin `required_attention_backend()="torch"` (the
  additive per-query mask needs materialized SDPA, exactly as GLM-DSA / V2 do).
- **Loader:** map the on-disk `language_model.*` weights directly (expert
  `w1=gate, w3=up, w2=down`), dropping vision/MTP keys.

The genuinely-new code is small: `swigluoai`, the block-pooled indexer, and the
additive-mask wiring in `packed_attention_torch`. Everything else is reuse or a
thin extension. The full mechanism, bit-parity-critical list, weight-name map, and
reuse map (with file:line) are in the spec doc.

## Parity strategy

The HF reference is the upstreamed `transformers` `minimax_m3_vl` (the HF repo's
`auto_map` maps only `AutoConfig`; the model resolves from installed transformers,
and a pure-PyTorch eager path exists, so the kernel is not required for
correctness). Validate the from-scratch text forward against a **tiny-random
text-only config on CPU** (3 dense + 2 MoE/MSA layers, small `index_block_size`),
loading identical weights into both and comparing per-layer top-down; index and
router selections are checked as exact integer sets, tensors at
`allclose(rtol=1e-4, atol=1e-5)` fp32. The real 428B model can't run a full
side-by-side, so its bar is layer bit-parity (the tiny harness) plus a real-model
coherence gate on GPU (Phase 6). Pins: HF model `MiniMaxAI/MiniMax-M3` @
`bfd6c97f0296da547f10ecb20102c5d51a5c462e`; `transformers==5.12.1` as a test-only
dependency.

## Alternatives Considered

- **Gather selected blocks (the efficient kernel path) as the reference.**
  Rejected for the parity target: the HF eager/sdpa reference builds a dense
  additive mask and runs full attention (it ignores `block_indices` in the attn
  fn). We match the dense-mask path bit-for-bit; the gather/kernel is a separate
  Phase-5 optimization validated against it.
- **A bespoke StateCache (the V4 route).** Unnecessary: MSA keeps the full KV
  cache, so the standard PagedKVCache + scheduler + prefix cache apply directly.
- **Mixtral MoE as the base.** Rejected: softmax routing, no selection-bias. GLM's
  `GlmNoAuxTcGate` already matches M3's sigmoid + bias + unbiased-weight + scaling
  + shared semantics.
- **swigluoai via the existing SiLU SwiGLU.** Rejected: the alpha/limit clamping
  changes the math; implement the formula directly.

## Consequences

- Far cheaper than V4: ~80% reuse, no new cache/scheduler. New = `swigluoai`, the
  block-pooled indexer, the mask wiring.
- Pinned to the `torch` attention backend for M3 (materialized mask), like the
  other DSA-style models. Acceptable; throughput is a non-goal.
- The MSA Triton kernel (Phase 5) is deferred and gated on an end-to-end A/B
  before shipping (the V4-decode-kernel lesson). Unlike V4 decode, MSA's value is
  at long context where attention is a large decode share, so the kernel may move
  end-to-end here; the A/B decides, and it ships default-on only if also
  token-identical.
- Open items to resolve during implementation (RoPE pairing convention; the
  exact transformers patch to pin; verbatim-code drift; indexer norm type; whether
  to add a third `idx_k` cache stream) are listed in the spec doc Section 6 and
  are settled by the parity harness, not by more upfront research.

## References

- ADRs: ADR-014 (V4 hybrid attention; the indexer cousin), ADR-018 (HC Sinkhorn
  Triton port; kernel-port precedent), the GLM-5.2 MoE-DSA work (#15; the closest
  template), ADR-020 (the V4-decode-kernel rejection lesson informing Phase 5).
- Reference: HF `transformers` `models/minimax_m3_vl/` (modeling +
  configuration), pinned per the spec; MSA report arXiv 2606.13392; kernel repo
  `github.com/MiniMax-AI/MSA`.
- Build reference: [docs/plans/minimax-m3-spec.md](../plans/minimax-m3-spec.md);
  phased plan: [docs/plans/minimax-m3.md](../plans/minimax-m3.md).
