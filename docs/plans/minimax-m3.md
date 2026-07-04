# Plan: MiniMax-M3 (MSA) support in mini-infer

> Status: proposed. Date: 2026-06-26.
> Goal: complete support for MiniMax-M3's text path, from-scratch and bit-parity
> validated, served over the existing HTTP stack, with the MSA Triton kernel and a
> real-model GPU gate. Multimodal (vision) is out of scope (text-only non-goal).

## What M3 is (from config.json + arXiv 2606.13392)

- 60 layers, hidden 6144, vocab 200064, untied LM head, 1M context.
- Attention: GQA, 64 query heads / 4 KV heads (G=16), head_dim 128. Per-head QK
  norm (`qk_norm_type=per_head`). Partial RoPE (`rotary_dim=64`,
  `partial_rotary_factor=0.5`, `rope_theta=5e6`). Gemma-style RMSNorm
  (`use_gemma_norm=true`, eps 1e-6).
- Activation: `swigluoai` (clipped/scaled SwiGLU, `swiglu_alpha=1.702`,
  `swiglu_limit=7.0`).
- MoE: 128 routed experts, top-4 per token, 1 shared expert, sigmoid scoring with
  routing bias, `routed_scaling_factor=2.0`. First 3 layers are dense FFN
  (`dense_intermediate_size=12288`); layers 3-59 are MoE (`intermediate_size=3072`
  per expert, shared expert 3072). `moe_layer_freq=[0,0,0,1,...]`.
- MSA (sparse attention): first 3 layers use DENSE attention; layers 3-59 use MSA
  (`sparse_attention_freq=[0,0,0,1,...]`). MSA params: `sparse_block_size=128`,
  `sparse_topk_blocks=16`, `sparse_num_index_heads=4` (one index query head per
  GQA group, one shared index key head), `sparse_index_dim=128`,
  `sparse_score_type=max` (max-pool block aggregation), `sparse_local_block=1`
  (current block always selected), `sparse_init_block=0`.
- `num_mtp_modules=7` / `num_nextn_predict_layers=1`: multi-token-prediction heads.
  Out of scope for base inference; a future spec-decoding hook.
- Released weights: bf16. Reference: HF transformers `minimax_m3_vl` modeling code
  (text path) + the kernel reference at `github.com/MiniMax-AI/MSA`.

## The key architectural insight (why this is cheaper than V4)

MSA **keeps the full KV cache** and only *selects* which 128-token blocks each
query attends to (top-16 blocks + the local block ≈ 2,048 tokens). There is no
compression, no sliding window, no compressor/indexer accumulator state. So MSA
fits the **existing `PagedKVCache`** (full K/V already lives in 128-ish-token
paged blocks) plus one new step: score blocks and gather the top-k. Picking
top-k blocks is close to gathering block-table entries.

Consequence: MSA reuses the **standard serving stack** (`ModelRunner` +
`ContinuousScheduler` + `PagedKVCache` + the RadixAttention prefix cache + the
existing tensor-parallel path), unlike V4 which forced a bespoke `StateCache` and
all-new schedulers. The new work concentrates in the attention module, a few MLP/
MoE/norm variants, the loader, and the kernel.

## Reuse vs new

**Reuse (already in mini-infer):**
- GQA + `PagedKVCache` + packed-varlen attention (Qwen/Llama path).
- Partial RoPE, per-head QK norm (V4), Gemma RMSNorm (Gemma 3/4).
- MoE scaffolding (Mixtral; DeepSeek-V2/V4 sigmoid-gated MoE with shared expert +
  routing bias is the closest match).
- `ContinuousScheduler`, prefix caching, tensor parallelism, the `/v1/completions`
  server route for `PagedKVCache` models, the bench harness.

**New (the actual port):**
1. MSA attention module: index branch (per-group index Q + shared index K,
   token scores, max-pool over 128-blocks, top-16 selection + forced local/init
   blocks) and main branch (gather selected blocks, exact causal attention).
2. `swigluoai` MLP (alpha/limit-clipped SwiGLU).
3. M3 MoE (sigmoid scoring + routing bias + `routed_scaling_factor` + shared
   expert), if it differs from the existing DeepSeek MoE.
4. Block-selection integration with the paged block table (decode + prefill).
5. From-scratch loader + `MiniMaxM3ForCausalLM` registration.
6. The MSA Triton kernel (exp-free top-k + KV-outer block-sparse attention).

## Phased plan (each phase shippable; GPU only at the end)

**Phase 0: reference study + ADR. DONE (2026-06-26).** Read the HF
`minimax_m3_vl` modeling code, the `MiniMax-AI/MSA` kernel repo, and the paper;
resolved the `swigluoai` formula, per-head QK-norm, MoE gate, the dense-mask
reference path, and the prefill/decode block-scoring. Findings: the build
reference [minimax-m3-spec.md](minimax-m3-spec.md) and the decision record
[ADR-021](../decisions/ADR-021-minimax-m3-msa.md). Pins: HF model
`MiniMaxAI/MiniMax-M3` @ `bfd6c97f0296da547f10ecb20102c5d51a5c462e`,
`transformers==5.12.1` (test-only). Key result: M3 is GLM-MoE-DSA + block-pooled
top-k + `swigluoai`, ~80% reuse, no StateCache.

**Phase 1: building blocks (CPU, from-scratch, unit + bit-parity tests).**
`swigluoai` MLP, M3 MoE (sigmoid + bias + shared + scaling), Gemma-norm reuse,
per-head QK-norm + partial RoPE wiring, and the dense GQA layer (layers 0-2).
Each bit-parity'd against the reference module on synthetic configs (cosine >
0.999). No GPU.

**Phase 2: MSA attention module (core, parity-critical).** PyTorch reference
implementation of index + main branch (the oracle). Bit-parity vs the HF MSA
module on synthetic configs for prefill AND decode (cosine > 0.999), including the
forced-local/init-block and causal-within-block edge cases. No GPU.

**Phase 3: full model + loader.** Assemble the 60-layer model (3 dense+dense-FFN,
57 MSA+MoE, final norm, untied LM head). `from_checkpoint` mapping HF bf16 weights
(incl. 128 experts) to our modules. Register `MiniMaxM3ForCausalLM`. CPU
`generate()` on a small synthetic config; golden vs HF transformers at temp=0 on a
small config (the local correctness anchor). No GPU.

**Phase 4: serving integration (mostly reuse).** Route M3 through the standard
`ModelRunner` + `ContinuousScheduler` + `PagedKVCache` + prefix cache (NOT the
StateCache path). Wire MSA's block selection to the paged block table for prefill
and ragged decode. Serve over `/v1/completions`; tensor parallelism for the 428B
size (reuse the GQA + MoE TP path). Parity: served output == single-request
`generate` (self-consistency) + golden. CPU/gloo validation; real GPU deferred to
Phase 6.

**Phase 5: MSA Triton kernel.** Port the block-sparse kernel (exp-free top-k +
KV-outer block-sparse attention, per the paper's §4). PyTorch path stays the
oracle. Validate parity (cosine > 0.99), microbench, AND an end-to-end A/B
(kernel on vs off) before shipping, per the V4-decode-kernel lesson. Unlike V4's
short-context decode, MSA's value is at long context where attention is a large
share of decode, so the kernel may actually move end-to-end here; the A/B decides.
Default-off if it diverges token-level from the PyTorch path. GPU (cheap tier for
parity/microbench; priced + approved per run).

**Phase 6: real-model GPU gate.** Fit real M3-428B on Modal with quant-resident
experts (FP4/FP8, the V4-Flash playbook); scope the cheapest GPU config (likely
2-4 B200) and per-run cost FIRST, approve before spend. Real-model coherence gate:
load the real checkpoint, greedy-generate coherent + rank-consistent text under TP
(the V4-Flash bar). GPU (from the Phase-4 reserve).

**Phase 7: docs.** `docs/architectures/minimax-m3.md` walkthrough (paper-section →
code mapping). Finalize ADR(s). Update the roadmap + paper-watch list.

## Parity strategy

Same as V4: the from-scratch PyTorch forward is bit-parity'd against the HF
reference module per layer on synthetic configs (cosine > 0.999), and golden vs HF
at temp=0 on a small config. The real 428B model can't run a full side-by-side
(size), so the real-model bar is attention-layer bit-parity (unit tests) plus
coherent + rank-consistent output (Phase 6). The MSA kernel is validated against
the PyTorch MSA path (the oracle).

## Budget + honest risks

- **Real-model gate (Phase 6) is the budget risk.** M3 is 428B (vs V4-Flash
  158B), so the gate needs more GPU than V4's runs (which were ~$0.5-1.7 each on
  2x B200). With ~$8.6 left in the Phase-4 reserve, a 428B coherence gate plus the
  Phase-5 kernel A/B could approach or exceed it. I will price the exact GPU
  config + per-run cost before any spend and get explicit approval each time;
  if it doesn't fit the budget, Phase 6 pauses at "priced, awaiting budget" while
  Phases 0-5 (all free or cheap) complete. The from-scratch + bit-parity port
  (the primary-metric milestone) does not depend on Phase 6.
- **Reference fidelity.** `swigluoai`, the per-head QK norm, and the MoE gate
  have exact-form details only the HF modeling code pins down (Phase 0 resolves
  them before any math is written).
- **MSA kernel end-to-end value is unproven** until the Phase-5 A/B (the V4
  lesson). It ships default-on only if it's both faster end-to-end and
  token-identical, else opt-in/off.
- **MTP heads** (`num_mtp_modules`) are not part of base inference; left as a
  future spec-decoding hook, not in this plan's scope.

## Timeline note (primary metric)

MSA weights + report landed ~June 7-11, 2026. Phases 0-3 (the from-scratch port +
bit-parity, the milestone the time-from-paper metric counts) are the priority and
are GPU-free; completing them in the next 1-2 weeks keeps the port inside the
≤30-day target. Phases 4-7 (serving, kernel, real-model gate, docs) follow.
