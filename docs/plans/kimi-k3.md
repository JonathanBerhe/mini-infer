# Kimi K3 port plan (via Kimi Linear)

> Status: Stage A implemented (KDA + hybrid stack, bit-parity vs the vendored reference); Stage B blocked on the K3 weights + tech report (2026-07-27)
> Date: 2026-07-18
> Goal: register the Kimi K3 family with bit-parity validation inside the 30-day window that opens when its reference is published. Text-only; the vision/audio towers are out of scope by design.

## Context

Moonshot announced Kimi K3 on 2026-07-16: a 2.8T-parameter sparse MoE (16 of 896 experts active, "Stable LatentMoE"), 1M context, built on Kimi Delta Attention (KDA) with "Gated MLA", "AttnRes" cross-depth attention residuals, a SiTU activation, and MXFP4-weight / MXFP8-activation quantization-aware training. Weights and the tech report land 2026-07-27; until then only blog-level detail exists.

The port splits in two because ~90% of K3's inference-relevant novelty is already published and runnable as **Kimi Linear** (arXiv:2510.26692, checkpoint `moonshotai/Kimi-Linear-48B-A3B-Instruct`, MIT): KDA linear attention, the ~3:1 KDA/NoPE-MLA hybrid, and the recurrent-state serving model. Porting `kimi_linear` first validates all of that against a reference that runs today; the K3 delta then lands on top within days of the release.

Kimi K2 already runs here via the `DeepseekV2ForCausalLM` path and K2.6 was deferred as a V3 relabel (roadmap). K3 is not a relabel: KDA is this repo's first linear-attention / recurrent-state architecture.

## Stage A: `kimi_linear` family (IMPLEMENTED)

| Phase | Content | Status |
|---|---|---|
| A0 | Reference study, this plan, [kimi-k3-spec.md](kimi-k3-spec.md), ADR-026, `scripts/clone_kimi_linear_reference.py` (pinned revision) | done |
| A1 | `blocks/kda.py`: gate, L2 norm, conv helpers, recurrent + chunkwise delta rule; `tests/unit/test_kda_block.py` pins chunkwise == recurrent | done |
| A2 | `models/kimi_linear.py`: config validation, KDA + NoPE-MLA layers, Kimi MoE gate (see spec: NOT the GLM gate), registry entry, weight remap | done |
| A3 | `cache/kimi_state_cache.py` + cache-aware forwards (prefill continuation, decode, ragged decode); serving through the generalized `StateCacheGenerator` / `StateCacheContinuousScheduler` seam | done |
| A4 | Bit-parity vs the vendored reference (`tests/unit/test_kimi_linear_parity.py`): prefill cos > 0.999 + argmax + allclose 1e-3, token-exact greedy / batched-ragged / chunked prefill; structural + serving smoke in `test_models_kimi_linear.py` | done |
| A5 | Real-checkpoint gate on Modal: coherence + greedy determinism on the 48B-A3B weights (single H200-class device or 2x 80GB). Priced and explicitly approved before any spend | pending |
| A6 | `docs/architectures/kimi-linear.md` walkthrough, README + roadmap registry rows, ADR finalization | pending |

Design decisions and trade-offs live in ADR-026; the pinned numerical semantics in the spec.

## Stage B: K3 delta (opens 2026-07-27, target within 30 days)

- **B0, day-one triage.** Pull K3's config, modeling code, and tech report. Update the spec with exact semantics for Gated MLA, AttnRes, SiTU, LatentMoE, quantile-balancing inference impact, MTP presence, and the multimodal layout (text-only extraction, Inkling-style tower dropping).
- **B1, new blocks.** SiTU in `blocks/activations.py`; Gated MLA as a config-gated variant of the Kimi MLA layer; AttnRes wiring in the decoder stack (nearest precedent: `blocks/hyper_connections.py`); MoE gate variant only if LatentMoE changes inference math.
- **B2, model + loader.** `models/kimi_k3.py` composing the kimi_linear stack; `dequantize_mxfp4_to_bf16` beside `quant/nvfp4.py` (MXFP4 is fp4-e2m1 data + e8m0 scale per 32-block; the loader already preserves both dtypes); streaming loader with MXFP4-resident-or-dequant experts (M3 FP8 pattern).
- **B3, parity.** Tiny-random parity vs K3's own reference code through the same vendored-oracle harness. A full 2.8T real-weight gate is out of reach; validation is synthetic parity plus (stretch) first-N-layers streaming parity against real shards, documented like GLM-753B's deferral.
- **B4, docs + registry.** ADR, walkthrough, README row, 30-day metric recorded.

## Risks

1. **K3 unknowns.** AttnRes / LatentMoE / Gated MLA / SiTU are blog-level today; Stage B scope is provisional until the tech report. Stage A is immune.
2. **Reference-vs-repo transformers drift.** The Kimi remote code targets transformers ~4.57; the repo pins 5.14. Mitigated: the reference is vendored at a pinned revision and the test helper bridges the three observed drifts with scoped patches (see the spec's corrections log). A pin bump re-runs the parity suite.
3. **2.8T is unvalidatable end-to-end.** Named up front; same bar GLM-753B shipped with.
4. **Prefix caching for recurrent state** is whole-prefix snapshots at best (what Moonshot upstreamed to vLLM as "KDA with prefill cache"). Stage A serves the family with prefix sharing disabled; the KDA snapshot cache is a follow-up.
5. **Dense MLA buffers.** The per-request MLA history is dense (sized to max_seq_len), not paged. Correct and simple; a paged-MLA + state hybrid (vLLM's mixed cache groups) is an engine-level follow-up, not part of the port.

## Timeline note (primary metric)

The 30-day paper-to-port clock for K3 starts 2026-07-27 (reference availability). Stage A ran off-clock against the Kimi Linear reference (published 2025-10). The counted K3 milestone is the from-scratch port with synthetic bit-parity; serving hardening and kernels are not on the clock.
