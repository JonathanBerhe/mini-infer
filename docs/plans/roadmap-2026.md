# Roadmap 2026: research-paper inference engine

> Status: accepted as the project's positioning.
> Date: 2026-05-16.
> Supersedes the original "portfolio piece" framing for internal planning.

## Positioning

mini-infer is a **from-scratch inference engine for newly-published LLM
architectures, with bit-parity correctness validated against the
upstream reference implementation**. We do not compete with vLLM or
SGLang on production throughput, hardware breadth, or enterprise
features. We compete on:

1. **Time-from-paper to from-scratch implementation.** When a new
   architecture (DeepSeek-V4, future V5, future Kimi variants, future
   Gemma generations) is published with a reference implementation, we
   ship a clean from-scratch port in mini-infer within 30 days,
   bit-parity validated against the reference.
2. **Readability.** The full engine fits in ~10k LoC of Python +
   selective Triton, organized so a research-paper reader can follow
   line-by-line correspondence between paper section, reference
   implementation, and our code.
3. **Architecture-breadth correctness.** Every model family in the
   registry passes a golden test against HuggingFace at temperature=0
   and (where a reference exists) a bit-parity test against the
   upstream inference reference.

The niche this fills, that neither vLLM nor SGLang fill today:

- vLLM and SGLang are production engines. They prioritize hardware
  breadth and throughput. They wait for upstream reference ports and
  then port them into their own production scaffolding. Their code is
  hard to read end-to-end (~200k+ LoC each); their abstractions are
  optimized for parallel hierarchies (multi-rank, multi-hardware,
  enterprise features) rather than for paper correspondence.
- Research-paper readers want a from-scratch implementation that
  shows how the paper's primitives compose into a runnable engine,
  without 200k LoC of production scaffolding obscuring the architecture.
- Inference engineers studying new techniques want a reference where
  bit-parity vs the original is the test contract.

## Current state (May 2026)

### What we ship today

**Architecture coverage (9 families, all bit-parity validated):**

- Qwen2 / Qwen3
- Llama (1/2/3/4, SmolLM2, TinyLlama, Llama-Nemotron variants)
- Mistral
- Gemma 3 / Gemma 4 (heterogeneous-KV, k_eq_v, dual-RoPE)
- Mixtral (top-k MoE)
- DeepSeek-V2 (MLA + interleaved RoPE + heterogeneous FFN)
- DeepSeek-V4 (HCA + CSA + AttentionSink + GroupedOutput +
  LightningIndexer + HashRoutedMoE + HyperConnections + YaRN)

**Production techniques (textbook implementations all of them):**

- Continuous batching scheduler
- PagedAttention with per-layer per-stream KV pool
- Prefix caching (chained-hash radix LRU)
- Speculative decoding (greedy, two-model)
- INT8 weight quantization (W8A16) + fused Triton kernel
- TurboQuant KV (V1 + V3 + fused dequant)
- FlashInfer FP8 + NVFP4 KV
- Tensor parallelism (Megatron-style column/row/vocab/expert)
- Prefill/decode disaggregation (worker-level + wire protocol + HTTP toggle)
- YaRN long-context RoPE

**Validation infrastructure:**

- ~9k LoC of source, ~6k LoC of tests
- 458 unit tests on every push (CPU-only path)
- Bit-parity tests against HF for every architecture
- Bit-parity tests against research-paper reference where applicable
  (DeepSeek-V4 against `deepseek-ai/DeepSeek-V4-Pro/inference`)
- 17 ADRs documenting every non-trivial decision

### Open gaps (statically clean, awaiting live hardware runs)

- **V4-Flash forward end-to-end on 2× B200.** Load path proven against
  real safetensors index; three dtype + meta-init fixes landed.
  Remaining: one live run, blocked on Modal spend cycle.
- **PD smoke end-to-end on 2× H100.** Wire protocol fix landed
  (NCCL CUDA device for handoff header). Remaining: one live run,
  blocked on Modal spend cycle (workspace limit reached).

## Goals 2026

### Primary metric

**Time-from-paper-publication to from-scratch-implementation-with-bit-parity.**

Target: ≤ 30 days from upstream reference availability.

Secondary metrics:

- Every registered architecture has a `docs/architectures/<family>.md`
  walkthrough mapping paper sections → reference file:line → our code.
- 0 known correctness regressions in the unit suite over a quarter.

### Immediate (next 2 months)

1. **V4-Flash forward live validation** (one Modal run, ~$3-5). Closes
   the V4 chapter end-to-end on real B200 hardware.
2. **PD smoke live validation** (one Modal run, ~$1-2). Closes the
   PD chapter end-to-end on 2× H100.
3. **Per-architecture walkthrough docs** (`docs/architectures/`). One
   markdown file per family, with paper-section → code-file:line
   mapping. Lead with V4 + MLA (the densest material).
4. **DeepSeek-V3 / Kimi-K2 registration.** Same MLA shape as V2;
   the registry path is wired but hasn't been exercised on >70B
   checkpoints. Effort: ~1 day of config + smoke on Modal when budget
   permits.

### Medium-term (next 6 months)

5. **New paper architectures as published.** When a notable new
   architecture lands (V5, new Kimi variant, MoE-with-attention-routing
   variants, etc.), implement from-scratch within 30 days. Track via
   a paper-watch list in this doc.
6. ~~**Continuous bit-parity CI.**~~ **Shipped May 2026** (commit
   `24517e4`). HF model commits pinned in `tests/_pinned_models.toml`;
   the bit-parity workflow runs `requires_model`-marked tests against
   those pins.
7. ~~**A `PDScheduler` with admission queue + multi-request batching.**~~
   **Shipped May 2026** (ADR-017). Serial + parallel modes; HTTP toggle
   via `MINI_INFER_USE_PD=1`, mode via `MINI_INFER_PD_MODE`.
8. **Uniform benchmark harness across techniques.** A single
   entry point that runs the same workload through every technique
   (baseline, INT8 W8A16, spec decoding, TurboQuant KV, FlashInfer
   FP8/NVFP4, tensor parallelism, PD disaggregation) and emits a
   comparable tok/s + per-request-latency table. Today each technique
   has its own bench script under `scripts/`; the numbers don't line
   up because the workloads differ. Lifts the bench surface from
   "per-technique ad-hoc" to "one apples-to-apples table."

   Supersedes the previous "HTTP streaming for every technique" item:
   under the research-paper-engine niche the actual gap is bench
   comparability, not API surface. The HTTP server stays minimal
   (PDScheduler + baseline); curl-reachability of every quant /
   spec-dec / KV-format toggle isn't load-bearing for a reader who's
   running pytest and bench scripts.

### Long-term (12+ months)

9. **Reference-implementation tooling.** A `mini_infer.reference` CLI
   that loads any of our families, runs both mini-infer's forward and
   the HF / reference reference, prints per-layer cosine-sim. Useful
   for any new architecture port.
10. **Architecture coverage doubling.** If we currently cover ~9
    families, target 18 by year 2 — primarily by adding new
    architectures as they're published rather than back-porting old
    ones.
11. **One real Triton kernel per quarter where the paper calls for it.**
    First ship landed June 2026: `hc_split_sinkhorn` (DeepSeek-V4
    Hyper-Connections) ported from the reference's tilelang kernel to
    Triton, ~14x per-call latency vs the PyTorch transcription on
    L40S, parity-validated across 18 configurations (ADR-018). The
    PyTorch transcription remains the oracle + CPU/MPS path. Next
    candidates as papers call for them.

## Non-goals (deliberate)

The following are out-of-scope for the research-paper-engine niche:

1. **Hardware breadth beyond CUDA.** AMD ROCm, Intel Gaudi, AWS
   Inferentia, Google TPU — these are vLLM / SGLang's responsibility.
   We support CUDA + CPU / MPS reference paths. Adding a third backend
   is multi-person-year work that doesn't move our niche.
2. **Custom production kernels we don't already have.** Outside of the
   specific paper-implementation kernels above, we use FlashAttention 2
   + FlashInfer for attention math, cuBLAS for GEMM. We have Triton
   kernels for INT8 W8A16 and TurboQuant KV dequant where the math
   isn't in any vendor library.
3. **Production reliability features.** Auto-scaling, multi-replica
   HA, observability (metrics/traces), enterprise auth/RBAC beyond
   the bearer-token gate, multi-tenant isolation. These would require
   a SRE-skilled team mini-infer doesn't have.
4. **Model coverage breadth as the metric.** We don't aim for the
   "supports 100+ models" list. We aim for "supports the architectures
   that introduced novel primitives, implemented from scratch."
5. **LoRA / multi-LoRA serving, structured output, function calling,
   embeddings, reranking.** All real production features. Not in our
   niche.
6. **Customer support, deployment-pattern docs (Helm/Kubernetes),
   multi-region patterns.** Production engine concerns.

## Resource constraints

These shape every decision:

- **1 contributor** (the project owner). Multi-person-year features
  are non-starters.
- **Modal hardware budget ≈ $30 total across the lifetime of the
  project.** Live multi-GPU runs cost $2-5 each; we can afford
  ~6-10 over a year. Local M1 is the everyday development environment;
  Modal is reserved for the final validation gate on multi-GPU features.
- **No GPU on the development box** beyond what M1 offers (MPS for
  small models, CPU for everything). The CPU/MPS reference paths in
  the engine aren't optional; they're how we develop.

## Success criteria

We're succeeding at the research-paper-engine niche if:

1. **A reader can pick a research paper from our family list, open
   `docs/architectures/<family>.md`, and follow the paper through to
   our code line-by-line.** Failure mode: the doc doesn't exist or
   doesn't match the code; remediation: write the doc as part of
   adding the family.
2. **A new paper publishes a notable architecture, and within 30 days
   we have a bit-parity port + walkthrough.** Failure mode: we skip
   a paper or take >30 days; remediation: this implies the niche is
   bigger than one person can sustain — at which point the project's
   value is the existing implementations + we revise the goal to
   per-quarter rather than per-paper.
3. **The full engine remains readable end-to-end.** Concrete: source
   stays under ~15k LoC; no module exceeds ~1.5k LoC; ADRs accompany
   every architectural decision; comments explain *why*, not *what*.
4. **Bit-parity tests don't bit-rot.** All upstream-pinned tests pass
   on a fresh checkout against the pinned reference.

## Paper watch list (informal)

Public architectures we'd port from-scratch when reference inference
code is available:

- **DeepSeek-V5** (TBD). Expected to extend V4's hybrid attention.
  Our V4 primitives should compose forward.
- **Future Gemma generations.** Gemma 4 is in; future versions likely
  refine the heterogeneous-KV + dual-RoPE pattern.
- **MoE-routing-novelty papers.** Anything that genuinely changes the
  expert-routing math (V4's hash routing was an example).
- **Novel attention variants** beyond the GQA/MLA/HCA/CSA family.
  Linear attention, state-space hybrids (Mamba, Jamba, Nemotron-H) —
  these change the cache abstraction entirely; would be a separate
  big-piece engineering effort.

### Evaluated and explicitly deferred

- **Kimi K2.6** (Moonshot AI, April 2026). Multimodal release. Text
  decoder is a relabeled `DeepseekV3ForCausalLM` (the K2.6 config's
  `text_config.architectures` literally says so). Two real gaps
  surfaced: (a) DeepSeek-V3's MoE routing primitives — `scoring_func:
  sigmoid`, `topk_method: noaux_tc`, group-limited routing — aren't
  in mini-infer's V2 path; (b) the published K2.6 weights are
  `compressed-tensors` INT4 (group_size=32), a quantization format
  not in our dequant inventory. The vision encoder (MoonViT) is
  text-only-out-of-scope. The agent-swarm / long-horizon coding
  features are training / system concerns, not inference
  architecture. **Deferred** because the underlying architectural
  innovation is DeepSeek-V3's, which is older than May 2026 and
  doesn't trigger our "new architecture within 30 days" rule.
  The V3-routing gap is tracked separately under "Tracked
  open architectural gaps" below.

## Tracked open architectural gaps

These are claims our existing code or docs implicitly makes that we
haven't fully delivered. Honest tracking, no immediate action.

- **DeepSeek-V3 MoE routing.** The `DeepseekV2ForCausalLM` registry
  entry covers V2-Lite / V2 / V3 / Kimi-K2 in terms of MLA attention,
  but the V3-specific routing primitives (`scoring_func: sigmoid`,
  `topk_method: noaux_tc`, group-limited routing via
  `n_group / topk_group`) aren't in `MoEFFN`. V4's `HashRoutedGate`
  has the `scoring_func` machinery; porting / sharing it with the
  V2/V3 path is the next architectural improvement to consider.
  Effort: ~3-5 days. Triggered if V5 or a future Kimi release uses
  the same routing AND we want bit-parity coverage.

## Open questions

1. **Whether to take state-space hybrids on.** Mamba / Jamba /
   Nemotron-H use SSM caches instead of KV caches. Adding them means
   a second cache abstraction. Big effort; aligned with the niche if
   the architectures matter. Re-evaluate quarterly.
2. **Whether to ship a `mini-infer-reference` Python package on
   PyPI.** Would broaden the audience but increases the documentation
   + onboarding burden. Defer until the architecture doc set is
   complete.

## Doc location

This document lives at `docs/plans/roadmap-2026.md`. Update it as the
niche evolves, the goals shift, or specific items land. ADRs in
`docs/decisions/` remain the per-decision record; this is the
top-level positioning + priorities doc.
