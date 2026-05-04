# Plan: multi-model support in mini-infer

## Context

mini-infer today supports one model family: Qwen2.5 (0.5B, 7B, 32B-Instruct).
The loader path is `AutoModelForCausalLM.from_pretrained(...)` plus a runtime
attention monkey-patch in [src/mini_infer/engine/attention_patches/qwen2.py](../../src/mini_infer/engine/attention_patches/qwen2.py)
that swaps HF's attention for our paged-varlen path. This works for one model
but doesn't scale: every new architecture needs its own patcher, and the
attention shapes that don't fit Qwen's pattern (MoE, MLA, hybrid CSA+HCA) need
much more than a patch.

The goal is to support a broader set of modern open models. Concretely:
**Llama 3/4, Qwen2.5/Qwen3, Gemma 2/3, Mistral / Mixtral, DeepSeek-V2/V3/V4,
Kimi-K2, Nemotron** as a starting set. Some are essentially-the-same-as-Qwen
and cost almost nothing once the plumbing exists. Others are genuinely new
architectures.

This plan is about the plumbing first, then the per-model work, in cheapest-
first order. The deepest single piece — V4's hybrid attention — is its own
sub-plan at [docs/plans/deepseek-v4-attention.md](deepseek-v4-attention.md).

## Status

| Phase | Status | Notes |
|---|---|---|
| 1: Registry + canonical blocks | **shipped** | Replaced HF-model-plus-patch with owned `nn.Module`s |
| 2: Llama-shape adds | **partial** | `LlamaForCausalLM` shipped (covers Llama 2/3/4 + SmolLM2/TinyLlama). Mistral, Qwen3, Nemotron pending — each a thin per-file add |
| 3a: SWA primitive + Gemma 3 | **shipped** | Per-layer attention type, dual RoPE, partial RoPE, GemmaRMSNorm/GeGLU/GemmaDecoderLayer; Gemma 3 1B-it greedy-parity with HF |
| 3b: Gemma 4 31B | **deferred** | Needs heterogeneous-KV (Stage C1, now shipped) plus `attention_k_eq_v` and `v_norm` (small follow-up). 26B-A4B and E2B/E4B remain out of scope |
| 4: MoE FFN + Mixtral | **shipped** | `MixtralForCausalLM`, top-k MoE primitive bit-parity vs HF |
| 5-prep (C1): heterogeneous-KV BlockPool | **shipped** | Per-layer `(num_kv_heads, head_dim)`. Foundation for MLA, V4, and Gemma 4 31B |
| 5: MLA + DeepSeek-V2/V3 + Kimi-K2 | **pending** | MLA is a multi-stream KV (latent + RoPE-K) — generalizes the per-layer-shape primitive into a per-layer storage descriptor |
| 6: DeepSeek-V4 hybrid attention | **pending** | See [V4 sub-plan](deepseek-v4-attention.md) |

## Architectural taxonomy

Sorting the target models by what they share:

| Family | Backbone | Attention | FFN | Notes |
|---|---|---|---|---|
| Llama 3/4 | RMSNorm + RoPE | GQA | SwiGLU | The "default modern" shape |
| Qwen2.5/Qwen3 | RMSNorm + RoPE | GQA | SwiGLU | Llama-shape; already supported |
| Mistral | RMSNorm + RoPE | GQA | SwiGLU | Llama-shape |
| Mixtral | RMSNorm + RoPE | GQA | **MoE SwiGLU** | First MoE we need |
| Gemma 2/3 | RMSNorm (+1 offset) + RoPE | GQA + **alternating SWA / global** | GeGLU | Sliding-window attention is mandatory |
| Gemma 4 31B | RMSNorm + dual RoPE | GQA + **heterogeneous attention shape per layer** (sliding=`head_dim 256, kv_heads 16`; global=`head_dim 512, kv_heads 4`, K=V) | GeGLU | Same heterogeneous-KV layout that V4 needs; supporting one paves the road for both |
| Gemma 4 E2B/E4B | RMSNorm + RoPE | GQA + SWA + **shared-KV tail layers** | GeGLU + **PLE** | Per-Layer Embeddings + shared KV layers; new primitives |
| Nemotron | RMSNorm + RoPE | GQA | **squared-ReLU** | Mostly Llama-shape; activation differs. Nemotron-H (state-space) out of scope |
| DeepSeek-V2/V3 | RMSNorm + RoPE | **MLA** | **MoE SwiGLU** | New attention shape (latent KV) |
| DeepSeek-V4 | RMSNorm + RoPE | **Hybrid CSA + HCA + SWA** | **MoE SwiGLU** | New attention + heterogeneous KV layout |
| Kimi-K2 | RMSNorm + RoPE | MLA | MoE SwiGLU | DeepSeek-V3-shape; piggybacks on V3 work |

Note on "Ollama": that's a runtime, not a model family. The Llama-shape row
covers what Ollama typically serves.

The plumbing falls out of this table:
- A **model registry**: HF `config.architectures[0]` → our model class.
- A **canonical building-block library**: shared modules (RMSNorm, RoPE,
  GQA, SwiGLU, SWA, MoE-FFN, MLA, CSA, HCA) that per-model files compose.
- A **weight loader contract**: each model class declares how HF weight
  names map into its module hierarchy.

## Today's state and what changes

```
Today:
  AutoModelForCausalLM.from_pretrained(...)
    -> patch_model_attention(...)  # Qwen2-only patcher
    -> ModelRunner wraps it

After Phase 1:
  ModelRegistry.from_pretrained(...)
    -> looks up architecture in registry
    -> instantiates our model class
    -> loads HF weights via the model class's mapping
    -> ModelRunner wraps it
```

Why move off the HF-model-plus-patch approach: each architecture's HF
implementation has its own attention shape, FFN shape, and rotary application
order. Patching is fine for one model and brittle for ten. Owning the model
code lets us share a single attention path across families and avoids the
patch-per-architecture sprawl.

## Phased shipping

### Phase 1: Registry + canonical building blocks (plumbing only)

**What ships:** No new model support. The existing Qwen path moves behind
a registry. The architectural shape is in place for everything below.

**Files (rough):**
- NEW `src/mini_infer/models/__init__.py` — `ModelRegistry`, `register`
  decorator, `from_pretrained` entry point.
- NEW `src/mini_infer/models/blocks/` — canonical modules:
  - `rmsnorm.py`, `rope.py`, `gqa.py`, `swiglu.py`
- NEW `src/mini_infer/models/qwen2.py` — Qwen2 model class composing
  blocks; weight-name mapping from HF.
- EDIT `src/mini_infer/engine/model_runner.py` — accept a model from the
  registry instead of doing its own `AutoModelForCausalLM` call.
- DELETE `src/mini_infer/engine/attention_patch.py` and
  `attention_patches/qwen2.py` once Qwen runs through the new path.
- EDIT golden tests — must still pass token-for-token vs HF reference.

**Risk:** golden-test regression. Mitigation: implement Qwen2 in the new
path and run the existing golden suite before removing the old path.

**Acceptance:** all current tests pass with the registry path; the old
patcher is removed; one model (Qwen2) is fully on the new path.

### Phase 2: Easy adds (Llama-shape family)

**What ships:** support for Llama 3/4, Mistral, Qwen3, Nemotron. All four
are RMSNorm + RoPE + GQA + SwiGLU with cosmetic differences (Nemotron uses
squared-ReLU, the rest are SwiGLU). Each is one small file: a model class
that composes Phase 1's blocks and declares the weight-name mapping.

**Files (rough):** one per model, ~50–100 lines each.
- `src/mini_infer/models/llama.py`
- `src/mini_infer/models/mistral.py`
- `src/mini_infer/models/qwen3.py`
- `src/mini_infer/models/nemotron.py` (with `squared_relu` block added to
  `models/blocks/`)

**Validation:** golden test (greedy parity vs HF reference at small size)
for each. Run on CPU-shaped synthetic configs first, then HF small models
where available (e.g. `Llama-3.2-1B`, `Mistral-7B-Instruct-v0.3`).

**Acceptance:** each model loads, runs, and produces token-for-token
matches against HF in greedy mode at small scale.

### Phase 3: Sliding-window attention + Gemma

**Why now:** Gemma 2/3 alternates sliding-window and global attention
layers. SWA is also a foundational primitive for V4 (CSA and HCA both
have sliding-window branches). Building it in Phase 3 means Phase 5 (V4)
can reuse it.

**Files:**
- NEW `src/mini_infer/models/blocks/swa.py` — sliding-window attention.
  Could wrap FlashInfer's SWA path or be a thin mask wrapper around the
  existing GQA path.
- NEW `src/mini_infer/models/gemma.py` — Gemma 2/3 model class. Adds a
  `gemma_rmsnorm` variant (the +1 offset).
- EDIT `block_pool.py` — per-layer attention type metadata, so the cache
  knows which layers are SWA-bounded vs global.

**Validation:** golden test against `gemma-2-2b` or `gemma-3-1b`.

**Acceptance:** Gemma loads and matches HF; SWA primitive is reusable.

### Phase 4: MoE FFN

**Why now:** unlocks Mixtral, DeepSeek-V2/V3, Kimi-K2, V4. One new
building block; per-model integration is then trivial.

**Files:**
- NEW `src/mini_infer/models/blocks/moe_ffn.py` — top-k expert routing,
  expert weights, softmax over selected experts. Start with the simple
  no-fused-grouped-gemm version (one matmul per expert per token); a
  fused implementation is a profile-driven follow-up.
- NEW `src/mini_infer/models/mixtral.py` — Mixtral 8x7B model class.

**Validation:** golden test against `Mixtral-8x7B-Instruct-v0.1` at small
batch (the model is large; greedy test on a single-prompt fixture).

**Acceptance:** Mixtral loads and matches HF; MoE block is reusable.

### Phase 5: MLA attention + DeepSeek-V2/V3 + Kimi-K2

**Why now:** MLA is the prerequisite for DeepSeek-V2/V3 and Kimi-K2.
The KV layout is fundamentally different (latent KV with separate RoPE
heads); building it correctly is its own piece of work.

**Files:**
- NEW `src/mini_infer/models/blocks/mla.py` — Multi-head Latent Attention.
  Q/K/V projections are low-rank with a small RoPE head appended.
  Decoder-side: cache the latent K/V (smaller than per-head K/V) and the
  RoPE K head, dequant on read.
- EDIT `block_pool.py` — accept a per-layer KV-shape descriptor (latent
  width vs RoPE width are distinct).
- NEW `src/mini_infer/models/deepseek_v3.py` — DeepSeek-V3 model class
  composing MLA + MoE blocks.
- NEW `src/mini_infer/models/kimi_k2.py` — same shape as V3, mostly
  weight-name remapping.

**Validation:** golden test against `DeepSeek-V3` at small batch
(671B is impractical; use `DeepSeek-V2-Lite` 16B as the primary
oracle). Greedy parity.

**Acceptance:** at least one DeepSeek model loads and matches a reference
implementation in greedy mode.

### Phase 6: DeepSeek-V4 hybrid attention

**Sub-plan:** [docs/plans/deepseek-v4-attention.md](deepseek-v4-attention.md).

This is the biggest single piece of work in the project. Builds on:
- Phase 1's registry and blocks
- Phase 3's SWA primitive (V4's sliding-window branch)
- Phase 4's MoE FFN
- Phase 5's per-layer KV-shape descriptor (V4's heterogeneous block layout
  is more aggressive but the abstraction is the same)

If Phases 1–5 are in place, Phase 6 is "implement CSA, implement HCA,
extend the per-layer cache descriptor to handle interleaved CSA/HCA
blocks, validate against DeepSeek's reference inference code." See the
sub-plan for the full breakdown.

## What's deliberately not in scope

- **Multi-GPU / tensor parallel.** Several target models (V4-Pro at 1.6T,
  Mixtral 8x22B) cannot fit on a single GPU. Adding TP is a separate plan
  that touches the model runner, the block pool, and the scheduler. We
  validate large models at small scale (V2-Lite, Mixtral on a slice) until
  TP exists.
- **Multimodal.** Gemma 3 has vision; we use the text-only configs.
- **State-space models** (Mamba, Nemotron-H, Jamba). Different cache
  abstraction entirely. Could be its own future plan.
- **On-disk KV cache** (V4 §3.6.2). Storage layer, orthogonal to the model
  work. Probably useful eventually; not blocking any model support.

## Open questions (decide before Phase 1)

1. **HF-model-plus-patch vs own-the-model.** The plan above assumes we
   move to owning the model code (vLLM / SGLang style). The alternative is
   to stay with `AutoModelForCausalLM` and write a patcher per architecture
   (`attention_patches/llama.py`, `attention_patches/gemma.py`, etc.).
   Patching is cheaper short-term, more brittle long-term — every HF
   transformers release can shift the layer names we patch. Recommend
   own-the-model. Decision blocks Phase 1.
2. **Weight-loading mechanism.** Three options for mapping HF safetensors
   into our modules: (a) explicit per-model maps (verbose but auditable),
   (b) convention-over-configuration with shared prefixes (terse but
   fragile), (c) HF's `state_dict` with rename rules (middle ground).
   Recommend (a) — explicit is best for a project where readability is a
   stated goal.
3. **Test budget per model.** Golden parity tests at small scale are
   essential; full integration tests on real weights are expensive. Pick a
   "representative small checkpoint" per family (e.g. `Llama-3.2-1B`,
   `gemma-2-2b`, `DeepSeek-V2-Lite`) and run greedy-parity goldens
   in CI-skipped tests, manually before merge.
4. **Model selection priority.** This plan covers a lot; not everything
   needs to ship in the next quarter. Most-valuable-first ordering depends
   on which model the user actually wants to demo. If the answer is
   "DeepSeek-V4", Phases 1, 3, 4, 5, 6 in that order. If the answer is
   "anything Llama-shaped", Phases 1, 2 are nearly the whole job.

## Critical files to read before Phase 1

- [src/mini_infer/engine/model_runner.py](../../src/mini_infer/engine/model_runner.py)
  — current loader and forward path.
- [src/mini_infer/engine/attention_patch.py](../../src/mini_infer/engine/attention_patch.py)
  — current patching strategy (to be replaced).
- [src/mini_infer/cache/block_pool.py](../../src/mini_infer/cache/block_pool.py)
  — KV layout assumptions; will need a per-layer descriptor.
- vLLM's [model registry](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models)
  — reference for the registry + per-model file pattern.
- SGLang's [models](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models)
  — alternative reference, similar structure.
