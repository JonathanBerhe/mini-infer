# Plan: DeepSeek-V4 hybrid attention support

Source: `docs/papers/deepseek-v4.pdf` §2.3 (Hybrid Attention) and §3.6
(Inference Framework). PDF is gitignored — read locally.

## Context

DeepSeek-V4 (released April 2026) is a 1.6T / 49B-active MoE family
with 1M-token context. The flagship efficiency claim is that at 1M
tokens, V4-Pro uses **27% of the FLOPs and 10% of the KV cache** of
V3.2. That win comes almost entirely from the new attention
architecture in §2.3: an interleaved hybrid of two compressed
attention modes (CSA and HCA) plus a sliding window branch.

Supporting V4 end-to-end (loader + MoE + multi-GPU + 1M context) is
out of scope for mini-infer — that's a quarter of work and most of
it is plumbing rather than the technical novelty. The portfolio-worthy
piece is the **attention contribution itself**. This plan covers
implementing CSA and HCA at small scale, validated against the
DeepSeek-AI reference at
`https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`.

## What V4's hybrid attention actually is

Two interleaved attention variants. Layers alternate between them:

### CSA — Compressed Sparse Attention

Three branches that all feed a final shared-KV multi-query attention:

1. **Compressed branch**. A learnable token-level compressor maps every
   `m` tokens into one compressed KV entry. Sequence length drops by
   `1/m`. Compression weights are softmax-normalized across `2m` raw
   entries with a learnable positional bias (formulas 11–12).
2. **Lightning Indexer + top-k**. A small low-rank attention computes
   per-(query, compressed-block) index scores; a top-k selector picks
   which compressed entries survive into core attention (formulas
   13–17). The indexer has its OWN small KV cache (different head dim,
   FP4 internal compute).
3. **Sliding window**. The most recent `n_win` uncompressed KV entries
   bypass compression entirely, for fine-grained local context.

Final core attention is MQA over `[selected compressed KV] + [sliding
window KV] + attention_sink`. Output is then split into `g` groups,
projected per-group, concatenated (Grouped Output Projection).

### HCA — Heavily Compressed Attention

Same machinery as CSA's compressed branch, but with a much larger
compression ratio `m' >> m` and no sparse selection — every compressed
entry is attended to. Plus the same sliding-window branch and grouped
output projection. **No Lightning Indexer, no top-k.**

### Shared mechanics

Both CSA and HCA use:
- **Mixed precision KV**: BF16 on the partial-RoPE dims (last 64), FP8
  on the rest. Cuts KV size ~half vs pure BF16.
- **Partial RoPE**: only on the last 64 dims of queries / KV / output.
  Output applies RoPE with position `−i` to recover relative encoding
  across compressed entries.
- **Attention sink**: a learnable per-head sink logit added to the
  softmax denominator (OpenAI 2025 trick).
- **MQA**: one shared KV head across all query heads inside each
  attention. Concretely small.

### Why "hybrid"

Layers alternate CSA / HCA. Different layers, different compression
ratios, different KV sizes. This is what breaks vanilla PagedAttention:
the per-layer KV layout is no longer homogeneous.

## What §3.6 says about inference

The paper explicitly calls out the conflict with PagedAttention and
proposes a two-tier cache layout (Figure 6, p. 23):

### State Cache (per-request, fixed-size pool)

Holds:
- Sliding-window KV (always uncompressed, every layer).
- "Tail tokens" — uncompressed KV for tokens not yet ready to be
  compressed (waiting for the next `m` or `m'` boundary).

These are sequence-local state, allocated once per request as a fixed
chunk.

### KV Cache (paged, hybrid block layout)

Each block holds `lcm(m, m')` original tokens worth of compressed
entries. A single block contains, mixed across layers:
- CSA Indexer KV (small head dim, FP4 inner)
- CSA Main KV (compressed at `m`)
- HCA KV (compressed at `m'`)

Sparse-attention kernel co-design: the CSA top-k path requires the
sparse kernel to operate on aligned blocks, so the block size MUST be
a multiple of `lcm(m, m')`.

### On-disk KV (§3.6.2)

Orthogonal optimization: persist the compressed KV cache to disk for
prefix-sharing across requests / restarts. SWA KV is ~8x larger than
compressed, so they offer three storage strategies trading disk for
recompute: full / periodic checkpoint / zero. "Zero SWA caching" is
the leanest — recompute the last `n_win × L` SWA entries from cached
CSA/HCA on a hit.

## Impact on mini-infer

| V4 requirement | Current state | Delta |
|---|---|---|
| Per-layer heterogeneous KV layout | `BlockPool` assumes one KV layout for all layers | Major: per-layer block schema |
| Compressed KV with learnable compressor | Fixed-shape KV per token | New: token-level compressor module + state buffer for incomplete blocks |
| Lightning Indexer (own KV cache) | One KV cache per layer | New: a second small KV cache per CSA layer |
| Top-k sparse attention | Dense paged FA | New attention kernel (FlashInfer roadmap?) |
| Sliding-window KV pool per request | Single paged pool | New: state-cache pool |
| Mixed BF16 / FP8 KV inside one tensor | bf16 / fp8 / nvfp4 all-or-nothing | New: per-layer split-precision |
| MoE FFN | Dense FFN only | New (separate from this plan; flag below) |

## Scope for this plan

**In scope:**
- CSA forward (compressor + indexer + top-k + sliding + sink + grouped output)
- HCA forward (compressor + sliding + sink + grouped output)
- A test harness that validates each at cosine-sim > 0.999 against
  the DeepSeek-AI reference inference code.
- A small hand-built backbone (3–4 layers, dense FFN) to wire CSA and
  HCA layers into something runnable end-to-end on a single GPU.

**Out of scope (flagged for user decision):**
- Loading actual V4-Pro / V4-Flash weights. Both need multi-GPU and
  V4-Pro at 1.6T won't fit anywhere we benchmark. We can implement
  the attention without loading the real weights.
- MoE FFN. Required to load real V4 weights. Substantial separate
  effort; would be its own plan.
- The hybrid block-pool layout from §3.6.1. Required only when CSA
  and HCA layers share a pool. Plan defers this to Stage 3.
- On-disk KV (§3.6.2). Pure storage layer, orthogonal to compute,
  lowest priority.
- 1M-token context. The attention works at any length; we'd just
  need to verify nothing blows up at scale. Probably defer to a
  follow-up bench.

## Staged shipping

### Stage 1: HCA only (simpler — no sparse selection)

Why first: HCA is CSA minus the Lightning Indexer + top-k, which is
the hardest piece. Validates the compressor, sliding window, attention
sink, partial RoPE, grouped output projection, and the
state-cache-for-tail-tokens pattern. All layers HCA, so the block
layout stays homogeneous.

**Files (rough):**
- NEW `src/mini_infer/models/deepseek_v4/hca.py` — `HCALayer` module.
- NEW `src/mini_infer/cache/state_cache.py` — fixed-size per-request
  pool for SWA KV + tail tokens.
- EDIT `src/mini_infer/cache/block_pool.py` — extend for compressed
  KV at compression ratio `m'`.
- NEW `tests/unit/test_hca.py` — cosine-sim parity vs DeepSeek-AI's
  HCA reference on random inputs.

### Stage 2: CSA (adds the sparse selection)

Why second: builds on Stage 1's machinery. Adds the Lightning Indexer
(its own small KV cache) and the top-k selector. The sparse attention
itself can either be built from primitives (gather + dense MQA) or
deferred to FlashInfer if/when it lands a top-k variant. Use the
gather-based path first; it's correct and easy to validate.

**Files:**
- NEW `src/mini_infer/models/deepseek_v4/csa.py` — `CSALayer`,
  `LightningIndexer`, `TopKSelector`.
- EDIT `state_cache.py` — extend for the indexer's separate KV pool.
- NEW `tests/unit/test_csa.py` — cosine-sim parity vs reference.

### Stage 3: Hybrid (CSA + HCA interleaved)

Why third: the heterogeneous-layer wiring is the hard part of §3.6.1.
Once Stage 1 and Stage 2 each work in isolation, the integration is
about block layout, not new compute. Block size = `lcm(m, m')`. Per-layer
schema lives on the model config; `BlockPool` learns to honor it.

**Files:**
- EDIT `block_pool.py` — accept a per-layer schema list rather than
  a uniform layout.
- EDIT `paged_kv_cache.py` — per-layer slot allocation respects the
  schema.
- NEW `tests/unit/test_v4_hybrid_layout.py` — verifies a 4-layer
  CSA/HCA/CSA/HCA backbone runs with valid KV state across all layers.

### Stage 4 (optional): On-disk KV

Pure storage. Implement "Zero SWA Caching" only — lowest disk, highest
recompute. Useful only when prefix sharing matters; punt unless we
have a concrete workload that needs it.

## Open questions (decide before Stage 1)

1. **Reference for parity tests.** The paper points at
   `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
   for the open-source implementation. Plan needs that code as a
   parity oracle. Action: clone it locally, confirm it's runnable on
   a small synthetic config without loading the full weights.
2. **Compression ratios `m` and `m'`.** Paper uses specific values for
   V4-Pro / V4-Flash. We can either match those (so config files
   align) or pick smaller values (`m=4`, `m'=16`?) that keep tests
   fast. Recommend match-the-paper for legitimacy.
3. **Sparse top-k attention path.** Build from gather + MQA (always
   correct, slower) or wait for a vendor kernel. Recommend gather
   for Stage 2; revisit when there's a profiled bottleneck.
4. **Hand-built backbone for end-to-end runs.** Use HF Transformers
   to build a minimal `LlamaForCausalLM`-shaped model with 3–4
   layers and swap in our CSA/HCA modules — that gets us a runnable
   forward pass without writing a tokenizer or loader. Or: standalone
   torch module, no HF dep. Recommend HF for minimal new surface.
5. **MoE.** Out of scope for this plan, but the user should be aware:
   loading any actual V4 checkpoint requires MoE. Without it, we
   demonstrate the attention only.

## Validation

For each stage:
- Cosine-sim > 0.999 vs DeepSeek-AI's reference `inference/` code on
  random inputs.
- For Stage 3, end-to-end greedy parity on a fixed prompt with a
  hand-built tiny model (deterministic — same weights, same RNG seed,
  same output tokens both sides).
- No GPU-only parts of CSA/HCA require CUDA at small scale; CPU
  parity tests should be feasible. CUDA tests gated behind
  `@pytest.mark.requires_cuda` for any FlashInfer integration.

## Acceptance for Stage 1 (the first commit)

- HCA forward implemented in pure PyTorch (no custom kernels).
- Cosine-sim > 0.999 vs reference HCA on a random `(B=2, T=512,
  d=512, m'=16, n_win=64)` config.
- All three branches (compression, sliding window, sink) covered in
  unit tests.
- Runs on CPU and CUDA both. No flash-attn / FlashInfer dependency
  yet.

## Critical files to read before coding

- `src/mini_infer/cache/block_pool.py` — how the homogeneous KV
  layout is currently wired.
- `src/mini_infer/cache/paged_kv_cache.py` — slot management
  abstraction.
- `src/mini_infer/cache/packed_attention.py` — attention dispatcher.
- DeepSeek-V4 paper §2.3 (pp. 9–14) — formulas for CSA and HCA.
- DeepSeek-V4 paper §3.6 (pp. 22–24) — hybrid block layout.
- `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference` —
  open-source reference (clone locally).
