# Architecture walkthroughs

Per-architecture line-by-line correspondence between research paper,
upstream reference implementation, and our from-scratch port.

The direct realization of mini-infer's research-paper-engine niche
(see [roadmap-2026.md](../plans/roadmap-2026.md)).

## Available walkthroughs

| Family | Paper | Reference | Our walkthrough |
|---|---|---|---|
| DeepSeek-V4 | DeepSeek-V4 technical report | [DeepSeek-V4-Pro/inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference) | [deepseek-v4.md](deepseek-v4.md) |
| DeepSeek-V2 / V3 / Kimi-K2 (MLA) | DeepSeek-V2 technical report | HF transformers `modeling_deepseek_v2.py` | [deepseek-v2-mla.md](deepseek-v2-mla.md) |
| Gemma 4 (text-only) | Gemma 4 release notes | HF transformers `modeling_gemma3.py` (text path) | [gemma-4.md](gemma-4.md) |
| Mixtral 8x7B / 8x22B | Mixtral of Experts paper | HF transformers `modeling_mixtral.py` | [mixtral.md](mixtral.md) |
| Qwen3 | Qwen3 technical report | HF transformers `modeling_qwen3.py` | [qwen3.md](qwen3.md) |
| Gemma 3 (text-only) | Gemma 3 technical report | HF transformers `modeling_gemma3.py` | [gemma-3.md](gemma-3.md) |

## Pending walkthroughs

All currently-registered architectures have walkthroughs. Future
additions trigger when a new paper architecture is implemented per
the [roadmap-2026 §Paper-watch-list](../plans/roadmap-2026.md).

Reading order suggestion if you're new to the codebase:

1. **[Qwen3](qwen3.md)** — smallest delta over the Llama baseline.
   Read this first to see what a minimal family file looks like.
2. **[Mixtral](mixtral.md)** — the simplest MoE. Adds top-k sparse
   FFN on top of the Llama baseline.
3. **[Gemma 3](gemma-3.md)** — sandwich norms, offset RMSNorm, dual
   RoPE, sliding+global alternation. Most family-specific quirks
   without changing the cache shape.
4. **[Gemma 4](gemma-4.md)** — adds heterogeneous-KV per layer-type
   (different `(num_kv_heads, head_dim)` between sliding and full
   layers) on top of Gemma 3's machinery.
5. **[DeepSeek-V2 (MLA)](deepseek-v2-mla.md)** — the dense material
   for the DeepSeek family. Two-stream KV cache (`kv_latent` +
   `k_rope`), low-rank Q, interleaved RoPE, asymmetric Q/K vs V
   head_dim, heterogeneous FFN (dense + MoE with shared experts).
6. **[DeepSeek-V4](deepseek-v4.md)** — the headline. Hybrid
   per-layer attention (SWA/CSA/HCA), Lightning Indexer, attention
   sink, grouped output projection, hash-routed MoE,
   Hyper-Connections. Read after V2 since V4 extends MLA.

## Walkthrough doc template

Every walkthrough should follow the structure of
[deepseek-v4.md](deepseek-v4.md):

1. **TL;DR**: one-paragraph summary of what makes this family novel.
2. **The map**: table of paper-primitive → reference file:line → our
   code file:line. No test references: tests churn faster than the
   architecture and a static doc that names test files will go stale
   on every refactor.
3. **Per-primitive walkthroughs**: paper section number → reference
   quote → our code quote, with commentary on what the math does and
   where we diverged.
4. **Decoder layer assembly**: how the primitives compose per block.
5. **Cache structure**: any family-specific cache contract beyond the
   standard paged KV.
6. **Validation contract**: how bit-parity is enforced for this
   family (cosine-sim threshold, dtype, what's exercised). Point at
   `tests/unit/` for the directory; do not enumerate test files.
7. **Where we diverged + why**: pointers to the ADR for design decisions.
8. **Pointers**: reference path, paper path, ADR, our model class,
   per-primitive blocks, loader.
9. **What's still open**: any remaining gap (live hardware validation,
   kernel ports, follow-ups).

## Where to start

If you're reading the V4 paper: open [deepseek-v4.md](deepseek-v4.md)
in one window and the paper in another. The walkthrough's section
ordering matches the paper.

If you want to add a walkthrough: pick one from the pending list,
copy the V4 doc's structure, fill in the cells.
