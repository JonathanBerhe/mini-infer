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

## Pending walkthroughs

The order tracks the roadmap's "medium-term" priorities. Each one is
~1-3 days of careful work mapping paper → reference → our code.

- **Qwen3** — per-head Q/K norm + tied embeddings (smallest delta over
  the Llama baseline, useful for "what does the family-specific block
  do" reading).
- **Gemma 3** — sliding-window + global alternating attention, dual
  RoPE, sandwich norms, GemmaRMSNorm (`(1+w)*x`), GeGLU
  (`gelu_pytorch_tanh`), embed scaling, Q/K norm.

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
