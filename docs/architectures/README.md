# Architecture walkthroughs

Per-architecture line-by-line correspondence between research paper,
upstream reference implementation, and our from-scratch port.

The direct realization of mini-infer's research-paper-engine niche.

## Where the code is

Each model family has one folder for its code, one for its tests, and one
for its scripts. A file that two or more families use is in
`src/mini_infer/models/blocks/`, and its tests are in
`tests/unit/models/blocks/`.

| Family | Code | Tests | Scripts |
|---|---|---|---|
| Llama | [`models/llama/`](../../src/mini_infer/models/llama/) | [`models/llama/`](../../tests/unit/models/llama/) | [`llama/`](../../scripts/llama/) |
| Mistral, Mixtral | [`models/mistral/`](../../src/mini_infer/models/mistral/) | [`models/mistral/`](../../tests/unit/models/mistral/) | none |
| Qwen2, Qwen3, Qwen3.5 | [`models/qwen/`](../../src/mini_infer/models/qwen/) | [`models/qwen/`](../../tests/unit/models/qwen/) | [`qwen/`](../../scripts/qwen/) |
| Gemma 3, Gemma 4 | [`models/gemma/`](../../src/mini_infer/models/gemma/) | [`models/gemma/`](../../tests/unit/models/gemma/) | [`gemma/`](../../scripts/gemma/) |
| DeepSeek-V2, V3 | [`models/deepseek_v2/`](../../src/mini_infer/models/deepseek_v2/) | [`models/deepseek_v2/`](../../tests/unit/models/deepseek_v2/) | [`deepseek_v2/`](../../scripts/deepseek_v2/) |
| DeepSeek-V4 | [`models/deepseek_v4/`](../../src/mini_infer/models/deepseek_v4/) | [`models/deepseek_v4/`](../../tests/unit/models/deepseek_v4/) | [`deepseek_v4/`](../../scripts/deepseek_v4/) |
| GLM-5.2, GLM-5.3 | [`models/glm/`](../../src/mini_infer/models/glm/) | [`models/glm/`](../../tests/unit/models/glm/) | [`glm/`](../../scripts/glm/) |
| MiniMax-M3 | [`models/minimax_m3/`](../../src/mini_infer/models/minimax_m3/) | [`models/minimax_m3/`](../../tests/unit/models/minimax_m3/) | [`minimax_m3/`](../../scripts/minimax_m3/) |
| Inkling | [`models/inkling/`](../../src/mini_infer/models/inkling/) | [`models/inkling/`](../../tests/unit/models/inkling/) | [`inkling/`](../../scripts/inkling/) |
| Kimi Linear, Kimi K3 | [`models/kimi_linear/`](../../src/mini_infer/models/kimi_linear/) | [`models/kimi_linear/`](../../tests/unit/models/kimi_linear/) | [`kimi_linear/`](../../scripts/kimi_linear/) |
| Nemotron-H | [`models/nemotron_h/`](../../src/mini_infer/models/nemotron_h/) | [`models/nemotron_h/`](../../tests/unit/models/nemotron_h/) | [`nemotron_h/`](../../scripts/nemotron_h/) |
| Shared blocks | [`models/blocks/`](../../src/mini_infer/models/blocks/) | [`models/blocks/`](../../tests/unit/models/blocks/) | none |

## Available walkthroughs

| Family | Paper | Reference | Our walkthrough |
|---|---|---|---|
| Nemotron-H (Nemotron 3.5 Lightning, Nemotron 3 Nano; Mamba-2 hybrid, text-only) | Mamba-2 / SSD paper (arXiv 2405.21060) | HF transformers `modeling_nemotron_h.py` | [nemotron-h.md](nemotron-h.md) |
| Qwen3.5 / Qwen3.6 / Qwen3.8 (Gated DeltaNet hybrid, text-only) | Gated DeltaNet paper (arXiv 2412.06464) | HF transformers `modeling_qwen3_5.py` (text path) | [qwen3-5.md](qwen3-5.md) |
| Kimi Linear 48B-A3B (KDA + NoPE-MLA hybrid) | Kimi Linear paper (arXiv 2510.26692) | Checkpoint `modeling_kimi.py` (trust_remote_code, vendored) | [kimi-linear.md](kimi-linear.md) |
| Kimi K3 / K3.1 Pro (SiTU, gated MLA, AttnRes, latent MoE, MXFP4; text-only) | Kimi K3 tech report | Checkpoint `modeling_kimi_linear.py` (trust_remote_code, vendored) | [kimi-k3.md](kimi-k3.md) |
| Inkling / Inkling-Small | [Release post](https://thinkingmachines.ai/news/introducing-inkling/) | HF transformers `modeling_inkling.py` (text path) | [inkling.md](inkling.md) |
| MiniMax-M3 (MSA) | MSA report (arXiv 2606.13392) | HF transformers `modeling_minimax_m3_vl.py` (text path) | [minimax-m3.md](minimax-m3.md) |
| GLM-5.2 / GLM-5.3 (GlmMoeDsa: MLA + DSA + IndexShare) | No GLM paper. DSA from DeepSeek-V3.2, IndexShare from arXiv 2603.12201 | HF transformers `modeling_glm_moe_dsa.py` | [glm-moe-dsa.md](glm-moe-dsa.md) |
| DeepSeek-V4 | DeepSeek-V4 technical report | [DeepSeek-V4-Pro/inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference) | [deepseek-v4.md](deepseek-v4.md) |
| DeepSeek-V2 / V3 / Kimi-K2 (MLA) | DeepSeek-V2 technical report | HF transformers `modeling_deepseek_v2.py` | [deepseek-v2-mla.md](deepseek-v2-mla.md) |
| Gemma 4 (text-only) | Gemma 4 release notes | HF transformers `modeling_gemma3.py` (text path) | [gemma-4.md](gemma-4.md) |
| Mixtral 8x7B / 8x22B | Mixtral of Experts paper | HF transformers `modeling_mixtral.py` | [mixtral.md](mixtral.md) |
| Qwen3 | Qwen3 technical report | HF transformers `modeling_qwen3.py` | [qwen3.md](qwen3.md) |
| Gemma 3 (text-only) | Gemma 3 technical report | HF transformers `modeling_gemma3.py` | [gemma-3.md](gemma-3.md) |
| Llama 2 / 3 / 3.1-3.3 / SmolLM2 / TinyLlama | Llama 2 paper + Llama 3 technical report | HF transformers `modeling_llama.py` | [llama.md](llama.md) |
| Qwen2 / Qwen2.5 | Qwen2 technical report | HF transformers `modeling_qwen2.py` | [qwen2.md](qwen2.md) |
| Mistral 7B / Small / Medium / Large / Codestral | Mistral 7B paper | HF transformers `modeling_mistral.py` | [mistral.md](mistral.md) |

## Pending walkthroughs

None. Every ported family has a walkthrough.

Future additions follow when a new paper architecture is implemented.

Reading order suggestion if you're new to the codebase:

1. **[Llama](llama.md)** — the canonical Llama-shape baseline. Read
   this first to learn the shared blocks (`GroupedQueryAttention`,
   `TransformerBlock`, `SwiGLU`, `RMSNorm`, `RotaryEmbedding`) that
   every other family reuses.
2. **[Mistral](mistral.md)** — the shortest delta on Llama: a 30-line
   type-distinct alias for the HF architecture key, plus an honest
   non-implementation of v0.1/v0.2 sliding-window attention.
3. **[Qwen2](qwen2.md)** — Llama-shape with `qkv_bias=True`. Drives
   most of the CI as the day-to-day smoke (Qwen2.5-0.5B-Instruct).
4. **[Qwen3](qwen3.md)** — smallest delta over Qwen2 / Llama. Adds
   per-head Q/K norm and drops the biases again.
5. **[Mixtral](mixtral.md)** — the simplest MoE. Adds top-k sparse
   FFN on top of the Llama baseline.
6. **[Gemma 3](gemma-3.md)** — sandwich norms, offset RMSNorm, dual
   RoPE, sliding+global alternation. Most family-specific quirks
   without changing the cache shape.
7. **[Gemma 4](gemma-4.md)** — adds heterogeneous-KV per layer-type
   (different `(num_kv_heads, head_dim)` between sliding and full
   layers) on top of Gemma 3's machinery.
8. **[DeepSeek-V2 (MLA)](deepseek-v2-mla.md)** — the dense material
   for the DeepSeek family. Two-stream KV cache (`kv_latent` +
   `k_rope`), low-rank Q, interleaved RoPE, asymmetric Q/K vs V
   head_dim, heterogeneous FFN (dense + MoE with shared experts).
9. **[DeepSeek-V4](deepseek-v4.md)** — the headline. Hybrid
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
