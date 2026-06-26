# ADR-021: GLM-5.2 (GlmMoeDsa) from-scratch port

Date: 2026-06-19
Status: Accepted

## Context

GLM-5.2 (z.ai, released June 2026) ships as `GlmMoeDsaForCausalLM`
(`model_type: glm_moe_dsa`). Stripped of branding it is **DeepSeek-V3.2's
architecture plus one GLM-original optimization (IndexShare)**:

- **MLA attention** (q_lora 2048, kv_lora 512, qk_nope 192 / qk_rope 64,
  v_head 256, 64 heads) - identical family to our existing `MLAAttention`.
- **DeepSeek Sparse Attention (DSA)**: a Lightning Indexer scores the query
  against past tokens, selects `index_topk` (2048) of them, and the main MLA
  attends only to those (the rest masked to `-inf`).
- **MoE** (DeepSeek-V3 style): 256 routed + 1 shared expert, top-8, a sigmoid
  gate with an aux-loss-free selection bias (`noaux_tc`) and grouped top-k.
- **IndexShare** (the GLM-original bit): the indexer runs only on `"full"`
  layers; the next `"shared"` layers reuse that selection (`index_topk_freq`
  layers share one indexer pass), cutting indexer FLOPs at long context.
- RoPE non-interleaved (theta 8e6, no YaRN), RMSNorm, SwiGLU dense layers
  (first 3), MoE after. A single MTP draft layer for speculative decoding.

The project's charter is to be the implementation venue for newly published
architectures, bit-parity-validated against the upstream reference. The
decisive enabler here: the pinned `transformers` (5.6.2) already ships
`modeling_glm_moe_dsa`, so every component can be validated on CPU against the
official implementation without a checkpoint download.

The one hard constraint: GLM-5.2 ships **only** at ~753B parameters (no Air/Flash
variant exists, and the small GLM-4.x models are a different architecture). So
running the trained weights is out of reach for modest hardware. Architecture
correctness therefore comes from synthetic-config bit-parity, not a checkpoint.

## Decision

Port GLM-5.2 end-to-end through the **existing PagedKVCache engine** (the same
path Qwen2/Llama/DeepSeek-V2 use), bit-parity-validated against the HF
`glm_moe_dsa` reference at every level: RoPE, indexer selection, MoE routing,
full-model logits, and multi-step greedy decode. Reuse the DeepSeek-V2
machinery; add only the three GLM-specific pieces.

What shipped:

1. **`MLAAttention` extensions** (`models/blocks/mla.py`), default-off so
   V2/V3/Kimi stay bit-identical:
   - `use_interleaved_rope` flag. GLM uses non-interleaved (NeoX/Llama) RoPE
     for both the main attention and the indexer. (The HF docstring says the
     main attention is "interleaved", but the actual code applies NeoX
     `apply_rotary_pos_emb`; we follow the code, which is the parity oracle.)
   - An optional `indexer` hook + `dsa_topk` argument threaded into the SDPA,
     and `compute_dsa_topk` for cache-aware selection.

2. **`GlmDsaIndexer`** (`models/blocks/glm_dsa_indexer.py`): the DSA top-k
   selector over raw tokens. Per-head Q from the shared q_lora latent, a single
   shared LayerNorm'd key, ReLU'd dot-product scaled and weighted per head,
   causal-masked top-k. A no-cache `forward` (prefill / block tests) and a
   cache-aware `forward_cached` that writes per-token keys to a PagedKVCache
   `index_k` stream and scores against the full history (decode).

3. **DSA sparse mask** (`cache/mla_attention.py`): an optional `dsa_topk` adds a
   per-query `-inf`-except-selected mask before the causal mask, mirroring HF's
   `index_mask + causal_mask`. `None` keeps the dense V2/V3 behavior.

4. **`GlmNoAuxTcGate` + `GlmMoeFFN`** (`models/blocks/glm_moe_gate.py`): the
   sigmoid `noaux_tc` router (selection bias is added for the choice only;
   expert weighting uses the unbiased sigmoid scores; grouped top-k) plus a thin
   FFN reusing `MixtralExpert` and the expert-parallel dispatch shape.

5. **`GlmMoeDsaForCausalLM`** (`models/glm_moe_dsa.py`): config + `from_hf`,
   per-layer dense-vs-MoE dispatch, IndexShare threading through the decoder
   stack, heterogeneous `per_layer_streams` (full layers carry an `index_k`
   stream, ordered first so layer 0 triggers block allocation), and
   `load_weights` that block-FP8-dequantizes the published checkpoint (e4m3 +
   `weight_scale_inv`, ceil-aware for partial-block shapes like the 576-row
   `kv_a_proj`), handles both the stacked (HF in-memory) and per-expert
   (published checkpoint) routed-expert layouts, applies the global->local
   expert-parallel remap so each rank loads only its expert slice under TP,
   renames the shared expert, and drops the MTP layer.

6. **FP8-resident experts** (`blocks/fp8_expert.py` + `GlmMoeFFN(expert_dtype=...)`):
   `Fp8Expert` stores each routed expert's `w1/w2/w3` as e4m3 + a ceil-block scale
   and dequantizes per matmul, so the FP8 checkpoint fits 8xH200 without the
   ~1.5 TB BF16 blow-up. `from_hf` sets `expert_dtype="fp8"` from the checkpoint's
   `quantization_config`; attention / dense / shared / indexer still dequantize to
   BF16. The ceil-aware block-FP8 dequant lives in `quant/nvfp4.py`, shared by the
   loader and `Fp8Expert`.

## Alternatives Considered

**StateCache (V4-style) instead of PagedKVCache.** V4's `StateCache` is built
around compressed history + SWA windows, which DSA does not have (it keeps full
K/V and just masks). Reject: it would fight the abstraction, and the HF
reference itself keeps full K/V and applies an additive mask. PagedKVCache is
the natural fit and gives GLM the existing scheduler/golden plumbing for free.

**Reuse the V4 `LightningIndexer`.** V4's indexer is compressor-coupled (it
scores compressed blocks via a `TokenLevelCompressor` with partial RoPE). GLM's
DSA scores raw tokens with full non-interleaved RoPE. Reject: different enough
that a dedicated, simpler block is clearer than overloading V4's.

**Overload `MoEFFN` for the gate.** `MoEFFN` is softmax-routed; GLM needs
sigmoid + `noaux_tc` + grouped top-k. Reject: a small dedicated gate keeps the
widely-shared Mixtral block clean. The expert dispatch loop is still reused.

**Cache indexer keys in a separate per-request buffer (like HF's `_cached_keys`).**
A standalone buffer does not fit continuous batching (multiple ragged requests).
Reject: a PagedKVCache `index_k` stream pages per request idiomatically and
reuses the existing append/materialize machinery. Ordering it first on full
layers makes layer 0 the allocation trigger, before any slot is written.

**Model the MTP draft layer.** It is a speculative-decoding accelerator, not
required for greedy correctness, and the HF model's own forward never runs it.
Reject: drop its checkpoint keys at load; add it only if/when spec-decode for
GLM is in scope.

## Consequences

**Positive:**

- mini-infer now loads, parity-validates, and **generates correct text** for
  GLM-5.2, all on CPU with no checkpoint download. The synthetic-config
  bit-parity exercises 100% of the architecture (MLA, DSA, IndexShare, noaux_tc
  MoE), which is the actual correctness gate.
- `MLAAttention` now carries an optional DSA hook (indexer + sparse mask + a
  cache-aware `index_k` stream) that any future DeepSeek-V3.2-family model can
  reuse. V2/V3/Kimi are untouched by default arguments.
- The port reused the entire DeepSeek-V2 path (config shape, decoder skeleton,
  MLA cache contract, weight-load structure), so the GLM-specific surface is
  one model file plus two block files.

**Negative:**

- Running the trained 753B weights end-to-end is out of scope for modest
  hardware (FP8 weights alone are ~753 GB, needing a multi-GPU node). The port
  is validated against the HF reference on synthetic configs; a real-checkpoint
  smoke is deferred and gated separately.
- Single-GPU CUDA (exact fp32 parity + bf16 generation) and 2-GPU NCCL tensor
  parallelism, including expert-parallel MoE (logits identical across ranks,
  exact vs the world_size=1 GPU reference), are validated on L4s. The only
  remaining unrun item is the 753B checkpoint itself (needs a multi-GPU node,
  beyond budget). Caveat surfaced in testing: the MoE expert dispatch has a
  GPU-vs-CPU fp32 gap (index_add_ / scaled matmuls), so TP correctness must be
  judged against a same-device reference, not CPU.
- The published checkpoint is **per-expert block-FP8** (~755 GB; confirmed via
  the index with `scripts/inspect_glm_safetensors.py`) and `load_weights`
  ingests it (ceil-aware dequant + per-expert + EP remap). To fit a single node,
  `expert_dtype="fp8"` keeps the routed experts (~96% of the weights)
  e4m3-resident via `Fp8Expert` (dequant per-matmul, like V4's `FP4Expert`) while
  attention / dense / shared / indexer dequantize to BF16, landing ~785 GB on
  8xH200; `from_hf` selects this automatically from the checkpoint's
  `quantization_config`. `scripts/modal_glm_stage_weights.py` stages the weights
  on a CPU-only function. The only thing left between here and generated text is
  the funded multi-GPU run itself (~$36/hr on 8xH200, beyond the current budget).

**Trade-offs:**

- On "full" indexer layers the q_lora latent is computed twice per step (once in
  `compute_dsa_topk`, once in the attention forward). Correctness-first; a
  shared-latent pass is a perf follow-up.
- The `index_k` stream stores RoPE'd keys at bf16 in production (same as the
  main k_rope stream), so decode introduces the same cache-precision delta the
  main attention already has. The CPU parity tests run fp32, so they are exact.

## Validation

All tests gate on the in-venv HF `glm_moe_dsa` reference via
`pytest.importorskip`. Synthetic tiny configs, CPU, fp32.

| Level | Test | Result |
|---|---|---|
| Main-attention RoPE | `test_glm_mla_rope_parity` | cosine > 0.999, allclose 1e-4 (+ V2 regression) |
| DSA indexer top-k | `test_glm_dsa_indexer_parity` | selected key set matches HF (selective + all-pass) |
| MLA + DSA attention | `test_glm_mla_dsa_attention_parity` | cosine > 0.999, allclose 1e-4 |
| noaux_tc gate + MoE | `test_glm_moe_gate_parity` | routing + FFN match (n_group 1 and 2) |
| Model assembly + IndexShare | `test_models_glm_moe_dsa` | structure, indexer reuse `[1,0,0,0]` / `[1,0,1,0]` |
| Full-model logits | `test_full_model_parity_vs_hf` | logits allclose 1e-3, greedy argmax identical |
| Greedy decode | `test_greedy_decode_parity_vs_hf` | tokens identical to HF over 6 steps |
| Batched decode | `test_batched_decode_matches_hf` | two ragged prompts, tokens match HF per request |
| TP indexer selection | `test_glm_dsa_indexer_tp_parity` | world_size=2 top-k identical across ranks + matches ws=1 |
| Prefix cache (index_k) | `test_glm_dsa_prefix_cache` | index_k publishes/reuses; prefix-hit logits match full prefill |
| Real GPU (NVIDIA L4) | `scripts/modal_glm_smoke.py` | fp32 parity exact (cos 1.0, max abs diff 0); fp32 decode tokens match HF; bf16 generation finite |
| TP model (CPU gloo, ws=2) | `test_glm_moe_tp_parity` | full MoE model: ranks identical, matches ws=1 (expert-parallel load) |
| Real GPU TP (2x L4, NCCL) | `scripts/modal_glm_tp_smoke.py` | MoE config: ranks identical (diff 0); exact vs ws=1 GPU ref (cos 1.0) |
| Block-FP8 + per-expert load | `test_glm_fp8_load` | ceil-aware dequant on partial blocks; per-expert BF16 load exact; FP8 round-trip recovers (cos > 0.97) |
| FP8-resident experts | `test_glm_fp8_load` | Fp8Expert == dequantized MixtralExpert; fp8-resident model logits == bf16-dequant (experts stay e4m3) |

## References

- HF reference: `transformers.models.glm_moe_dsa.modeling_glm_moe_dsa`
  (transformers 5.6.2, the pinned project version).
- Model card / config: `huggingface.co/zai-org/GLM-5.2`
  (`GlmMoeDsaForCausalLM`, 78 layers, 256 routed + 1 shared experts, 1M context).
- DeepSeek-V2 template this port reused: `src/mini_infer/models/deepseek_v2.py`
  and [ADR-014](ADR-014-deepseek-v4-hybrid-attention.md) (MLA + sparse-attention
  precedent).
- Real-GPU smokes: `scripts/modal_glm_smoke.py` (single L4) and
  `scripts/modal_glm_tp_smoke.py` (2x L4 NCCL TP), both validated PASS.
- Real-checkpoint readiness: `scripts/inspect_glm_safetensors.py` (confirmed the
  per-expert block-FP8 format from the index, $0) and
  `scripts/modal_glm_stage_weights.py` (CPU-only Volume staging for a funded run).
- Implementation on branch `glm-moe-dsa`.
