# DeepSeek-V2 / V3 / Kimi-K2 walkthrough (MLA)

A line-by-line correspondence between the DeepSeek-V2 paper, the
HuggingFace transformers reference implementation
(`transformers/models/deepseek_v2/modeling_deepseek_v2.py`), and
mini-infer's from-scratch MLA port.

This is the realization of mini-infer's research-paper-engine niche
for **the entire DeepSeek-V2 / V3 / Kimi-K2 family**, which share the
same MLA shape. Reading the walkthrough once covers all four models:
the same `MLAAttention` + `_DeepseekV2DecoderLayer` + `DeepseekV2ForCausalLM`
serve V2-Lite (16B), V2 (236B), V3 (671B), and Kimi-K2 (1T) — they
differ in scale + a handful of config bits (`q_lora_rank`,
`n_routed_experts`, `qk_nope_head_dim`), not in architecture.

> **Audience**: someone with the V2 paper open who wants to follow the
> MLA + heterogeneous-FFN implementation. Read this before the V4 doc
> if you're new to DeepSeek; V4's HCA/CSA extend MLA's per-stream KV
> idea.

## TL;DR

DeepSeek-V2's signature contribution is **Multi-head Latent Attention
(MLA)**: cache TWO compressed streams per layer instead of per-head K
and V tensors. Standard MHA stores `(num_heads, head_dim) * 2` per
token; MLA stores `(1, kv_lora_rank=512) + (1, qk_rope_head_dim=64)`
per token. ~7x smaller KV cache at V2-Lite scale, larger savings at
V3 / Kimi-K2.

The five primitives that distinguish V2 from a vanilla Llama-shape
model:

1. **MLA** with low-rank Q (`q_a_proj → q_a_layernorm → q_b_proj` for
   V2/V3, direct `q_proj` for V2-Lite) and decoupled `kv_latent` +
   `k_rope` streams.
2. **Interleaved RoPE** (DeepSeek convention; pairs `(x[2i], x[2i+1])`
   rotate together, vs Llama's "first-half / second-half" split).
3. **Asymmetric Q/K vs V `head_dim`** (192 vs 128 at V2-Lite). Rules
   out flash-attn 2 / FlashInfer prefill; forces the PyTorch SDPA
   reference backend.
4. **Heterogeneous FFN per layer**: SwiGLU for the first
   `first_k_dense_replace` layers, MoE (with shared experts + routed
   scaling factor) after.
5. **Per-stream KV cache** with `kv_latent` + `k_rope` instead of
   `(K, V)`, enabling the cache shrink.

## The map

| Paper primitive | Reference (HF `modeling_deepseek_v2.py`) | Our code |
|---|---|---|
| Backbone | `DeepseekV2Model` + `DeepseekV2ForCausalLM` | `DeepseekV2ForCausalLM` + `_DeepseekV2InnerModel` in `src/mini_infer/models/deepseek_v2.py` (L201, L171) |
| Decoder layer | `DeepseekV2DecoderLayer` | `_DeepseekV2DecoderLayer` in `src/mini_infer/models/deepseek_v2.py` (L112) |
| Multi-head Latent Attention | `DeepseekV2Attention` | `MLAAttention` in `src/mini_infer/models/blocks/mla.py` (L68) |
| Low-rank Q (V2/V3/Kimi-K2 path) | `q_a_proj` + `q_a_layernorm` + `q_b_proj` | same names in `MLAAttention.__init__` |
| Direct Q (V2-Lite path) | `q_proj` (when `q_lora_rank is None`) | same name in `MLAAttention.__init__` |
| Joint KV-and-RoPE projection | `kv_a_proj_with_mqa` | same name in `MLAAttention.__init__` |
| KV latent + RoPE split | inside `DeepseekV2Attention.forward` | inside `MLAAttention.forward` (L148) |
| KV latent norm + decompression | `kv_a_layernorm` + `kv_b_proj` | same names in `MLAAttention.__init__`, called on read |
| Interleaved RoPE | `apply_rotary_pos_emb` (interleaved variant) | `apply_interleaved_rotary_pos_emb` in `blocks/rope.py` (L269) |
| MLA attention (Q/K/V different head_dim) | `torch.nn.functional.scaled_dot_product_attention` after manual concat | `mla_packed_attention_forward` in `cache/mla_attention.py` |
| Heterogeneous FFN dispatch | `if layer_idx < first_k_dense_replace` | `cfg.is_moe_layer(layer_idx)` in `_DeepseekV2DecoderLayer.__init__` (L138) |
| SwiGLU dense FFN | `DeepseekV2MLP` | `SwiGLU` in `blocks/swiglu.py` |
| Top-k MoE FFN with shared experts | `DeepseekV2MoE` | `MoEFFN` in `blocks/mixtral_moe.py` (config-extended for shared experts) |
| Per-stream KV cache | manual `past_key_values` tuple (single-request) | `PagedKVCache` with per-layer `StreamSpec` list (`kv_latent`, `k_rope`) in `cache/paged_kv_cache.py` |
| YaRN long-context RoPE | `precompute_yarn_freqs` (optional) | `RotaryEmbedding` in `blocks/rope.py` (L144), YaRN active past `yarn_original_seq_len` |

Bit-parity is exercised against HF's reference inside `tests/unit/`;
the test layout is not enumerated here (tests churn faster than the
architecture). The `MLAAttention` block matches `DeepseekV2Attention`
at cosine-sim > 0.999 on synthetic configs.

## §2.1 — Multi-head Latent Attention

The defining trick. Standard MHA caches per-head K and V (huge memory
for many-head models). MLA factorises K and V through a low-rank
latent (`kv_lora_rank=512`) and a small decoupled RoPE channel
(`qk_rope_head_dim=64`); the cache holds those two compressed streams
and reconstructs per-head K + V at attention time.

### Why two streams (and not one fused stream)

RoPE is position-dependent. If RoPE were applied to the full K, the
cache would have to store position-encoded K — which means
re-rotating on every read (slow) or duplicating per query position
(huge). DeepSeek's answer: split K into a position-free part
(`k_nope`) and a position-dependent part (`k_rope`). Apply RoPE only
to `k_rope`. Cache `kv_latent` (compresses to both `k_nope` and the
full V on decompression) and `k_rope` (pre-rotated). On read,
`k_nope` is decompressed fresh; `k_rope` is broadcast to all heads
and concatenated with `k_nope`. Same final K shape as MHA, but ~7x
less memory in cache.

### The forward shape walk

From the `blocks/mla.py` docstring (single-token, batch=1):

```
hidden_states (1, 1, H)
→ q (1, num_heads, 1, qk_head_dim)        # via q_proj OR low-rank
→ split q_nope (qk_nope_head_dim) + q_pe (qk_rope_head_dim); RoPE on q_pe
→ kv_a_proj_with_mqa → (1, 1, kv_lora_rank + qk_rope_head_dim)
  → split kv_latent (kv_lora_rank) + k_rope (qk_rope_head_dim); RoPE on k_rope
→ write kv_latent + k_rope to cache as 1-head streams
→ read FULL kv_latent + k_rope from cache (concatenated past + present)
→ kv_a_layernorm + kv_b_proj on kv_latent → split to k_nope (per-head) + v (per-head)
→ broadcast k_rope to all heads, cat with k_nope → K
→ SDPA(Q, K, V) where V has different head_dim than Q/K
→ o_proj
```

**Reference**: `DeepseekV2Attention.forward` in HF transformers'
`modeling_deepseek_v2.py`. Same logical sequence.

**Our code**: `MLAAttention.forward` (`blocks/mla.py:148`).

### Low-rank Q path (V2 / V3 / Kimi-K2)

For models with `q_lora_rank` set (V2, V3, Kimi-K2): Q is computed via
two matmuls — `q_a_proj` shrinks `hidden_size → q_lora_rank`,
`q_a_layernorm` normalises the latent, `q_b_proj` projects back up to
`num_heads * qk_head_dim`. The intermediate `q_lora_rank` is much
smaller than `hidden_size`, saving parameters in the per-head Q.

**Reference**: `q_a_proj` + `q_a_layernorm` + `q_b_proj` in
`DeepseekV2Attention`.

**Our code**: `q_a_proj` + `q_a_layernorm` + `q_b_proj` (same names)
in `MLAAttention.__init__`.

### Direct Q path (V2-Lite)

V2-Lite (16B) doesn't use the low-rank Q: `q_lora_rank` is `None`
in the HF config, and Q is computed by a single `q_proj` matmul.
Both paths produce the same `Q` shape; the dispatch is at
`__init__` time.

### Shared MQA stream + per-head K/V decompression

`kv_a_proj_with_mqa` produces a single tensor of shape
`(B, T, kv_lora_rank + qk_rope_head_dim)`. The first slice is the KV
latent (un-normed); the second is the RoPE channel. Both are
**shared across heads** in the cache (1 head, full dim) — this is
why the cache shrinks. On read:

- `kv_a_layernorm(kv_latent) → kv_b_proj(...)` produces a tensor of
  shape `(B, T, num_heads, qk_nope_head_dim + v_head_dim)` which is
  then split into per-head `k_nope` (no positional encoding) and
  per-head `v`.
- `k_rope` is the single-head RoPE channel broadcast across heads and
  concatenated with `k_nope` to form per-head K.

### Asymmetric Q/K vs V head_dim

V2-Lite: `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64
= 192`, but `v_head_dim = 128`. Q and K have head_dim 192, V has
head_dim 128.

This rules out flash-attn 2 (assumes Q/K/V head_dim symmetry) and
FlashInfer prefill. mini-infer dispatches MLA through
`mla_packed_attention_forward` in `cache/mla_attention.py`, which is
a PyTorch SDPA reference path. The model class declares
`required_attention_backend() = "torch"` so the runner forces this
backend regardless of caller preference.

### Interleaved RoPE

DeepSeek uses **interleaved RoPE**: pairs `(x[2i], x[2i+1])` rotate
together. Llama's RoPE rotates `(x[i], x[i + dim/2])` (first-half /
second-half split). They produce the same math after a permutation of
the channel layout — but the stored RoPE weights and the
checkpoint-loading order assume the interleaved convention.

**Reference**: `apply_rotary_pos_emb` in HF transformers'
`modeling_deepseek_v2.py` (or `modular_deepseek_v2.py`).

**Our code**: `apply_interleaved_rotary_pos_emb` in `blocks/rope.py`
(L269). The complementary half-split function `apply_rotary_pos_emb`
(L254) is what Llama uses; both live in the same file so the
distinction is one import away.

## §2.2 — Heterogeneous FFN (dense + MoE with shared experts)

V2 uses dense SwiGLU for the first few layers (`first_k_dense_replace`,
typically 1 in V2-Lite) and a top-k MoE FFN with shared experts after.

### Layer dispatch

**Reference**: `DeepseekV2DecoderLayer.__init__` branches on
`layer_idx < first_k_dense_replace`.

**Our code**: `cfg.is_moe_layer(layer_idx)` in
`_DeepseekV2DecoderLayer.__init__` (`deepseek_v2.py:138`). Same logic.

### Dense FFN (early layers)

**Reference**: `DeepseekV2MLP` is a SwiGLU.

**Our code**: `SwiGLU` from `blocks/swiglu.py`. Same shape (`gate_proj`,
`up_proj`, `down_proj` with SiLU activation).

### MoE FFN (later layers)

V2's MoE is top-k routing + **shared experts** that run on every
token (the "always-on" experts) + routed experts (selected per token
by the gate's top-k). A `routed_scaling_factor` rescales the routed
contribution post-aggregation; `norm_topk_prob` controls renormalization
of the top-k weights.

**Reference**: `DeepseekV2MoE` in `modeling_deepseek_v2.py`. Gate +
routed-expert dispatch + shared-expert path + final scaling.

**Our code**: `MoEFFN` from `blocks/mixtral_moe.py` (with config
extensions for `n_shared_experts`, `shared_intermediate_size`,
`routed_scaling_factor`, `renormalize_topk`). The shared-experts
trick is: collapse N shared MLPs into a single MLP whose
intermediate_size = `n_shared * shared_intermediate`. The aggregated
shared output is mathematically identical to summing N separate small
MLPs but uses one matmul instead of N.

This collapse trick is what V4's `HashRoutedMoEFFN` also uses (`blocks/hash_routed_moe_ffn.py`).

## Per-stream KV cache

The cache shrink is the headline of MLA. We generalize the cache
abstraction so a layer can declare arbitrary named streams instead of
the legacy `(K, V)` pair.

### Stream declaration

Each layer publishes a `list[StreamSpec]` describing its KV streams.
Standard MHA/GQA layers declare `["k", "v"]` (the streams alias the
rectangular `(num_kv_heads, head_dim)` layout — no extra memory).
MLA layers declare `["kv_latent", "k_rope"]` with different shapes per
stream (`1 × kv_lora_rank` vs `1 × qk_rope_head_dim`).

**Reference**: HF stores past_key_values as a tuple per layer; MLA's
two-stream layout is folded into a single tensor via concatenation.
Less flexible than the stream API but works for a single architecture.

**Our code**: `cache/paged_kv_cache.py` + `cache/block_pool.py`. The
`per_layer_streams()` hook on a model class returns the per-layer
stream list. `PagedKVCache.append_stream_packed` /
`materialize_packed_stream` are the per-stream variants of the
legacy `append_kv_packed` / `materialize_packed_kv`.

### Block pool allocation

Per-layer per-stream storage lives in `BlockPool`. For MLA at
V2-Lite: 27 layers × (`kv_latent` 1×512 + `k_rope` 1×64) = 27 × 576
bytes per token per block-slot. Compared to a hypothetical MHA at the
same model size (27 layers × 2 streams × 16 heads × 128 head_dim =
27 × 4096 = 7.1x more), the cache savings are real and measurable.

### Why "1 head" for both streams

Both `kv_latent` and `k_rope` are per-token, **shared across heads**.
The MLA decompression on read produces per-head K and V from the
shared latent. So the cache stores a single "head" per stream;
broadcast happens at attention time, not at storage time.

## §3.1 — YaRN long-context RoPE

V2 supports YaRN for context lengths past `yarn_original_seq_len`.
The same `RotaryEmbedding` block serves V2, V3, V4, and Kimi-K2;
YaRN kicks in when the requested `seq_len > yarn_original_seq_len`.

**Reference**: HF transformers has an optional YaRN-aware RoPE
precomputation when `rope_scaling.type == "yarn"`. The `beta_fast`,
`beta_slow`, `mscale`, `mscale_all_dim` parameters control the
wave-frequency correction ramp.

**Our code**: `RotaryEmbedding` in `blocks/rope.py:144` consumes the
same YaRN parameters from the V2 config. The `apply_yarn_correction`
helper (`blocks/rope.py:74`) builds the per-frequency correction
factor.

## Decoder layer assembly

**Reference (`DeepseekV2DecoderLayer.forward`)**: standard pre/post-
norm decoder shape.

```python
residual = h
h = input_layernorm(h)
h = self_attn(h, ...)
h = residual + h
residual = h
h = post_attention_layernorm(h)
h = mlp(h)  # dense SwiGLU or MoEFFN
h = residual + h
```

**Our code (`_DeepseekV2DecoderLayer.forward`)**: same shape
(`deepseek_v2.py:152`). The pre-norm pattern (normalise → sub-block
→ add residual) is the standard one; V4 replaces this with the
Hyper-Connections residual scheme, which is one of the reasons V4
needed a separate decoder layer file.

## Validation contract

Bit-parity is enforced against HF transformers'
`DeepseekV2Attention` + `DeepseekV2MoE` on synthetic inputs at FP32.
The MLA block matches at cosine-sim > 0.999; the MoE FFN matches at
cosine-sim > 0.999. End-to-end golden output against HF at
temperature=0 is the integration gate.

Tests live under `tests/unit/`; the specific layout isn't enumerated
here. Run `pytest tests/unit/ -k 'mla or deepseek_v2'` to see what's
exercised.

## Where we diverged + why

1. **No vendored reference**. We don't vendor an MLA reference (unlike
   V4 which has `third_party/deepseek_v4_reference/`). HF transformers'
   `DeepseekV2Attention` IS the upstream reference for the family;
   we test directly against it. Reasons: (a) HF transformers is on
   every dev machine, (b) the V2 paper's reference code is the same
   as HF's, (c) pinning a specific reference version creates
   maintenance burden the V2 architecture doesn't justify.
2. **Three model variants, one model class**. V2-Lite (16B), V2 (236B),
   V3 (671B), Kimi-K2 (1T) all share the `DeepseekV2ForCausalLM`
   class. The differences are config bits (`q_lora_rank`,
   `n_routed_experts`, `qk_nope_head_dim`, `first_k_dense_replace`);
   the architecture is identical.
3. **`PagedKVCache` per-stream abstraction generalised**. We extended
   the cache to support named streams per layer (`["kv_latent",
   "k_rope"]` for MLA, `["k", "v"]` for standard, V4's richer set).
   This is a project-level abstraction not specific to the V2 paper.
4. **`"torch"` attention backend forced** at the model layer.
   `MLAAttention.required_attention_backend()` returns `"torch"` so
   the runner picks the PyTorch SDPA path regardless of what the
   caller asked for (flash-attn doesn't support asymmetric head_dim).
   Same pattern Gemma 4 uses for its `head_dim=512` constraint.

## Pointers

- **Reference**: HuggingFace transformers
  `transformers/models/deepseek_v2/modeling_deepseek_v2.py`
  (and `modular_deepseek_v2.py`). Live in the installed
  transformers package; we don't vendor it.
- **Paper**: `docs/papers/` (not vendored; the V2 paper is on arXiv).
- **Our model class**: `src/mini_infer/models/deepseek_v2.py`.
- **MLA block**: `src/mini_infer/models/blocks/mla.py`.
- **MLA attention dispatch**: `src/mini_infer/cache/mla_attention.py`.
- **Per-stream cache**: `src/mini_infer/cache/paged_kv_cache.py` +
  `cache/block_pool.py::StreamSpec`.
- **Interleaved RoPE**: `src/mini_infer/models/blocks/rope.py`.

## What's still open

- **Large-scale validation on V3 / Kimi-K2 weights**. V2-Lite (16B)
  is validated locally; V3 (671B) and Kimi-K2 (1T) checkpoints would
  need multi-GPU loading via TP. Same `DeepseekV2ForCausalLM` class;
  validation is a budget question, not a code question.
- **Mid-cache-block prefix-cache integration for MLA**. The standard
  prefix cache works (chained-hash, per-block). The MLA-specific
  shape (1-head streams) just plugs in via the stream abstraction.
  No known gap; flagged as "exercise it on real workloads".

## How this doc extends to V3 / Kimi-K2

Both V3 and Kimi-K2 use the same `MLAAttention` block + the same
`DeepseekV2ForCausalLM` class. The diffs are config-only:

- **V3 (671B)**: 61 layers, 128 routed experts, 8 active per token,
  3 shared experts, MTP heads (Multi-Token Prediction). The MTP
  heads aren't yet wired; the base model loads + runs.
- **Kimi-K2 (1T)**: similar shape, larger expert count, different
  routing config.

Both load through the same path. The walkthrough above covers their
architecture in full; the only V3-specific concern that doesn't
generalise from V2 is MTP, which is its own primitive (separate
walkthrough when we wire it up).
