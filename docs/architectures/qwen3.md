# Qwen3 walkthrough

A line-by-line correspondence between Alibaba's Qwen3 release,
HuggingFace transformers' `Qwen3ForCausalLM`, and mini-infer's
from-scratch port.

Qwen3 is the **smallest family-specific delta in mini-infer's
registry**. Read this if you want to see what a minimal family
implementation looks like — almost everything is the shared Llama
baseline; only two things make it Qwen3.

## TL;DR

Qwen3 is a Llama-shape backbone (GQA + RoPE + RMSNorm + SwiGLU)
with two architectural changes vs Qwen2:

1. **No QKV biases** (`attention_bias=False`). Qwen2 had biased Q/K/V
   projections; Qwen3 dropped them.
2. **Per-head Q/K norm before RoPE.** Same RMSNorm shape Gemma 3+
   uses (one RMSNorm per attention head), but with the standard
   RMSNorm (no `+1` weight offset like Gemma 3's).

Plus a couple of bookkeeping bits:

- **Tied embeddings** between input and output (`embed_tokens.weight
  == lm_head.weight`).
- **Single RoPE theta**, no dual-RoPE / partial RoPE.

That's it. The entire family implementation file is ~160 lines.

## The map

| Primitive | Reference (HF `modeling_qwen3.py`) | Our code |
|---|---|---|
| Backbone | `Qwen3Model` + `Qwen3ForCausalLM` | `Qwen3ForCausalLM` + `_Qwen3InnerModel` in `src/mini_infer/models/qwen3.py` (L104, L78) |
| Decoder layer | `Qwen3DecoderLayer` | shared `TransformerBlock` from `blocks/transformer_block.py` |
| GQA attention | `Qwen3Attention` | `GQAAttention` in `blocks/gqa.py` (shared) |
| **Per-head Q norm** | `Qwen3Attention.q_norm` (RMSNorm, per head) | per-head `q_norm` flag in `GQAAttention.__init__` |
| **Per-head K norm** | `Qwen3Attention.k_norm` (RMSNorm, per head) | per-head `k_norm` flag in `GQAAttention.__init__` |
| **No QKV bias** | `attention_bias=False` in config | `attention_bias=False` threaded into `GQAAttention` |
| Standard RMSNorm | `Qwen3RMSNorm` (no `(1+w)` offset) | shared `RMSNorm` in `blocks/rmsnorm.py` |
| Single-theta RoPE | `Qwen3RotaryEmbedding` | shared `RotaryEmbedding` in `blocks/rope.py` |
| SwiGLU FFN | `Qwen3MLP` | shared `SwiGLU` in `blocks/swiglu.py` |
| Tied embeddings | `tie_word_embeddings=True` in config | `Qwen3ForCausalLM` reuses `embed_tokens.weight` for `lm_head` |

Bit-parity is exercised against HF transformers' `Qwen3Attention` on
synthetic configs at FP32 with cosine-sim > 0.999. Golden tests at
temperature=0 validate end-to-end against HF. Tests under
`tests/unit/`; layout intentionally not enumerated.

## Per-head Q/K norm

The one *actually-novel* primitive in Qwen3. Before RoPE, each head
of Q and K passes through its own RMSNorm. Different from a single
shared norm; each head learns its own per-channel weight vector.

### Why per-head

The standard pattern (`norm(x)` once, then split into heads) is what
Llama / Mistral / earlier Qwen do. Per-head norm normalises each head
independently, which:
- bounds per-head magnitudes regardless of channel allocation
- helps numerical stability when `head_dim` is small or unusual
- absorbs the `1/sqrt(d_head)` softmax scale into the learned norm
  weights (similar to Gemma 4's softmax-scale=1.0 trick, though Qwen3
  keeps the standard `1/sqrt(d_head)` scale)

### Where in the forward pass

```
x (B, T, hidden_size)
→ q_proj(x), k_proj(x), v_proj(x)      # no biases
→ reshape Q to (B, T, num_heads, head_dim)
→ reshape K to (B, T, num_kv_heads, head_dim)
→ q_norm(Q)                            # per-head RMSNorm, weight (num_heads, head_dim)
→ k_norm(K)                            # per-head RMSNorm, weight (num_kv_heads, head_dim)
→ apply_rotary(Q), apply_rotary(K)     # standard RoPE
→ SDPA(Q, K, V)
→ o_proj(...)
```

The norms happen AFTER the linear projections (so they normalise the
projected heads, not the raw hidden state) but BEFORE RoPE (so the
position encoding rotates already-normalised heads).

**Reference**: `Qwen3Attention.forward` applies `self.q_norm(q)` and
`self.k_norm(k)` between the projection and the RoPE rotation.

**Our code**: `GQAAttention.__init__` accepts `q_norm=True` /
`k_norm=True` flags. When set, it instantiates `RMSNorm` modules with
per-head weight shape `(num_heads, head_dim)` / `(num_kv_heads,
head_dim)` and applies them in `forward` at the right place. At
`q_norm=False` / `k_norm=False` (Llama / Qwen2), the norms are
no-ops and there's no parameter overhead.

### Standard RMSNorm, not Gemma 3's offset variant

The norm classes share the underlying `RMSNorm` math: normalise by
RMS, multiply by learned weight. Gemma 3's `(1 + w) * normalised_x`
twist (where weights are stored close to 0 and the `+1` makes them
behave like small adjustments around 1.0) is NOT used by Qwen3.
Qwen3's stored weights are in their final form, multiplied directly.

## No QKV biases

Qwen2 had `q_proj`, `k_proj`, `v_proj` with bias terms. Qwen3 dropped
them. Loses a tiny number of parameters (3 × `hidden_size` per
layer); not a big deal architecturally, but the loader and the GQA
block both have to know.

**Reference**: `attention_bias=False` in `Qwen3Config` →
`Qwen3Attention.__init__` builds the linear layers without bias.

**Our code**: `attention_bias=False` in `Qwen3Config` →
`GQAAttention.__init__` passes `bias=False` to the column-parallel
linears.

Qwen2's loader is the same shape but with `attention_bias=True`;
both families use the same `GQAAttention` block with this single
config flag toggled.

## Tied embeddings

Input embedding (`embed_tokens.weight`) and output projection
(`lm_head.weight`) share parameters. Saves `vocab_size * hidden_size`
parameters; standard trick for small / medium-sized models where the
embedding matrix is a big fraction of total parameters.

**Reference**: `Qwen3ForCausalLM.__init__` ties via
`self.lm_head.weight = self.model.embed_tokens.weight`.

**Our code**: `Qwen3ForCausalLM` does the same tying after constructing
the inner model.

The vocab-parallel embedding from `distributed/embedding.py` carries
the shared weight; the LM head reads through the same vocab-parallel
slice. At `world_size=1` this is a single shared matrix; at
`world_size>1` each rank holds the same vocab slice for both purposes.

## Decoder layer assembly

Same pre-norm decoder shape as Llama:

```python
residual = h
h = input_layernorm(h)
h = self_attn(h, ...)        # GQA with per-head q_norm, k_norm, no biases
h = residual + h
residual = h
h = post_attention_layernorm(h)
h = mlp(h)                   # standard SwiGLU
h = residual + h
```

**Reference (`Qwen3DecoderLayer.forward`)**: standard.

**Our code**: uses the shared `TransformerBlock` from
`blocks/transformer_block.py`. No Qwen3-specific decoder layer needed;
the family-specific behaviour lives entirely inside `GQAAttention`
via the `q_norm` / `k_norm` / `attention_bias` flags.

## Validation contract

Bit-parity against HF transformers' `Qwen3Attention` on synthetic
configs at FP32 with cosine-sim > 0.999. Golden output at
temperature=0 against HF. Qwen3-0.6B / 1.7B / 4B fit M1 fp16 comfortably
and run as the everyday smoke. Larger variants (8B, 14B, 32B) validate
on Modal when budget permits.

Tests under `tests/unit/`; layout intentionally not enumerated.

## Why Qwen3's implementation is so short

The whole `src/mini_infer/models/qwen3.py` is ~160 lines because:

1. **Decoder layer is shared**. `TransformerBlock` from
   `blocks/transformer_block.py` is the standard pre-norm shape that
   serves Llama / Mistral / Qwen2 / Qwen3 / SmolLM2 / etc.
2. **Attention is shared**. `GQAAttention` from `blocks/gqa.py` is the
   shared GQA primitive; family-specific behaviour (q_norm, k_norm,
   bias) lives as constructor flags.
3. **RoPE / RMSNorm / SwiGLU are shared**. Standard primitives in
   `blocks/`.
4. **No family-specific decoder file**. Compare to Gemma 3 / Gemma 4
   / V4 / V2, which all have family-specific decoder layer files
   because their per-block math diverges from the standard.

The Qwen3 file is the **minimum viable model class**: config dataclass,
inner model wiring, the forward method, the `register_model` decoration,
and the `from_hf` config parser. About 90% of the file is config
shuffling.

This is the model file to use as a template when adding a new
Llama-shape architecture with one or two family quirks.

## Where we diverged + why

1. **`GQAAttention` carries Qwen3's quirks as flags**. `q_norm`,
   `k_norm`, `attention_bias` are all constructor flags on the shared
   GQA block. Same pattern Gemma 4 uses for its `attention_k_eq_v`
   and `query_scale=1.0` quirks. Keeps the family-specific behaviour
   localised; the shared GQA block stays the single source of truth.
2. **Per-head norm weight shape `(num_heads, head_dim)`**. Stored on
   disk in this shape (one weight vector per head, channel-wise).
   The RMSNorm block accepts this shape and applies it per-head.
   Different from layer-wide norms which use a `(hidden_size,)`
   weight shape.

## Pointers

- **Reference**: HF transformers
  `transformers/models/qwen3/modeling_qwen3.py`.
- **Our model class**: `src/mini_infer/models/qwen3.py`.
- **GQA block** (shared with Llama / Qwen2 / Mistral / SmolLM2):
  `src/mini_infer/models/blocks/gqa.py`.
- **TransformerBlock** (shared decoder shape):
  `src/mini_infer/models/blocks/transformer_block.py`.

## What's still open

- **Qwen3-MoE variants**. Not yet implemented; would reuse the
  Qwen3 attention path + Mixtral's `MoEFFN`. Straightforward extension
  when the demand or paper-watch trigger comes.
- **Larger checkpoint validation** (Qwen3-32B). Local validation runs
  on 0.6B / 1.7B / 4B; 32B needs Modal hardware. Same loader path;
  not blocked on code.
