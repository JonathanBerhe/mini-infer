# Llama walkthrough

A line-by-line correspondence between Meta's Llama releases, HuggingFace
transformers' `LlamaForCausalLM`, and mini-infer's from-scratch port.

Llama is the **canonical Llama-shape baseline** for everything else in
the registry. Read this if you want to understand the shared building
blocks (`GroupedQueryAttention`, `TransformerBlock`, `SwiGLU`, `RMSNorm`,
`RotaryEmbedding`) that Qwen2 / Qwen3 / Mistral / Mixtral / Gemma /
DeepSeek all reuse with family-specific flags.

## TL;DR

Llama is the canonical RMSNorm + RoPE + GQA + SwiGLU stack:

1. **Grouped-Query Attention** (Llama 2 70B onwards). Q has
   `num_attention_heads` heads, K/V share `num_key_value_heads` groups.
   Each KV head serves `num_attention_heads / num_key_value_heads`
   query heads.
2. **RoPE** with a single rotation table over all `head_dim` channels.
   Plain `base=10000.0` for Llama 2; Llama 3.1+ uses YaRN-style frequency
   scaling via the same `RotaryEmbedding` block.
3. **RMSNorm** in the standard form (no `(1+w)` offset like Gemma 3),
   pre-norm pattern at the block boundary.
4. **SwiGLU** FFN: `down_proj(silu(gate_proj(x)) * up_proj(x))`. No biases.
5. **No biases** on Q/K/V projections (`attention_bias=False`) or MLP
   (`mlp_bias=False`). Refusing those configs is a `NotImplementedError`
   on load so a biased Llama checkpoint fails loudly rather than silently
   loading wrong.
6. **Optional tied embeddings** (`tie_word_embeddings`). True for small
   variants (SmolLM2-135M, TinyLlama, Llama-3-1B); False for Llama-3-8B+
   where the embedding matrix is dwarfed by the rest of the model.

That's it. The entire family implementation is ~180 lines, of which
~90% is config parsing and weight-load wiring.

Covers Llama 2, Llama 3 / 3.1 / 3.2 / 3.3, plus the Llama-shape
derivatives that ship under HF architecture string `LlamaForCausalLM`
(SmolLM2, TinyLlama, unsloth Llama clones, Nemotron-Llama variants
that don't change the architecture).

## The map

| Primitive | Reference (HF `modeling_llama.py`) | Our code |
|---|---|---|
| Backbone | `LlamaModel` + `LlamaForCausalLM` | `LlamaForCausalLM` + `_LlamaInnerModel` in [src/mini_infer/models/llama.py:113](../../src/mini_infer/models/llama.py#L113), [:87](../../src/mini_infer/models/llama.py#L87) |
| Decoder layer | `LlamaDecoderLayer` | shared `TransformerBlock` in [blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py) |
| GQA attention | `LlamaAttention` | shared `GroupedQueryAttention` in [blocks/gqa.py](../../src/mini_infer/models/blocks/gqa.py) |
| Standard RoPE | `LlamaRotaryEmbedding` | shared `RotaryEmbedding` in [blocks/rope.py:144](../../src/mini_infer/models/blocks/rope.py#L144) |
| RMSNorm | `LlamaRMSNorm` | shared `RMSNorm` in [blocks/rmsnorm.py](../../src/mini_infer/models/blocks/rmsnorm.py) |
| SwiGLU FFN | `LlamaMLP` | shared `SwiGLU` in [blocks/swiglu.py](../../src/mini_infer/models/blocks/swiglu.py) |
| Tied embeddings | `tie_word_embeddings=True` in config | `LlamaForCausalLM.__init__` aliases `lm_head.weight = embed_tokens.weight` ([llama.py:127](../../src/mini_infer/models/llama.py#L127)) |
| Vocab-parallel embedding | `nn.Embedding` | `VocabParallelEmbedding` (ws=1 identity to `nn.Embedding`) |
| Column-parallel LM head | `nn.Linear` | `ColumnParallelLinear(..., gather_output=True)` |

Bit-parity is exercised against HF transformers' `LlamaAttention` on
synthetic configs at FP32 with cosine-sim > 0.999. Golden tests at
temperature=0 validate end-to-end against HF using SmolLM2-135M as the
pinned smoke model. Tests under `tests/unit/`; layout intentionally
not enumerated.

## Grouped-Query Attention

The one *actually-novel* primitive Llama introduced (at 70B in Llama 2).
Multi-Head Attention has `num_heads` Q heads and `num_heads` K/V heads.
Multi-Query Attention has `num_heads` Q heads and 1 shared K/V head.
GQA is the middle ground: `num_heads` Q heads, `num_kv_heads` K/V heads
with `num_heads / num_kv_heads` Q heads sharing each KV head.

### Why GQA

The KV cache size grows linearly with `num_kv_heads`. Going from
`num_kv_heads = num_heads = 64` (MHA) to `num_kv_heads = 8` (GQA-8)
shrinks the cache 8x while keeping most of MHA's quality. The standard
trade-off Llama 2 70B chose; every Llama-shape model since copies it.

### Where in the forward pass

```
x (B, T, hidden_size)
→ q_proj(x), k_proj(x), v_proj(x)      # no biases
→ reshape Q to (B, T, num_heads, head_dim)
→ reshape K to (B, T, num_kv_heads, head_dim)
→ reshape V to (B, T, num_kv_heads, head_dim)
→ apply_rotary(Q), apply_rotary(K)     # standard RoPE
→ append_kv_packed(K, V) into paged cache
→ packed_attention_forward(Q, cache)   # FlashInfer / FlashAttention / SDPA
→ o_proj(...)
```

**Reference**: `LlamaAttention.forward` calls
`apply_rotary_pos_emb(q, k, cos, sin)` then `eager_attention_forward`
(or the FlashAttention path).

**Our code**: [blocks/gqa.py:93](../../src/mini_infer/models/blocks/gqa.py#L93)
projects, reshapes, applies RoPE, appends to the paged cache, then
dispatches the packed varlen attention kernel. One layer forward = one
call to `packed_attention_forward`.

The shared `GroupedQueryAttention` block carries every family-specific
quirk as a constructor flag (`q_norm`, `k_norm`, `attention_k_eq_v`,
`v_norm`, `query_scale`); for plain Llama all flags are off / None.

## RoPE

Standard Llama-shape rotary embeddings. Single rotation table over
`head_dim` channels, base 10000.0, full `partial_rotary_factor=1.0`
(all channels rotate).

**Reference**: `LlamaRotaryEmbedding` builds an `inv_freq` buffer and
generates `cos`/`sin` per forward call against `position_ids`.

**Our code**: [blocks/rope.py:144](../../src/mini_infer/models/blocks/rope.py#L144)
holds the same buffer; the `forward` method matches HF's math line for
line including the autocast-disable boundary (RoPE tables must be fp32
to avoid drift on long contexts).

### Llama 3.1+ long context

Llama 3.1 introduced 128k-context variants via frequency-scaled RoPE.
HF's `LlamaRotaryEmbedding` reads `rope_scaling.type` and dispatches
to one of several schemes (linear / dynamic / Llama3). Our
`RotaryEmbedding` supports the YaRN-style correction used by DeepSeek
(V2/V3/V4/Kimi-K2) and is the building block; the Llama 3.1
`type="llama3"` scaling is a close cousin we don't currently parse out
of the HF config. Llama 2 and Llama 3 base models load straight through
the default branch.

This is a tracked gap, not a blocker: SmolLM2 / TinyLlama / Llama-2-7B /
Llama-3-8B all use the default `rope_theta` path and validate
bit-parity end-to-end. Long-context Llama-3.1 variants would need the
`rope_scaling` dict parsed in `LlamaConfig.from_hf` and threaded into
`RotaryEmbedding`'s YaRN args.

## SwiGLU FFN

Three-projection gated FFN, same shape Llama 2 introduced and every
Llama-shape model uses unchanged:

```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```

`gate_proj` and `up_proj` go from `hidden_size` to `intermediate_size`,
`down_proj` goes back. No biases. SiLU is the activation; the elementwise
multiply is the "glu" gate.

**Reference**: `LlamaMLP.forward`.

**Our code**: [blocks/swiglu.py](../../src/mini_infer/models/blocks/swiglu.py).
`gate_proj` and `up_proj` are column-parallel along `intermediate_size`;
`down_proj` is row-parallel along its input. Standard Megatron pairing,
one all-reduce per FFN.

## RMSNorm + pre-norm decoder

The decoder block is pre-norm: normalize, then attention, then add
residual; normalize, then FFN, then add residual.

**Reference (`LlamaDecoderLayer.forward`)**:

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states, ...)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

**Our code**: [blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py)
expresses the same shape with HF-aligned attribute names
(`input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp`) so
weight loading is identity rename.

The `RMSNorm` block itself is the standard form: `x * rsqrt(mean(x²) + eps) * weight`,
compute in fp32 then cast back. No `(1+w)` offset (that's Gemma 3+).

## Tied embeddings

The input embedding (`embed_tokens.weight`) and output projection
(`lm_head.weight`) optionally share parameters. Saves
`vocab_size * hidden_size` parameters; standard trick for small models
where the embedding matrix dominates the parameter count.

| Variant | `tie_word_embeddings` | Why |
|---|---|---|
| SmolLM2-135M | True | Embedding is ~50% of total params |
| TinyLlama-1.1B | True | Embedding is ~25% of total params |
| Llama-3.2-1B / 3B | True | Same rationale |
| Llama-2-7B / Llama-3-8B / Llama-3-70B | False | Embedding is <5% of total params |

**Our code**: [llama.py:127](../../src/mini_infer/models/llama.py#L127)
aliases `lm_head.weight = self.model.embed_tokens.weight` when the
config says to. Both tensors are stored as `(vocab_per_rank, hidden_size)`
so aliasing still works under tensor parallelism; at world_size=1
it's exactly the unsharded tying.

[llama.py:156](../../src/mini_infer/models/llama.py#L156) whitelists
`lm_head.weight` from the load-mismatch check when tying is on, because
HF checkpoints with `tie_word_embeddings=True` only ship `embed_tokens.weight`.

## Loud failure on biased Llama variants

A handful of community fine-tunes ship under `LlamaForCausalLM` with
`attention_bias=True` or `mlp_bias=True`. Our `GroupedQueryAttention`
and `SwiGLU` both build their linears at construction time and the
constructor takes a single `qkv_bias` flag (none for MLP). Adding biased
variants would mean a second construction path on both blocks.

Rather than silently load the bias-bearing weight names into a no-bias
model (which would put numerical errors into every layer and surface
as a golden-test failure thousands of tokens later), `LlamaConfig.from_hf`
raises `NotImplementedError` when either flag is set in the HF config.

This is the same "fail loud at the boundary" pattern that runs through
the codebase: weight-load mismatches are loud (`load_weights` raises on
missing/unexpected keys), config mismatches are loud (refuse to construct),
and a biased Llama variant is rejected at parse time with a message that
points at the actual config field.

## Decoder layer assembly

```python
residual = h
h = input_layernorm(h)
h = self_attn(h, ...)        # GQA, no biases, no q_norm/k_norm
h = residual + h
residual = h
h = post_attention_layernorm(h)
h = mlp(h)                   # SwiGLU, no biases
h = residual + h
```

**Reference (`LlamaDecoderLayer.forward`)**: standard.

**Our code**: shared `TransformerBlock` from
[blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py).
No Llama-specific decoder layer needed; the family-specific behaviour
lives entirely inside `GroupedQueryAttention` (and there's none for
plain Llama, every flag is off).

## Cache structure

Standard paged KV cache: one block table per layer, `block_size` tokens
per block, `(num_kv_heads, head_dim)` per token. No family-specific
cache contract; Llama's GQA fits the canonical paged shape.

## Validation contract

- **Bit-parity vs HF**: synthetic `LlamaAttention` configs at FP32
  produce activations within cosine-sim > 0.999 of our
  `GroupedQueryAttention` output.
- **Golden test vs HF at temperature=0**: SmolLM2-135M is the pinned
  smoke. Same generated tokens for a fixed prompt seed.
- **Weight-load round-trip**: a Llama state_dict loaded through
  `load_state_dict_with_tp` reproduces the original tensors per layer.

The pinned HF revision for SmolLM2-135M lives in
[`tests/_pinned_models.toml`](../../tests/_pinned_models.toml); the
bit-parity CI workflow runs the `requires_model`-marked tests against
that revision.

Tests under `tests/unit/`; layout intentionally not enumerated.

## Why Llama is the template

The entire `src/mini_infer/models/llama.py` is ~180 lines because every
non-config piece is shared:

1. **Decoder layer is shared.** `TransformerBlock` is the standard
   pre-norm shape that serves Llama / Qwen2 / Qwen3 / Mistral / SmolLM2.
2. **Attention is shared.** `GroupedQueryAttention` carries family
   quirks as constructor flags; plain Llama uses none of them.
3. **RoPE / RMSNorm / SwiGLU are shared.** Standard primitives.
4. **No family-specific decoder file.** Compare to Gemma 3 / 4 / V4 / V2
   which each have a family-specific decoder layer file because their
   per-block math diverges.

The Llama file is the **minimum viable model class** plus the
loud-failure boundary for biased variants. About 50% of the file is
the `from_hf` config parser, 30% is module wiring, and the rest is
weight-loading plumbing.

Every other Llama-shape walkthrough in this directory reads as a delta
on this file:

- **Qwen2** adds `attention_bias=True` (QKV biases). Same blocks, one flag.
- **Qwen3** drops the biases again and adds `q_norm` / `k_norm` flags.
- **Mistral** is identical at the block level; the family file is a
  ~10-line rename for the HF architecture string.
- **Mixtral** swaps `SwiGLU` for `MoEFFN`, keeps everything else.

## Where we diverged + why

1. **`GroupedQueryAttention` is shared, not per-family.** Every flag a
   family needs (`q_norm`, `k_norm`, `attention_k_eq_v`, `v_norm`,
   `query_scale`) is a constructor argument. Plain Llama uses none. Same
   reason as ADR-007: family quirks compose better as flags on the
   shared block than as forked attention classes.
2. **Refuse biased Llama variants at config-parse time.** The
   alternative (load anyway, drop the bias terms) would corrupt the
   numerical contract and surface as a golden-test failure far from
   the cause. Loud failure at the boundary is the project standard.
3. **Long-context Llama 3.1 RoPE scaling is a tracked gap.** Our
   `RotaryEmbedding` has the YaRN building block; parsing
   `rope_scaling.type="llama3"` and threading the `low_freq_factor` /
   `high_freq_factor` / `original_max_position_embeddings` args is
   straightforward but not yet wired. Llama 2 / Llama 3 base / SmolLM2 /
   TinyLlama all validate on the default path.

## Pointers

- **Reference**: HF transformers `transformers/models/llama/modeling_llama.py`.
- **Our model class**: [src/mini_infer/models/llama.py](../../src/mini_infer/models/llama.py).
- **GQA block** (shared with Qwen2 / Qwen3 / Mistral / SmolLM2): [blocks/gqa.py](../../src/mini_infer/models/blocks/gqa.py).
- **TransformerBlock** (shared decoder shape): [blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py).
- **RoPE** (shared with every Llama-shape family): [blocks/rope.py](../../src/mini_infer/models/blocks/rope.py).
- **RMSNorm** (shared): [blocks/rmsnorm.py](../../src/mini_infer/models/blocks/rmsnorm.py).
- **SwiGLU FFN** (shared): [blocks/swiglu.py](../../src/mini_infer/models/blocks/swiglu.py).
- **Pinned smoke**: SmolLM2-135M-Instruct, see [tests/_pinned_models.toml](../../tests/_pinned_models.toml).

## What's still open

- **Llama 3.1+ rope_scaling parsing.** The math is in `RotaryEmbedding`
  (via the YaRN correction); the HF-config parser in `LlamaConfig.from_hf`
  doesn't yet read `rope_scaling`. Adding it is ~20 lines + a parity
  test against HF on a 3.1 checkpoint. Not blocked on hardware.
- **Biased Llama variants.** Some community fine-tunes set
  `attention_bias=True` or `mlp_bias=True`. Supporting them is a
  block-constructor change in `GroupedQueryAttention` / `SwiGLU` plus
  threading the flag through `LlamaConfig.from_hf`. Triggered if a
  notable biased Llama variant lands; otherwise the loud failure is
  the right default.
- **Larger checkpoint validation** (Llama-3-70B). Local validation
  runs on SmolLM2-135M / TinyLlama-1.1B / Llama-3.2-1B. 70B needs
  Modal hardware; same loader path, not blocked on code.
