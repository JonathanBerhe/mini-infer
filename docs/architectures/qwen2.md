# Qwen2 walkthrough

A line-by-line correspondence between Alibaba's Qwen2 / Qwen2.5 releases,
HuggingFace transformers' `Qwen2ForCausalLM`, and mini-infer's
from-scratch port.

Qwen2 is the **primary smoke model family**: Qwen2.5-0.5B-Instruct is
the smallest pinned checkpoint and drives most of the project's
day-to-day CI (PD scheduler, spec decoding, API tests, the API
streaming path). Architecturally it's a Llama-shape backbone with one
twist: Q/K/V projections carry learnable biases.

## TL;DR

Qwen2 / Qwen2.5 is a Llama-shape backbone (RMSNorm + RoPE + GQA +
SwiGLU) with one architectural change vs Llama:

1. **QKV biases.** `q_proj`, `k_proj`, `v_proj` each have a learnable
   bias vector. The output projection (`o_proj`) and the MLP
   projections stay bias-free. Qwen3 dropped these again; the bias
   is the visible artifact of Qwen2's lineage.

Plus a couple of bookkeeping bits:

- **Tied embeddings** (`tie_word_embeddings=True`) for the small
  variants (0.5B, 1.5B); not tied for the larger ones (7B, 14B, 32B,
  72B).
- **Single RoPE theta**, no dual-RoPE / partial RoPE.
- **Same shared blocks as Llama.** `TransformerBlock`,
  `GroupedQueryAttention`, `SwiGLU`, `RMSNorm`, `RotaryEmbedding`.

Family file is ~170 lines, of which ~50% is the config parser and
weight loader, ~30% is module wiring, and ~20% is the forward.

Covers Qwen2 (all sizes) and Qwen2.5 (all sizes). Qwen3 is a separate
family because of its per-head Q/K norm; see [qwen3.md](qwen3.md).

## The map

| Primitive | Reference (HF `modeling_qwen2.py`) | Our code |
|---|---|---|
| Backbone | `Qwen2Model` + `Qwen2ForCausalLM` | `Qwen2ForCausalLM` + `_Qwen2InnerModel` in [src/mini_infer/models/qwen2.py:94](../../src/mini_infer/models/qwen2.py#L94), [:69](../../src/mini_infer/models/qwen2.py#L69) |
| Decoder layer | `Qwen2DecoderLayer` | shared `TransformerBlock` in [blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py) |
| GQA attention with QKV biases | `Qwen2Attention` (`attention_bias=True`) | shared `GroupedQueryAttention` with `qkv_bias=True` in [blocks/gqa.py](../../src/mini_infer/models/blocks/gqa.py) |
| Standard RoPE | `Qwen2RotaryEmbedding` | shared `RotaryEmbedding` in [blocks/rope.py](../../src/mini_infer/models/blocks/rope.py) |
| RMSNorm | `Qwen2RMSNorm` | shared `RMSNorm` in [blocks/rmsnorm.py](../../src/mini_infer/models/blocks/rmsnorm.py) |
| SwiGLU FFN | `Qwen2MLP` | shared `SwiGLU` in [blocks/swiglu.py](../../src/mini_infer/models/blocks/swiglu.py) |
| Tied embeddings | `tie_word_embeddings=True` in config | `Qwen2ForCausalLM.__init__` aliases `lm_head.weight = embed_tokens.weight` ([qwen2.py:112](../../src/mini_infer/models/qwen2.py#L112)) |

Bit-parity is exercised against HF transformers' `Qwen2Attention` on
synthetic configs at FP32 with cosine-sim > 0.999, via the shared
`GroupedQueryAttention` path. Golden tests at temperature=0 validate
end-to-end against HF using Qwen2.5-0.5B-Instruct as the pinned smoke.

Tests under `tests/unit/`; layout intentionally not enumerated.

## QKV biases

The one *actually-novel* primitive in Qwen2 (vs Llama). Before
projecting hidden states into Q / K / V, each projection adds a
learnable bias:

```python
q = q_proj.weight @ x + q_proj.bias
k = k_proj.weight @ x + k_proj.bias
v = v_proj.weight @ x + v_proj.bias
```

The output projection `o_proj` and the FFN projections stay bias-free,
matching Llama.

### Why biases on QKV

Mostly an architectural choice the Qwen team made early; the bias adds
`3 * num_kv_heads * head_dim` parameters per layer (negligible) but
helps train-time stability in their setup. Qwen3 dropped them again
without a quality regression on their reported benchmarks, which
suggests the bias was more incidental than load-bearing.

### Where the support hooks in

The shared `GroupedQueryAttention` constructor takes a `qkv_bias` bool
flag:

```python
self.q_proj = ColumnParallelLinear(hidden_size, num_q_heads * head_dim, bias=qkv_bias)
self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
```

When `qkv_bias=True`, each linear constructs its bias parameter; when
False, no bias. The o_proj is hardcoded to `bias=False` either way.

`_Qwen2InnerModel.__init__` passes `qkv_bias=True` ([qwen2.py:84](../../src/mini_infer/models/qwen2.py#L84));
`_LlamaInnerModel.__init__` passes `qkv_bias=False`. That single line
flip is the architectural difference between the two families.

## Why a separate family file (not inheritance)

Mistral inherits from `LlamaForCausalLM` because the architectures are
literally identical. Qwen2 has its own `_Qwen2InnerModel` and its own
`Qwen2ForCausalLM`, both very close to Llama's but not subclassing.
Three reasons:

1. **HF parameter naming.** Qwen2's checkpoints include
   `q_proj.bias` / `k_proj.bias` / `v_proj.bias` tensors that Llama's
   load path would reject as "unexpected keys". The two loaders diverge
   on what counts as a valid state dict.
2. **lm_head construction comment.** Qwen2's lm_head is always built as
   a `Linear` (then optionally aliased to `embed_tokens.weight`),
   because downstream int8-quant walkers expect a consistent module
   shape. That construction note lives in `Qwen2ForCausalLM.__init__`
   ([qwen2.py:103](../../src/mini_infer/models/qwen2.py#L103)) and
   would be a footgun to inherit silently.
3. **Independent evolution headroom.** Qwen2 → Qwen2.5 didn't change
   the architecture, but Qwen3 did (per-head Q/K norm). Having Qwen2
   stand alone makes the family-by-family delta visible.

If we were tightening the codebase today, Qwen2 *could* subclass Llama
the way Mistral does, by adding a `qkv_bias=True` flag to `LlamaConfig`
and toggling it. The current shape exists for the reasons above; the
duplication is ~80 lines, which is below the threshold where the
inheritance trade-off becomes obviously better.

## Decoder layer assembly

Same pre-norm shape as Llama:

```
residual = h
h = input_layernorm(h)
h = self_attn(h, ...)        # GQA with QKV biases
h = residual + h
residual = h
h = post_attention_layernorm(h)
h = mlp(h)                   # SwiGLU, no biases
h = residual + h
```

**Reference (`Qwen2DecoderLayer.forward`)**: standard.

**Our code**: shared `TransformerBlock` from
[blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py).
No Qwen2-specific decoder layer; the family-specific behaviour
(QKV biases) lives entirely inside `GroupedQueryAttention` via the
`qkv_bias=True` flag.

## Cache structure

Standard paged KV cache. Same shape Llama uses. No family-specific
cache contract.

## Validation contract

- **Bit-parity vs HF**: synthetic `Qwen2Attention` configs at FP32
  produce activations within cosine-sim > 0.999 of our
  `GroupedQueryAttention` output with `qkv_bias=True`.
- **Golden test vs HF at temperature=0**: Qwen2.5-0.5B-Instruct is the
  pinned smoke and the project's *primary* CI model (PD scheduler,
  spec decoding, API tests all run through it). Same generated tokens
  for a fixed prompt seed.
- **Weight-load round-trip**: a Qwen2 state_dict (including the
  `.bias` entries) loaded through `load_state_dict_with_tp`
  reproduces the original tensors per layer.

The pinned HF revision for Qwen2.5-0.5B-Instruct lives in
[`tests/_pinned_models.toml`](../../tests/_pinned_models.toml); the
bit-parity CI workflow runs `requires_model`-marked tests against
that revision.

## Why Qwen2 is the day-to-day smoke

Three things make Qwen2.5-0.5B-Instruct the right CI model:

1. **Fits M1 fp16 trivially.** 0.5B parameters ~= 1 GB at fp16, which
   M1 Macs handle as a normal RAM allocation. CI runs without GPU.
2. **Real instruction-tuned model.** Greedy generation produces
   coherent output, so a golden-test failure is obviously wrong
   (catastrophic divergence) rather than ambiguous (slight
   perplexity wobble).
3. **Architecturally identical to the larger Qwen2.5 family.**
   Bit-parity at 0.5B gives reasonable confidence that the same
   path works at 7B / 32B / 72B, modulo memory-bound concerns that
   show up only on Modal.

This is why almost every scheduler / API / spec-decode test in the
suite reaches for Qwen2.5-0.5B. The architecture's main role in the
project isn't its novelty (it has very little); it's that it's the
**ergonomic smoke**.

## Where we diverged + why

1. **QKV bias is a constructor flag, not a forked attention class.**
   Same pattern Qwen3 uses for per-head Q/K norm, Gemma 4 uses for
   `attention_k_eq_v`, etc. Family quirks compose better as flags on
   the shared block than as forked attention classes.
2. **lm_head always constructed as `Linear`, then optionally aliased.**
   Comment at [qwen2.py:103](../../src/mini_infer/models/qwen2.py#L103)
   explains why: the int8-quant walker expects to see a consistent
   module type for the LM head regardless of tying.

## Pointers

- **Reference**: HF transformers `transformers/models/qwen2/modeling_qwen2.py`.
- **Our model class**: [src/mini_infer/models/qwen2.py](../../src/mini_infer/models/qwen2.py).
- **GQA block** (shared with Llama / Qwen3 / Mistral / SmolLM2):
  [blocks/gqa.py](../../src/mini_infer/models/blocks/gqa.py).
- **TransformerBlock** (shared decoder shape):
  [blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py).
- **Pinned smoke**: Qwen2.5-0.5B-Instruct, see
  [tests/_pinned_models.toml](../../tests/_pinned_models.toml).

## What's still open

- **Larger checkpoint validation** (Qwen2.5-32B, Qwen2.5-72B). Local
  validation runs on 0.5B / 1.5B. The 7B and larger variants need
  Modal hardware; same loader path, not blocked on code.
- **Qwen2-MoE variants** (Qwen2-MoE / Qwen2.5-MoE). Not yet
  implemented. Would reuse the Qwen2 attention path + Mixtral's
  `MoEFFN`. The HF architecture key is `Qwen2MoeForCausalLM`,
  which would register as a separate family.
- **Qwen2-VL (multimodal).** Out of scope per the project's text-only
  positioning. Mentioned for completeness.
