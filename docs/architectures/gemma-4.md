# Gemma 4 (text-only) walkthrough

A line-by-line correspondence between Google's Gemma 4 release, the
HuggingFace transformers reference (`Gemma4ForConditionalGeneration` /
`Gemma3ForCausalLM` text decoder paths), and mini-infer's from-scratch
port of the Gemma 4 31B text decoder.

> **Scope**: text-only. Vision and audio towers in the published
> `google/gemma-4-31B-it` checkpoint are filtered out at load time;
> mini-infer does not implement multimodal towers.

## TL;DR

Gemma 4 text introduces the most family-specific machinery of any
model in mini-infer's registry. Every primitive below is *different
from a Llama baseline*:

1. **Heterogeneous-KV per layer-type.** Sliding layers carry
   `(num_kv_heads=16, head_dim=256)`; full ("global") layers carry
   `(num_kv_heads=4, head_dim=512)`. First model in the registry to
   actually exercise mini-infer's heterogeneous-KV `BlockPool`.
2. **Dual RoPE with different `head_dim` per type.** Sliding RoPE:
   `head_dim=256`, `theta=10000`, full rotation. Full-layer RoPE:
   `head_dim=512`, `theta=1000000`, **partial rotation**
   (`partial_rotary_factor=0.25` — first 64 dims rotate, rest pass
   through).
3. **`attention_k_eq_v=True` on full layers.** No separate `v_proj`;
   V reuses the post-`k_proj` tensor BEFORE `k_norm` and BEFORE RoPE.
4. **Unscaled `v_norm` per layer.** RMSNorm with no learnable weight
   applied to V after capture. Every layer.
5. **`layer_scalar` per layer.** A `(1,)` buffer applied to the block
   output as `hidden_states *= layer_scalar`.
6. **Final logit softcap.** `logits = tanh(logits / 30) * 30` after
   `lm_head`. Bounded output for numerical stability at long context.
7. **Softmax scale of 1.0.** `query_scale=1.0`; `q_norm` / `k_norm`
   absorb the magnitude. Different from the standard `1/sqrt(d_head)`.
8. **Multimodal weight filter at load time.** Vision / audio / MM
   projector / embed_vision prefixes are stripped before
   `load_state_dict`.
9. **Model-side attention-backend override.** `head_dim=512` on full
   layers exceeds flash-attn 2's and FlashInfer prefill's supported
   range; the model forces the `"torch"` SDPA reference backend.

This is the same conclusion vLLM and SGLang reach for Gemma 4: their
Triton unified kernel is the answer; we use the materialized SDPA
fallback because that's what the project's non-goal list says.

## The map

| Primitive | Reference (HF `modeling_gemma3.py` / `modeling_gemma4_text.py`) | Our code |
|---|---|---|
| Backbone (text decoder) | `Gemma3TextModel` + `Gemma4ForConditionalGeneration` text head | `Gemma4ForCausalLM` + `_Gemma4InnerModel` in `src/mini_infer/models/gemma4.py` (L181, L151) |
| Decoder layer | `Gemma3DecoderLayer` | `Gemma4DecoderLayer` in `src/mini_infer/models/blocks/gemma4_decoder_layer.py` (L39) |
| GQA with heterogeneous head shapes | `Gemma3Attention` | `GQAAttention` in `blocks/gqa.py` (shared with Llama/Qwen/Mistral; Gemma 4 declares per-layer shape via `per_layer_kv_shape()`) |
| Sliding-window / global alternation | `Gemma3Attention` `sliding_window` field | per-layer `LayerAttentionSpec` from `Gemma4Config.per_layer_attention()` |
| Dual RoPE (sliding theta=10000, global theta=1000000 partial 0.25) | `Gemma3RotaryEmbedding` (two instances) | two `RotaryEmbedding` instances in `_Gemma4InnerModel` (one per type) |
| `attention_k_eq_v` (V = post-`k_proj` tensor) | `if config.attention_k_eq_v` branch in `Gemma3Attention.forward` | `attention_k_eq_v` flag threaded through `GQAAttention.forward` |
| Unscaled `v_norm` (no learnable weight) | `Gemma3Attention.v_norm` (RMSNorm without scale) | per-layer `v_norm` in `Gemma4DecoderLayer` |
| Per-layer `layer_scalar` | `Gemma3DecoderLayer.layer_scalar` (buffer) | `layer_scalar` buffer in `Gemma4DecoderLayer` (L39+) |
| Standard RMSNorm (not Gemma 3's `(1+w)*x`) | `Gemma3RMSNorm` (with the `(1+w)*x` transform on Gemma 3 only) | `RMSNorm` in `blocks/rmsnorm.py` (standard form for Gemma 4; the offset variant is `GemmaRMSNorm` in `blocks/gemma_rmsnorm.py` and is Gemma 3-only) |
| Final logit softcap | `tanh(logits / 30) * 30` inside `Gemma4ForConditionalGeneration` | applied in `Gemma4ForCausalLM.forward` after `lm_head` |
| Softmax scale 1.0 | hard-coded scale in `Gemma3Attention.forward` | `query_scale=1.0` passed into `GQAAttention` |
| Multimodal weight filter | not applicable (HF holds vision tower separately) | `_MULTIMODAL_PREFIX_RE` in `gemma4.py:65`; filters at load time |
| Attention backend override | not applicable (HF runs reference SDPA / Triton unified) | `Gemma4ForCausalLM.required_attention_backend()` returns `"torch"` (line near L181) |

Bit-parity is exercised against HF transformers' `Gemma3Attention`
(the text path Gemma 4 31B inherits in the published HF version) on
synthetic configs at FP32; the `GQAAttention` block matches at
cosine-sim > 0.999 when Gemma 4's per-layer shapes + `attention_k_eq_v`
+ `v_norm` + dual-RoPE config are wired in. End-to-end golden output
against HF is the integration gate. Tests live under `tests/unit/`;
specific layout intentionally not enumerated.

## Heterogeneous-KV per layer-type

Gemma 4's signature systems contribution: **two different KV
shapes inside one model**.

| Layer type | `num_kv_heads` | `head_dim` | RoPE | Q heads |
|---|---|---|---|---|
| Sliding | 16 | 256 | theta=10000, full rotation | 32 |
| Full ("global") | 4 | 512 | theta=1000000, partial 0.25 | 32 |

**Why two shapes?** Sliding layers are local (windowed attention); a
larger number of KV heads at smaller `head_dim` packs well for the
local context. Full layers are global; fewer KV heads at larger
`head_dim` reduces the per-token KV cost on the unbounded context.

**Reference**: HF transformers carries per-layer config via the
`sliding_window` / `is_sliding` flag on each layer; the attention
class branches on it.

**Our code**: mini-infer's `BlockPool` was extended (project commit
history; ADRs 003 + 005) to support per-layer `(num_kv_heads,
head_dim)` via the `per_layer_kv_shape()` hook. `Gemma4Config`
implements that hook; the pool then allocates per-layer storage of
the right shape, no special-casing inside the attention kernel.

```python
# Gemma4Config.per_layer_kv_shape() returns:
[
    (16, 256),  # layer 0: sliding
    (16, 256),  # layer 1: sliding
    ...
    (4, 512),   # layer N-1: full (every Nth layer)
    ...
]
```

The `LayerAttentionSpec` lives in `cache/block_pool.py` and is the
generic primitive; Gemma 4 is its first real consumer.

## Dual RoPE

Two separate `RotaryEmbedding` instances. The dispatcher picks one
based on the layer's type at forward time.

**Sliding RoPE** (`head_dim=256`, theta=10000, full rotation): standard
form. Identical to Llama RoPE at the same `head_dim`.

**Full RoPE** (`head_dim=512`, theta=1000000, partial rotation
`factor=0.25`): rotates only the first `head_dim * factor = 128 / 2 =
64` channel pairs (interleaved) of the 256 RoPE channels = 64 rotated
dims out of 256. The rest pass through unchanged.

**Reference**: HF transformers instantiates two `Gemma3RotaryEmbedding`
modules; the attention forward picks the right one based on
`is_sliding`.

**Our code**: two `RotaryEmbedding` instances in `_Gemma4InnerModel.__init__`,
each parameterised with the appropriate `head_dim`, `theta`, and
`partial_rotary_factor`. The decoder layer is told its type at
construction and references the right rotary buffer.

### Why partial rotation on full layers

The Gemma 4 paper's framing: full-layer attention spans the entire
context. RoPE's encoding becomes less informative at large position
deltas. Partial rotation lets the model carry some channels as
position-free (good for global semantics) and some channels as
rotated (good for relative-position cues). Empirically this combines
well with the larger `theta=1000000` which slows the rotation rate
across the rotated channels.

## `attention_k_eq_v` (V is K pre-norm pre-RoPE)

Full layers don't have a separate `v_proj`. Instead, V is **the same
tensor that came out of `k_proj`**, captured BEFORE `k_norm` and
BEFORE RoPE.

**Reference**: `Gemma3Attention.forward` branches on
`config.attention_k_eq_v`. When true: `k = k_proj(x); v = k.clone();
k = k_norm(k); k = apply_rope(k, ...)`. V skips both transformations.

**Our code**: `GQAAttention.forward` accepts an `attention_k_eq_v`
flag; when true, V is captured from the post-`k_proj` activation
before any further transformation.

**Why this works**: K and V have the same shape in standard MHA/GQA
(`num_kv_heads × head_dim`). If a single shared projection can serve
both, you save `hidden_size × num_kv_heads × head_dim` parameters per
layer (an entire `v_proj` matrix). On full Gemma 4 layers with
`head_dim=512` that's not trivial — `v_proj` would be one of the
largest matrices in the model.

### What about norm + RoPE on V?

`k_norm` and RoPE are *applied to K only*. V (= pre-norm K) keeps
the un-normed pre-rotated values. This is the trick: K needs the
positional encoding for the attention dot product; V doesn't (V is
just a value lookup once attention has decided which K to attend to).

## Unscaled `v_norm`

Every layer (sliding AND full) applies an unscaled RMSNorm to V
after capture. Unscaled = no learnable weight, just the
normalization step. The point is numerical: V's magnitude gets
bounded before it enters the SDPA matmul.

**Reference**: `Gemma3Attention.v_norm` — RMSNorm with `weight=None`
(or equivalently, weight is the all-ones tensor and not learned).

**Our code**: `v_norm` parameter on `Gemma4DecoderLayer` configured
to skip the scale path. The RMSNorm primitive supports the unscaled
path via a `use_weight=False` flag.

## Per-layer `layer_scalar`

A scalar buffer `(1,)` per layer that multiplies the post-block
hidden state. Effectively a per-layer gain control on the residual
flow.

**Reference**: `Gemma3DecoderLayer.layer_scalar` (registered as a
buffer, not a parameter — value is from the checkpoint, not trained
during fine-tuning).

**Our code**: `layer_scalar` buffer on `Gemma4DecoderLayer`. Applied
to the final residual addition (`h = h + block_out * layer_scalar`).

## Standard RMSNorm (not Gemma 3's offset variant)

Gemma 3 uses an unusual RMSNorm: `(1 + weight) * normalised_x` instead
of the standard `weight * normalised_x`. The weights on disk are
small (close to 0) and the `+1` makes them function as "small
adjustments around 1.0".

**Gemma 4 reverts this**: the weights on disk are in their "final"
form (close to 1.0 directly), and the RMSNorm is the standard
`weight * normalised_x`.

**Reference**: `Gemma3RMSNorm` is the offset variant (Gemma 3 only).
Gemma 4 reuses the standard RMSNorm class.

**Our code**: `RMSNorm` in `blocks/rmsnorm.py` is the standard form
(used by Gemma 4). `GemmaRMSNorm` in `blocks/gemma_rmsnorm.py` is
the offset variant (used by Gemma 3 only). Two classes, no
conditional branches inside one.

## Final logit softcap

After `lm_head`, logits are bounded via `tanh(x / 30) * 30`. Smooth
saturation past `|x| ≈ 30`; unchanged in the small-magnitude region.

**Reference**: `Gemma4ForConditionalGeneration.forward` applies the
softcap inline after the LM head.

**Our code**: `Gemma4ForCausalLM.forward` applies the same softcap.
The constant 30 is exposed as `final_logit_softcapping` in the config
(per HF naming).

## Softmax scale of 1.0

Standard attention uses `softmax(QK / sqrt(d_head))`. Gemma 4 uses
`softmax(QK)` directly — scale of 1.0.

Why? `q_norm` and `k_norm` are applied to Q and K respectively
before the dot product. They normalise per-head, which absorbs the
magnitude that `1/sqrt(d_head)` would have scaled. The effective
softmax temperature is similar to a scaled version of unnormalised
Q/K.

**Reference**: hard-coded `scale=1.0` in `Gemma3Attention.forward`.

**Our code**: `query_scale=1.0` passed into `GQAAttention`. Default
in `GQAAttention` is `1/sqrt(d_head)`; Gemma 4 overrides.

## Multimodal weight filter at load time

The published `google/gemma-4-31B-it` checkpoint contains:
- text decoder weights (we want these)
- `vision_tower`, `audio_tower`, `multi_modal_projector`,
  `embed_vision`, `embed_audio` (we filter these out)

**Our code**: `_MULTIMODAL_PREFIX_RE` (`gemma4.py:65`) is a regex
applied during `load_state_dict`. Keys matching the multimodal
prefixes are dropped; only the text decoder loads.

When `attention_k_eq_v` is true, the full-layer `v_proj` weight is
also unused and gets filtered out (the regex `_LAYER_IDX_RE` plus a
layer-type check). This keeps the loader strict (no unexpected /
unused keys) while still loading the same checkpoint.

## Model-side attention-backend override

`head_dim=512` on full layers is the killer detail: neither
flash-attn 2 nor FlashInfer prefill supports head dims that large.
Both crash or fall back silently. mini-infer's response: force the
PyTorch SDPA reference backend at the model level.

**Reference**: HF transformers picks between SDPA and a Triton
unified kernel based on availability. vLLM and SGLang both ship a
custom Triton kernel for this case.

**Our code**: `Gemma4ForCausalLM.required_attention_backend()`
returns `"torch"`. The `ModelRunner.from_pretrained` checks this hook
and overrides `attention_backend="torch"` regardless of the caller's
choice. Same pattern V2's MLA uses for its asymmetric Q/K vs V
head_dim case.

## Decoder layer assembly

**Reference (`Gemma3DecoderLayer.forward`)**: pre-norm decoder shape.
Attention block: input_layernorm → attention → post_attention_layernorm
→ residual → block_out *= `layer_scalar`. MLP block: similar shape.

**Our code (`Gemma4DecoderLayer.forward`)**: same shape. The
`layer_scalar` multiply happens at the end of the residual path.

## Validation contract

Bit-parity against HF transformers' `Gemma3Attention` (the text path
the published Gemma 4 31B uses) at FP32 with cosine-sim > 0.999 on
synthetic configs. Golden test against HF generates the same tokens
at temperature=0. Tests under `tests/unit/`; layout intentionally not
enumerated here.

Real-hardware validation: Gemma 4 31B (62 GB at bf16) doesn't fit a
single M1; B200 Modal run validated end-to-end. This was the model
that proved out the heterogeneous-KV BlockPool path on real weights.

## Where we diverged + why

1. **Two RMSNorm classes**. `RMSNorm` (standard) for Gemma 4 + Llama +
   Qwen + everyone else; `GemmaRMSNorm` (offset, `(1+w)*x`) for
   Gemma 3 only. No conditional flag inside one class; two classes,
   two imports. Easier to reason about.
2. **`GQAAttention` carries Gemma 4's quirks as flags, not a subclass**.
   `attention_k_eq_v`, `query_scale`, dual-RoPE awareness are all
   constructor parameters on the shared GQA block. The dispatch in
   `forward` branches on them. This keeps Gemma 4's variations
   localised to its config + decoder-layer construction; the
   `GQAAttention` block is the shared primitive.
3. **Multimodal filtering at load time, not at config time**. HF's
   `Gemma4ForConditionalGeneration` config has the multimodal sub-
   configs nested; we read the text-decoder sub-config and filter the
   checkpoint at load time. Same outcome (text-only model loaded),
   cleaner separation.

## Pointers

- **Reference**: HF transformers
  `transformers/models/gemma3/modeling_gemma3.py` (the text path
  Gemma 4 uses in the published HF version; some later transformers
  versions split this into `models/gemma4_text/`).
- **Our model class**: `src/mini_infer/models/gemma4.py`.
- **Decoder layer**: `src/mini_infer/models/blocks/gemma4_decoder_layer.py`.
- **GQA block (shared)**: `src/mini_infer/models/blocks/gqa.py`.
- **Heterogeneous-KV pool**: `src/mini_infer/cache/block_pool.py::LayerAttentionSpec`.
- **Standard RMSNorm**: `src/mini_infer/models/blocks/rmsnorm.py`.
- **Gemma 3 offset RMSNorm**: `src/mini_infer/models/blocks/gemma_rmsnorm.py`.

## What's still open

- **Gemma 4 MoE variants (26B-A4B)**. Not yet implemented; would
  reuse most of the text decoder path plus the MoE block from
  `blocks/mixtral_moe.py`. Listed in the Gemma 4 docstring as out of
  scope for the current text-only port.
- **Gemma 4 PLE variants (E2B / E4B)**. Per-Layer Embeddings + shared
  KV across layers. Separate code path, different from the current
  text decoder. Out of scope today.
