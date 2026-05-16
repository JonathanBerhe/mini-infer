# Gemma 3 walkthrough

A line-by-line correspondence between Google's Gemma 3 release,
HuggingFace transformers' `Gemma3ForCausalLM` (text-only), and
mini-infer's from-scratch port.

> **Scope**: text-only. Gemma 3 multimodal variants (vision / audio)
> ship under a separate HF architecture string and aren't covered here.

## TL;DR

Gemma 3 packs more family-specific quirks than Llama / Qwen3 /
Mistral but stays homogeneous-KV (same `(num_kv_heads, head_dim)` on
every layer, unlike Gemma 4). The novelty is in the layer-type
alternation and the norm scheme.

Five primitives that distinguish Gemma 3 from a Llama baseline:

1. **Sliding-window + global alternating attention.** Typically 5:1
   sliding-to-global ratio. Sliding layers attend over a fixed window;
   global layers attend to the full context.
2. **Dual RoPE with the same `head_dim`, different `theta`.** Sliding
   RoPE uses `theta=10000` (standard); global RoPE uses
   `theta=1000000` (slower rotation rate for long contexts).
3. **Per-head Q/K norm.** Same shape Qwen3 uses but with Gemma 3's
   offset RMSNorm.
4. **Sandwich norms.** Norms surround BOTH the attention block AND
   the FFN block (`pre-norm + sub-block + post-norm`). Llama has only
   pre-norms.
5. **GemmaRMSNorm: `(1 + weight) * normalised_x`.** The weights on
   disk are small (close to 0) and the `+1` makes them function as
   small adjustments around 1.0. Distinctive to Gemma 3; Gemma 4
   reverts to the standard form.
6. **GeGLU activation in the FFN** (`gelu_pytorch_tanh` variant),
   not SwiGLU.
7. **Embedding scaling by `sqrt(hidden_size)`** on the embed_tokens
   output. Tied embeddings throughout.

## The map

| Primitive | Reference (HF `modeling_gemma3.py`) | Our code |
|---|---|---|
| Backbone | `Gemma3Model` + `Gemma3ForCausalLM` | `Gemma3ForCausalLM` + `_Gemma3InnerModel` in `src/mini_infer/models/gemma3.py` (L116, L88) |
| Decoder layer | `Gemma3DecoderLayer` | `GemmaDecoderLayer` in `blocks/gemma_decoder_layer.py` |
| GQA attention | `Gemma3Attention` | `GQAAttention` in `blocks/gqa.py` (shared) with `q_norm` / `k_norm` / `query_scale=1.0` flags |
| Sliding-window vs global per-layer | `Gemma3Attention.is_sliding` flag | per-layer `LayerAttentionSpec` from `Gemma3Config.per_layer_attention()` |
| Dual RoPE (same head_dim, different theta) | two `Gemma3RotaryEmbedding` instances | two `RotaryEmbedding` instances in `_Gemma3InnerModel` |
| Per-head Q/K norm (offset RMSNorm) | `Gemma3Attention.q_norm` / `k_norm` (`Gemma3RMSNorm`) | `q_norm=True` / `k_norm=True` flag on `GQAAttention` with `GemmaRMSNorm` |
| **Sandwich norms** | `Gemma3DecoderLayer` has `pre_feedforward_layernorm` + `post_feedforward_layernorm` etc. | `GemmaDecoderLayer` carries the same four-norm shape (pre + post around attention, pre + post around FFN) |
| GemmaRMSNorm `(1+w)*x` | `Gemma3RMSNorm` | `GemmaRMSNorm` in `blocks/gemma_rmsnorm.py` |
| GeGLU FFN | `Gemma3MLP` (gate/up/down with `gelu_pytorch_tanh`) | `GeGLU` (or `SwiGLU(activation=...)`) — see `blocks/swiglu.py` |
| Embedding scaling | `hidden_states *= sqrt(hidden_size)` after embed lookup | same scaling in `_Gemma3InnerModel.forward` |
| Tied embeddings | `tie_word_embeddings=True` | `Gemma3ForCausalLM` reuses `embed_tokens.weight` for `lm_head` |

Bit-parity is exercised against HF transformers' `Gemma3Attention` on
synthetic configs at FP32 with cosine-sim > 0.999. Golden output at
temperature=0 against HF validates end-to-end. Tests under
`tests/unit/`; layout intentionally not enumerated.

## Sliding-window + global alternating attention

The dominant axis of variation in Gemma 3.

| Layer type | Ratio (typical) | Attention | RoPE theta |
|---|---|---|---|
| Sliding | 5 of every 6 | windowed (sliding_window tokens) | 10000 |
| Global | 1 of every 6 | full (entire context) | 1000000 |

The sliding-window primitive bounds the per-token attention cost on
long contexts (the model can't see past `sliding_window` tokens on
sliding layers), while the global layers ensure full-context
information mixes in periodically.

**Reference**: HF transformers carries `is_sliding` per layer; the
attention class branches on it for the window mask.

**Our code**: `Gemma3Config.per_layer_attention()` returns a
`list[LayerAttentionSpec]` where each layer is one of
`AttentionType.SLIDING` or `AttentionType.FULL`. `BlockPool` allocates
homogeneous KV (same `(num_kv_heads, head_dim)` for both types — this
is what makes Gemma 3 simpler than Gemma 4), and the attention
dispatcher applies the window mask only on sliding layers.

Unlike Gemma 4, **the `(num_kv_heads, head_dim)` is the same** on
both layer types. The only difference is the attention pattern (windowed
vs full) and the RoPE theta.

## Dual RoPE with same head_dim, different theta

**Sliding RoPE**: `head_dim`, `theta=10000`, full rotation.

**Global RoPE**: same `head_dim`, `theta=1000000`, full rotation.

Same shape on both sides; only the rotation rate (governed by `theta`)
differs. The slower rotation rate on global layers makes the relative-
position encoding more uniform across long distances.

**Reference**: two `Gemma3RotaryEmbedding` instances at construction
time; the attention forward picks one based on `is_sliding`.

**Our code**: same — two `RotaryEmbedding` instances in
`_Gemma3InnerModel.__init__`, each parameterised with its theta.

Compare to Gemma 4 which also uses dual RoPE but with **different
`head_dim`** per type (and partial rotation on the global side).
Gemma 3's dual RoPE is the simpler version.

## Per-head Q/K norm with offset RMSNorm

Same per-head norm shape as Qwen3 (one weight vector per attention
head, shape `(num_heads, head_dim)` or `(num_kv_heads, head_dim)`).
But where Qwen3 uses the standard `RMSNorm` (`weight * normalised_x`),
Gemma 3 uses `GemmaRMSNorm` (`(1 + weight) * normalised_x`).

The weights on disk are stored close to 0; `(1 + weight)` makes them
behave as small multiplicative adjustments around 1.0. Numerically
equivalent to "store weights close to 1 and use standard RMSNorm"
(which is what Gemma 4 does), but easier to fine-tune (small
deviations from 0 are smoother than small deviations from 1).

**Reference**: `Gemma3RMSNorm`'s forward is
`return self.weight + 1.0) * normalised_x` (or the equivalent
in-place form).

**Our code**: `GemmaRMSNorm` in `blocks/gemma_rmsnorm.py` implements
the offset form. `GQAAttention` accepts a `norm_class` parameter so
the per-head norm uses `GemmaRMSNorm` for Gemma 3 and `RMSNorm` for
Qwen3 / Gemma 4.

### Softmax scale

Gemma 3 uses softmax scale 1.0 (same as Gemma 4), letting the q/k
norm magnitudes absorb the standard `1/sqrt(d_head)` scale.

## Sandwich norms

Llama / Qwen / Mistral use **pre-norm only**: normalise before each
sub-block, add residual after.

```
# Pre-norm (Llama)
h = h + attn(input_layernorm(h))
h = h + mlp(post_attention_layernorm(h))
```

Gemma 3 uses **sandwich norms**: normalise BEFORE AND AFTER each
sub-block.

```
# Sandwich norm (Gemma 3)
h = h + post_attn_layernorm(attn(pre_attn_layernorm(h)))
h = h + post_ffn_layernorm(mlp(pre_ffn_layernorm(h)))
```

Four norms per layer instead of two. The extra norms stabilise the
residual stream's magnitude through the block.

**Reference**: `Gemma3DecoderLayer` has `pre_attention_layernorm`,
`post_attention_layernorm`, `pre_feedforward_layernorm`, and
`post_feedforward_layernorm`.

**Our code**: `GemmaDecoderLayer` in `blocks/gemma_decoder_layer.py`
carries the same four-norm shape. The four norms are all
`GemmaRMSNorm` (offset variant).

## GeGLU instead of SwiGLU

Same shape as SwiGLU (`gate_proj`, `up_proj`, `down_proj`) but with
GELU (`gelu_pytorch_tanh` variant) instead of SiLU on the gate.

**Reference**: `Gemma3MLP.forward`:
`down(gelu_pytorch_tanh(gate(x)) * up(x))`.

**Our code**: same shape. `SwiGLU` accepts an activation parameter;
Gemma 3 passes `gelu_pytorch_tanh`. Cleaner than carrying a separate
`GeGLU` class for one activation difference.

## Embedding scaling by sqrt(hidden_size)

After the embedding lookup, the hidden state is multiplied by
`sqrt(hidden_size)`. Old trick from the original Transformer paper;
Gemma 3 carries it from Gemma 2's lineage.

**Reference**: applied inside `Gemma3Model.forward` right after
`embed_tokens(input_ids)`.

**Our code**: same step in `_Gemma3InnerModel.forward`.

## Decoder layer assembly

The sandwich-norm shape is the headline:

```python
# Attention block (sandwich)
residual = h
h = pre_attention_layernorm(h)
h = self_attn(h, ...)
h = post_attention_layernorm(h)
h = residual + h

# FFN block (sandwich)
residual = h
h = pre_feedforward_layernorm(h)
h = mlp(h)
h = post_feedforward_layernorm(h)
h = residual + h
```

**Reference (`Gemma3DecoderLayer.forward`)**: exactly this shape.

**Our code (`GemmaDecoderLayer.forward`)**: same.

## Validation contract

Bit-parity against HF transformers' `Gemma3Attention` on synthetic
configs at FP32 with cosine-sim > 0.999. Golden output against HF at
temperature=0 validates end-to-end. Gemma 3 1B + 4B variants fit
M1 fp16 dev hardware and run as everyday smokes.

Tests under `tests/unit/`; layout intentionally not enumerated.

## Where we diverged + why

1. **Sandwich norms are part of `GemmaDecoderLayer`, not added as a
   flag to `TransformerBlock`**. The four-norm shape is structurally
   different enough from the pre-norm shape that a flag on
   `TransformerBlock` would either complicate the standard path or
   add unused norms in the Llama case. A separate decoder layer file
   keeps both paths clean.
2. **`GemmaRMSNorm` is a separate class from `RMSNorm`**. Same math
   modulo the `(1+w)` offset, but used by Gemma 3 only. Gemma 4 reverts
   to the standard form; two classes, no conditional flag.
3. **`SwiGLU` accepts an activation parameter** rather than carrying a
   separate `GeGLU` class. The MLP shape is identical; only the
   activation differs.

## Pointers

- **Reference**: HF transformers
  `transformers/models/gemma3/modeling_gemma3.py` (text path).
- **Our model class**: `src/mini_infer/models/gemma3.py`.
- **Decoder layer**: `src/mini_infer/models/blocks/gemma_decoder_layer.py`.
- **GemmaRMSNorm (offset)**: `src/mini_infer/models/blocks/gemma_rmsnorm.py`.
- **GQA block (shared)**: `src/mini_infer/models/blocks/gqa.py`.

## What's still open

- **Gemma 3 multimodal**. Vision + audio towers register under a
  separate HF architecture string. Out of scope for the text-only
  port today; would compose on top of this base model class.
- **Gemma 3 27B / larger checkpoints**. Local validation runs on 1B
  and 4B; 27B needs Modal hardware. Same loader path; not blocked
  on code.
