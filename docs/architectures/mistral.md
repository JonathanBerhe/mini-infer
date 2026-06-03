# Mistral walkthrough

A line-by-line correspondence between Mistral AI's Mistral 7B / Small /
Large releases, HuggingFace transformers' `MistralForCausalLM`, and
mini-infer's from-scratch port.

Mistral is the **shortest family file in the registry** (~30 lines).
It's structurally identical to Llama; the file exists to register the
HF architecture string and a type-distinct config class. Read this if
you want to see what a "nothing-new" family looks like, and what
mini-infer deliberately doesn't implement (sliding-window attention).

## TL;DR

Mistral 7B v0.1 introduced two architectural ideas:

1. **Grouped-Query Attention.** Mistral 7B was a GQA-from-day-one
   release; Llama 2 only added GQA at the 70B size. By the time of
   this writeup, GQA is standard across every Llama-shape family and
   is shared in `GroupedQueryAttention`.
2. **Sliding-Window Attention** (SWA, window=4096). Each token only
   attends to the previous 4096 tokens, not the full prefix. Cheaper
   per-token attention at long context; trades global context for
   compute.

Of those, only GQA carried forward. **Mistral 7B v0.3 dropped SWA**;
the `sliding_window` field is still in the HF config for v0.1 / v0.2
but most production deployments treat it as full attention anyway.
Subsequent Mistral releases (Mistral Small, Mistral Large, Codestral)
also don't use SWA.

Our `MistralForCausalLM` therefore reuses everything Llama-shape via
inheritance:

- Same `GroupedQueryAttention`, `TransformerBlock`, `SwiGLU`, `RMSNorm`,
  `RotaryEmbedding`.
- Same forward / load / weight machinery.
- Different `HF_ARCHITECTURE` string so the registry dispatch works.
- Different config type so isinstance checks distinguish the two.

The entire family file is 31 lines, of which 24 are imports and class
boilerplate.

## The map

| Primitive | Reference (HF `modeling_mistral.py`) | Our code |
|---|---|---|
| Backbone | `MistralModel` + `MistralForCausalLM` | `MistralForCausalLM` extends `LlamaForCausalLM` in [src/mini_infer/models/mistral.py:28](../../src/mini_infer/models/mistral.py#L28) |
| Decoder layer | `MistralDecoderLayer` | shared `TransformerBlock` (via `_LlamaInnerModel`) |
| GQA attention | `MistralAttention` | shared `GroupedQueryAttention` in [blocks/gqa.py](../../src/mini_infer/models/blocks/gqa.py) |
| Standard RoPE | `MistralRotaryEmbedding` | shared `RotaryEmbedding` in [blocks/rope.py](../../src/mini_infer/models/blocks/rope.py) |
| RMSNorm | `MistralRMSNorm` | shared `RMSNorm` in [blocks/rmsnorm.py](../../src/mini_infer/models/blocks/rmsnorm.py) |
| SwiGLU FFN | `MistralMLP` | shared `SwiGLU` in [blocks/swiglu.py](../../src/mini_infer/models/blocks/swiglu.py) |
| **Sliding-window attention** (v0.1 / v0.2) | `sliding_window=4096` honored in attention mask | **Not implemented**: see "What's still open" below |

Bit-parity is exercised against HF transformers' `MistralAttention` on
synthetic configs at FP32 via the shared `GroupedQueryAttention` parity
tests. No Mistral-specific golden test today (Mistral 7B doesn't fit
M1 fp16 comfortably and isn't in `tests/_pinned_models.toml`); the
class itself is exercised through the shared `LlamaForCausalLM` paths.

## How short is the file?

```python
@dataclass
class MistralConfig(LlamaConfig):
    """Identical surface to LlamaConfig; type-distinct for registry clarity."""


@register_model
class MistralForCausalLM(LlamaForCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "MistralForCausalLM"
    Config: ClassVar[type] = MistralConfig
```

That's the whole module body. Everything else (the `from_hf` parser,
the loud failure on `attention_bias=True` / `mlp_bias=True`, the
forward, the weight loader) is inherited unchanged from
`LlamaForCausalLM`.

### Why a separate class at all

Three reasons:

1. **HF dispatch.** `load_model()` looks up `config.architectures[0]`
   in the registry. A Mistral checkpoint declares `"MistralForCausalLM"`,
   not `"LlamaForCausalLM"`. Without a separate registration, the
   registry would `KeyError` on every Mistral checkpoint.
2. **Type distinctness.** `isinstance(model, MistralForCausalLM)` is
   useful for debugging and for future family-specific work. The
   alternative (register `LlamaForCausalLM` under both architecture
   keys) hides the fact that we know Mistral is a separate family.
3. **A future-proof seam.** If Mistral ever ships a new architectural
   primitive (or if we ever wire SWA), the override point already
   exists. The file goes from 31 lines to maybe 60 without churning
   the registry shape.

## Sliding-Window Attention: what we don't implement

Mistral 7B v0.1 and v0.2 set `sliding_window=4096` in their HF config.
The reference attention mask masks out positions further than
`window_size` in the past, so each token only attends to the last
`window_size` keys.

**Why we skip it:**

1. **Mistral 7B v0.3 dropped it.** The headline Mistral 7B variant most
   users load no longer needs SWA. v0.1 / v0.2 still work numerically
   without it for prompts shorter than 4096 tokens (which is most
   prompts in practice).
2. **Honest correctness boundary.** For prompts longer than 4096
   tokens on Mistral 7B v0.1 / v0.2, our output will diverge from HF
   (we attend over the full prefix, HF attends over a 4096-window).
   This is a documented limitation rather than a silent bug. Golden
   tests against HF on these variants would surface it immediately
   if we ever added one.
3. **No clean per-layer SWA path yet.** Our `GroupedQueryAttention`
   dispatches to a single `packed_attention_forward` kernel that
   doesn't accept a per-request window argument. Adding SWA means
   either (a) a window-aware variant of the packed attention kernel
   or (b) routing SWA layers through `SWAAttention` (which exists
   today for DeepSeek-V4's `compress_ratio=0` layers but is built
   around V4's low-rank Q + grouped output + sink, not the plain
   GQA path Mistral wants). Both are real engineering, not config
   tweaks.

Reference to where SWA lives in our codebase for V4:
[`blocks/swa.py`](../../src/mini_infer/models/blocks/swa.py).
That file's purpose is V4's pure-SWA attention layers (no compressor,
no indexer), so it carries V4's low-rank Q machinery and isn't a
drop-in for Mistral's plain GQA + window combination.

### What we do honor

Mistral checkpoints loaded through `MistralForCausalLM` produce
bit-parity output vs HF for:

- Mistral 7B v0.3 (dropped SWA, plain GQA, full attention).
- Mistral Small / Medium / Large (no SWA in those configs).
- Codestral 22B (no SWA).
- Prompts shorter than `sliding_window` on v0.1 / v0.2 (the mask is a
  no-op when the prefix fits in the window).

## Decoder layer assembly

Same as Llama (`TransformerBlock` from
[blocks/transformer_block.py](../../src/mini_infer/models/blocks/transformer_block.py)):

```
residual = h
h = input_layernorm(h)
h = self_attn(h, ...)        # GQA, no biases, no q_norm/k_norm
h = residual + h
residual = h
h = post_attention_layernorm(h)
h = mlp(h)                   # SwiGLU
h = residual + h
```

No Mistral-specific decoder shape. The reference's
`MistralDecoderLayer.forward` is the same pre-norm pattern as
`LlamaDecoderLayer.forward` modulo the SWA mask construction (which
we don't apply, see above).

## Cache structure

Standard paged KV cache. Same shape Llama uses; one block table per
layer, `block_size` tokens per block, `(num_kv_heads, head_dim)` per
token. No family-specific cache contract.

(If we ever wire SWA, the cache itself stays the same shape; the
sliding window is enforced at attention-mask time, not by evicting
keys from the cache. The cache eviction would be a *separate* memory
optimization that vLLM ships and we don't.)

## Validation contract

- **Bit-parity vs HF**: synthetic `MistralAttention` configs at FP32
  via the shared `GroupedQueryAttention` parity test. Identical math
  to the Llama parity; the only difference HF sees is the architecture
  string.
- **Weight-load round-trip**: a Mistral state_dict loaded through
  `load_state_dict_with_tp` reproduces the original tensors. Inherited
  from `LlamaForCausalLM.load_weights`.
- **No pinned Mistral checkpoint** in `tests/_pinned_models.toml`.
  Mistral 7B doesn't fit M1 fp16 comfortably; the family is covered
  by the shared Llama machinery rather than its own smoke. Adding a
  pinned Mistral checkpoint is a follow-up if a smaller Mistral
  variant becomes available, or as part of the SWA work.

Tests under `tests/unit/`; layout intentionally not enumerated.

## Why this writeup is so short

Because the implementation is so short. Mistral 7B was a genuinely
novel release (GQA at 7B, then SWA) but both of its primitives have
either become universal (GQA) or been dropped from the family (SWA).
What remains is the Llama-shape backbone everyone uses.

The honest read of mini-infer's Mistral support is:

- **Yes** for every Mistral checkpoint that uses plain attention (v0.3,
  Small, Medium, Large, Codestral, derivatives).
- **Caveated** for v0.1 / v0.2 on prompts longer than 4096 tokens
  (we'll diverge from HF; the divergence is upstream of the SWA we
  don't implement).
- **No** if a future Mistral release introduces something new. That
  would be a paper-watch trigger.

## Pointers

- **Reference**: HF transformers `transformers/models/mistral/modeling_mistral.py`.
- **Our model class**: [src/mini_infer/models/mistral.py](../../src/mini_infer/models/mistral.py).
- **Inherited from**: [src/mini_infer/models/llama.py](../../src/mini_infer/models/llama.py). See [llama.md](llama.md) for the substance.
- **GQA block** (shared): [blocks/gqa.py](../../src/mini_infer/models/blocks/gqa.py).
- **V4's pure-SWA block** (not Mistral's; here for cross-reference): [blocks/swa.py](../../src/mini_infer/models/blocks/swa.py).

## What's still open

- **Mistral 7B v0.1 / v0.2 sliding-window attention.** Would need a
  window-aware path through `GroupedQueryAttention` (or routing SWA
  layers through a Mistral-shape SWA block, distinct from V4's
  low-rank-Q SWA). Triggered if a notable Mistral variant brings SWA
  back, or if a user reports the divergence on long v0.1 / v0.2
  prompts. Not blocked on hardware.
- **Pinned Mistral smoke.** A small Mistral-shape checkpoint
  (Mistral-Small, when small enough, or one of the Codestral
  variants) added to `tests/_pinned_models.toml` would give the
  family its own golden test. Today it's covered by inheritance.
- **Mixtral 8x7B / 8x22B.** Sparse-MoE Mistral; covered separately
  in [mixtral.md](mixtral.md). The base Mistral 7B is what this
  walkthrough describes.
