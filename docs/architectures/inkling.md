# Inkling: relative-bias hybrid attention, short convolutions, shared-expert-sink MoE

**Reference:** HF transformers 5.14 `modeling_inkling.py` (text path;
`InklingForCausalLM`). Release announcement:
[thinkingmachines.ai/news/introducing-inkling](https://thinkingmachines.ai/news/introducing-inkling/).
**Our port:** [`src/mini_infer/models/inkling.py`](../../src/mini_infer/models/inkling.py)
plus three blocks:
[`inkling_rel_bias.py`](../../src/mini_infer/models/blocks/inkling_rel_bias.py),
[`inkling_sconv.py`](../../src/mini_infer/models/blocks/inkling_sconv.py),
[`inkling_moe.py`](../../src/mini_infer/models/blocks/inkling_moe.py).
**Parity suite:** [`tests/unit/test_inkling_parity.py`](../../tests/unit/test_inkling_parity.py).

Inkling is a 975B-total / 41B-active decoder-only MoE (66 layers, hidden
6144, vocab 201024 padded / 200058 real, 1M-token context). Inkling-Small
(276B/12B) shares the architecture. The released checkpoints are multimodal
(`InklingForConditionalGeneration`, `model.language_model.*` weight prefix);
we port the text path only and drop the vision/audio towers and MTP draft
weights at load.

## The decoder layer

```
x ── input_layernorm ── attention ── attn_sconv ──(+)── post_attention_layernorm ── mlp ── mlp_sconv ──(+)──
│                                                  │                                                     │
└──────────────────────── residual ────────────────┘──────────────────── residual ───────────────────────┘
```

Two things are nonstandard: the SConvs on the branch outputs (inside the
residual add), and the fact that the embedding output is RMS-normed
(`embed_norm`) before layer 0.

## 1. Attention without RoPE

Every layer is GQA with per-head RMS QK-norm; because q and k are unit-RMS
per head, logits scale by `1/d` instead of `1/sqrt(d)`. Layers interleave
5:1: five `hybrid_sliding` layers (window 512; 64 q / 16 KV heads) then one
`hybrid` global layer (64 q / 8 KV heads). Note the head counts differ per
layer TYPE; the `BlockPool` gets per-layer KV shapes via
`per_layer_kv_shape()`.

Position enters through a learned relative bias, not RoPE:

```
rel_states = r_proj(x).view(T, heads, d_rel)        # d_rel = 16
profiles   = rel_states @ proj                      # proj: (d_rel, rel_extent)
bias[t, h, s] = profiles[t, h, t - s]   if 0 <= t - s < rel_extent else 0
```

`proj` is a trained bank of bias-vs-distance profiles; each token mixes them
into one bias value per backward distance. `rel_extent` is the window size
on sliding layers and 1024 on global layers, so even at 1M context the bias
touches only the most recent 1024 keys; beyond that, global attention is
position-free. Causality and the sliding window stay in the mask.

Global layers additionally apply log length scaling in fp32 to q and to the
bias:

```
tau = 1 + alpha * log(max(1, (pos + 1) / n_floor))      # alpha=0.1, n_floor=128000
```

Our implementation folds bias + causal + window into one per-request
additive mask and feeds it through `packed_attention_torch`'s `block_mask`
path (the same hook MiniMax-M3's MSA uses); the model pins the `torch`
attention backend, mirroring HF shipping the family with flash-attn
disabled.

## 2. Short convolutions (SConv)

Four per layer, each a depthwise causal `Conv1d` (kernel 4, no bias, no
activation) computed in fp32 with the residual folded inside:

```
out = (causal_conv1d(x.float()) + x.float()).to(x.dtype)
```

Placement: after `k_proj` and `v_proj` (BEFORE the KV cache write; the cache
stores what attention consumes: post-conv V and post-conv-post-norm K), and
on the attention/MLP branch outputs before their residual adds.

Serving-side this makes decode stateful: step `t` needs the previous 3
pre-conv inputs of each conv. We store the pre-conv inputs as per-token
streams in the `PagedKVCache` (per layer: `conv_k`, `conv_v`, `k`, `v`,
`conv_attn`, `conv_mlp`) and gather the 3-token tail each step. That is
correct under chunked prefill, batched ragged decode, `truncate_to`, and
prefix-cache reuse, at the cost of extra pool memory (see ADR-025; a rolling
conv-state buffer is the known follow-up).

## 3. MoE with a shared-expert sink

256 routed + 2 shared experts, top-6, on every layer past `dense_mlp_idx`
(the dense layers use SwiGLU times a learned `global_scale` scalar).
Selection is DeepSeek-V3-style (sigmoid scores + aux-loss-free
`e_score_correction_bias`), but the WEIGHTS are new:

```
logits      = x @ W.T                     # W has n_routed + n_shared rows
chosen      = topk(sigmoid(routed) + bias)
lp          = logsigmoid(cat([routed_logits[chosen], shared_logits]))
weights     = softmax-normalized exp(lp) * route_scale * global_scale
routed_w, shared_gammas = weights[:6], weights[6:]
```

The shared experts sit inside the normalization (the "sink"): when routed
experts score high, the shared gammas shrink. Each shared expert multiplies
its gamma into the activated intermediate (before down-proj) and the
cross-expert sum runs in fp32. This differs enough from `GlmNoAuxTcGate`
(weights from unbiased sigmoid scores, grouped top-k, shared expert outside
the normalization) that it gets its own gate in `inkling_moe.py`.

## 4. Unembedding

muP-style: `logits = lm_head(h / logits_mup_width_multiplier)` with
multiplier 24, lm_head untied from the embedding, and logits sliced to
`unpadded_vocab_size` (200058 of 201024).

## What the parity suite pins

Tiny-random config with both layer types, both MLP types, sliding window (8)
and rel_extent (6) both shorter than the context, active log scaling, and
MORE KV heads on sliding layers than global ones (4 vs 2, like the real 16
vs 8):

- one-shot prefill logits: cosine > 0.999, argmax-equal, allclose 1e-3;
- greedy decode and two-request batched ragged decode through one shared
  `PagedKVCache`: token-equal with HF incremental decode (HF carries its own
  conv states; we rebuild from the conv streams);
- chunked prefill (7+5) equals one-shot, crossing a conv-kernel and window
  boundary;
- component parity: gate (indices, weights, gammas) vs `InklingTopkRouter`;
  SConv full-sequence vs stepped-with-tails.

Known HF quirk: transformers 5.14 crashes on prompts shorter than the conv
kernel (4 tokens) in its cached path; our port handles them (zero left-pad),
but the batched parity test uses a length-4 prompt as the shortest
HF-checkable raggedness.
