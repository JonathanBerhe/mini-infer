# GLM-MoE-DSA walkthrough (GLM-5.2, GLM-5.3)

A line-by-line correspondence between the HF transformers reference
(`transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`) and mini-infer's
from-scratch port of `GlmMoeDsaForCausalLM`. GLM-5.2 and GLM-5.3 share this
one architecture and one configuration, so one walkthrough covers both.

GLM ships no paper of its own. DeepSeek Sparse Attention (DSA) comes from
DeepSeek-V3.2, and the HF code cites arXiv 2603.12201 for IndexShare. Where a
description and the HF code disagree, the HF code wins. Reference line numbers
are for transformers 5.14.0, the version that `pyproject.toml` pins.

> Audience: someone who knows MLA from the
> [DeepSeek-V2 walkthrough](deepseek-v2-mla.md) and wants to follow what GLM
> adds. Pair this page with
> [ADR-021](../decisions/ADR-021-glm-moe-dsa-port.md) for the port
> decisions.

## TL;DR

GLM-5.2 is DeepSeek-V3.2 plus one GLM addition. It has 78 layers, about 753B
parameters, and four parts that matter:

1. MLA attention, the same latent attention as DeepSeek-V2/V3, with
   interleaved RoPE (theta 8e6, no YaRN) on 64 of the 256 query/key dims.
2. DSA. A small Lightning Indexer (a second, cheap attention that only
   scores keys) picks the 2048 past tokens that each query attends to. All
   other keys get a score of `-inf` before the softmax.
3. IndexShare. Only 21 of the 78 layers run the indexer. Each of the other
   57 layers reuses the selection of the nearest earlier indexer layer.
4. A DeepSeek-V3 `noaux_tc` MoE. A sigmoid router picks 8 of 256 experts.
   It uses a learned bias to choose the experts but not to weight them. One
   shared expert runs on every token.

The port runs on the standard `PagedKVCache` engine. DSA keeps the full K/V
history and only masks it, so no special state cache is needed. The indexer
keys live in one more cache stream, `index_k`.

Parity is tested against HF on tiny synthetic configurations on the CPU in
fp32. The full model matches HF logits within 1e-3, and greedy decode is
token-identical.

## The map

| Primitive | Reference (`modeling_glm_moe_dsa.py`, transformers 5.14.0) | Our code |
|---|---|---|
| Backbone | `GlmMoeDsaModel` (L657), `GlmMoeDsaForCausalLM` (L736) | `GlmMoeDsaForCausalLM` in `src/mini_infer/models/glm/glm_moe_dsa.py` (L307), `_GlmMoeDsaInnerModel` (L269) |
| Configuration and layer kinds | `config.mlp_layer_types`, `config.indexer_types` | `GlmMoeDsaConfig.from_hf` (`glm/glm_moe_dsa.py:93`), `is_moe_layer` (L164), `indexer_is_shared` (L167) |
| Decoder layer | `GlmMoeDsaDecoderLayer` (L577) | `_GlmMoeDsaDecoderLayer` (`glm/glm_moe_dsa.py:171`) |
| MLA attention | `GlmMoeDsaAttention` (L309) | `MLAAttention` in `src/mini_infer/models/blocks/mla.py` (L71) |
| Interleaved RoPE | `apply_rotary_pos_emb_interleave` (L133) | `apply_interleaved_rotary_pos_emb` in `blocks/rope.py` (L316) |
| RoPE tables | `GlmMoeDsaRotaryEmbedding` (L68) | `RotaryEmbedding` in `blocks/rope.py` (L163), built at `glm/glm_moe_dsa.py:319` |
| Softmax scale | `GlmMoeDsaAttention.__init__` (L360-366) | `GlmMoeDsaConfig.attention_softmax_scale` (`glm/glm_moe_dsa.py:149`) |
| DSA Lightning Indexer | `GlmMoeDsaIndexer` (L172) | `GlmDsaIndexer` in `src/mini_infer/models/glm/dsa_indexer.py` (L82) |
| fp32 indexer head weights | `_keep_in_fp32_modules` (L643) | `_Fp32ColumnParallelLinear` (`glm/dsa_indexer.py:56`) |
| DSA sparse mask | `index_mask` in `GlmMoeDsaAttention.forward` (L423-432) | `mla_packed_attention_forward` in `src/mini_infer/cache/mla_attention.py` (L21), mask at L88-94 |
| IndexShare | `skip_topk` (L368), top-k loop in `GlmMoeDsaModel.forward` (L715-724) | `skip_topk` / `next_skip_topk` (`glm/glm_moe_dsa.py:230-233`), loop in `GlmMoeDsaForCausalLM.forward` (L336-338) |
| `noaux_tc` router | `GlmMoeDsaTopkRouter` (L472) | `GlmNoAuxTcGate` in `src/mini_infer/models/blocks/moe/glm_moe_gate.py` (L42) |
| MoE block | `GlmMoeDsaMoE` (L553), `GlmMoeDsaExperts` (L514) | `GlmMoeFFN` (`blocks/moe/glm_moe_gate.py:119`), `MixtralExpert` in `blocks/moe/mixtral_moe.py` (L58) |
| Dense FFN (layers 0-2) | `GlmMoeDsaMLP` (L456) | `SwiGLU` in `blocks/swiglu.py` (L23) |
| FP8-resident experts | none (HF dequantizes at load) | `Fp8Expert` in `blocks/moe/fp8_expert.py` (L36) |
| KV cache | `DynamicCache`: per-head K/V plus indexer keys | `PagedKVCache` streams from `GlmMoeDsaForCausalLM.per_layer_streams` (`glm/glm_moe_dsa.py:351`) |
| Weight loader | `from_pretrained` | `GlmMoeDsaForCausalLM.load_weights` (`glm/glm_moe_dsa.py:379`) |

Paths that start with `glm/` or `blocks/` are under `src/mini_infer/models/`.

## MLA attention

GLM's main attention is DeepSeek-V3 MLA. The
[DeepSeek-V2 walkthrough](deepseek-v2-mla.md) explains the two cached streams
(`kv_latent` and `k_rope`) and the decompression on read. GLM-5.2 uses the
low-rank Q path with these sizes:

| Field | Value |
|---|---|
| `num_attention_heads` | 64 |
| `q_lora_rank` | 2048 |
| `kv_lora_rank` | 512 |
| `qk_nope_head_dim` / `qk_rope_head_dim` | 192 / 64 |
| `v_head_dim` | 256 |

So Q/K and V both have a head dim of 256, unlike DeepSeek-V2-Lite. The port
still forces the `"torch"` attention backend, because no flash or FlashInfer
kernel takes the DSA mask (`glm/glm_moe_dsa.py:367`).

Reference (`modeling_glm_moe_dsa.py:385-396`):

```python
q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
...
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
```

Our code (`blocks/mla.py:220-250`):

```python
q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
q = self.q_b_proj(q_latent)
...
if self.use_interleaved_rope:
    q_pe, k_rope_rotated = apply_interleaved_rotary_pos_emb(q_pe, k_rope_for_rope, cos, sin)
```

GLM added three options to the shared `MLAAttention`. Each one is off by
default, so DeepSeek-V2, V3 and Kimi-K2 stay bit-identical:

- `use_interleaved_rope` (`blocks/mla.py:87`). GLM passes `True` at
  `glm/glm_moe_dsa.py:209`.
- `softmax_scale` (`blocks/mla.py:88`, used at L161-163). GLM passes
  `attention_softmax_scale()`.
- `indexer` (`blocks/mla.py:89`). An attached indexer needs the low-rank Q
  path (L118-119), because it scores from the shared `q_lora` latent. The
  `forward` takes a matching `dsa_topk` argument (L202).

### Interleaved RoPE

GLM rotates the pairs `(x[2i], x[2i+1])` together, like DeepSeek. Old
transformers versions applied the NeoX split-half instead. HF fixed that in
5.7 for the main attention and in 5.14 for the indexer. The ADR-021
amendments record both fixes.

HF's helper writes the rotated pairs back as two concatenated halves. Our
helper writes them back interleaved (`blocks/rope.py:316`). The same
permutation applies to Q and to K, so each `q.k` dot product does not change.

### Softmax scale

The scale is `qk_head_dim ** -0.5`, so `256 ** -0.5` for GLM-5.2. A
configuration with a non-default `rope_type` and `mscale_all_dim` set also
multiplies the scale by `yarn_get_mscale(factor, mscale_all_dim) ** 2`. HF
does this at L360-366, and the port at `glm/glm_moe_dsa.py:158-162`. The
released configuration uses `rope_type: "default"`, so the plain scale
applies.

## DSA Lightning Indexer

The indexer is a small extra attention that only ranks keys. It has 32 heads
of width 128, and the first 64 dims of each head get RoPE. For each query it
returns the 2048 past tokens with the highest scores.

The math, from the `glm/dsa_indexer.py` docstring (L9-24):

```
q_t,h     = wq_b(q_resid_t)                # per head, RoPE dims first
k_j       = k_norm(wk(h_j))                # one key shared by all heads
s_t,h,j   = ReLU((q_t,h . k_j) * d^-0.5)
w_t,h     = weights_proj(h_t) * n_heads^-0.5
score_t,j = sum_h(s_t,h,j * w_t,h)
topk(t)   = top index_topk of score_t,j, causal-masked
```

The ReLU before the head sum makes each head a vote. A key survives only if
some head scores it above zero. The query reuses the main attention's
`q_lora` latent (`q_resid`), so the indexer adds only `wq_b`, `wk`, `k_norm`
and `weights_proj`.

Reference (`modeling_glm_moe_dsa.py:236-263`):

```python
k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
...
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
...
scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
scores = F.relu(scores)
weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
...
return index_scores.topk(topk, dim=-1).indices.to(torch.int32)
```

Our code splits the same steps into two methods:

- `_project` (`glm/dsa_indexer.py:125`) builds the RoPE'd queries, the shared
  key and the head weights.
- `_causal_topk` (`glm/dsa_indexer.py:166`) scores each request, masks future
  keys to `-inf` and takes the top-k (L194-201).

Some details differ from the main attention and are easy to get wrong:

- The RoPE dims come first in each indexer head (`q_pe, q_nope`, L142).
  The main MLA puts them last.
- `k_norm` is a LayerNorm with eps 1e-6 (L118), not an RMSNorm.
- `weights_proj` stays fp32, also when the rest of the model is bf16. HF
  marks it with `_keep_in_fp32_modules` (L643). The port subclasses the
  linear so that a dtype cast keeps the fp32 bits (`_Fp32ColumnParallelLinear`,
  `glm/dsa_indexer.py:56`).
- The `n_heads ** -0.5` factor uses the full head count, also under tensor
  parallelism (L161-163).

The indexer has two entry points. `forward` (L204) scores the current tokens
only and serves the block-level tests. `forward_cached` (L221) appends this
step's keys to the `index_k` cache stream and scores the full history, so a
one-token decode step ranks every past token. The serving path calls
`forward_cached` through `MLAAttention.compute_dsa_topk` (`blocks/mla.py:165`).

Under tensor parallelism, `wq_b` and `weights_proj` are split by head. Each
rank sums its own heads, and an all-reduce completes the sum before the top-k
(L196). So every rank picks the same keys.

## DSA sparse mask

The indexer output becomes a mask on the main attention scores.

Reference (`modeling_glm_moe_dsa.py:423-432`):

```python
index_mask = (
    topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
    .scatter(-1, topk_indices.long(), False)
    .unsqueeze(1)
)
...
attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
```

Our code (`cache/mla_attention.py:88-94`):

```python
if dsa_topk is not None:
    topk_r = dsa_topk[batch_idx]
    index_mask = torch.full(
        (q_len, k_len), -float("inf"), dtype=scores.dtype, device=q.device
    )
    index_mask.scatter_(1, topk_r, 0.0)
    scores = scores + index_mask[:, None, :]
```

The causal mask follows at L95-100. HF fills masked slots with
`finfo(dtype).min` and the port uses `-inf`. Both give a softmax weight of
zero.

A prompt shorter than 2048 tokens selects every past token. Then DSA equals
dense attention. The parity tests use a small `index_topk` so that the
selection really drops keys.

The mask saves no compute today. Each query still scores every key, and only
then are the unselected ones dropped. A kernel that gathers only the selected
keys is open work (see the end of this page).

## IndexShare

The configuration marks each layer `"full"` or `"shared"` in `indexer_types`.
In the released configuration, layers 0, 1, 2, 6, 10, ..., 74 are full (21
layers). Every other layer is shared. So after layer 2, one full layer serves itself and the
next three layers.

A shared layer has no indexer weights at all. HF sets `self.indexer = None`
(L369), and the port does the same (`glm/glm_moe_dsa.py:186-197`). Loading
an indexer on a shared layer fails the weight load.

Reference (`GlmMoeDsaModel.forward`, L715-724):

```python
topk_indices = None  # MAIN DIFF with DSV3.2
for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    hidden_states, topk_indices = decoder_layer(
        ...
        prev_topk_indices=topk_indices,  # MAIN DIFF with DSV3.2
    )
```

Our code (`GlmMoeDsaForCausalLM.forward`, `glm/glm_moe_dsa.py:336-338`):

```python
prev_topk: list[torch.Tensor] | None = None
for layer in self.model.layers:
    x, prev_topk = layer(x, (cos, sin), past_key_values, cu_seqlens_q, prev_topk)
```

HF returns the selection from every layer. The port returns it only while the
next layer shares it (`next_skip_topk`, L231-233 and L266), and returns
`None` otherwise. A shared layer that gets `None` raises an error (L248-252).
The configuration also carries `index_topk_freq` 4 and `index_skip_topk_offset` 3.
`from_hf` reads the explicit `indexer_types` list instead, and both give the
same 21 full layers.

## noaux_tc MoE

Layers 0-2 use a dense SwiGLU of width 12288. Layers 3-77 use the MoE: 256
routed experts of width 2048, top-8, and one shared expert.

The router is the DeepSeek-V3 `noaux_tc` gate. The name means "no auxiliary
loss": a learned bias balances the expert load instead of a training loss.

Reference (`GlmMoeDsaTopkRouter.forward`, L485-510):

```python
router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
scores = router_logits.sigmoid()
scores_for_choice = scores + self.e_score_correction_bias
...
topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
topk_weights = scores.gather(1, topk_indices)
```

Our code (`GlmNoAuxTcGate.forward`, `blocks/moe/glm_moe_gate.py:87-116`):

```python
logits = functional.linear(x_flat.float(), self.weight.float())
scores = logits.sigmoid()
choice = scores + self.e_score_correction_bias
...
topk_indices = torch.topk(choice_masked, k=self.top_k, dim=-1, sorted=False)[1]
# Expert weights come from the UNBIASED sigmoid scores.
topk_weights = scores.gather(1, topk_indices)
```

The bias changes which experts win, but the weights come from the unbiased
scores. Then the weights are normalized to sum to 1 and multiplied by
`routed_scaling_factor` (2.5). The grouped top-k code is complete, but
`n_group = topk_group = 1` makes it a no-op on the released configuration.

Kimi Linear uses a router that looks the same but weights the experts by the
biased scores. See [kimi-linear.md](kimi-linear.md#moe-router). Do not reuse
one gate for the other.

`GlmMoeFFN` (`blocks/moe/glm_moe_gate.py:119`) runs the experts. Each expert
is a `MixtralExpert` SwiGLU. The one shared expert is one `MixtralExpert` of
width `n_shared_experts * moe_intermediate_size`. Under expert parallelism,
each rank owns a contiguous range of experts. The partial sums are
all-reduced, and then the shared expert output is added once (L207-212).

On Hopper and newer GPUs, the routed experts run as grouped matmuls
(`_grouped_pack`, L215), for bf16 experts
([ADR-036](../decisions/ADR-036-grouped-expert-matmuls.md)) and for FP8
experts ([ADR-037](../decisions/ADR-037-block-fp8-grouped-kernel.md)).
Other devices loop over the experts (`_loop_experts`, L236).

## Decoder layer assembly

Reference (`GlmMoeDsaDecoderLayer.forward`, L591-621):

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _, topk_indices = self.self_attn(..., prev_topk_indices=prev_topk_indices)
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
return hidden_states, topk_indices
```

Our code (`_GlmMoeDsaDecoderLayer.forward`, `glm/glm_moe_dsa.py:235-266`):

```python
residual = hidden_states
x = self.input_layernorm(hidden_states)
if self.skip_topk:
    topk = prev_topk
else:
    topk = self.self_attn.compute_dsa_topk(x, position_embeddings, past_key_values, cu_seqlens_q)
x = self.self_attn(x, position_embeddings, past_key_values, cu_seqlens_q, dsa_topk=topk)
hidden_states = residual + x
residual = hidden_states
x = self.post_attention_layernorm(hidden_states)
x = self.mlp(x)
out = residual + x
return out, (topk if self.next_skip_topk else None)
```

The block is the standard pre-norm shape. The only GLM part is the top-k
selection before the attention call. HF runs the indexer inside the attention
module. The port runs it from the decoder layer, so that the layer can pass
the result to the next one.

## Cache structure

Each layer declares its cache streams in `per_layer_streams`
(`glm/glm_moe_dsa.py:351`). All streams have one head, because MLA shares the
latent across heads.

| Stream | Heads x dim | Layers |
|---|---|---|
| `index_k` | 1 x 128 | full indexer layers only (21) |
| `kv_latent` | 1 x 512 | all 78 |
| `k_rope` | 1 x 64 | all 78 |

So one token costs 576 cached values on a shared layer and 704 on a full one.
HF caches the decompressed per-head K and V instead: 64 heads x (256 + 256)
values, which is 32768 per token per layer.

`index_k` comes first in the list (L358-363). Layer 0 is always a full layer,
so its `index_k` append is the first write of a step, and that write
allocates the cache block.

Prefix caching works with no extra code. A prefix hit reuses the shared
blocks, `index_k` included, and the suffix's indexer scores against them.

## Validation contract

The tests compare each part with the HF `glm_moe_dsa` reference on tiny
synthetic configurations, on the CPU in fp32 unless a row names a GPU. They load the
same weights into both models, so a mismatch is a math bug, not a load bug.

| What | Result |
|---|---|
| MLA with RoPE and with a selective DSA mask | cosine > 0.999, allclose 1e-4 |
| MLA under a yarn rope configuration | allclose 1e-4 |
| Indexer | the same selected key set as HF, also at bf16 and under TP |
| Router and MoE FFN | same routing, FFN allclose 1e-4 |
| Full model | logits allclose 1e-3, greedy argmax identical |
| Greedy decode | tokens identical to HF over 6 steps |
| Two ragged prompts in one batch | match HF |
| Prefix-cache hit | logits match a full prefill |
| TP, world size 2 (gloo) | ranks identical, cosine > 0.999 with world size 1 |
| FP8-resident experts | logits allclose 1e-4 with the bf16-dequantized model |
| NVIDIA L4 (`scripts/glm/modal_glm_smoke.py`) | fp32 parity exact (cosine 1.0) |
| 2x L4, NCCL (`scripts/glm/modal_glm_tp_smoke.py`) | ranks identical |

The GLM-5.3 configuration is a byte-for-byte fixture in `tests/fixtures/glm53/`. A
test parses it with `from_hf` and loads the full 78-layer key layout on the
meta device. The tests are in `tests/unit/models/glm/`. Run them with
`uv run pytest tests/unit/models/glm/`.

## Where we diverged + why

[ADR-021](../decisions/ADR-021-glm-moe-dsa-port.md) records the decisions.

1. Compressed KV cache. The port caches `kv_latent` and `k_rope` and
   decompresses on read, like the DeepSeek-V2 port. HF caches per-head K and
   V. The compressed form keeps MLA's memory saving.
2. Indexer keys in a cache stream. HF keeps them in a per-layer buffer. The
   port puts them in the `index_k` stream, so they work with continuous
   batching and prefix caching.
3. A dense masked attention. The port computes all scores in the torch
   backend and masks them, with `-inf` where HF uses `finfo(dtype).min`.
4. Interleaved write-back. The RoPE helper keeps the rotated pairs
   interleaved, and HF writes them as halves. The dot products are the same.
5. IndexShare hand-off. The decoder layer runs the indexer and passes the
   top-k on only while the next layer shares it.
6. FP8-resident experts. `from_hf` sets `expert_dtype="fp8"` for an FP8
   checkpoint. `Fp8Expert` keeps each routed expert in e4m3 with a scale per
   128x128 block, so the model fits one 8xH200 node (about 785 GB).
7. Load by rank. `load_weights` (`glm/glm_moe_dsa.py:379`) dequantizes the
   block-FP8 weights that stay bf16, with a ceiling for partial blocks such
   as the 576-row `kv_a_proj_with_mqa`. It maps global expert numbers to
   local ones and skips the experts of other ranks. It reads both expert
   layouts: HF's stacked 3D tensors and the checkpoint's per-expert tensors.
8. No MTP layer. The checkpoint's draft layer (`model.layers.78.*`) is
   dropped at load (L422-430). Greedy decode does not need it.

## Pointers

- Reference: `transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` in the
  installed transformers 5.14 package. It is not vendored.
- ADR: [ADR-021](../decisions/ADR-021-glm-moe-dsa-port.md), with amendments
  for interleaved RoPE and GLM-5.3.
- Model class and configuration:
  [`glm/glm_moe_dsa.py`](../../src/mini_infer/models/glm/glm_moe_dsa.py).
- Indexer: [`glm/dsa_indexer.py`](../../src/mini_infer/models/glm/dsa_indexer.py).
- MLA block: [`blocks/mla.py`](../../src/mini_infer/models/blocks/mla.py).
- DSA mask: [`cache/mla_attention.py`](../../src/mini_infer/cache/mla_attention.py).
- Router and MoE FFN:
  [`blocks/moe/glm_moe_gate.py`](../../src/mini_infer/models/blocks/moe/glm_moe_gate.py).
- FP8 experts: [`blocks/moe/fp8_expert.py`](../../src/mini_infer/models/blocks/moe/fp8_expert.py).
- Loader: `GlmMoeDsaForCausalLM.load_weights`. The scripts in
  [`scripts/glm/`](../../scripts/glm/) inspect and stage the real checkpoint.

## What is still open

- The real 753B checkpoint never ran. It needs a multi-GPU node such as
  8xH200. No GLM-5.3 weights ran on a GPU either.
- DSA saves no attention compute yet. A kernel that gathers only the selected
  keys is the next step for long contexts.
- On full indexer layers, the `q_lora` latent is computed twice per step:
  once in `compute_dsa_topk` and once in the attention forward.
- `index_k` is stored in the cache dtype (bf16 on a GPU). The CPU tests run
  in fp32 and are exact.
- The model's RoPE tables have no YaRN. A configuration with
  `rope_type: "yarn"` gets the mscale softmax correction but plain RoPE
  tables (`glm/glm_moe_dsa.py:315-319`). The released configurations use the
  default rope, so this does not affect GLM-5.2 or GLM-5.3.
- The MTP draft layer is not modeled, so speculative decoding with it is not
  available.
