# Qwen hybrid (Qwen3.5 / Qwen3.6 / Qwen3.8)

## TL;DR

Qwen3.8 is not its own architecture. `Qwen/Qwen3.8-27B` declares
`architectures: ["Qwen3_5ForConditionalGeneration"]` and `model_type
"qwen3_5"`, the same architecture as Qwen3.5-4B/9B/27B and Qwen3.6-27B, so
one port covers the whole line. The novelty is a 3:1 hybrid decoder: three
Gated DeltaNet layers, a linear attention whose decode state is a
fixed-size matrix rather than a growing KV cache, for every one Gated
Attention layer, an ordinary grouped-query attention carrying a sigmoid
output gate fused into its query projection. For the 27B that is 48 linear
and 16 full over 64 layers, with the split given explicitly by the config's
`layer_types` list.

Two details do most of the damage if you get them wrong. The output gate is
packed into `q_proj` PER HEAD, so each head's slice is `[query | gate]`;
reading the projection as `[all queries | all gates]` produces a model that
loads, runs, and is numerically wrong. And the Gated DeltaNet broadcasts key
heads up to value heads by an integer ratio that is 3 on the 27B but 2 on
the 4B and 9B, so the ratio has to come from the config.

The expensive math was already in the repo. Gated DeltaNet's recurrence is
the Kimi Delta Attention recurrence with a coarser decay, so
[blocks/kda.py](../../src/mini_infer/models/blocks/kda.py) is reused
unchanged and [blocks/gdn.py](../../src/mini_infer/models/blocks/gdn.py)
holds only the parts that genuinely differ.

## The map

| Primitive | Reference (`transformers/models/qwen3_5/modeling_qwen3_5.py`) | Our code |
| --- | --- | --- |
| Gated DeltaNet layer | `Qwen3_5GatedDeltaNet` | [qwen3_5.py](../../src/mini_infer/models/qwen3_5.py) `_QwenGatedDeltaNet` |
| Delta-rule recurrence | `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule` | [kda.py](../../src/mini_infer/models/blocks/kda.py) `kda_chunkwise`, `kda_recurrent` |
| Decay gate | inline `-A_log.exp() * softplus(a + dt_bias)` | [gdn.py](../../src/mini_infer/models/blocks/gdn.py) `gdn_gate` |
| Gated output norm | `Qwen3_5RMSNormGated` | [gdn.py](../../src/mini_infer/models/blocks/gdn.py) `gated_rmsnorm_silu` |
| Fused causal conv | `causal_conv1d_fn` / `F.silu(conv1d(...))` fallback | [kda.py](../../src/mini_infer/models/blocks/kda.py) `causal_conv1d_prefill`, `causal_conv1d_step` |
| Gated attention | `Qwen3_5Attention` | [qwen3_5.py](../../src/mini_infer/models/qwen3_5.py) `_QwenGatedAttention` |
| Decoder norm | `Qwen3_5RMSNorm` (the `1 + w` form) | [gemma_rmsnorm.py](../../src/mini_infer/models/blocks/gemma_rmsnorm.py) `GemmaRMSNorm` |
| Partial rotary | `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb` | [rope.py](../../src/mini_infer/models/blocks/rope.py) `RotaryEmbedding`, `apply_rotary_pos_emb_partial` |
| Decode state | `Qwen3_5DynamicCache` | [qwen3_5_state_cache.py](../../src/mini_infer/cache/qwen3_5_state_cache.py) `Qwen35StateCache` |

## Gated DeltaNet

The layer projects the hidden state four ways. One fused projection
produces q, k and v together (`2 * key_dim + value_dim` channels, which is
10240 on the 27B) and passes through a single depthwise causal convolution
with SiLU. Three narrow projections produce the output gate `z`, the
delta-rule step size `b`, and the decay features `a`, and these read the
PRE-convolution hidden state. That asymmetry is easy to miss.

The recurrence is the gated delta rule:

    S_t = S_{t-1} * exp(g_t)                          decay, g_t <= 0
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T      rank-1 correction
    o_t = S_t^T q_t

with `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`.
`A_log` and `dt_bias` are flat per-value-head vectors, so every head decays
its whole state by ONE scalar per step.

This is where the reuse comes from. Kimi Delta Attention runs the same
recurrence but decays by a diagonal, one entry per key channel. A scalar is
the diagonal whose entries are all equal, so broadcasting the Qwen scalar
across the key channels lets `kda_recurrent` and `kda_chunkwise` compute the
Gated DeltaNet update with no change. That is an exact identity, not an
approximation: measured against the reference in fp32 the max absolute error
is round-off, around 1e-7, across square and non-square head dims, sequences
crossing the 64-token chunk boundary, ragged padding and non-zero initial
states.

Before the recurrence, q and k are L2-normalized per head and q is scaled by
`head_k_dim ** -0.5`. The reference does this inside its kernel via
`use_qk_l2norm_in_kernel=True`; we do it at the call site so the shared
recurrence stays the bare delta rule. Then q and k are broadcast up to the
value-head count with `repeat_interleave`, which duplicates adjacently.

The output is normalized and gated by `z`. The gate activation is SiLU, and
the affine weight is applied in the INPUT dtype before the fp32 gate
multiply. Both matter: KDA's equivalent gates with sigmoid and does its
affine in fp32, so the two functions are not interchangeable even though
they look alike.

## Gated attention

Ordinary grouped-query attention, 24 query heads over 4 KV heads at head_dim
256 on the 27B, with per-head RMS norms on q and k, plus two twists.

`q_proj` is `num_attention_heads * head_dim * 2` wide (12288 on the 27B).
Viewing it as `(batch, seq, heads, 2 * head_dim)` and chunking the last
dimension gives each head its query and its gate. The gate is applied as
`attn_output * sigmoid(gate)` on the head-major flattened activation, after
the attention core and before `o_proj`. There is no separate gate tensor in
the checkpoint, which is the clue that it must be fused.

Rotary embedding is partial: with `partial_rotary_factor` 0.25 and head_dim
256, only the leading 64 dimensions of each head rotate and the remaining
192 pass through untouched. The inverse-frequency table is built over the
ROTARY dim, so the exponent denominator is 64 and not 256. Building it the
other way is a silently plausible table that produces wrong angles.

The text backbone nominally uses a three-component mrope for interleaved
image and video tokens. For text-only input all three position streams carry
the same values, and the interleave then rewrites every section with
identical numbers, so the result is bit-identical to plain 1D partial rope.
This port computes plain partial rope and pins that equivalence against the
reference rather than assuming it.

## Decoder layer assembly

Pre-norm throughout, with `GemmaRMSNorm` (the `1 + weight` form over
zero-stored weights) for `input_layernorm`, `post_attention_layernorm` and
the final norm. The mixer is `linear_attn` or `self_attn` depending on
`layer_types[i]`, and the FFN is a standard SwiGLU. The one norm that is NOT
the `1 + w` form is the Gated DeltaNet's internal output norm, which is a
plain ones-init affine.

## Cache structure

Neither layer kind fits PagedAttention's one-entry-per-token model, so this
family sets `USES_STATE_CACHE = True` and rides the per-request state-cache
seam that the Kimi Linear port generalized.

`Qwen35StateCache` holds, per layer:

- Gated DeltaNet: an fp32 recurrent state `(num_v_heads, head_k_dim,
  head_v_dim)` plus ONE fused conv tail `(conv_channels, kernel)` carrying
  the last `kernel` raw pre-conv inputs, newest last. Constant size
  regardless of context length. Kimi's KDA layers carry three tails here
  because Kimi does not fuse its q/k/v projections.
- Gated attention: append-only per-token K and V buffers `(max_seq_len,
  num_kv_heads, head_dim)`. Plain GQA, not Kimi's compressed MLA buffer, so
  nothing is re-decompressed on read. K is stored after its norm and after
  rotary, so a decode step only has to position its own query.

Only the full-attention layers grow with context, so `max_seq_len` bounds
admission and `check_capacity` fails loudly before a write would run past
the buffers.

## Loading

`StateCacheGenerator.from_pretrained` calls
`Qwen35ForConditionalGeneration.from_checkpoint`. It builds the model
directly in the target dtype on the target device, then copies the
checkpoint in one safetensors shard at a time. Peak memory is the model plus
the largest shard: for the 4B, about 8.5 GB of fp16 weights plus a 5.3 GB
shard. The generic `load_model` builds in fp32 first, which for the 4B alone
is about 17 GB.

## Decode step

A one-token step attends over the whole fixed-length K/V buffer and masks
the unwritten slots on the device, with the query heads of each KV head
stacked on the query axis. The step therefore has no host sync and no
data-dependent shape, and `decode_step_static` can be captured in a CUDA
graph. On CUDA the serving scheduler replays it from the graph
([ADR-033](../decisions/ADR-033-cuda-graph-state-cache-decode.md)): on the
real 27B on an A100 that is 46 ms per step at batch 1 instead of 107 ms,
with identical tokens.

## Validation contract

Bit-parity runs against the pinned transformers implementation of this exact
architecture, in fp32 on CPU, with weights pushed through the production
`load_weights` using CHECKPOINT-shaped keys so the prefix strip, the
vision/MTP drop and the conv squeeze are exercised rather than bypassed. The
synthetic configs deliberately use a 3x linear head ratio, matching the 27B,
because hardcoding the 4B's 2x would hide a whole class of broadcast bug.

Covered: one-shot prefill, greedy decode across the chunk boundary, chunked
prefill split mid-conv-kernel, ragged batched decode after `copy_row_from`,
the mrope collapse, and the weight-drop path. See `tests/unit/`.

Real weights: `tests/unit/test_qwen3_5_real_weights.py` loads
`Qwen/Qwen3.5-4B` at a pinned revision, the smallest checkpoint with this
architecture. It goes through `StateCacheGenerator.from_pretrained` and the
state-cache scheduler, and checks the layout and a greedy "Paris" completion.
It passed on an M1 Pro (16 GB, MPS, fp16) in about 4 minutes, mostly
loading. The download is 9.3 GB, so the test carries the `large_model`
marker and skips unless you pass `--run-large-models`.

The 27B is 55.6 GB and does not fit the local development box.
[scripts/modal_qwen38_smoke.py](../../scripts/modal_qwen38_smoke.py) runs it
on one A100 80 GB GPU in bf16 against transformers on the same weights, with
four prompts from 4 to 83 tokens. Result on `Qwen/Qwen3.8-27B` at revision
`1d4bf0f`:

- Prefill logits: cosine 0.99985 to 0.99998 against the reference, the same
  last-token top-1 for every prompt, and the same argmax at 99 to 100 percent
  of positions.
- Greedy decode through the state-cache scheduler: 32 of 32 tokens equal to
  the reference for all four prompts, and identical across two runs. The
  first prompt completes as " Paris.".
- Two requests in flight: three prompts match the one-at-a-time run token for
  token; one diverges after 14 tokens. Batched bf16 matrix products round
  differently, so this is reported and not asserted.
- `from_checkpoint` loads the model in 11 seconds from a warm volume, at
  50.1 GiB of GPU memory.

## Where we diverged and why

See [ADR-032](../decisions/ADR-032-qwen38-hybrid-port.md) for the decisions:
reusing the KDA recurrence rather than reimplementing it, a separate state
cache rather than extending Kimi's, keeping the reference's fused
projections so weights load without remapping, collapsing the multimodal
rope on purpose, and staying single-rank for this port.

## Pointers

- Reference: `transformers/models/qwen3_5/modeling_qwen3_5.py` (pinned 5.14)
- Model: [src/mini_infer/models/qwen3_5.py](../../src/mini_infer/models/qwen3_5.py)
- Blocks: [gdn.py](../../src/mini_infer/models/blocks/gdn.py), [kda.py](../../src/mini_infer/models/blocks/kda.py)
- Cache: [qwen3_5_state_cache.py](../../src/mini_infer/cache/qwen3_5_state_cache.py)
- ADRs: [ADR-032](../decisions/ADR-032-qwen38-hybrid-port.md) (port),
  [ADR-033](../decisions/ADR-033-cuda-graph-state-cache-decode.md) (CUDA-graph decode)
- Graph runner: [cuda_graph_decode.py](../../src/mini_infer/engine/cuda_graph_decode.py)

## What is still open

- Tensor parallelism. `load_weights` raises on world_size != 1; sharding the
  fused conv channels and the per-head matrix state is a separate piece of
  work, as it was for Kimi Linear.
- MoE variants of this family, including the 2.4T-A95B Max tier.
- `Qwen3.8-Flash-Next` is NOT this architecture. It declares
  `Qwen4ExpForConditionalGeneration` / `qwen4_exp` and is a 512-expert MoE,
  so it needs its own port.
- Cross-request prefix sharing for the recurrent state.
- Fused decode kernels. With the CUDA graph, a 27B decode step runs at
  about 57 percent of the weight-bandwidth bound on an A100; the rest is
  thousands of small kernels. See
  [ADR-033](../decisions/ADR-033-cuda-graph-state-cache-decode.md).
