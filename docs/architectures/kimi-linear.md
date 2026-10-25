# Kimi Linear walkthrough (Kimi Linear 48B-A3B)

A line-by-line correspondence between the Kimi Linear paper
(arXiv 2510.26692), Moonshot's reference modeling code, and mini-infer's
from-scratch port of `KimiLinearForCausalLM`.

The reference is the checkpoint's own `modeling_kimi.py` from
`moonshotai/Kimi-Linear-48B-A3B-Instruct` at revision
`e1df551a447157d4658b573f9a695d57658590e9`. It ships as `trust_remote_code`,
not as a transformers class, so the repo vendors it at a fixed revision with
`scripts/kimi_linear/clone_kimi_linear_reference.py`. Reference line numbers
below are for that file in `third_party/kimi_linear_reference/`.

The reference calls GPU kernels from the FLA library (flash-linear-attention)
for the linear-attention math. Where the paper, a blog post and the code
disagree, the code wins.

> Audience: someone who wants to follow a linear-attention hybrid from paper
> to code. Read the [DeepSeek-V2 walkthrough](deepseek-v2-mla.md) first for
> MLA. Pair this page with
> [ADR-026](../decisions/ADR-026-kimi-linear-kda-port.md) for the port
> decisions and [the Kimi spec](../plans/kimi-k3-spec.md) for the pinned
> semantics.

## TL;DR

Kimi Linear is a 27-layer hybrid with two kinds of attention layer:

- 20 Kimi Delta Attention (KDA) layers. KDA is a linear attention. A KV
  cache grows by one entry per token, but a KDA layer keeps a fixed-size
  matrix state per request instead. A gated delta rule updates it, with a
  separate decay rate for each key channel.
- 7 MLA layers at the 1-indexed positions 4, 8, 12, 16, 20, 24 and 27. They
  have the DeepSeek latent shape but no rotary embedding at all (NoPE). All
  position information comes from the KDA layers.

Layer 0 has a dense SwiGLU. Every other layer has a 256-expert top-8 MoE with
one shared expert.

Three details matter most:

- The MoE router weights the experts by the biased scores. The reference adds
  the bias in place, through a view, so the bias reaches the weights. The GLM
  router looks the same but uses unbiased weights.
- The MLA layers keep the `qk_rope_head_dim` split but never rotate it. Do
  not add RoPE to "fix" this.
- The configuration lists layer kinds with 1-indexed numbers (`kda_layers`,
  `full_attn_layers`). The port keeps the lists and converts at lookup.

The port serves the family on a per-request `KimiStateCache`, not on the paged
KV cache. Against the vendored reference on a tiny CPU configuration, prefill
logits reach cosine > 0.999 and greedy decode is token-exact.

## The map

| Primitive | Reference (`modeling_kimi.py`) | Our code |
|---|---|---|
| Backbone | `KimiLinearModel` (L899), `KimiLinearForCausalLM` (L1009) | `KimiLinearForCausalLM` in `src/mini_infer/models/kimi_linear/kimi_linear.py` (L574), `_KimiInnerModel` (L555) |
| Configuration and layer kinds | `KimiLinearConfig.is_kda_layer` (`configuration_kimi.py:136`) | `KimiLinearConfig.from_hf` (`kimi_linear/kimi_linear.py:115`), `is_kda_layer` (L173), `is_moe_layer` (L177) |
| Decoder layer | `KimiDecoderLayer` (L783) | `_KimiDecoderLayer` (`kimi_linear/kimi_linear.py:512`) |
| KDA layer | `KimiDeltaAttention` (L444) | `_KimiDeltaAttention` (`kimi_linear/kimi_linear.py:229`) |
| Short causal conv | FLA `ShortConvolution` (called at L541-558) | `causal_conv1d_prefill` in `src/mini_infer/models/blocks/kda.py` (L93), taps in `_KimiConv1d` (`kimi_linear/kimi_linear.py:211`) |
| Decay gate | FLA `fused_kda_gate` (called at L560) | `kda_gate` (`blocks/kda.py:40`) |
| q/k L2 norm | inside the FLA kernels (`use_qk_l2norm_in_kernel=True`, L576, L588) | `l2norm` (`blocks/kda.py:68`), called at `kimi_linear/kimi_linear.py:287-289` |
| Delta rule, spans of 64 tokens or fewer | FLA `fused_recurrent_kda` (called at L580) | `kda_recurrent` (`blocks/kda.py:156`) |
| Delta rule, longer spans | FLA `chunk_kda` (called at L568) | `kda_chunkwise` (`blocks/kda.py:200`) |
| Gated output norm | FLA `FusedRMSNormGated` (`o_norm`, called at L598) | `gated_rmsnorm` (`blocks/kda.py:74`) |
| NoPE MLA | `KimiMLAAttention` (L326) | `_KimiMlaAttention` (`kimi_linear/kimi_linear.py:312`) |
| MoE router | `KimiMoEGate` (L608) | `_KimiMoeGate` (`kimi_linear/kimi_linear.py:410`) |
| MoE block | `KimiSparseMoeBlock` (L701), `KimiBlockSparseMLP` (L245), `KimiMLP` (L265) | `_KimiMoe` (`kimi_linear/kimi_linear.py:464`), `MixtralExpert` in `blocks/moe/mixtral_moe.py` (L58), `SwiGLU` in `blocks/swiglu.py` (L23) |
| Decode state | `KimiDynamicCache` (L118) | `KimiStateCache` in `src/mini_infer/models/kimi_linear/state_cache.py` (L101), specs from `build_kimi_state_cache_specs` (`kimi_linear/kimi_linear.py:185`) |
| Weight loader | `from_pretrained` | `KimiLinearForCausalLM.load_weights` (`kimi_linear/kimi_linear.py:725`), `_remap_kimi_state` (L749) |

Paths that start with `kimi_linear/` or `blocks/` are under
`src/mini_infer/models/`. The FLA rows name kernels that the reference
imports. The test helpers replace them with plain PyTorch on the CPU (see
the validation section).

## Kimi Delta Attention

KDA belongs to the DeltaNet family. Each head keeps a state matrix `S` of
shape `(128, 128)`. Per token, the state decays, then gets a rank-1
correction toward the new value, and then the query reads it. From the
`blocks/kda.py` docstring (L9-11):

```
S_t = S_{t-1} * Diag(exp(g_t))                       # decay, g_t <= 0 in log space
S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T         # delta-rule rank-1 correction
o_t = S_t^T q_t
```

The decay `g_t` has one value per key channel. Gated DeltaNet (used by
Qwen3.5, see [qwen3-5.md](qwen3-5.md)) has one value per head. That per-channel
decay is the main change that KDA makes.

Kimi Linear has 32 KDA heads of width 128 in each KDA layer, and a conv
kernel of 4 taps.

Reference (`KimiDeltaAttention.forward`, L541-598, shortened):

```python
q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states), cache=conv_state_q, ...)
k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states), cache=conv_state_k, ...)
v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states), cache=conv_state_v, ...)
g = self.f_b_proj(self.f_a_proj(hidden_states))
g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
beta = self.b_proj(hidden_states).float().sigmoid()
...
o, recurrent_state = chunk_kda(q=q, k=k, v=v, g=g, beta=beta,
                               initial_state=recurrent_state,
                               use_qk_l2norm_in_kernel=True, ...)
...
g = self.g_b_proj(self.g_a_proj(hidden_states))
o = self.o_norm(o, g)
```

Our code (`_KimiDeltaAttention.forward`, `kimi_linear/kimi_linear.py:278-307`,
shortened):

```python
q_raw, new_conv_q = causal_conv1d_prefill(self.q_proj(x), self.q_conv1d.weight, conv_q)
k_raw, new_conv_k = causal_conv1d_prefill(self.k_proj(x), self.k_conv1d.weight, conv_k)
v_raw, new_conv_v = causal_conv1d_prefill(self.v_proj(x), self.v_conv1d.weight, conv_v)
decay = kda_gate(self.f_b_proj(self.f_a_proj(x)), self.A_log, self.dt_bias, self.head_dim)
beta = self.b_proj(x).float().sigmoid()
q = l2norm(q_raw.view(batch, seq_len, self.num_heads, self.head_dim))
q = q * self.head_dim**-0.5
k = l2norm(k_raw.view(batch, seq_len, self.num_heads, self.head_dim))
...
if seq_len <= 64:
    out, final_state = kda_recurrent(q, k, v, decay, beta, initial_state=initial)
else:
    out, final_state = kda_chunkwise(q, k, v, decay, beta, initial_state=initial)
...
gate = self.g_b_proj(self.g_a_proj(x)).view(batch, seq_len, self.num_heads, self.head_dim)
out = gated_rmsnorm(out, gate, self.o_norm_weight, self.o_norm_eps)
```

The steps, in order:

1. Short convolution. Each of q, k and v goes through a depthwise causal
   conv with 4 taps, then SiLU (`blocks/kda.py:93`). The conv state holds
   the last 4 raw inputs, newest last, which is FLA's layout. The layer uses
   this one function for every span, also for a one-token decode step.
2. Decay gate. `g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)` in fp32
   (`blocks/kda.py:40-65`). Every value is at most 0, so `exp(g)` is a true
   decay. `A_log` has one value per head, and `dt_bias` has one per channel.
3. Step size. `beta = sigmoid(b_proj(x))` in fp32, one value per head.
4. Normalization. q and k are L2-normalized per head with eps 1e-6
   (`blocks/kda.py:68`), and q is scaled by `128 ** -0.5`. The reference
   kernels do this inside, because of `use_qk_l2norm_in_kernel=True`. The
   port does it at the call site, so the recurrence functions are the bare
   delta rule.
5. Recurrence. A span of 64 tokens or fewer runs one token at a time
   (`kda_recurrent`). A longer span runs in chunks (`kda_chunkwise`). The
   reference switches at the same length (L523). The state stays fp32.
6. Output. An RMSNorm in fp32 over each head, then a shared weight, then a
   sigmoid gate from `g_b(g_a(x))`, and only then a cast back
   (`blocks/kda.py:74-90`). `o_proj` follows.

### Recurrent form

`kda_recurrent` (`blocks/kda.py:156`) is the loop over tokens. It serves
decode and it is the oracle for the chunked form. The core is L188-196:

```python
for t in range(seq_len):
    state = state * g32[:, t].exp().unsqueeze(-1)
    predicted = (k32[:, t].unsqueeze(-1) * state).sum(dim=-2)
    update = torch.einsum(
        "bhk,bhv->bhkv", beta32[:, t].unsqueeze(-1) * k32[:, t], v32[:, t] - predicted
    )
    state = state + update
    out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
```

### Chunked form

`kda_chunkwise` (`blocks/kda.py:200`) computes the same result for prefill,
in chunks of 64 tokens. Inside a chunk, the per-token rank-1 corrections fold
into one matrix through the WY representation. That representation writes a
product of rank-1 updates as one low-rank update. Across chunks, the state
advances once per chunk.

The port solves the unit lower-triangular system with
`torch.linalg.solve_triangular` (L272-274). FLA's naive reference uses a
forward-substitution loop. The matrix is the same, but the float operations
run in a different order. A test pins the chunked form to the recurrent form
within 1e-5.

The last chunk is padded with zeros. A padded position has `beta = 0` and
`g = 0`, so it leaves the state unchanged.

## NoPE MLA

The 7 full-attention layers use the DeepSeek-V3 latent shape with two
changes. Q comes from one plain `q_proj` (`q_lora_rank` is `None`), and no
rotary embedding exists anywhere. Each layer has 32 heads, `kv_lora_rank`
512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64 and `v_head_dim` 128. The
softmax scale is `(128 + 64) ** -0.5`.

Reference (`KimiMLAAttention.forward`, L394-410):

```python
q_pass, q_rot = torch.split(
    q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
k_pass, k_rot = torch.split(
    compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
...
k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
query_states = torch.cat((q_pass, q_rot), dim=-1)
key_states = torch.cat((k_pass, k_rot), dim=-1)
```

Our code (`_KimiMlaAttention.forward`, `kimi_linear/kimi_linear.py:386-392`):

```python
kv_latent, k_rot = torch.split(kv_hist, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
decompressed = self.kv_b_proj(self.kv_a_layernorm(kv_latent)).view(
    batch, -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
)
k_nope, values = torch.split(decompressed, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
# NoPE: the rope-dim split concatenates UNROTATED, broadcast to heads.
keys = torch.cat([k_nope, k_rot.unsqueeze(2).expand(-1, -1, self.num_heads, -1)], dim=-1)
```

In both, `q_rot` and `k_rot` keep their names but are never rotated. The
configuration still carries `rope_theta`, but nothing reads it. `from_hf`
rejects a configuration with `mla_use_nope` off or with `q_lora_rank` set
(`kimi_linear/kimi_linear.py:122-128`), because that is different math.

This layer is a separate class, not the shared `MLAAttention` of
`blocks/mla.py`. It has no RoPE, and it reads and writes a dense per-request
buffer instead of paged cache streams. The decompression on read is the same
idea. The causal mask uses each row's own start position (L394-403), so one
batch can hold requests of different lengths.

The reference forces `flash_attention_2` when the model is built. The parity
tests set the eager implementation after construction, for CPU runs.

## MoE router

The router is a sigmoid gate with an `e_score_correction_bias`, like the
DeepSeek-V3 `noaux_tc` gate. One line makes it different.

Reference (`KimiMoEGate.forward`, L664-689):

```python
scores_for_choice = scores.view(bsz * seq_len, -1)
scores_for_choice += self.e_score_correction_bias.unsqueeze(0)
...
tmp_scores = scores_for_choice.masked_fill(
    ~score_mask.bool(), 0.0)  # [n, e]
_, topk_idx = torch.topk(
    tmp_scores, k=self.top_k, dim=-1, sorted=False,
)
topk_weight = scores.gather(1, topk_idx)
```

`scores_for_choice` is a view of `scores`, so the `+=` also changes `scores`.
The final `scores.gather` therefore returns biased scores. The DeepSeek-V3
and GLM gates gather the unbiased scores
([glm-moe-dsa.md](glm-moe-dsa.md#noaux_tc-moe)).

Our code (`_KimiMoeGate.forward`, `kimi_linear/kimi_linear.py:439-461`)
writes the effect out:

```python
logits = functional.linear(x_flat.float(), self.weight.float())
scores = logits.sigmoid() if self.activation_func == "sigmoid" else logits.softmax(dim=1)
choice = scores + self.e_score_correction_bias.float()
...
masked_choice = choice.masked_fill(~score_mask.bool(), 0.0)
topk_indices = torch.topk(masked_choice, k=self.top_k, dim=-1, sorted=False)[1]
topk_weights = choice.gather(1, topk_indices)
if self.top_k > 1 and self.moe_renormalize:
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
return topk_indices, topk_weights * self.routed_scaling_factor
```

Two more details follow the reference. Groups that lose are masked with
`0.0`, not `-inf`. And the weights are normalized and then multiplied by
`routed_scaling_factor` (2.446). The released configuration has one expert group,
so the group mask changes nothing there.

`_KimiMoe` (`kimi_linear/kimi_linear.py:464`) runs the experts. The 256
routed experts are `MixtralExpert` SwiGLUs of width 1024, with the
checkpoint's names `w1`, `w2` and `w3`. The routed outputs add up in fp32, as
in the reference. The shared expert is one SwiGLU of width 1024, added after
the routed sum.

## Decoder layer assembly

Reference (`KimiDecoderLayer.forward`, L811-870): a pre-norm block. The
attention is KDA or MLA, and the FFN is `block_sparse_moe` or `mlp`.

Our code (`_KimiDecoderLayer.forward`, `kimi_linear/kimi_linear.py:532-552`):

```python
residual = hidden_states
x = self.input_layernorm(hidden_states)
if self.is_kda:
    x = self.self_attn(x, layer_state)
else:
    x = self.self_attn(x, layer_state, row_starts)
hidden_states = residual + x
residual = hidden_states
x = self.post_attention_layernorm(hidden_states)
x = self.block_sparse_moe(x) if hasattr(self, "block_sparse_moe") else self.mlp(x)
out = residual + x
```

The KDA layer takes no position. Only the MLA layer needs `row_starts`, the
position of each row's first new token. The module names mirror the
reference (`block_sparse_moe` and `mlp`), so the weights load by direct copy.

`KimiLinearForCausalLM._run` (`kimi_linear/kimi_linear.py:591`) walks the
layers and hands each one its slice of the state cache. Three entry points
call it: `forward_prefill_with_cache` (L653), `forward_decode_with_cache`
(L673) and `forward_decode_with_cache_ragged` (L691). They follow the same
contract as DeepSeek-V4.

## Cache structure

`KimiStateCache` (`kimi_linear/state_cache.py:101`) holds one state per
layer and per batch row:

| Layer kind | Tensors | Shape at 48B-A3B |
|---|---|---|
| KDA (20 layers) | `recurrent_state`, always fp32 | `(B, 32, 128, 128)` |
| KDA | `conv_q`, `conv_k`, `conv_v`, the last 4 raw inputs | `(B, 4096, 4)` each |
| MLA (7 layers) | `kv`, the raw `kv_a_proj_with_mqa` output | `(B, max_seq_len, 576)` |

The KDA state does not grow with the context. One request needs 2 MiB of fp32
state per KDA layer, so 40 MiB for all 20, at any context length. Only the 7
MLA layers grow.

The MLA buffer stores the compressed 576 values per token (512 latent plus
64 unrotated dims) and decompresses them on read. The reference
`KimiDynamicCache` (L118) stores the decompressed per-head K and V instead:
32 heads x (192 + 128), so 10240 values per token per layer. NoPE puts
nothing position-dependent into the cache, so both give the same result.

The cache follows the life cycle of the DeepSeek-V4 `StateCache`. The caller
advances `start_pos` after each forward. `copy_row_from` (L191) moves a
prefilled request into a row of the batched cache, for the
continuous-batching scheduler. The two cache classes share no tensors, so they
stay separate classes (ADR-026).

The whole family is free of positions apart from the MLA offset. So
`forward_prefill_with_cache` can continue from `start_pos > 0`, and chunked
prefill works. The DeepSeek-V4 state path cannot do this.

The MLA buffer has a hard capacity of `max_seq_len` tokens per request.
Admission control must respect it, and an overflow raises an error
(`kimi_linear/kimi_linear.py:368-372`).

## Validation contract

The parity tests build a tiny random configuration with the same features as
the real one. It has KDA at layers 1-3 and MLA at layer 4 (1-indexed). Its
conv kernel is shorter than the prompts, its MLA is NoPE, and a dense layer 0
comes before the MoE layers. Both
models load the same weights through `load_weights`, so a mismatch is a math
bug. The tests also randomize the router bias, so a GLM-style unbiased gate
fails them.

The vendored reference imports FLA's GPU kernels. The test helpers replace
each kernel with plain PyTorch that follows FLA's own naive reference
(`fla/ops/kda/naive.py`). The helpers do not import `blocks/kda.py`, so the
port is not compared with itself.

| What | Result |
|---|---|
| Prefill logits vs the reference | cosine > 0.999, allclose 1e-3 |
| Greedy decode | token-exact |
| Two requests, ragged decode through one `KimiStateCache` | token-exact |
| Chunked prefill across a conv-kernel boundary | allclose 1e-3 |
| Router | same experts, weights allclose 1e-6 |
| Router weights are the biased scores | a regression test |
| Chunked KDA vs recurrent KDA, with and without a start state | allclose 1e-5 |
| Conv prefill vs one step at a time | allclose 1e-6 |
| Scheduler serving | matches the scalar oracle |

Run them with `uv run pytest tests/unit/ -k "kimi or kda"`. The reference
parity tests skip when `third_party/kimi_linear_reference/` is missing. In a
git worktree, symlink it from the main checkout, or run the clone script.

## Where we diverged + why

[ADR-026](../decisions/ADR-026-kimi-linear-kda-port.md) records the decisions.

1. KDA math in plain PyTorch. FLA's kernels run only under Triton, with no
   CPU or MPS path. So the port writes the recurrence from scratch in
   `blocks/kda.py`, and FLA's semantics serve only as a test oracle.
2. A triangular solve. The chunked form calls `solve_triangular` where FLA's
   naive reference runs forward substitution. Same matrix, different float
   order.
3. Normalization at the call site. The port L2-normalizes q and k and scales
   q before the recurrence. The reference kernels do it inside.
4. Compressed MLA cache. The port caches the 576-wide compressed entry, and
   the reference caches per-head K and V.
5. A Kimi-specific router. `_KimiMoeGate` copies the biased-weight
   behavior. Reusing `GlmNoAuxTcGate` gives wrong expert weights.
6. A separate `KimiStateCache`. It shares the life cycle of the V4
   `StateCache` but no tensors, so each cache keeps its own indexing.
7. Chunked prefill. Prefill can continue from `start_pos > 0`.
8. Defined start values. The reference leaves `dt_bias` and
   `e_score_correction_bias` uninitialized (`torch.empty`). The port starts
   both at zero, so a model without loaded weights stays finite. A checkpoint
   load overwrites them.
9. Load mapping. The module tree mirrors the checkpoint names. Only two keys
   change: the conv weights lose their `(C, 1, W)` singleton dim, and
   `o_norm.weight` maps to `o_norm_weight` (`kimi_linear/kimi_linear.py:749-768`).

## Pointers

- Reference: `third_party/kimi_linear_reference/modeling_kimi.py` and
  `configuration_kimi.py`, vendored by
  [`scripts/kimi_linear/clone_kimi_linear_reference.py`](../../scripts/kimi_linear/clone_kimi_linear_reference.py).
  The folder is not in git.
- Paper: Kimi Linear, arXiv 2510.26692.
- ADR: [ADR-026](../decisions/ADR-026-kimi-linear-kda-port.md).
- Spec and plan: [kimi-k3-spec.md](../plans/kimi-k3-spec.md),
  [kimi-k3.md](../plans/kimi-k3.md).
- Model class and configuration:
  [`kimi_linear/kimi_linear.py`](../../src/mini_infer/models/kimi_linear/kimi_linear.py).
- KDA math: [`blocks/kda.py`](../../src/mini_infer/models/blocks/kda.py).
- State cache:
  [`kimi_linear/state_cache.py`](../../src/mini_infer/models/kimi_linear/state_cache.py).
- Loader: `KimiLinearForCausalLM.load_weights`. It raises an error under
  tensor parallelism.

## What is still open

- The real 48B-A3B checkpoint never ran. The gate (step A5 of the
  [Kimi plan](../plans/kimi-k3.md)) is pending.
- Single rank only. Tensor parallelism needs the KDA conv channels and the
  per-head state split across ranks.
- No prefix sharing across requests. That needs a snapshot of the KDA state.
- The MLA history is a dense buffer per request, not paged.
- The KDA math runs as plain PyTorch loops, with no fused kernel. The MLA
  layer reads `row_starts` on the host (L363, L368, L384), so the decode
  step does not run under a CUDA graph.
- The MoE runs its experts one at a time in a loop, with no grouped-matmul
  path.
- The Kimi K3 port (stage B of the plan) builds on this one and does not
  exist yet.
