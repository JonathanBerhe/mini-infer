# Kimi K3 walkthrough (the delta over Kimi Linear)

Kimi K3 (`moonshotai/Kimi-K3`) is the Kimi Linear stack with six
config-gated additions and two changed conventions. This page covers only
that delta. Read the [Kimi Linear walkthrough](kimi-linear.md) first for
KDA, the NoPE-MLA layers, the router and the state cache.

The reference is the checkpoint's own `modeling_kimi_linear.py` at revision
`f831ab66814297da540d832a5235f8e904f29d06`, vendored by
`scripts/kimi_linear/clone_kimi_k3_reference.py` into
`third_party/kimi_k3_reference/`. The vision wrapper `modeling_kimi_k3.py`
is vendored too, but only read for the weight layout. Where the tech report,
the blog and the code disagree, the code wins.

Kimi K3.1 Pro (`moonshotai/Kimi-K3.1-Pro`, weights out on 2026-11-04) is
not the same architecture, although the announcement said so. Its 24 MLA
layers are Gated DeepSeek Sparse Attention layers. The section
[Kimi K3.1 Pro: Gated DSA](#kimi-k31-pro-gated-dsa) covers that delta, and
[ADR-055](../decisions/ADR-055-kimi-k31-gated-dsa.md) records the decisions.

> Pair this page with [ADR-049](../decisions/ADR-049-kimi-k3-port.md) for
> the port decisions.

## TL;DR

| Reference | Config field | Ours |
|---|---|---|
| `SituAndMul` | `hidden_act = "situ"`, `activation_situ_beta`, `activation_situ_linear_beta` | `blocks/activations.situ` |
| `KimiMLAAttention.q_a_proj / q_a_layernorm / q_b_proj` | `q_lora_rank` | `_KimiMlaAttention` |
| `KimiMLAAttention.g_proj` | `mla_use_output_gate` | `_KimiMlaAttention` |
| `KimiDeltaAttention.g_proj` | `linear_attn_config.use_full_rank_gate` | `_KimiDeltaAttention` |
| FLA kernels with `lower_bound` | `linear_attn_config.gate_lower_bound` | `blocks/kda.kda_lower_bound_gate` |
| `KimiSparseMoeBlock.routed_expert_*` | `routed_expert_hidden_size`, `latent_moe_use_norm` | `_KimiMoe` |
| `_apply_attn_res`, `*_res_norm / *_res_proj` | `attn_res_block_size` | `kimi_linear/attn_res.py`, `_KimiDecoderLayer` |
| `KimiK3ForConditionalGeneration` | `text_config` | `kimi_linear/kimi_k3.py` |
| compressed-tensors `mxfp4-pack-quantized` | `quantization_config` | `quant/mxfp4.py` |
| K3.1: DeepSeek-V3.2 indexer (SGLang `Indexer`) | `text_config.dsa_config` | `kimi_linear/dsa_indexer.py`, `_KimiMlaAttention` |
| K3.1: `KimiK31ForConditionalGeneration` | `text_config` | `kimi_linear/kimi_k3.py` |

The real config: 93 layers (69 KDA, 24 MLA at every fourth layer plus the
last), hidden 7168, 96 heads, `q_lora_rank` 1536, `kv_lora_rank` 512, 896
experts with 16 active and 2 shared, expert width 3072 in a latent of 3584,
`attn_res_block_size` 12, `gate_lower_bound` -5.0, SiTU betas 4.0 and 25.0,
no MTP head, vocabulary 163840, context 1048576.

## SiTU-GLU

The reference concatenates the gate and up projections and feeds the pair to
`SituAndMul`:

    a   = beta * tanh(gate / beta) * sigmoid(gate)
    u   = linear_beta * tanh(up / linear_beta)
    out = a * u

Both branches run in fp32; the product is cast back to the input dtype. The
gate branch is a SiLU whose magnitude is bounded by `beta`; the up branch is
bounded by `linear_beta`. The dense MLP of layer 0, the 896 routed experts
and the shared experts all use it. Ours is a `GateUpActivation` combiner
like `swigluoai`, so `SwiGLU` and `MixtralExpert` take it unchanged.

## Gated MLA with low-rank Q

Two changes on the NoPE-MLA layers. Queries go through `q_a_proj` (7168 to
1536), an RMSNorm, and `q_b_proj` (1536 to 96 heads of 192), the
DeepSeek-V3 shape. After attention, the per-head outputs are multiplied by
`sigmoid(g_proj(x))` before `o_proj`, where `x` is the layer's normalized
input. The KV path, the NoPE treatment and the compressed cache are the
same as Kimi Linear.

Correction: the reference builds `q_a_layernorm` and `kv_a_layernorm` with
the `KimiRMSNorm` default eps of 1e-6, not `rms_norm_eps` (1e-5). Stage A
used 1e-5 for `kv_a_layernorm`; the parity tolerance hid it. Both now use
1e-6.

## KDA: full-rank output gate and the bounded decay

`use_full_rank_gate` replaces the low-rank `g_a_proj / g_b_proj` output
gate with one `g_proj` (7168 to 96 x 128). The reference `A_log` is flat
`(H,)`, but the released checkpoint stores 128 entries: the 96 head values,
then 32 zeros. The loader drops the zeros and fails if they are not zero.

`gate_lower_bound = -5.0` changes the decay gate. Kimi Linear computes
`-exp(A_log) * softplus(raw + dt_bias)`, unbounded below. K3 passes the raw
features to the FLA kernels with `use_gate_in_kernel=True` and
`lower_bound=-5.0`, and the kernel computes

    g = lower_bound * sigmoid(exp(A_log) * (raw + dt_bias))

so every channel's log-decay stays in `(-5, 0)`. The kernels also apply
`sigmoid(beta)` inside and keep the state as `[V, K]`
(`transpose_state_layout=True`). None of that changes the delta rule; our
recurrence takes the computed gate and `sigmoid(beta)` as before, and the
test stub mirrors the in-kernel forms independently.

## Latent MoE

The routed path runs in a 3584-wide latent space instead of the 7168 hidden
width:

1. the gate scores the full-width token (unchanged);
2. `routed_expert_down_proj` maps the token to the latent width once;
3. the selected experts run at the latent width with intermediate 3072;
4. the weighted sum is RMS-normalized (`routed_expert_norm`) and mapped
   back by `routed_expert_up_proj`;
5. the two shared experts run on the full-width token and are added.

The combine stays in fp32, then casts, then the norm runs. Both shared
projections and the norm ship unquantized.

## AttnRes

With `attn_res_block_size = 12` the residual stream is not one running sum.
Every 12 layers (at layers 0, 12, ..., 84) the running sum is snapshotted
into a list of blocks and restarted from zero. Before each attention,
before each FFN and once before the final norm, the layer picks its input
by a per-token softmax over the blocks plus the current running sum. The
query is `norm.weight * proj.weight`, one learned vector per site. The
candidates are RMS-normalized only for scoring; the softmax mixes the raw
candidates. At layer 0 the list is empty, so the attention input is the
embedding itself and the first snapshot is the embedding.

This is per token, so decode carries no extra state and the state cache is
unchanged. The decoder layer returns `(running_sum, blocks)`; `_run` threads
the pair and applies the output mixer before `model.norm`.

## Router convention

The K3 file adds `e_score_correction_bias` out of place. The bias still
steers selection, but the gathered expert weights are the UNBIASED sigmoid
scores, the DeepSeek-V3 convention. Kimi Linear's file adds it in place and
gathers the biased scores. Dropped groups are masked with `-inf` in K3 and
`0.0` in Kimi Linear (no effect with one group). `_KimiMoeGate` keeps one
code path and a flag; `from_hf` sets the flag from the K3-only fields.
`test_k3_gate_weights_are_unbiased_scores` pins the K3 side and
`test_gate_weights_are_biased_scores` pins the Kimi Linear side.

## Checkpoint layout and MXFP4

The checkpoint has 497220 tensors under three prefixes: `language_model.`
(the text model), `vision_tower.` and `mm_projector.`. The loader drops the
last two and strips the first. Only the routed experts are quantized:
`experts.N.w{1,2,3}.weight_packed` (uint8, two E2M1 values per byte, low
nibble first) and `.weight_scale` (uint8, one E8M0 exponent per 32 values,
`scale = 2 ** (byte - 127)`). Attention, the shared experts, the dense MLP,
the latent projections, the AttnRes queries and `lm_head` are BF16.
`quant/mxfp4.py` dequantizes a pair to BF16 through the NVFP4 nibble path.
The loader does this only when `quantization_config` declares
`mxfp4-pack-quantized`. Any other declared format fails when the config
loads.

## Kimi K3.1 Pro: Gated DSA

The K3.1 Pro checkpoint (revision `6d0a0202956796543364d666ab19c6ad1a494ee1`)
has the K3 configuration plus one block, `text_config.dsa_config`. The block
turns each of the 24 MLA layers into a DeepSeek Sparse Attention (DSA)
layer. A DSA layer first runs an indexer, a small scorer that picks for each
query the `index_topk` earlier tokens that the attention can read. The
attention masks every other token out of its softmax. The output gate
`g_proj` stays, so Moonshot calls these layers Gated DSA.

The repository ships no modeling code, only `configuration_kimi_k3.py`. The
math is the DeepSeek-V3.2 indexer with the K3.1 values that SGLang sets in
its K3.1 support (sgl-project/sglang#48415, `KimiLinearConfig._init_dsa`):

    q_t,h     = wq_b(q_lora_t)                    # q_lora: the q_a_layernorm output
    k_j       = LayerNorm_fp32(wk(x_j))           # one key per token, eps = rms_norm_eps
    w_t,h     = weights_proj(x_t) * n_heads^-0.5  # rounded to the model dtype
    score_t,j = sum_h w_t,h * ReLU(q_t,h . k_j * head_dim^-0.5)
    keep_t    = the index_topk best score_t,j over the keys j <= t

The configuration has 32 indexer heads of width 128 and a top-k of 2048. The
indexer has no RoPE (`index_rope_dim` 0) and no Hadamard rotation, and its
key norm is a LayerNorm with a bias. `index_topk_freq` is 1, so every DSA
layer runs its own indexer. vLLM's IndexShare rule skips a layer when
`max(layer - offset + 1, 0) % freq != 0`. That is never true for a frequency
of 1, so `index_skip_topk_offset` 1 has no effect.

The checkpoint adds 120 tensors, five per DSA layer:
`self_attn.indexer.{wq_b,wk,weights_proj}.weight` and
`self_attn.indexer.k_norm.{weight,bias}`. They are BF16, because the
quantization ignore list covers `self_attn`. No K3 tensor is gone.

`KimiLinearConfig.from_hf` reads `dsa_config` and rejects any other
frequency, RoPE width, norm type or layer list. `KimiDsaIndexer`
(`kimi_linear/dsa_indexer.py`) holds the five tensors. `_KimiMlaAttention`
runs it and masks its attention scores with the selection. The state cache
keeps one indexer key per token next to the compressed KV
(`KimiMlaLayerState.index_k`).

While a request holds at most 2048 tokens, each query keeps every earlier
key. The layer then equals the K3 layer bit for bit, and it skips the
scoring. The serving engines quantize the indexer to FP8, while ours scores
in fp32. So the two selections can differ only where two scores tie at the
top-k boundary.

## Validation contract

`tests/unit/models/kimi_linear/test_kimi_k3_parity.py` builds a tiny-random
K3 with every feature on (6 layers, AttnRes block 2, latent 48 in hidden
64, two shared experts, low-rank Q, both gates, the bounded decay) and loads
the same weights into the reference and ours. It pins full-sequence logits,
greedy decode, batched ragged decode through one state cache, chunked
prefill across a conv-kernel boundary, and the gate. The structural tests
pin the config contract, the loader translation with MXFP4 experts, and the
block formulas against direct transcriptions.

K3.1 has no reference to run. `tests/unit/models/kimi_linear/test_models_kimi_k31.py`
pins the indexer against its formula and the DSA layer against a loop over
the kept keys. It also pins K3.1 against K3 while every key stays (bit for
bit). With a top-k of 3, it pins decode, chunked prefill and ragged decode
against the stateless forward. `tests/fixtures/kimi_k31/config.json` is the
released configuration.

`scripts/kimi_linear/check_kimi_k3_checkpoint.py` checks a real checkpoint
without downloading its weights. It reads every tensor's name, dtype and
shape from the safetensors headers, diffs the config against the pinned K3
release, builds the model on the meta device, and runs the loader on meta
tensors. It takes two to four minutes on a laptop CPU. It passes on K3, on
Kimi Linear 48B and on K3.1 Pro at revision `6d0a0202`, where 249876 text
tensors map onto 249876 parameters. To run it:

```bash
uv run python scripts/kimi_linear/check_kimi_k3_checkpoint.py moonshotai/Kimi-K3.1-Pro
```

## What is still open

- No real-weight run. The 2.8T checkpoint needs a streaming loader and a
  multi-GPU script.
- No logits parity for K3.1, because no modeling code exists. The first GPU
  run must compare with vLLM or SGLang output on the same prompts.
- The DSA layer builds the full score matrix and masks it. It saves no
  compute. A kernel that reads only the kept keys is the speed path.
- The MXFP4 experts dequantize to BF16 at load. A resident FP4 grouped GEMM
  is the DeepSeek-V4 FP4 path and is not wired for this family yet.
- Everything open for Kimi Linear: single rank, no prefix sharing, dense
  MLA history, no fused KDA kernel, experts run one at a time.
