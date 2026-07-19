# Kimi Linear / Kimi K3 implementable spec

> Status: Stage A (Kimi Linear) verified against the reference; Stage B sections marked TBD until the K3 tech report (2026-07-27)
> Reference: `moonshotai/Kimi-Linear-48B-A3B-Instruct` `modeling_kimi.py` at revision `e1df551a447157d4658b573f9a695d57658590e9` (vendored by `scripts/clone_kimi_linear_reference.py`), driving FLA (`fla-org/flash-linear-attention`) kernels whose naive reference is `fla/ops/kda/naive.py`.
> Rule: **HF code wins** over the paper and over blog descriptions. Everything below is pinned from the reference source, not from prose.

## Architecture (Kimi Linear 48B-A3B)

27 layers, hidden 2304, vocab 163840, 1M context. Per the config's **1-indexed** `linear_attn_config.kda_layers` / `full_attn_layers` lists: 20 KDA layers, 7 MLA layers at {4, 8, 12, 16, 20, 24, 27} (~3:1). `is_kda_layer(idx)` checks `(idx + 1) in kda_layers`; the two lists must partition `1..num_layers` (our `from_hf` enforces this). Layer 0 is dense SwiGLU (`first_k_dense_replace = 1`); every other layer is a 256-expert top-8 MoE with 1 shared expert, `routed_scaling_factor = 2.446`. No MTP (`num_nextn_predict_layers = 0`), untied embeddings.

## KDA layer (per layer: 32 heads x head_dim 128, conv kernel 4)

Module names (checkpoint layout): `q_proj`, `k_proj`, `v_proj`, `q/k/v_conv1d` (FLA `ShortConvolution`, an `nn.Conv1d` with weight `(C, 1, W)`), `A_log` `(1, 1, H, 1)` fp32, `dt_bias` `(H*128,)` fp32, `f_a_proj`/`f_b_proj` (low-rank decay-gate features), `b_proj`, `g_a_proj`/`g_b_proj` (low-rank output gate), `o_norm` (FLA `FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation='sigmoid')`), `o_proj`.

Forward, in reference order:

1. `q/k/v = proj(x)` each through its depthwise causal conv + SiLU. The conv cache is `(N, C, W)` holding the last W RAW pre-conv inputs, newest last; a step rolls left and writes at `[-1]`.
2. Decay gate: `g = fused_kda_gate(f_b(f_a(x)), A_log, head_dim, g_bias=dt_bias)` which computes `-exp(A_log) * softplus(raw + dt_bias)` in fp32, output shape `(..., H, 128)`, always `<= 0`. Note the fla-core 0.4.x signature: `head_dim` is the third POSITIONAL arg (GitHub main has a different signature).
3. `beta = sigmoid(b_proj(x))` in fp32, `(..., H)`.
4. Recurrence (`chunk_kda` for spans > 64, `fused_recurrent_kda` otherwise; both with `use_qk_l2norm_in_kernel=True`): q and k are L2-normalized per head (`x * rsqrt(sum(x^2) + 1e-6)`, fp32), q is scaled by `128 ** -0.5`, then per token over state `S` `(B, H, 128, 128)` **fp32** (the kernels assert fp32 initial state):

       S = S * exp(g_t)[..., None]            # per-CHANNEL decay on the key dim
       S = S + (beta_t * k_t) outer (v_t - (k_t[..., None] * S).sum(-2))
       o_t = q_t . S

5. Output: `o_norm(o, g_b(g_a(x)))` = fp32 RMSNorm over head_dim, times the (head-shared) weight, times `sigmoid(gate)`, cast back only at the end; then `o_proj`.

Our split: `blocks/kda.py` holds the pure math (`kda_gate`, `l2norm`, `gated_rmsnorm`, `causal_conv1d_prefill/step`, `kda_recurrent`, `kda_chunkwise`); the layer lives in `models/kimi_linear.py`. Our chunkwise form solves the unit-lower-triangular WY system with `torch.linalg.solve_triangular` instead of FLA's in-place forward substitution (same matrix, different fp op order); `test_kda_block.py` pins chunkwise == recurrent, and the parity suite pins the whole layer against the reference.

## MLA layers (NoPE)

`q_lora_rank` is None (plain `q_proj`, asserted by the reference); `kv_lora_rank 512`, `qk_nope_head_dim 128`, `qk_rope_head_dim 64`, `v_head_dim 128`; softmax scale `(128+64) ** -0.5`; eager attention with fp32 softmax.

**There is no rotary embedding anywhere in this family** (`mla_use_nope` is asserted True; no rotary module exists in the file). The `qk_rope_head_dim` splits of q and of the shared `kv_a_proj_with_mqa` output are concatenated UNROTATED (the k split broadcast to all heads). All position information comes from the KDA layers' convs and decay. Do not "fix" this by adding RoPE; a roped variant is a different family and `from_hf` rejects it.

Cache: we store the raw `kv_a_proj_with_mqa` output (`512 + 64` per token, shared across heads) and re-decompress on read via `kv_a_layernorm + kv_b_proj`, like `blocks/mla.py`. The reference caches the decompressed per-head K/V instead; the math is per-token and deterministic, so both give identical results (parity-verified) and ours keeps the MLA memory win.

## MoE

Gate (`KimiMoEGate`): fp32 `logits = x @ W^T` (no bias), `scores = sigmoid(logits)` (config also allows softmax). Then, the load-bearing divergence from DeepSeek-V3 / GLM:

- the reference adds `e_score_correction_bias` to the scores **in place** (`scores_for_choice = scores.view(...); scores_for_choice += bias` mutates `scores` through the view), so the final `scores.gather(...)` returns **BIASED** weights. GLM's `GlmNoAuxTcGate` gathers UNBIASED scores; that is why `_KimiMoeGate` is a separate class and why `test_gate_weights_are_biased_scores` pins the difference.
- non-kept groups are masked with `0.0` (not `-inf`) before the final top-k. Degenerate here (`num_expert_group = topk_group = 1`) but mirrored faithfully.
- weights renormalize to sum 1 (`+ 1e-20`) when `top_k > 1 and moe_renormalize`, then scale by `routed_scaling_factor`.

Experts are SwiGLU (`experts.N.w1/w2/w3`); the shared expert is one `KimiMLP` of width `num_shared_experts * moe_intermediate_size` (`shared_experts.gate/up/down_proj`), added AFTER the routed sum. Routed contributions combine in fp32 (the reference's `moe_infer` casts to the fp32 weights dtype for the weighted sum). We reuse `MixtralExpert` and `SwiGLU`; only the gate is Kimi-specific.

## Decode state and serving

Per KDA layer per request: `S` `(H, 128, 128)` fp32 + three conv tails `(C, 4)`. Per MLA layer: the per-token compressed buffer. Ours is `cache/kimi_state_cache.py` (`KimiStateCache`), driven through the same `forward_prefill_with_cache` / `forward_decode_with_cache[_ragged]` contract as DeepSeek-V4 and served by the generalized `StateCacheGenerator` + `StateCacheContinuousScheduler` (models now expose `build_state_cache`, caches expose `copy_row_from`).

Because the family is position-free, `forward_prefill_with_cache` continues from `state_cache.start_pos > 0`: chunked prefill works by carrying `S`, the conv tails, and the MLA buffer offset. Prefix sharing (`StatePrefixCache`) stays V4-only; `snapshot_from_cache` raises a clear TypeError for other cache types.

## Weight mapping

The module tree mirrors the checkpoint names, so `load_weights` is a direct copy except: conv weights `(C, 1, W) -> (C, W)` (squeeze the Conv1d singleton) and `o_norm.weight -> o_norm_weight`. Single-rank only for now (TP would shard the conv channels and the per-head state).

## Corrections log (found while porting; HF code wins)

1. **Blog/paper say "NoPE MLA", the config still carries `qk_rope_head_dim` and `rope_theta`.** Resolved by reading the code: the split exists structurally, rotation never happens, `rope_theta` is vestigial.
2. **Kimi MoE weights are biased sigmoid scores**, not DeepSeek-V3's unbiased convention, due to an in-place `+=` through a view. Easy to mis-port by reusing the GLM gate; pinned by a dedicated regression test.
3. **`fused_kda_gate` signature drift**: the checkpoint calls the fla-core 0.4.x form `(g, A_log, head_dim, g_bias=...)`; FLA main has `(g, A_log, dt_bias, ...)`. The test stub implements the 0.4.x form the reference actually calls.
4. **The reference leaves `dt_bias` and `e_score_correction_bias` uninitialized** (`torch.empty`; its `_init_weights` covers only Linear/Embedding). Any fresh-model comparison must overwrite them first or risk inf/NaN flakes; our modules init them to zeros (checkpoints overwrite).
5. **`KimiLinearModel.__init__` force-overrides `_attn_implementation` to flash_attention_2**, even when eager is requested. Set `model.config._attn_implementation = "eager"` AFTER construction for CPU runs; the layers read the config at call time.
6. **transformers 5.14 drift** (the reference targets ~4.57, asserts >= 4.56): `OutputRecorder` moved out of `transformers.utils.generic`; `create_causal_mask` renamed `input_embeds -> inputs_embeds` and dropped `cache_position`; the Cache mask protocol now wants `get_query_offset(layer_idx)` and `get_mask_sizes(q_length: int, ...)`. All three are bridged in `tests/unit/_kimi_reference_helpers.py`, and the `create_causal_mask` patch is **restored immediately after the reference imports**: leaving it installed breaks every 5.14-native family in the same pytest process (it broke MiniMax-M3's parity test until scoped).
7. **`config.head_dim` (72) is unused** by both attention kinds (MLA uses the nope+rope dims, KDA uses `linear_attn_config.head_dim`); ignore it.

## Kimi K3 delta (TBD, tech report 2026-07-27)

Blog-level only, to be pinned from the released code: Gated MLA (output gating on the MLA layers), AttnRes (selective cross-depth retrieval, ~25% training-efficiency claim), Stable LatentMoE (896 experts, 16 active) + Quantile Balancing router (likely training-time), SiTU activation, MXFP4 weights + MXFP8 activations from QAT (fp4-e2m1 data, e8m0 scale per 32-block; sibling of `quant/nvfp4.py`), native multimodality (text-only port, towers dropped at load), 1M context.
