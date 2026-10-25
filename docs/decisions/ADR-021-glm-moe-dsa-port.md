# ADR-021: GLM-5.2 (GlmMoeDsa) from-scratch port

Date: 2026-06-19
Status: Accepted

> Amendment (2026-07-02): the first text of this ADR said that the main
> attention uses non-interleaved RoPE. That came from a bug in transformers
> <= 5.6, which PR huggingface/transformers#46372 fixed. GLM's main MLA uses
> INTERLEAVED (DeepSeek-style) RoPE. The port passes
> `use_interleaved_rope=True` and matches transformers 5.12. Like HF, it no
> longer creates indexer weights on "shared" IndexShare layers.

> Amendment (2026-10-20): GLM-5.3 runs on this port with no code change.
> `zai-org/GLM-5.3` (revision `ba32f8c19c8738e2c57c71798c601d477a7bf844`) is
> another `GlmMoeDsaForCausalLM`, under the GLM-5.3 License (GLM-5.2 is
> MIT). Read the license before you serve the weights.
>
> - The configuration equals GLM-5.2-FP8 except `transformers_version`, and
>   the index is byte-identical. All 141 shards have the same size and a new
>   hash.
> - The fields that `from_hf` ignores do not change the math. For example,
>   `index_topk_freq` 4 with `index_skip_topk_offset` 3 gives the same 21
>   full layers.
> - The index holds the 58794 text tensors of the reference plus 1569 MTP
>   tensors. `load_weights` accepts that key set on meta tensors.
> - transformers 5.15 changed `modeling_glm_moe_dsa.py` in structure only,
>   with bit-identical logits. The pin stays at 5.14.
> - Correction: since transformers 5.14 the indexer also uses interleaved
>   RoPE, and so does the port.
>
> No GLM-5.3 weights ran on a GPU. `tests/unit/test_glm53_config.py` keeps
> these facts under test.

## Context

GLM-5.2 (z.ai, June 2026) ships as `GlmMoeDsaForCausalLM`
(`model_type: glm_moe_dsa`). It is DeepSeek-V3.2 plus one GLM addition:

- MLA attention (q_lora 2048, kv_lora 512, qk_nope 192 / qk_rope 64,
  v_head 256, 64 heads), the family of our `MLAAttention`.
- DeepSeek Sparse Attention (DSA). A Lightning Indexer selects
  `index_topk` (2048) past tokens for each query, and the main MLA attends
  only to those.
- MoE with 256 routed + 1 shared expert, top-8, and a sigmoid `noaux_tc`
  gate (an aux-loss-free selection bias) with grouped top-k.
- IndexShare. The indexer runs only on `"full"` layers, and the next
  `"shared"` layers reuse its selection.
- RoPE theta 8e6, 3 dense SwiGLU layers, then MoE, and one MTP draft layer.

The pinned `transformers` (5.6.2) ships `modeling_glm_moe_dsa`, so each
part can be tested on the CPU without a checkpoint. GLM-5.2 exists only at
~753B parameters. So correctness comes from bit-parity (equal outputs, bit
for bit) on synthetic configurations.

## Decision

Port GLM-5.2 through the existing PagedKVCache engine and reuse the
DeepSeek-V2 code (`src/mini_infer/models/deepseek_v2.py`, with
[ADR-014](ADR-014-deepseek-v4-hybrid-attention.md) as the MLA and
sparse-attention precedent). Test each level against the HF `glm_moe_dsa`
reference, from RoPE to multi-step greedy decode.

1. `MLAAttention` extensions (`models/blocks/mla.py`). They are off by
   default, so V2, V3 and Kimi stay bit-identical. They add the
   `use_interleaved_rope` flag, an optional `indexer` hook, a `dsa_topk`
   argument and `compute_dsa_topk`.
2. `GlmDsaIndexer` (`models/blocks/glm_dsa_indexer.py`) selects the top-k
   raw tokens. `forward_cached` writes keys to a PagedKVCache `index_k`
   stream and scores the full history in decode.
3. A DSA sparse mask (`cache/mla_attention.py`). An optional `dsa_topk`
   masks all keys except the selected ones to `-inf`, as HF's
   `index_mask + causal_mask` does. `None` keeps the dense V2/V3 behavior.
4. `GlmNoAuxTcGate` and `GlmMoeFFN` (`models/blocks/glm_moe_gate.py`). The
   sigmoid `noaux_tc` router uses the selection bias only to choose the
   experts, and weights them by the unbiased scores. The FFN reuses
   `MixtralExpert`.
5. `GlmMoeDsaForCausalLM` (`models/glm_moe_dsa.py`). Full layers carry an
   `index_k` stream in `per_layer_streams`, ordered first so that layer 0
   allocates the blocks. `load_weights` dequantizes the block-FP8
   checkpoint (e4m3 + `weight_scale_inv`, with ceil for partial blocks such
   as the 576-row `kv_a_proj`). It maps global to local experts, so each
   rank loads only its experts under TP. It drops the MTP layer.
6. FP8-resident experts (`blocks/fp8_expert.py`). `Fp8Expert` keeps each
   routed expert in e4m3 with a ceil-block scale and dequantizes per
   matmul. So the FP8 checkpoint fits 8xH200 without the ~1.5 TB BF16 copy.
   `from_hf` sets `expert_dtype="fp8"` from the checkpoint's
   `quantization_config`.

## Alternatives Considered

- `StateCache` (V4-style) instead of PagedKVCache. Rejected, because DSA
  keeps full K/V and only masks it, as the HF reference does.
- Reuse the V4 `LightningIndexer`. Rejected, because it scores compressed
  blocks through a `TokenLevelCompressor`, and GLM scores raw tokens.
- Extend `MoEFFN` for the gate. Rejected, because it routes with softmax.
- Keep indexer keys in a separate buffer per request, like HF's
  `_cached_keys`. Rejected, because it does not work with continuous
  batching.
- Build the MTP draft layer. Rejected, because greedy correctness does not
  need it, and the HF forward never runs it.

## Consequences

mini-infer loads GLM-5.2, matches the HF reference and generates correct
text on the CPU with no checkpoint download. Later DeepSeek-V3.2-family
models can reuse the DSA hook in `MLAAttention`.

On L4 GPUs, the MoE expert dispatch has a GPU-vs-CPU fp32 gap. So TP
results must be compared with a reference on the same device.

The published checkpoint is per-expert block-FP8 (~755 GB, read from the
index by `scripts/inspect_glm_safetensors.py`). The routed experts are ~96%
of the weights, and the FP8-resident model takes ~785 GB on 8xH200. `scripts/modal_glm_stage_weights.py` stages the weights
on a CPU-only function. The 753B checkpoint itself did not run, because it
needs a multi-GPU node.

Limits:

- On "full" indexer layers the q_lora latent is computed twice per step,
  in `compute_dsa_topk` and in the attention forward.
- The `index_k` stream stores RoPE'd keys in bf16, like the main k_rope
  stream. The CPU tests run in fp32 and are exact.

The tests compare with the HF reference on tiny synthetic configurations,
on the CPU in fp32 unless the row names a GPU:

| Test | Result |
|---|---|
| `test_glm_mla_rope_parity`, `test_glm_mla_dsa_attention_parity` | cosine > 0.999, allclose 1e-4 |
| `test_glm_dsa_indexer_parity` | selected key set matches HF |
| `test_glm_moe_gate_parity` | routing + FFN match |
| `test_full_model_parity_vs_hf` | logits allclose 1e-3, greedy argmax identical |
| `test_greedy_decode_parity_vs_hf` | tokens identical to HF over 6 steps |
| `test_batched_decode_matches_hf` | two ragged prompts match HF |
| `test_glm_dsa_prefix_cache` | prefix-hit logits match full prefill |
| `test_glm_moe_tp_parity` | CPU gloo, ws=2: ranks identical, matches ws=1 |
| `scripts/modal_glm_smoke.py` | NVIDIA L4: fp32 parity exact (cos 1.0) |
| `scripts/modal_glm_tp_smoke.py` | 2x L4, NCCL: ranks identical (diff 0) |
| `test_glm_fp8_load` | fp8-resident logits == bf16-dequant |
