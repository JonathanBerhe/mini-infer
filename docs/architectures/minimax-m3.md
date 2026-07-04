# MiniMax-M3 (MSA) walkthrough

A line-by-line correspondence between the MSA report (arXiv 2606.13392), the
upstream reference (HF transformers `models/minimax_m3_vl/`, the upstreamed
modeling code; NOT trust_remote_code), and mini-infer's from-scratch text-only
port.

> **Audience**: someone with the MSA paper open who wants to follow the
> implementation. Pair this doc with
> [ADR-021](../decisions/ADR-021-minimax-m3-msa.md) (port decisions) and
> [ADR-022](../decisions/ADR-022-msa-block-sparse-decode-kernel.md) (the
> opt-in decode kernel). Where the paper and the HF code disagree, the HF
> code wins; every such spot is flagged below.

## TL;DR

MiniMax-M3 (428B total / 23B active, 60 layers, text path of a multimodal
release) is architecturally a GQA transformer plus three distinctive pieces:

1. **MiniMax Sparse Attention (MSA)**: a small indexer scores 128-token KV
   *blocks* and selects the top-16 (plus the query's own block) per query;
   the main attention then attends only those blocks. The KV cache stays
   FULL (standard paged cache, no compression, no window, no recurrent
   state), which is why M3 runs on mini-infer's stock PagedKVCache +
   continuous-batching + prefix-cache stack, unlike DeepSeek-V4 which needed
   a bespoke StateCache.
2. **`swigluoai` activation**: a clamped GLU
   (`(clamp(up,-7,7)+1) * clamp(gate,max=7) * sigmoid(1.702*clamp(gate,max=7))`)
   used by the dense MLPs and every expert.
3. **DeepSeek-V3-style MoE** (128 experts, top-4, one shared expert, sigmoid
   scoring with an aux-loss-free selection bias): reused verbatim from the
   GLM-MoE-DSA port; only the expert activation differs.

Three properties that took source-reading (and one full parity harness) to pin
down, because the paper or a first reading suggests otherwise:

- **Selection is GLOBAL per query, not per GQA group.** The block scores
  max-pool over the block's tokens AND the index heads, so every main head of
  a query shares one selected-block set. The mask is `[B, 1, S, k_len]`.
- **RoPE is PARTIAL: the first `rotary_dim` (64) of 128 dims.** The
  deployment config ships a FLAT `partial_rotary_factor: 0.5` that HF's
  `standardize_rope_params` folds into rope_parameters; the apply slices to
  the cos width. Trap for parity harnesses: passing an explicit
  `rope_parameters` dict without the factor silently degenerates HF to full
  rope, which briefly convinced us the field was inert; the real 428B gate
  (fluent-but-degenerate output) exposed it. Harness configs must mirror the
  deployment config's flat fields.
- **Index scores carry NO `1/sqrt(d)` scale.** Raw fp32 dot products.

Full-model logits match the HF reference bit-for-bit on the tiny-random CPU
harness (cos, argmax, and allclose at 1e-3), and greedy decode is
token-identical through the paged cache.

## The map

| Primitive | Reference (`modeling_minimax_m3_vl.py`) | Our code |
|---|---|---|
| Backbone | `MiniMaxM3VLForCausalLM` (text path) | `MiniMaxM3ForCausalLM` in `src/mini_infer/models/minimax_m3.py` |
| Decoder layer | `MiniMaxM3VLDecoderLayer` | `_MiniMaxM3DecoderLayer` (same file) |
| Main attention | `MiniMaxM3VLAttention` | `_MiniMaxM3Attention` (same file) |
| MSA indexer | `MiniMaxM3VLIndexer` | `MiniMaxM3Indexer` in `blocks/minimax_m3_indexer.py` |
| Selection -> mask | eager/sdpa mask construction | `MiniMaxM3Indexer.build_block_mask` |
| Gemma RMSNorm | `MiniMaxM3VLRMSNorm` (`(1+w)`, fp32) | `GemmaRMSNorm` in `blocks/gemma_rmsnorm.py` |
| swigluoai | `ACT2FN["swigluoai"]` composition | `swigluoai` in `blocks/activations.py` |
| MoE router | `MiniMaxM3VLTopKRouter` | `GlmNoAuxTcGate` in `blocks/glm_moe_gate.py` (reused) |
| MoE block | `MiniMaxM3VLSparseMoeBlock` | `GlmMoeFFN` (reused, `activation=swigluoai`) |
| RoPE | `MiniMaxM3VLRotaryEmbedding` + `apply_rotary_pos_emb` | `RotaryEmbedding` + `apply_rotary_pos_emb` in `blocks/rope.py` |
| KV cache | HF cache + separate indexer key cache | `PagedKVCache` streams `k`/`v`/`index_k` |
| Block-sparse decode kernel (opt-in) | none upstream (repo kernel's sparse decode is a stub) | `msa_paged_decode` in `cache/msa_paged_attention.py` |

Layer routing: layers 0-2 are dense full attention + dense SwiGLU; layers
3-59 are MSA + MoE (`mlp_layer_types` / `layer_types` in the config,
collapsed to `first_dense_layers` in our config).

## The indexer (paper section 2, the heart of MSA)

The scoring branch is a miniature attention: 4 index heads (`q_proj`
6144->512), one shared index key head (`k_proj` 6144->128), per-head Gemma
RMSNorm on both, applied BEFORE the transpose and BEFORE RoPE, then partial
RoPE with the same width-64 tables as the main branch.

```python
idx_q = q_norm(q_proj(x).view(B, S, 4, 128)).transpose(1, 2)
idx_k = k_norm(k_proj(x).view(B, S, 1, 128)).transpose(1, 2)
idx_q, idx_k = apply_rotary_pos_emb_partial(idx_q, idx_k, cos, sin)
scores = idx_q.float() @ idx_k.float().mT        # RAW fp32 dot, NO 1/sqrt(d)
```

Selection (`MiniMaxM3Indexer._select_blocks`) then, in this exact order:

1. causal mask at TOKEN granularity (`k_pos > q_pos` -> `-inf`), BEFORE any
   pooling;
2. right-pad the last partial block to `block_size` with `-inf` (blocks are
   slot-anchored: block `b` owns key slots `[128b, 128b+128)`; only right
   padding preserves that, which is why HF skips left-padding tests);
3. max-pool over the block's tokens AND over the index heads:
   `scores.view(B, h, S, n_blocks, 128).amax(-1).amax(1)`. This is the
   global-selection property: one set per `(batch, query)`;
4. force-include the local block(s) via `scatter_(+inf)` at
   `clamp(q_block - arange(local_blocks), min=0)`;
5. `topk(min(topk_blocks, n_blocks))`; slots whose score stayed `-inf`
   (all-future blocks) become `-1` padding.

`[PAPER-CONTRADICTION]` twice here: the paper describes a scaled score and a
register-heap top-k; the shipped code uses the raw dot and (in the kernel
repo) a histogram select. We follow the code.

## Selection -> dense additive mask (the parity target)

`[KERNEL-CONTRADICTION]` The HF eager/sdpa reference does NOT gather the
selected blocks. It expands the selection into a dense additive mask and runs
ordinary full attention:

```python
mask = build_block_mask(block_indices, k_len, position_ids, dtype)
# keep (bias 0) iff block selected AND key_pos <= query_pos; else finfo.min
```

Two details that matter for bit-parity: the mask REPLACES the causal mask
(causality is folded in; it is not added on top of a separate causal mask),
and masked scores use `finfo(dtype).min`, not `-inf`. Our serving path feeds
this mask into `packed_attention_torch(block_mask=...)`
(`cache/packed_attention.py`), which is why M3 pins
`required_attention_backend() == "torch"`: no flash/FlashInfer kernel accepts
an arbitrary per-query additive mask.

## Main attention

Standard GQA (64 q heads / 4 kv heads / head_dim 128, scale `128**-0.5`) with
per-head Gemma QK-norm applied on the `(B, S, H, 128)` view before transpose
and RoPE (V is not normed), then partial RoPE over the first 64 dims (NeoX
split-half pairing, theta 5e6, width-64 cos/sin tables):

```python
q = q_norm(q_proj(x).view(B, S, 64, 128)).transpose(1, 2)
k = k_norm(k_proj(x).view(B, S, 4, 128)).transpose(1, 2)
q, k = apply_rotary_pos_emb_partial(q, k, cos, sin)  # rotate [:64], pass [64:]
```

On sparse layers the indexer's mask replaces the causal mask; softmax runs in
fp32 and casts back (`softmax(..., dtype=fp32).to(bf16)`), matching HF.

## MoE (reused from GLM-MoE-DSA)

`MiniMaxM3VLTopKRouter` is semantically `GlmNoAuxTcGate` with
`n_group = topk_group = 1`:

```python
w = sigmoid(router_logits.float())                    # scoring
top4 = topk(w + e_score_correction_bias).indices      # SELECTION uses the bias
weights = w.gather(top4) / w.gather(top4).sum(-1)     # WEIGHTS do not
out = 2.0 * sum(weights_e * expert_e(x)) + shared_expert(x)
```

The only new piece is `swigluoai` inside each expert and the dense MLPs. On
disk the fused `gate_up_proj` splits into contiguous halves (gate first, up
second, NOT interleaved), matching `swigluoai`'s `chunk(2, -1)`.

## Cache layout and serving

Because MSA keeps the full KV, the port rides the standard serving stack. Per
layer the pool holds named streams:

| stream | heads x dim | layers |
|---|---|---|
| `k`, `v` | 4 x 128 | all |
| `index_k` | 1 x 128 | sparse layers (3-59) |

The indexer caches its RoPE'd keys in `index_k` so a decode step scores the
new query against the full history without recomputing past keys (HF keeps an
equivalent separate cache). Decode re-selects the top-k fresh every step from
all cached blocks; there is no incremental block-score state. The savings are
in the main branch (at most 17 blocks attended); the index branch stays
O(context) per step.

Serving gates, all CPU-validated in `tests/unit/test_minimax_m3_parity.py`
and `test_minimax_m3_tp_parity.py`: greedy decode and batched ragged decode
token-equal with HF's incremental decode; world_size=2 tensor+expert parallel
logits identical across ranks and matching the single-rank reference; a
prefix-cache hit reuses the shared blocks (including `index_k`, which the
suffix's selection scores against) bit-exactly; and the documented on-disk
428B layout (`block_sparse_moe.*`, per-expert `w1/w3/w2`, block-level router
bias) loads to the same model as HF's in-memory state_dict.

## The block-sparse decode kernel (opt-in, ADR-022)

The torch path pays O(context) per layer per decode step (materialize +
dense mask + full attention). `msa_paged_decode`
(`cache/msa_paged_attention.py`) reads ONLY the selected blocks from the
paged pool: one Triton program per `(request, KV head)` computes the whole
16-head GQA group tile with online softmax in fp32. Selection comes from the
same `select_cached` routine that builds the oracle's mask, so the two paths
cannot disagree about WHICH tokens are attended, only about float rounding.

Measured (A10, bf16, M3 head geometry; details in
[2026-07-03-msa-decode-kernel.md](../benchmarks/2026-07-03-msa-decode-kernel.md)):
parity cosine 1.000 vs the reference with a bit-identical sparsity probe;
attention-op speedup 2.3x at 1K context to 105.9x at 64K; end-to-end on a
114M synthetic model 0.95x (host overhead dominates at toy scale). Per the
V4-decode-kernel lesson the flag ships OFF; the real-model end-to-end A/B is
the ship gate.

## Parity ladder (how correctness was established)

All layers live under `tests/unit/` (run
`uv run pytest tests/unit/ -k 'minimax or msa or swigluoai'`); the
individual files are deliberately not enumerated here (they churn faster
than the architecture).

1. Mechanism tests against loop-based references: the indexer's selection
   set, the mask builder, and the collapse-to-causal invariant when every
   block is selectable.
2. Activation and mask plumbing: the `swigluoai` forms and the `block_mask`
   extension of the packed torch attention.
3. Full-model bit-parity vs HF on a tiny-random config (3 dense + 2 MSA/MoE
   layers), identical weights loaded through `load_weights`: cos > 0.999,
   argmax-equal, allclose 1e-3. The config must be DEPLOYMENT-shaped (flat
   partial_rotary_factor, no hand-built rope_parameters); a hand-built dict
   once masked the partial-rope path until the real-model gate exposed it.
4. Serving parity: greedy / batched / TP / prefix-cache / disk-layout, plus
   the kernel-path variants (the CPU dispatcher runs the pure-torch
   block-sparse reference, so the wiring is exercised without a GPU).
5. GPU: kernel parity + microbench on A10; the real-checkpoint coherence
   gate runs the pre-quantized (block-FP8 experts) staging via the streaming
   shard loader (`load_weights_streaming`).

## What's still open

- The real-428B gate PASSED (coherent reasoning output, 4-rank TP/EP
  consistency, block-FP8-resident experts on 4x H200; see
  [2026-07-03-minimax-m3-428b-gate.md](../benchmarks/2026-07-03-minimax-m3-428b-gate.md)),
  and the kernel A/B on it came back 0.98x token-identical at 16K, settling
  the decode kernel as off by default (ADR-022).
- A block-sparse PREFILL kernel (the reference's CSR KV-outer path); the
  materialized torch path is the only prefill today.
- The index branch stays O(context) per decode step (full re-score); an
  incremental scoring scheme is the next lever if a long-context profile
  shows it dominating.
