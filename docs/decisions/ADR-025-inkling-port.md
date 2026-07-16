# ADR-025: Inkling port (relative-bias hybrid attention, SConv, shared-expert-sink MoE)

Date: 2026-07-16
Status: Accepted

## Context

Thinking Machines released Inkling (975B total / 41B active MoE, Apache 2.0,
`thinkingmachines/inkling`; a 276B/12B Inkling-Small preview shares the
architecture). It is the first prominent open-weights model with no RoPE:
position enters attention through a learned, query-conditioned relative bias.
It also introduces short convolutions in the decoder block and a new MoE
weighting scheme. Reference implementation: transformers 5.14
(`InklingForCausalLM` / `InklingForConditionalGeneration`). Text path only;
the vision/audio towers are out of scope by design.

The distinctive components, and where each lives:

1. **Relative position bias** (`blocks/inkling_rel_bias.py`): `r_proj` emits
   `d_rel=16` features per token per head; a trained `(d_rel, rel_extent)`
   bank maps them to a bias per backward distance, zero beyond the extent
   (window size on sliding layers, 1024 on global layers). Global layers also
   scale q and the bias by `tau = 1 + alpha*log(max(1, (pos+1)/n_floor))`.
2. **Hybrid attention 5:1**: `hybrid_sliding` layers (window 512, 64q/16kv
   heads) interleaved with `hybrid` global layers (64q/8kv). Per-head RMS
   QK-norm and 1/d logit scaling.
3. **Short convolutions** (`blocks/inkling_sconv.py`): four depthwise causal
   convs (kernel 4, fp32, residual inside) per layer: on the K and V
   projections and on the attention/MLP branch outputs.
4. **Shared-expert-sink MoE** (`blocks/inkling_moe.py`): sigmoid router with
   aux-loss-free selection bias (DeepSeek-V3 style), but expert weights come
   from a log-sigmoid softmax over the top-6 routed logits JOINTLY with the 2
   shared experts' logits, scaled by `route_scale=8 * global_scale`.
5. **muP unembedding**: hidden / 24 before the untied lm_head; logits sliced
   to `unpadded_vocab_size`.

## Decision

- **Attention path**: fold relative bias + causality + window into a
  per-request additive mask consumed by `packed_attention_torch`'s
  `block_mask` path (the MiniMax-M3 MSA precedent), and pin
  `required_attention_backend() = "torch"`. HF makes the same call: the
  family ships with flash-attn disabled because no fused kernel takes a
  per-head additive bias.
- **Conv state**: store each conv's PRE-conv inputs as per-token streams in
  the `PagedKVCache` (`conv_k`/`conv_v`/`conv_attn`/`conv_mlp` next to the
  post-conv `k`/`v`), and gather the `kernel_size - 1` tail per step.
- **MoE**: new `InklingGate`/`InklingMoE` rather than extending
  `GlmNoAuxTcGate`; the weighting math and gamma'd shared experts share no
  load-bearing code with the DeepSeek-V3 gate.
- **Single-rank only** for now; `load_weights` raises under TP.
- **transformers pin bumped to 5.14.x** (the Inkling modeling code's first
  release), which changed two existing families' reference numerics; both
  ports were realigned in the same change (see Consequences).

## Alternatives Considered

- **Per-request rolling conv states** (Mamba-style `(channels, K-1)` ring
  buffers). Less memory, but a new per-request state type in the cache layer,
  and it breaks under `truncate_to` (OOM recovery) and prefix-cache rollback,
  where per-token streams stay trivially correct. Deferred as a
  benchmark-gated optimization, not skipped on merit.
- **Extending `packed_attention_forward` with a bias argument** instead of
  building masks model-side. The existing `block_mask` hook already composes
  bias + mask per request and is proven by M3; a second bias pathway in the
  dispatcher would duplicate it.
- **Pinning transformers to 5.12 and vendoring Inkling's modeling code** to
  avoid realigning M3/GLM. Rejected: HF is the parity reference by policy,
  and 5.14's M3/GLM changes are corrections toward the actual releases, not
  regressions.

## Consequences

- Inkling decodes through the standard `PagedKVCache`/continuous-batching
  path, including chunked prefill and batched ragged decode (parity tests pin
  both token-for-token against HF incremental decode).
- The conv streams cost roughly `2*hidden + 2*kv_dim` extra pool bytes per
  token per layer (~4-7x the raw KV footprint at Inkling's shapes). Fine for
  parity work and small contexts; a rolling conv-state buffer is the known
  follow-up before any long-context serving claim.
- Every attention read goes through the materialized torch path; no CUDA fast
  path yet. Candidates in order of value: a paged decode kernel that applies
  the bias inline (the bias only touches the last `rel_extent` keys), and
  flex-attention/score-mod on the GPU backend.
- The 5.12 -> 5.14 bump surfaced two reference corrections, both realigned
  and re-validated in this change:
  - **GLM-MoE-DSA**: the DSA indexer's RoPE is interleaved (5.12 shipped it
    non-interleaved). One-line fix in `blocks/glm_dsa_indexer.py`; scores are
    permutation-invariant to our interleave-back layout, so parity holds
    bit-for-bit. The GPU-validated 5.12 behavior was wrong per upstream.
  - **MiniMax-M3**: MSA block selection is per index head (one per KV/GQA
    group), not one shared set per query. `minimax_m3_indexer.py` now returns
    `(B, h, S, topk)`, `build_block_mask` expands each head's selection
    across its group, and `msa_paged_attention.py` (torch + Triton) walks one
    block list per KV head. TP slices the replicated indexer's mask to the
    rank-local query heads. The Triton kernel change is CPU-validated via
    the torch reference; the GPU kernel A/B on Modal should be re-run before
    the next M3 real-model claim.
- Loading the real 975B checkpoint (or the 276B Inkling-Small) is out of
  budget for now, same posture as GLM's 753B: needs `load_weights_streaming`
  plus a multi-GPU staging script. The tiny-config parity suite is the
  correctness gate.
