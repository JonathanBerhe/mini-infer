# ADR-014: DeepSeek-V4 hybrid attention contribution

Date: 2026-05-08 (initial), 2026-05-10 (full primitive integration)
Status: Accepted

> Update (2026-05-10): the three primitives that this ADR deferred are now
> in the model. The sections below keep the original trade-offs. The last
> section gives the later state.

## Context

DeepSeek-V4 (April 2026) mixes Compressed Sparse Attention (CSA), Heavily
Compressed Attention (HCA), a sliding-window branch and a per-head attention
sink. At 1M context, V4-Pro uses 27% of the FLOPs and 10% of the KV cache of
V3.2 (paper §2.3, §4.2.1). The published inference reference
(`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`) uses
fused tilelang FP4 and FP8 kernels. It is a good parity oracle but hard to
read.

This stage implements the attention math end to end, on synthetic
configurations that exercise every formula. The other V4 features are
large, separate pieces of work.

## Decision

Implement the V4 attention at block level, plus a runnable hybrid backbone.
Each step has a bit-parity test against the published reference. The work
is five commits, `b632256` … `4dabeb1`:

1. Six primitives in `src/mini_infer/models/deepseek_v4/`:
   - `TokenLevelCompressor` (§2.3.2) pools every `m` tokens into one KV
     entry with softmax weights. HCA uses `overlap_mode=False`, `m'=128`.
     CSA uses `overlap_mode=True`, `m=4`, and also pools the previous
     block.
   - `LightningIndexer` (§2.3.2) is the top-k selector of CSA.
   - `AttentionSink` (§2.3.3) adds a learned per-head logit to the softmax
     denominator. It prevents attention collapse at long context.
   - `GroupedOutputProjection` (§2.3.2) projects the `n_h` head outputs in
     `g` groups. This cuts compute when `n_h` is large (V4-Pro: 128 heads).
   - `apply_partial_rope_last_n_dims` rotates only the last
     `rope_head_dim` dimensions.
   - `hca_mqa_with_sink` (`models/deepseek_v4/hca_attention.py`) is the PyTorch core
     attention: many Q heads, one KV head, sink logits.
2. Two attention blocks. `HCAAttention` (`hca.py`) is SWA, heavy
   compression, sink and grouped output, and attends to all causally valid
   compressed entries. `CSAAttention` (`csa.py`) adds a `LightningIndexer`
   that picks `top_k` of them.
3. A per-request `StateCache` (`cache/state_cache.py`), a fixed-size pool
   per layer. It holds the SWA circular buffer, the append-only compressed
   history, the in-flight accumulator of the compressor and, for CSA, an
   indexer sub-cache.
4. Decode-step compressors (`forward_decode_step`) and
   `HCAAttention.forward_decode` / `CSAAttention.forward_decode`, which
   decode one token from a `StateCache`. A compressed entry comes out at
   each block boundary.
5. `DeepseekV4ForCausalLM` (`models/deepseek_v4/deepseek_v4.py`), registered under the
   HF architecture string. `compress_ratios` drives the attention class of
   each layer: ratio 4 gives `CSAAttention`, any other ratio gives
   `HCAAttention`. This per-layer dispatch selects the type from the
   configuration, so no subclass is necessary for each layer mix.

## Alternatives Considered

- Implement Hyper-Connections (§2.5) now. The reference replaces standard
  residuals with a Sinkhorn-mixed `(B, T, hc_mult, dim)` multi-residual.
  Rejected for this stage: it is separate research and does not change the
  attention.
- Implement MoE FFN with hash routing (§2.2) now. Rejected for this stage:
  the backbone runs with SwiGLU, and the MoE becomes the gate for real V4
  weights.
- Plug compressed entries into PagedKVCache instead of StateCache. The
  Stage C3 BlockPool (commit `c9015fe`) already supports per-stream
  allocation. Rejected for this stage: the SWA circular buffer and the
  in-flight accumulator (`kv_state` / `score_state`) do not fit the paged
  "one slot per token" model. A follow-up can move the compressed history
  into a paged stream once cross-request prefix-sharing becomes a workload.
- Implement YaRN long-context RoPE, which V4 uses past 4k context.
  Rejected: it does not change the attention math, and the parity tests
  run at T=512.
- Implement on-disk KV (§3.6.2). Rejected: it is storage, not compute, and
  no workload needs it yet.
- Ship the blocks without a model class. Rejected: the class is about 200
  lines. It tests the hybrid layer dispatch and the per-layer StateCache
  wiring, and it runs the demo (`scripts/deepseek_v4/demo_deepseek_v4_hybrid.py`).

## Consequences

- `StateCache` serves any architecture with per-request SWA and compressor
  state. The dispatch in `DeepseekV4DecoderLayer` is a template for later
  hybrid-attention models.
- Stage C3's per-stream BlockPool (`StreamSpec`) now has two consumers:
  MLA + V4's compressed branch.
- The `StateCache` lives outside `PagedKVCache`. A request that spans V4
  attention and a paged-KV model needs two cache lifecycles. It is not a
  concern today. A real V4-serving workload must unify them.
- Prefix sharing between requests does not work for V4 layers yet.
- The backbone uses pre-norm residuals and SwiGLU, so it does not match a
  full V4 reference with HC and MoE. The per-layer attention parity holds.

Each block has its own `tests/unit/test_v4_*_parity.py` test:

| Stage | Cosine sim | Max abs diff |
|---|---|---|
| HCA prefill | 1.0000 | 5.4e-9 |
| CSA prefill | 1.0000 | 1.2e-10 |
| HCA decode (8 steps, 1 flush) | 0.9999999 | 5.8e-11 |
| CSA decode (8 steps, 2 flushes) | 0.9999999 | 8.7e-11 |

With the narrower unit tests and the model-level tests, all 53 V4 tests
pass. The paper is at `docs/papers/deepseek-v4.pdf` (gitignored). The
reference is vendored read-only at `third_party/deepseek_v4_reference/`. The
original 4-stage plan is
[docs/plans/deepseek-v4-attention.md](../plans/deepseek-v4-attention.md).

## Update: full primitive integration (2026-05-10)

- MoE FFN with hash routing (§2.2). `HashRoutedGate` uses a per-token-id
  `tid2eid` lookup, or the top-k of `score_func(scores) + bias`. It matches
  the reference `Gate` in all six (mode, score_func) cells (cosine sim ≥
  0.99999). `HashRoutedMoEFFN` adds `MixtralExpert`-shaped MLPs and one
  shared-expert MLP. `cfg.is_hash_routed_layer(layer_idx)` picks hash
  routing for the first `cfg.num_hash_routed_layers` layers.
- Hyper-Connections (§2.5). `hc_split_sinkhorn` (`hyper_connections.py`)
  is a pure-PyTorch transcription of the tilelang kernel of the reference.
  The parity tests call it, so `Block.hc_pre` / `Block.hc_post` of the
  reference run in tests. `HyperConnections` matches the reference for
  `hc_mult ∈ {2, 3, 4}`. `HCHeadReduction` collapses the residual streams
  before the LM head.
- YaRN long-context RoPE (§3.1). `RotaryEmbedding` applies it past
  `yarn_original_seq_len` (commit `6ca690d`).

The loader supports the full V4-Flash storage format:

- `compress_ratios` has three modes (`0 → SWA`, `4 → CSA`, `128 → HCA`).
- `src/mini_infer/distributed/` holds the tensor-parallel linears,
  expert-parallel MoE and a vocab-parallel embedding and LM head. At
  `world_size=1` they are plain `nn.Linear` / `nn.Embedding`, so
  single-device tests stay bit-identical. A Qwen2.5-7B two-rank TP smoke
  passed on H100.
- `DeepseekV4ForCausalLM.load_weights` dequantizes block-FP8 non-MoE
  weights (`e4m3fn`, 128×128 `e8m0fnu` scales) and NVFP4 MoE experts
  (block-32 `e8m0fnu` scales). It renames the compact keys of the reference
  and remaps expert indices. It streams slices to the GPU, so peak memory
  is one rank's share, never the full 158 GB.
- A dry run against the real V4-Flash safetensors index matches all 34,223
  parameters. FP8 and NVFP4 dequant match a real shard.

At this update, the open work was the full forward pass through TP on 2×
B200. More than 100 V4 tests pass.
`test_hc_backbone_combines_with_moe_ffn` runs one `DeepseekV4ForCausalLM`
with all primitives together.
