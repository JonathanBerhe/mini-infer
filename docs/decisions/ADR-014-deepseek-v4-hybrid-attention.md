# ADR-014: DeepSeek-V4 hybrid attention contribution

Date: 2026-05-08 (initial), 2026-05-10 (full primitive integration)
Status: Accepted

> **Update (2026-05-10):** the original ADR scoped to "the attention math
> end-to-end + a runnable hybrid backbone", deferring three primitives
> as orthogonal stages: MoE FFN with hash routing (§2.2),
> Hyper-Connections residuals (§2.5), and YaRN long-context RoPE (§3.1).
> All three have since been implemented and integrated through the
> registered model class. The "Decision" / "Alternatives Considered" /
> "Consequences" sections below are kept verbatim as the original
> trade-off record; the post-update state is summarised in
> "Update — full primitive integration" at the bottom.

## Context

DeepSeek-V4 (released April 2026) ships a fundamentally new attention
architecture: a hybrid of **Compressed Sparse Attention (CSA)** and
**Heavily Compressed Attention (HCA)** plus a sliding-window branch and
a per-head attention sink. V4-Pro at 1M context uses 27% of the FLOPs
and 10% of the KV cache of V3.2 thanks to this design (paper §2.3,
§4.2.1). The published inference reference at
`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference` is
heavily kernel-fused (tilelang FP4 / FP8 paths, Hadamard rotations,
sparse-attention kernels) — useful as a parity oracle, hard to read.

Prior to this ADR, mini-infer had eight model families wired but no
attention research contribution of its own. The portfolio framing
this work targets is "you can read recent research and turn it into
shippable inference code" — an attention contribution, not another
model implementation.

What's worth implementing: the attention math itself, end-to-end, on
synthetic configs that exercise every formula. What's NOT worth
implementing inside this stage: the surrounding plumbing that's
orthogonal to the attention contribution and substantial in its own
right (Hyper-Connections, MoE FFN with hash routing, YaRN
long-context RoPE, on-disk KV, multi-GPU weight sharding).

## Decision

Implement V4's attention contribution end-to-end at the block level
plus a runnable hybrid backbone, **bit-parity-validated** against the
published inference reference at every step. Defer the surrounding V4
features — they're independent stages whose absence doesn't
compromise the attention work.

Scope shipped (five commits, `b632256` … `4dabeb1`):

1. **Six canonical primitives** under `src/mini_infer/models/blocks/v4/`:
   - `TokenLevelCompressor` (paper §2.3.2, formulas 20–23): softmax-
     weighted compression of every `m` consecutive tokens into one KV
     entry. Two modes: HCA (`overlap_mode=False`, `m'=128`) and CSA
     (`overlap_mode=True`, `m=4` — each block also pools the previous
     block's tokens via a doubled `2m` softmax).
   - `LightningIndexer` (paper §2.3.2, formulas 13–17): top-k selector
     for CSA. Per-head Q from the shared `q_lora` latent, ReLU'd
     dot-product against an indexer-private compressor's output, per-
     token per-head weighting collapses heads.
   - `AttentionSink` (paper §2.3.3, formula 27): per-head learnable
     scalar logit added to the softmax denominator. Prevents
     attention collapse at long contexts.
   - `GroupedOutputProjection` (paper §2.3.2): `n_h` head outputs
     split into `g` groups, projected per-group through `n_h/g * c
     -> o_lora_rank` then concatenated and projected to `hidden_size`.
     Cuts compute when `n_h` is large (V4-Pro: 128 heads).
   - `apply_partial_rope_last_n_dims`: rotates only the last
     `rope_head_dim` of a tensor, for the partial-RoPE Q/K/output
     scheme (V4 paper §2.3.3).
   - `hca_mqa_with_sink`: PyTorch reference for the asymmetric MQA-
     with-sink core attention. Handles the `(B, T, n_h, c)` Q against
     a single-head `(B, n_kv, c)` KV with per-head sink logits in the
     softmax denominator. Lives in `cache/hca_attention.py`.

2. **Two attention block modes** under `src/mini_infer/models/blocks/`:
   - `HCAAttention` (`hca.py`): SWA + heavy compression + sink +
     grouped output. **Cosine-sim 1.0000, max abs diff 1.2e-10** vs
     the reference `Attention(compress_ratios=(128,))` on synthetic
     `(B=2, T=512, dim=128, m'=128)` config.
   - `CSAAttention` (`csa.py`): adds a `LightningIndexer` for the
     compressed-branch top-k pick (HCA attends to all causally-valid
     compressed entries; CSA picks `top_k`). **Cosine-sim 1.0000,
     max abs diff 1.2e-10** vs the reference `Attention(compress_ratios=(4,))`
     on synthetic `(B=2, T=64, dim=64, m=4)` config.

3. **Per-request `StateCache`** (`cache/state_cache.py`): per-layer,
   fixed-size pool for SWA + compressor state + indexer state.
   - SWA circular buffer at `start_pos % n_win`.
   - Append-only compressed history.
   - Compressor in-flight accumulator. HCA: `(B, m, c)`. CSA:
     `(B, 2m, 2c)` — slots `[0, m)` hold the previous block's overlap
     data, slots `[m, 2m)` hold the current in-flight block; the
     accumulator slides on flush.
   - Optional indexer sub-cache (CSA layers only) with its own
     compressor state at the indexer head_dim.

4. **Decode-step compressors** (`forward_decode_step` on
   `TokenLevelCompressor`): single-token incremental compression.
   Writes the new token's KV/score to slot `start_pos % m` (HCA) or
   `m + start_pos % m` (CSA), emits a compressed entry on block
   boundaries `((start_pos + 1) % m == 0)`. CSA's slide-on-flush
   mirrors the reference's `kv_state[:, :m] = kv_state[:, m:]`.

5. **`HCAAttention.forward_decode` + `CSAAttention.forward_decode`**:
   single-token decode that reads SWA + compressed history from a
   `StateCache`, runs the compressor decode step, builds decode-time
   `topk_idxs` (window section uses circular indices wrapping at
   `n_win - 1`; compressed section is `arange(0, (start_pos+1)//m)`
   for HCA or the indexer's top-k for CSA), runs the same MQA-with-
   sink kernel. **Cosine-sim 0.9999998–1.0000001, max abs diff < 9e-11
   across 8 decode steps** (one HCA block-flush, two CSA flushes) vs
   the reference.

6. **`DeepseekV4ForCausalLM`** (`models/deepseek_v4.py`): registered
   under `DeepseekV4ForCausalLM` HF architecture string. Per-layer
   attention dispatch driven by `compress_ratios`: ratio==4 wires
   `CSAAttention`, anything else wires `HCAAttention`. End-to-end
   `forward(input_ids, ...)` for prefill + `forward_decode_with_cache(
   input_id, *, start_pos, state_cache)` for one-token decode through
   the full hybrid stack (CSA/HCA layers interleaved).

53 new tests across 8 files. All bit-parity assertions hold.

## Alternatives Considered

**Implement Hyper-Connections (V4 paper §2.5):** The reference's
`Block` replaces standard residuals with a Sinkhorn-mixed
`(B, T, hc_mult, dim)` multi-residual. This is its own piece of
research — orthogonal to the attention contribution. Rejected:
substantial separate body of work; doesn't compromise the attention
demonstration; the eventual "implement HC" stage adds it cleanly.

**Implement MoE FFN with hash routing (V4 paper §2.2):** V4 uses MoE
with hash-routed experts for the first `n_hash_layers` layers, then
softmax-topk after. mini-infer already has Mixtral-style softmax-topk
MoE; the hash-routing piece is a separate primitive. Rejected for
this stage: ships a V4 backbone that runs (with SwiGLU), defers the
MoE to a follow-up so loading real V4 weights becomes the gate.

**Plug compressed entries into PagedKVCache instead of StateCache:**
The Stage C3 BlockPool (commit `c9015fe`) already supports per-stream
allocation. Compressed entries fit as a single-head stream with one
entry per `m` tokens. Rejected for this stage: SWA's circular buffer
and the compressor's in-flight accumulator (`kv_state` /
`score_state`) don't fit the paged "one slot per token" model — they
need per-request fixed-size pools regardless. A `StateCache` makes
the parity tests self-contained; a follow-up can move the compressed
history into a paged stream once cross-request prefix-sharing
becomes a workload.

**Implement YaRN long-context RoPE:** V4 uses YaRN past 4k context.
Rejected: orthogonal RoPE variant, doesn't change the attention math
demonstrated here. The standalone parity tests run at T=512.

**Implement on-disk KV (V4 paper §3.6.2):** Pure storage layer for
prefix sharing across requests / restarts. Rejected: orthogonal to
compute, lowest priority on the original C4 plan, no concrete
workload that needs it yet.

**Skip the model class, just ship the blocks:** Simpler, but loses
the integration test of the hybrid layer dispatch + per-layer
StateCache wiring. Rejected: the model class is ~200 lines of
plumbing on top of the blocks and produces a runnable artifact (the
demo script).

## Consequences

**Positive:**

- mini-infer now has a from-scratch attention contribution validated
  against a published inference reference. All six V4 attention
  primitives implemented; both modes work for both prefill and decode.
- The `StateCache` abstraction is reusable for any architecture with
  per-request SWA + compressor state (a category that includes V4 +
  any future overlap-compression schemes).
- The per-layer attention dispatch in `DeepseekV4DecoderLayer` is a
  template for future hybrid-attention models; the registry pattern
  scales to compress_ratios-driven type selection without
  proliferating subclasses.
- Stage C3's per-stream BlockPool (`StreamSpec`) is now concretely
  motivated by two consumers (MLA + V4's compressed branch) — the
  abstraction earned its keep.

**Negative:**

- The V4-Flash artefact is "TP-ready backbone with a loader that
  matches the published storage format end-to-end against real
  safetensors", not "runs DeepSeek-V4-Pro out of the box". A full
  forward pass through TP on 2× B200 is the remaining gate — load
  is proven, runtime forward-path debugging is open work.
- The `StateCache` lives outside `PagedKVCache`. A request that
  spans both V4 attention and a model that uses paged KV would need
  two separate cache lifecycles. Not a concern today; a real
  V4-serving workload would unify them.

**Trade-offs:**

- Decided to use `StateCache`-based per-request storage rather than
  paging the compressed entries. The decode parity test suite runs
  cleanly because state lifecycle is single-request; the cost is
  that prefix sharing between requests is not yet possible for V4
  layers (would need a paged compressed stream + per-request SWA +
  in-flight state, three cache classes for one model).
- Vanilla pre-norm residuals + SwiGLU mean the backbone parity
  against a "full-V4" reference (with HC + MoE) would NOT hold. The
  per-layer attention parity holds (each block validated against the
  reference's `Attention`); the HC + MoE deltas are orthogonal.

## Validation

Each block has its own `tests/unit/test_v4_*_parity.py` parity test:

| Stage | Test | Cosine sim | Max abs diff |
|---|---|---|---|
| HCA prefill | `test_hca_block_matches_v4_reference` | 1.0000 | 5.4e-9 |
| CSA prefill | `test_csa_block_matches_v4_reference` | 1.0000 | 1.2e-10 |
| HCA decode (8 steps, 1 flush) | `test_hca_decode_matches_v4_reference` | 0.9999999 | 5.8e-11 |
| CSA decode (8 steps, 2 flushes) | `test_csa_decode_matches_v4_reference` | 0.9999999 | 8.7e-11 |

Plus 35 narrower unit tests covering the primitives and the StateCache
+ 13 model-level tests for the hybrid backbone. **53 V4-related tests
total**; all pass.

## References

- DeepSeek-V4 paper §2.3 (attention), §3.6 (cache layout) — local at
  `docs/papers/deepseek-v4.pdf` (gitignored).
- Reference inference code:
  `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
  (vendored read-only at `third_party/deepseek_v4_reference/`).
- Original 4-stage plan: [docs/plans/deepseek-v4-attention.md](../plans/deepseek-v4-attention.md).
- Implementation commits on `main`: `b632256` (HCA prefill),
  `8f90528` (CSA prefill), `c08f3e1` (HCA decode + StateCache),
  `0110253` (CSA decode + StateCache overlap + indexer),
  `4dabeb1` (DeepseekV4ForCausalLM hybrid backbone).
- Follow-up commits (2026-05-10): `93add37` (cache-aware prefill +
  unaligned), `6ca690d` (YaRN), `67345ad` (HashRoutedGate),
  `6b2ddf3` (HashRoutedMoEFFN), `0e02fdf` (decoder MoE integration),
  `408e65a` (HyperConnections primitive), `eeb1b89` (HC decoder
  integration).
- Demo: `scripts/demo_deepseek_v4_hybrid.py`.

## Update — full primitive integration (2026-05-10)

All three primitives originally listed under "Out of scope" / "Alternatives
Considered" have since been implemented and wired through the model:

**MoE FFN with hash routing** (paper §2.2):

- `HashRoutedGate` (`src/mini_infer/models/blocks/hash_routed_gate.py`):
  per-token-id `tid2eid` lookup OR top-k of `score_func(scores) + bias`.
  Three score functions (softmax / sigmoid / softplus_sqrt) with the
  right renorm rule per V4. Bit-parity (cosine sim ≥ 0.99999, max abs
  diff < 3e-9) vs reference `Gate` across all six (mode, score_func) cells.
- `HashRoutedMoEFFN` (`hash_routed_moe_ffn.py`): composes the gate with
  `MixtralExpert`-shaped MLPs + the V2/V3-style "collapse N shared
  experts into one MLP" pattern. fp32 routed accumulator matches the
  reference's `MoE.forward`.
- Decoder integration: `DeepseekV4DecoderLayer` gains `ffn_type ∈
  {"swiglu", "hash_moe"}`; per-layer routing mode (hash for the first
  `cfg.num_hash_routed_layers`, score_topk after) derived from
  `cfg.is_hash_routed_layer(layer_idx)`. The model's forward methods
  thread `input_ids` to layers when MoE is enabled.

**Hyper-Connections** (paper §2.5):

- `hc_split_sinkhorn` (`hyper_connections.py`): pure-PyTorch transcription
  of the reference's tilelang kernel. The kernel stub used by the V4
  parity tests delegates to this PyTorch impl, making the reference's
  `Block.hc_pre` / `Block.hc_post` actually runnable in test environments.
- `HyperConnections`: `(hc_pre, hc_post)` wrapping one sublayer's
  multi-residual mediation, with the trainable `(fn, base, scale)`
  parameters. Bit-parity vs reference `Block.hc_pre` / `Block.hc_post`
  for `hc_mult ∈ {2, 3, 4}`.
- `HCHeadReduction`: collapses `(B, T, hc_mult, dim) → (B, T, dim)`
  before the LM head — mirrors `ParallelHead.hc_head`.
- Decoder integration: `DeepseekV4DecoderLayer` gains
  `use_hyper_connections=True` mode owning per-sublayer
  `HyperConnections` instances. Model expand/reduce: embed output
  `(B, T, dim)` is unsqueezed + expanded to `(B, T, hc_mult, dim)` on
  entry and reduced via `HCHeadReduction` before the LM head.

**YaRN long-context RoPE** (paper §3.1):

- Wired into `RotaryEmbedding` as a wave-frequency correction that
  activates past `yarn_original_seq_len`. Same primitive serves V2 / V3
  / V4. (Stage `6ca690d`, separate from the V4 attention work — the
  `RotaryEmbedding` block now carries it.)

**Status of "load real V4 weights":** the full V4-Flash storage
format is supported end-to-end and validated against the published
`deepseek-ai/DeepSeek-V4-Flash` safetensors:

- **Architecture**: every published V4 primitive (CSA + HCA + SWA +
  AttentionSink + GroupedOutputProjection + LightningIndexer +
  TokenLevelCompressor + HashRoutedMoEFFN + HyperConnections +
  HCHeadReduction + YaRN RoPE) is implemented and integrated through
  `DeepseekV4ForCausalLM`. V4-Flash's three-mode `compress_ratios`
  (`0 → SWA`, `4 → CSA`, `128 → HCA`) dispatches correctly per layer.
- **Tensor parallelism**: column/row-parallel linears, expert-parallel
  MoE, and vocab-parallel embedding/LM-head live in
  `src/mini_infer/distributed/`. All TP-aware modules degrade to
  plain `nn.Linear` / `nn.Embedding` at `world_size=1`, so existing
  single-device tests stay bit-identical. Validated on real H100
  hardware via a Qwen2.5-7B two-rank TP smoke (each rank produces
  finite per-head sliced outputs with the expected
  `hidden_size / world_size` head dim).
- **Loader** (`DeepseekV4ForCausalLM.load_weights`): block-FP8 dequant
  for non-MoE weights (`e4m3fn` weight + 128×128 `e8m0fnu` scales),
  NVFP4 dequant for MoE experts (int8/uint8 packed weight + block-32
  `e8m0fnu` scales — the way V4-Flash's safetensors actually ship
  them), V4-reference compact key rename (`layers.X.attn.wq_a` →
  `model.layers.X.self_attn.q_a_proj.weight`, etc.), expert-parallel
  global→local index remap, meta-device construction + CPU-resident
  state_dict + per-slice GPU streaming (so peak memory is bounded by
  one rank's share, never the full 158 GB).
- **Validation**: dry-run against the real V4-Flash safetensors index
  matches all 34,223 model parameters exactly (zero missing, zero
  unexpected after dequant pairing consumes 33,389 sibling `.scale`
  companions). Dequant numerically verified against a real V4-Flash
  shard for both FP8 (`(1024, 4096) e4m3fn → bf16`) and NVFP4
  (`(2048, 2048) int8 → (2048, 4096) bf16`) paths.

**Remaining open work**: the full forward pass through TP on 2× B200.
The load contract is proven; further runtime debugging of the
forward path (additional meta buffers, dispatch quirks, cross-rank
synchronization) is the last piece before a "V4-Flash generates
coherent text" claim.

**Test count:** 100+ V4-related tests across the primitives + the
integration matrix. The `test_hc_backbone_combines_with_moe_ffn` test
in `tests/unit/test_models_deepseek_v4.py` exercises a single
`DeepseekV4ForCausalLM` with all primitives composed (CSA + HCA + sink
+ grouped output + cache + hash MoE + Hyper-Connections) end-to-end.
