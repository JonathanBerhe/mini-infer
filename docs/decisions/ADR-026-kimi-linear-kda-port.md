# ADR-026: Kimi Linear port (KDA linear attention, hybrid NoPE-MLA, KimiStateCache)

Date: 2026-07-18
Status: Accepted

## Context

Kimi K3 (announced 2026-07-16, weights + tech report 2026-07-27) is built on
Kimi Delta Attention (KDA) from Kimi Linear (arXiv:2510.26692), a hybrid
linear-attention architecture with a runnable reference today
(`moonshotai/Kimi-Linear-48B-A3B-Instruct`, trust_remote_code + FLA kernels).
Porting `kimi_linear` first validates ~90% of K3's inference-relevant novelty
against a real oracle before the K3 clock starts; see
[docs/plans/kimi-k3.md](../plans/kimi-k3.md) and the
[spec](../plans/kimi-k3-spec.md).

KDA is this repo's first linear-attention family: decode state is a
fixed-size fp32 matrix per layer (delta rule with per-channel decay) plus
short-conv tails, not per-token KV. The 1-in-4 full-attention layers are
MLA with **no rotary embedding anywhere** (NoPE; position information comes
entirely from the KDA layers). Neither state kind fits PagedAttention's
one-entry-per-token model.

## Decision

1. **From-scratch KDA math in `blocks/kda.py`**: an explicit one-token
   recurrence (decode / oracle) and a chunked WY-form evaluation (prefill),
   pinned equal by unit tests. The FLA Triton kernels stay a test-time
   oracle, not a dependency.
2. **Per-request `KimiStateCache`** (`cache/kimi_state_cache.py`): KDA
   layers hold the fp32 matrix state + three conv tails; MLA layers hold a
   dense compressed-KV buffer (`kv_a` output, re-decompressed on read like
   `blocks/mla.py`). A separate class from V4's `StateCache`, same
   lifecycle contract.
3. **Generalize the state-cache serving seam instead of forking it**:
   models expose `build_state_cache(...)`, caches expose `copy_row_from`,
   and `StateCacheGenerator` / `StateCacheContinuousScheduler` are typed
   against a small protocol. V4 behavior is unchanged (pure delegation);
   Kimi rides the same scheduler, and its scheduler output is
   parity-tested against the single-request oracle.
4. **A Kimi-specific MoE gate** (`_KimiMoeGate`): the reference gathers
   expert weights from BIASED sigmoid scores (an in-place `+=` through a
   view), unlike DeepSeek-V3 / GLM where the bias tilts selection only.
   Reusing `GlmNoAuxTcGate` would be silently wrong; a regression test pins
   the difference.
5. **Vendored, revision-pinned reference + stubbed FLA** for parity: the
   remote modeling code is cloned at a fixed SHA
   (`scripts/clone_kimi_linear_reference.py`), its FLA imports replaced by
   transcriptions of FLA's own naive reference (independent of our
   implementation), and the three transformers-5.14 drifts bridged by
   scoped patches in the test helper (restored after import so 5.14-native
   families are unaffected).
6. **Chunked prefill on the state path**: because the family is
   position-free, `forward_prefill_with_cache` continues from
   `start_pos > 0` by carrying the KDA state, conv tails, and MLA buffer
   offset. V4's state path does not support this; Kimi's math makes it
   natural, and a parity test crosses a conv-kernel boundary mid-prompt.

## Alternatives considered

- **Store per-token pre-state inputs as PagedKVCache streams** (the Inkling
  conv pattern) and recompute the recurrence per step. Defeats linear
  attention's O(1) decode, which is the architecture's point.
- **Extend V4's `StateCache` with new layer kinds.** The V4 state (SWA ring,
  compressor accumulators, indexer sub-state) and the Kimi state share no
  tensors; one class holding both would bury each cache's indexing, which
  the state-cache design deliberately keeps visible.
- **Depend on `fla-core` for the kernels.** Triton-only on the paths that
  matter (no CPU/MPS), and contrary to the from-scratch charter; kept as
  the semantics oracle instead.
- **Wait for K3 (no kimi_linear family).** Compresses KDA, the cache design,
  and the K3 delta into the same 30-day window and forfeits the only real
  checkpoint KDA can be validated against (2.8T is out of budget-reach).

## Consequences

- New family: `KimiLinearForCausalLM` (registry key `KimiLinearForCausalLM`,
  `USES_STATE_CACHE = True`), serving through the existing state-cache
  scheduler with prefix sharing disabled (whole-state snapshots for KDA,
  vLLM's "KDA prefill cache" idea, are a follow-up).
- MLA history is dense per request (bounded by `max_seq_len`), not paged; a
  mixed paged-KV + recurrent-state cache is an engine-level follow-up.
- Single-rank only (TP would shard KDA conv channels and per-head state);
  `load_weights` raises under TP, like Inkling did at first.
- `einops` joins the dev group (the vendored reference imports it); the
  runtime engine gains no new dependency.
- Validation: `test_kda_block.py` (chunkwise == recurrent, conv
  prefill == step), `test_kimi_linear_parity.py` (prefill cos > 0.999 +
  argmax + allclose 1e-3; token-exact greedy, batched-ragged, chunked
  prefill; gate component + biased-weights pin),
  `test_models_kimi_linear.py` (config validation, cache layout, scheduler
  smoke vs oracle). Real-checkpoint gate (48B-A3B on Modal) is Phase A5,
  priced and approved before spend.

## References

- Kimi Linear: arXiv:2510.26692; reference code
  `moonshotai/Kimi-Linear-48B-A3B-Instruct` @ `e1df551a4471`.
- FLA naive semantics: `fla-org/flash-linear-attention`,
  `fla/ops/kda/naive.py` (MIT).
- Kimi K3 announcement: kimi.com blog, 2026-07-16.
- Precedents: ADR-014/019 (V4 StateCache), ADR-021 (GLM noaux_tc gate),
  ADR-025 (Inkling conv streams, transformers pin realignment).
