# ADR-026: Kimi Linear port (KDA linear attention, hybrid NoPE-MLA, KimiStateCache)

Date: 2026-07-18
Status: Accepted

## Context

Kimi K3 (announced 2026-07-16) uses Kimi Delta Attention (KDA) from Kimi
Linear (arXiv:2510.26692), which has a runnable reference
(`moonshotai/Kimi-Linear-48B-A3B-Instruct`). A `kimi_linear` port first
tests about 90% of the inference novelty of K3 against a real oracle. See
[docs/plans/kimi-k3.md](../plans/kimi-k3.md) and the
[spec](../plans/kimi-k3-spec.md).

KDA is the first linear-attention family here. Its decode state is a
fixed-size fp32 matrix per layer (a delta rule with per-channel decay) plus
short-conv tails. One layer in four is MLA with no rotary embedding (NoPE).
Neither state fits the one-entry-per-token model of PagedAttention.

## Decision

1. KDA math from scratch in `blocks/kda.py`: a one-token recurrence for
   decode and a chunked WY-form for prefill, kept equal by
   `test_kda_block.py`.
2. A per-request `KimiStateCache` (`models/kimi_linear/state_cache.py`) holds the
   fp32 KDA state, three conv tails and a dense compressed-KV buffer for
   MLA.
3. Generalize the state-cache serving seam. Models expose
   `build_state_cache(...)`, caches expose `copy_row_from`, and
   `StateCacheGenerator` / `StateCacheContinuousScheduler` use a small
   protocol. V4 behavior does not change.
4. A Kimi-specific gate, `_KimiMoeGate`. The reference takes expert weights
   from the BIASED sigmoid scores, but `GlmNoAuxTcGate` uses the bias only
   for selection. A regression test pins the difference.
5. Chunked prefill: `forward_prefill_with_cache` can start from
   `start_pos > 0`. The V4 state path cannot.

## Alternatives considered

- Store per-token inputs as PagedKVCache streams and recompute the
  recurrence. This loses the O(1) decode of linear attention.
- Extend the V4 `StateCache`. The two states share no tensors, and one
  class hides the indexing of each cache.
- Depend on `fla-core`. It is Triton-only (no CPU/MPS), so its kernels stay
  a test oracle.
- Wait for K3. This loses the only real checkpoint to test KDA against.

## Consequences

- New family `KimiLinearForCausalLM` on the state-cache scheduler, with
  prefix sharing off.
- MLA history is dense per request (up to `max_seq_len`), not paged.
- Single rank only: `load_weights` raises under TP.
- `einops` joins the dev group only.
- `test_kimi_linear_parity.py` compares with a reference vendored at a
  fixed SHA (`scripts/kimi_linear/clone_kimi_linear_reference.py`): prefill cos > 0.999
  and token-exact greedy. The real-checkpoint gate (48B-A3B on Modal) is
  Phase A5.
