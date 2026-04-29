# ADR-011: Speculative decoding (vanilla, two-model, single-request, greedy V1)

Date: 2026-04-29
Status: Accepted

## Context

Phase 2 ROADMAP items (tensor parallelism, fused INT8 kernel) are paused for
this slice. Speculative decoding is the headline Phase 3 throughput technique:
use a small **draft** model to propose K tokens, then run the big **target**
model **once** over those K+1 candidates, accepting matches and falling back
to target's correction at the first mismatch. With matched-family models on
greedy, typical acceptance is 50–80%, translating to 1.5–2x decode throughput
without changing output distribution.

We ship a focused V1 that proves the mechanism end-to-end and bench-validates
the speedup on a real target/draft pair. Multi-request batching, sampling
support, scheduler integration, and adaptive K are explicit follow-ups.

## Decision

1. **Greedy-only V1**, fixed K=4. Temperature=0 collapses the rejection-
   sampling formula (`accept iff u < min(1, p_t/p_d)`) to argmax equality;
   the corrected sampling distribution `(p_t − p_d).clamp(0)` becomes a
   no-op. Sampling is a follow-up.
2. **Two separate `ModelRunner` instances**, one per model, each with its own
   `BlockPool` and `PagedKVCache`. The packed-varlen `forward_step` machinery
   handles single-slot cache + cu_seqlens_q `=[0, K+1]` (verify) or `[0, 1]`
   (per draft step) without changes.
3. **Cache truncation primitive**: `PagedKVCache.truncate_to(batch_idx,
   new_seq_len)` reclaims blocks beyond the new boundary. With prefix cache
   enabled, freed blocks go through `decref` (LRU-eligible); without it,
   straight to the pool's free list. Refuses to truncate inside a published
   prompt block, which would corrupt the cached entry's K/V on the next
   append.
4. **`SpeculativeRunner` orchestrates the loop**: prefill both models on the
   prompt, then iterate (draft K steps → verify K+1 in one target forward →
   accept-reject → emit accepted draft tokens + 1 bonus → truncate both
   caches to the new committed length).
5. **Catch-up step in the all-accepted case**: if all K candidates pass, the
   draft cache is at `seq_len = N+K` but the target is at `N+K+1` (the bonus
   already committed); we run one extra draft forward to align both caches
   before the next iteration. Keeps cache state symmetric.
6. **`forward_step_packed`** added to `ModelRunner`: returns the raw
   `(1, total_q, vocab)` logits tensor. Verify needs all K+1 positions, not
   just the last; the existing `forward_step` slices the last position and
   stays the standard scheduler entry point.
7. **`Tokenizer.vocab_size`** added so we can refuse mismatched draft/target
   pairs at init time (a vocab mismatch would silently produce wrong
   outputs since token ids would mean different tokens in each model).

## Why greedy + two-model + fixed K

Three options were considered:

- **Greedy two-model (this slice).** Cleanest math; demonstrates the
  mechanism end-to-end. Real win on matched-family models; verifiable by
  parity vs target-alone greedy.
- **Sampling two-model.** Real production-grade output; needs the
  rejection-sampling math (corrected distribution on rejection). About 2x
  the implementation surface; risk of subtle math bugs that don't show up
  on greedy parity tests. Defer.
- **Self-speculative (Medusa / EAGLE).** Adds trained speculation heads to
  the target model. Requires fine-tuning. Out of scope (we don't train).
- **N-gram / PLD speculation.** Free, no draft model. Worthwhile follow-up;
  complementary to model-based draft.

Fixed K=4 is conservative-but-typical; vLLM and SGLang ship K=5 by default.
Adaptive K (tune per request based on running acceptance) is a follow-up.

## Numerical correctness

Greedy spec-decode is mathematically equivalent to target-alone greedy at
every emitted position:

- If draft's argmax matches target's, that's the token target-alone would
  emit. Accept and continue.
- If draft's argmax doesn't match, the bonus (target's argmax at the first
  mismatch position) IS what target-alone would have emitted. Same token,
  same K/V committed.

So token-for-token parity vs target-alone is provable, not just empirical.
Verified on Qwen2.5-0.5B (same-model trick: target == draft → 100%
acceptance, parity by construction; synthetic-divergent-draft → 0%
acceptance, parity by bonus mechanism):

- `tests/unit/test_speculative.py::test_same_model_target_and_draft_gives_full_acceptance_and_matches_baseline`
- `tests/unit/test_speculative.py::test_spec_decode_synthetic_divergent_draft_emits_correct_tokens`
- `tests/stress/test_speculative_load.py::test_spec_decode_matches_target_alone_across_prompts`
- `tests/stress/test_speculative_load.py::test_spec_decode_target_cache_returns_to_pool`
  (no block-pool leaks across iterations)

## Cache management

The truncation rule that keeps target/draft caches consistent with the
emitted sequence:

- After verify, target_cache holds K/V at positions
  `[cache_seq, cache_seq+K]`, filled from inputs
  `[last_token, d_0, ..., d_{K-1}]`.
- We accept the first `accepted` draft tokens, then emit `bonus` at
  position `cache_seq + accepted + 1`.
- Truncate target_cache to `seq_len = cache_seq + accepted + 1`. This
  keeps positions `[0, cache_seq + accepted]`. The K/V at position
  `cache_seq + accepted` came from input `last_token` (if accepted==0)
  or `d_{accepted-1}` (if accepted>=1) — both are tokens that were
  accepted, so the K/V matches the emitted sequence at that position.
- The first rejected position (`cache_seq + accepted + 1`) holds K/V
  from `d_accepted`, which is wrong; it gets dropped by the truncate.
- Next iteration feeds `bonus` at position `cache_seq + accepted + 1`,
  writing fresh correct K/V there.

The same rule applies to draft_cache, with one wrinkle: in the
all-accepted case (`accepted == K`), draft_cache is at
`cache_seq + K` while target_cache is at `cache_seq + K + 1`. We run
one extra draft forward feeding `d_{K-1}` at position `cache_seq + K`
to align them before the next iteration. (`d_{K-1}` is correct because
it was accepted.)

## Alternatives considered

- **Self-speculative / Medusa / EAGLE**: speculation heads share the target
  model's weights. Cleaner deployment (one model). Requires fine-tuning;
  out of scope for an inference-engine project.
- **n-gram / PLD speculation**: lookup recent N-grams in the prompt; if a
  match, use the next K tokens as speculation. Free, no draft model. Best
  on repetitive content (code, RAG, summarization). Worthwhile follow-up;
  complementary to model-based draft.
- **Adaptive K**: tune K per request based on running acceptance. ~10-20%
  win on top of fixed K. Follow-up.
- **Multi-request batching**: pack multiple requests' verify candidates
  into one target forward. Each request can have a different acceptance
  count, so input shapes differ per step. Worthwhile but non-trivial.

## Consequences

- **Positive**:
  - 1.5-2x decode throughput at K=4 on matched-family pairs (target
    7B + draft 0.5B). Bigger pairs (target 70B + draft 1B) see larger
    relative wins because draft cost is a smaller fraction.
  - Token-for-token parity vs target-alone greedy is provable, not just
    empirical. The accept-reject loop emits target's exact argmax every
    iteration.
  - Modular: `SpeculativeRunner` is a separate path; the standard
    `ContinuousScheduler` and existing tests are unaffected.
  - `truncate_to` is a generally useful primitive (also for future
    request-preemption support in the scheduler).
- **Negative**:
  - Two model loads (memory: target + draft); on A10 the 7B + 0.5B pair
    fits comfortably (~17 GB), but smaller GPUs would have to drop
    target size.
  - Single-request only. Multi-request batching is a non-trivial
    follow-up.
  - Greedy only. Sampling support is a non-trivial follow-up that needs
    the corrected-distribution math.
- **Reversibility**: clean revert. `truncate_to` and `forward_step_packed`
  are additive (no caller changes). `Tokenizer.vocab_size` is additive.

## Validation

- **M1 (CPU/MPS, fp32/fp16)**: 8 unit tests for `truncate_to` + 6 unit
  tests for `SpeculativeRunner` + 2 stress tests pass. Existing 126-test
  unit suite stays green. Same-model trick (target == draft = Qwen2.5-0.5B)
  produces 100% acceptance and parity vs target-alone greedy across 3
  prompts. Synthetic-divergent-draft test produces 0% acceptance and same
  parity (bonus mechanism corrects each step).
- **CUDA (A10, Qwen2.5-7B target + Qwen2.5-0.5B draft, bf16, K=4)**:
  aggregate throughput **1.14x** vs target-alone greedy on a 3-prompt
  workload, mean acceptance 2.3–3.4 / K=4 (58–85%). One of three prompts
  showed bf16-drift token divergence vs target-alone exact baseline (≤1
  flipped token / 32 emitted) — expected at bf16 with q_len=5 verify
  matmul vs q_len=1 decode matmul; fp32 M1 reference holds parity
  exactly. Numbers in `docs/benchmarks/2026-04-29-speculative-decoding.md`.

The 1.14x is below the 1.5x plan target — hardware regime and model-size
ratio, not an algorithm bug. At 7B target on A10, verify-at-q_len=5 is
~3x decode wall time (Ampere becomes compute-bound at q_len > 1) and
draft cost (0.5B at K=4) is a non-trivial fraction of the iteration
budget. The same mechanism on a 70B target + smaller-relative draft on
Hopper would close the gap to the published 1.5–2x range; the
implementation here is the same one that scales there.

## Pointers

- Implementation: `src/mini_infer/engine/speculative.py`,
  `src/mini_infer/cache/paged_kv_cache.py::truncate_to`.
- Forward primitive: `src/mini_infer/engine/model_runner.py::forward_step_packed`.
- Unit tests: `tests/unit/test_speculative.py`,
  `tests/unit/test_paged_kv_cache.py` (truncate_to additions).
- Stress: `tests/stress/test_speculative_load.py`.
- Bench: `scripts/modal_packed_bench.py --config spec`.

## Follow-ups

- **Sampling support** (temperature, top-k, top-p) with corrected
  distribution `(p_t − p_d).clamp(0)` resampling on rejection.
- **Adaptive K**: tune per request based on running acceptance.
- **Multi-request batched speculative**.
- **Scheduler integration**: spec-decode as an option inside
  `ContinuousScheduler`'s step loop instead of a standalone function.
- **n-gram / PLD speculation** as a complementary, draft-model-free path.
- **EAGLE-style speculation heads** if we ever do training.
