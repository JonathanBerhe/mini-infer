# ADR-011: Speculative decoding (vanilla, two-model, single-request, greedy V1)

Date: 2026-04-29
Status: Accepted

## Context

Speculative decoding is the main Phase 3 throughput technique. A small draft
model proposes K tokens. The big target model then runs once over the K+1
candidates. It accepts the tokens that match and uses its own correction at
the first mismatch. With models of the same family on greedy, acceptance is
typically 50–80%, which gives 1.5–2x decode throughput with the same output.
This V1 proves the mechanism on a real target and draft pair.

## Decision

1. Greedy only, with a fixed K=4. At temperature 0 the rejection-sampling
   rule (`accept iff u < min(1, p_t/p_d)`) becomes argmax equality, and the
   corrected distribution `(p_t − p_d).clamp(0)` has no effect.
2. Two separate `ModelRunner` instances, one per model, each with its own
   `BlockPool` and `PagedKVCache`. The packed-varlen `forward_step` runs the
   verify and draft steps with no change.
3. `PagedKVCache.truncate_to(batch_idx, new_seq_len)` frees the blocks past
   the new length. It refuses to cut inside a published prompt block,
   because the next append can corrupt the cached K/V.
4. `SpeculativeRunner` (`src/mini_infer/engine/speculative.py`) runs the
   loop. It prefills both models. Each iteration drafts K steps, verifies
   K+1 tokens in one target forward, and accepts or rejects. It then emits
   the accepted tokens plus one bonus token and truncates both caches.
5. Catch-up step when all K candidates pass. The draft cache is then at
   `seq_len = N+K`, but the target is at `N+K+1`. One extra draft forward
   feeds `d_{K-1}` to align the two caches before the next iteration.
6. `forward_step_packed` in `ModelRunner` returns the raw
   `(1, total_q, vocab)` logits, because verify needs all K+1 positions.
   `forward_step` stays the scheduler entry point.
7. `Tokenizer.vocab_size` lets the runner refuse a draft and target pair
   with different vocabularies. A mismatch gives wrong output with no error.

After verify, the runner truncates the target cache to
`seq_len = cache_seq + accepted + 1`, where `accepted` counts the accepted
draft tokens. The last kept position holds K/V from an accepted token. The
first rejected position holds K/V from `d_accepted`, which is wrong, so the
truncate drops it. The next iteration writes the bonus token there.

The output is the same as target-alone greedy. If the draft argmax matches
the target argmax, the target emits that token. If not, the bonus is the
target argmax at that position. So token-for-token parity is provable, not
only measured.

## Alternatives considered

- Sampling with two models. It needs the corrected-distribution math, and
  its bugs do not show in greedy parity tests. Deferred.
- Self-speculative heads (Medusa, EAGLE). They need fine-tuning, and this
  project does not train.
- N-gram / PLD speculation. It needs no draft model. It is a complementary
  follow-up.
- Adaptive K, tuned per request from the running acceptance. It gives about
  10-20% over fixed K. Follow-up. vLLM and SGLang use K=5 by default.
- Multi-request batching. Each request accepts a different count, so input
  shapes differ per step. Follow-up.

## Consequences

- On A10 and H100 (Qwen2.5-7B target, Qwen2.5-0.5B draft, bf16, K=4),
  aggregate throughput against target-alone greedy is 1.14x on A10 and 1.00x
  on H100. Mean acceptance is 2.3–3.4 of K=4 (58–85%) on both GPUs. Numbers
  are in `docs/benchmarks/2026-04-29-speculative-decoding.md`.
- Both results are below the 1.5x plan target. On H100 the faster baseline
  decode leaves less overhead to amortize. Speculation wins 1.29x on the
  longest, lowest-acceptance prompt and loses 0.83x on the shorter ones. The
  cause is the regime, not a bug. The published 1.5–2x applies at 70B+
  target size, where decode stays HBM-bound and the draft costs less.
- On A10, one of three prompts differed from the exact target-alone baseline
  by bf16 drift (at most 1 flipped token in 32). H100 matched all three.
- `tests/unit/engine/test_speculative.py` and
  `tests/stress/test_speculative_load.py` cover parity. With target == draft
  (Qwen2.5-0.5B), acceptance is 100%. A synthetic divergent draft gives 0%
  acceptance. Both match target-alone greedy, and no blocks leak.
- Costs: two models in memory (the 7B + 0.5B pair takes about 17 GB on A10),
  one request at a time, and greedy only.
- `SpeculativeRunner` is a separate path, so `ContinuousScheduler` does not
  change. `truncate_to`, `forward_step_packed` and `Tokenizer.vocab_size`
  are additive. The bench is `scripts/bench/modal_packed_bench.py --config spec`.
