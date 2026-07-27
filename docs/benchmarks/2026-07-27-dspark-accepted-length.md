# DSpark drafter: accepted length, calibration, and the confidence threshold

Date: 2026-07-27
Hardware: NVIDIA L4 (24 GB), bf16, materialized-SDPA attention backend
Target: `Qwen/Qwen3-4B`
Drafter: `deepseek-ai/dspark_qwen3_4b_block7` (block size 7, 5 layers, 1.39B)
Workload: 20 prompts per dataset from DeepSpec's bundled `eval_datasets/`,
128 new tokens each, greedy
Engine: mini-infer @ ADR-027 (Stage C)
Script: `scripts/modal_dspark_bench.py`

As far as I can tell this is the first third-party measurement of any DSpark
claim; the paper shipped with no independent replication.

## Headline: accepted length by domain

Accepted length (tau) is mean committed tokens per verification round,
counting the bonus, which is DeepSpec's own definition
(`base_evaluator.py`: `acceptance_lengths.append(accepted_draft_tokens + 1)`).
It is the right headline because it is hardware-independent: ADR-011 found
wall-clock speedup for the two-model V1 collapsing from 1.29x to 0.83x across
prompts on one machine, so a wall-clock number mostly measures the GPU.

| dataset | domain | tau (ours) | paper Table 1 | accept rate | ECE | AUROC |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | math | 3.70 | ~5.57 | 38.6% | 0.338 | 0.842 |
| humaneval | code | 3.79 | ~5.12 | 39.8% | 0.375 | 0.811 |
| mt-bench | chat | 2.85 | ~3.49 | 26.4% | 0.250 | 0.855 |

**Gate: partially met.** The plan asked whether tau lands in the paper's
ballpark. Chat is close (2.85 vs ~3.49, 18% low). Math and code are further
off (3.70 vs ~5.57, 34% low; 3.79 vs ~5.12, 26% low). The domain ORDERING the
paper reports is reproduced (code and math clearly above chat), and the
implementation is bit-parity-validated against DeepSpec's own code on these
exact weights (ADR-027), so this is unlikely to be a port bug. More plausible
differences, none yet ruled out:

- **Prompt formatting.** We feed the raw `turns[0]` text with no chat
  template. The drafter was trained on target-regenerated responses in the
  model's chat format, and an off-distribution prompt shape depresses
  acceptance. This is the single most likely cause and the cheapest to test.
- **Sample size and truncation.** 20 prompts per set at 128 tokens; the paper
  evaluates the full sets to completion. Long math/code answers become
  formulaic (repeated derivation or boilerplate) precisely in their later
  tokens, which is where acceptance is highest, so cutting at 128 tokens
  should bias math and code down more than chat. That matches the pattern in
  the gap.
- **Greedy vs temperature 1.0.** The paper's Table 1 is at temperature 1.0
  with rejection sampling; ours is greedy. These are different acceptance
  criteria, not the same measurement at a different setting.

Do not read the absolute gap as a refutation of the paper. Read it as: on this
setup, with these caveats, the drafter delivers 2.85-3.79 tokens per target
forward.

## Per-position survival

Fraction of rounds whose accepted prefix reached past each position (so this
is cumulative survival, not per-position conditional acceptance):

| dataset | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.78 | 0.59 | 0.45 | 0.32 | 0.23 | 0.19 | 0.14 |
| humaneval | 0.78 | 0.61 | 0.47 | 0.37 | 0.26 | 0.18 | 0.12 |
| mt-bench | 0.70 | 0.45 | 0.27 | 0.18 | 0.12 | 0.08 | 0.05 |

Position 1 survives at 0.70-0.78, below the 0.88 the paper quotes for DFlash
on math (its argument being that a parallel backbone buys capacity at the
position where a rejection costs the whole block), consistent with our lower
overall tau.

Dividing successive entries gives the per-step conditional acceptance, which
is the quantity the Markov head is meant to keep flat rather than let decay:

- gsm8k: 0.78, 0.76, 0.76, 0.72, 0.72, 0.80, 0.73 (flat, no tail decay)
- humaneval: 0.78, 0.78, 0.77, 0.77, 0.71, 0.69, 0.68 (mild decay late)
- mt-bench: 0.70, 0.64, 0.61, 0.67, 0.69, 0.63, 0.61 (lower throughout, but
  not monotonically decaying either)

So the paper's qualitative claim holds here: the tail does not collapse the
way it reports for DFlash (0.72 -> 0.63 on chat). Chat sits uniformly lower
rather than degrading with depth, which is a difference in difficulty, not in
the drafter's ability to hold a block together.

The practical read: past position 4 or 5, fewer than a third of rounds are
still alive, so the last two or three slots of a block-7 proposal mostly
widen the verify forward without paying for themselves. That is exactly the
waste confidence scheduling exists to remove.

## Confidence threshold sweep (mt-bench)

| threshold | tau | draft tokens offered/round | accept rate |
|---:|---:|---:|---:|
| off (0.00) | 2.85 | 7.00 | 26.4% |
| 0.30 | 2.86 | 4.89 | 38.0% |
| 0.50 | 2.51 | 2.61 | 57.8% |
| 0.70 | 1.89 | 1.14 | 78.4% |

This reproduces the qualitative trade the paper describes, and the mechanism
is visible: raising the threshold discards tokens that were going to be
rejected, so acceptance RATE climbs steeply (26.4% -> 78.4%) while tokens per
round falls. The paper's 45.7% -> 95.7% chat sweep is the same shape at a
higher absolute level, consistent with our lower baseline acceptance.

The interesting point is threshold 0.30: it cuts verified tokens by 30%
(7.00 -> 4.89) at no cost to tau (2.85 -> 2.86). That is a free 30% reduction
in verify width, which is the whole thesis of confidence scheduling in
miniature. At 0.50 and above the threshold starts eating real tokens.

Caveat: tau alone does not decide the optimum. Narrower verify forwards are
cheaper, so the throughput-optimal threshold depends on the target's cost
curve, which is what the paper's batch-global scheduler solves against and
what we cannot evaluate without multi-request batching (Stage D).

## Calibration of the raw confidence head

ECE 0.25-0.375 with AUROC 0.81-0.86. The head **ranks** well (AUROC in the
paper's reported 0.81-0.90 band) but is badly **calibrated** in absolute
terms: the paper reports ECE 3-8% uncalibrated, and we measure 25-37%.

The most likely explanation is that the released checkpoints ship no
Sequential Temperature Scaling parameters (verified: the repos contain only
`config.json` and `model.safetensors`), and the paper's ECE figures come from
its own held-out calibration setup. A head trained against acceptance rates
under temperature-1.0 rejection sampling will also be systematically
overconfident when scored against greedy argmax equality, which is a stricter
criterion, and that direction matches what we see.

Practical consequence: absolute confidence values from these checkpoints
should not be treated as probabilities. Threshold choice should be tuned
empirically per deployment (as the sweep above does) rather than read off as
"keep tokens above 70% survival". Ranking quality is good enough for the
prefix-truncation decision, which only needs an ordering.

## Wall-clock, reported but not load-bearing

| dataset | spec (s) | baseline (s) | speedup | target forwards / 2560 tokens |
|---|---:|---:|---:|---:|
| gsm8k | 75 | 202 | 2.68x | 723 |
| humaneval | 74 | 205 | 2.78x | 702 |
| mt-bench | 95 | 201 | 2.12x | 925 |

These 2.1-2.8x numbers are flattering and should not be quoted as DSpark's
speedup. The baseline here decodes one token per forward through the
materialized-SDPA torch backend with no CUDA graphs, which is a slow baseline;
the win is mostly "3.5x fewer target forwards" times "a per-forward cost that
barely grows with width on an underutilized L4". A production baseline with a
fused attention kernel would narrow this considerably. The forward-count
column is the durable number: 2560 tokens in ~700-925 target forwards.

## Correctness: bf16 divergence from target-alone, explained

Greedy spec-decode must reproduce target-alone greedy exactly. On GPU it did
not, on 6-14 of 20 prompts per dataset. That was worth chasing, and it is not
a bug in the loop.

The evidence:

1. **Divergence does not scale with speculation.** Across the threshold sweep,
   offered tokens per round fall 6x (7.00 -> 1.14) while the divergence count
   stays flat (12, 14, 14, 12 of 20). A broken accept/reject or cache rollback
   would diverge more when it speculates more.
2. **fp32 is exact on a real model.** `tests/unit/test_dspark_speculative_real_model.py`
   runs Qwen3-0.6B on CPU across the all-rejected, partial-accept (2 of 7),
   and full-accept (7 of 7) paths, at prompt lengths chosen to land the cache
   truncation at different offsets inside a block. Every case matches
   target-alone token-for-token in fp32.
3. **bf16 reproduces the divergence on CPU**, prompt-dependently, first
   differing partway into the generation rather than at the start.

Mechanism: spec-decode verifies at `q_len > 1` while target-alone decodes at
`q_len == 1`. Different matmul shapes take different reduction orders, and
bf16's 8-bit mantissa is narrow enough that two near-tied logits can swap
across the argmax boundary. One flipped token changes the context for
everything after it, so a single tie-break produces a fully different suffix,
which is why a binary "did the sequences match" flag overstates the problem so
dramatically.

This is the same effect ADR-011 recorded for the two-model V1 (visible on A10,
not on H100), and the same stance applies: fp32 parity is the strict oracle,
bf16 parity is a sanity check, matching how vLLM and SGLang treat it.

## Reproduce

```bash
uv run modal run scripts/modal_dspark_bench.py
```

Raw data: `docs/benchmarks/data/dspark-stage-c.json`.

## Caveats

- 20 prompts per dataset, 128 new tokens, greedy, one seed. Enough to
  establish magnitude and ordering, not enough for tight confidence intervals
  on the tau gap.
- Prompts are raw dataset text with no chat template applied. Fixing this is
  the first thing to try before drawing conclusions about the tau gap.
- Single request throughout. The paper's headline is a batch-global scheduler
  under concurrency, which this cannot evaluate at all.
- The confidence sweep was run on chat only.

## Pointers

- ADR: [ADR-027](../decisions/ADR-027-dspark-drafter-port.md)
- Plan: [dspark-evaluation.md](../plans/dspark-evaluation.md) Stage C
- Implementation: `src/mini_infer/engine/dspark/speculative.py`,
  `src/mini_infer/engine/dspark/proposal.py`
- Prior spec-decode result: [ADR-011](../decisions/ADR-011-speculative-decoding.md),
  [2026-04-29 benchmark](2026-04-29-speculative-decoding.md)
