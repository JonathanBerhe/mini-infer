# DSpark drafter: accepted length, calibration, and the confidence threshold

Date: 2026-07-27
Hardware: NVIDIA L4 (24 GB), bf16, materialized-SDPA attention backend
Target: `Qwen/Qwen3-4B`
Drafter: `deepseek-ai/dspark_qwen3_4b_block7` (block size 7, 5 layers, 1.39B)
Workload: 50 prompts per dataset from DeepSpec's bundled `eval_datasets/`,
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

Templated prompts are the honest configuration for an instruct model, so they
are the headline; raw is reported alongside because the comparison turned out
to be informative.

| dataset | domain | tau (templated) | tau (raw) | paper Table 1 | accept | ECE | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| gsm8k | math | 3.79 | 3.91 | ~5.57 | 39.9% | 0.407 | 0.804 |
| humaneval | code | 3.14 | 3.59 | ~5.12 | 30.6% | 0.425 | 0.817 |
| mt-bench | chat | 3.07 | 3.27 | ~3.49 | 29.6% | 0.324 | 0.843 |

The drafter delivers roughly 3.1-3.8 tokens per target forward. Chat lands
close to the paper (3.07 vs ~3.49); math and code sit well below (3.79 vs
~5.57, 3.14 vs ~5.12).

## Two hypotheses tested, both refuted

The 20-prompt version of this benchmark listed three candidate explanations
for the shortfall. This run tested the two leading ones. Neither survived.

**Chat templating: predicted to help, actually hurts.** The reasoning was that
the drafter trained on target-regenerated responses in chat format, so raw
dataset text should be off-distribution for it. Measured effect on tau, same
prompts, only the formatting changed:

| dataset | raw | templated | delta |
|---|---:|---:|---:|
| gsm8k | 3.91 | 3.79 | -3% |
| humaneval | 3.59 | 3.14 | -12% |
| mt-bench | 3.27 | 3.07 | -6% |

Templating lowers accepted length in all three domains. The prediction was
backwards. A plausible reading, not tested here, is that raw prompts invite
the model to continue or restate the prompt text, which is unusually easy to
draft, so the raw arm was mildly inflated rather than the templated arm
depressed. Confirming that needs the generated text, which this run does not
store.

**Truncation at 128 tokens: real but small.** The reasoning was that math and
code answers get formulaic exactly in their later tokens, where acceptance is
highest, so a short cap should bias those domains down hardest. Rerunning
gsm8k templated at 384 tokens:

| generation budget | tau |
|---|---:|
| 128 tokens | 3.79 |
| 384 tokens | 3.90 |

A 3x longer budget buys 3%. Directionally as predicted, nowhere near enough to
explain a 32% shortfall.

## What most likely remains: greedy vs temperature 1.0

The paper's Table 1 is measured at temperature 1.0 with rejection sampling.
This benchmark is greedy. Those are not the same measurement at a different
setting, they are different acceptance criteria, and the difference runs in
the direction of the gap.

- Greedy accepts a draft token only if it is exactly the target's argmax, so
  expected acceptance is `P(argmax_draft == argmax_target)`.
- Temperature-1.0 rejection sampling accepts with probability
  `min(1, p_target(x) / p_draft(x))`, giving expected acceptance
  `sum_x min(p_draft(x), p_target(x))`, which is `1 - TV(p_draft, p_target)`.

When the target is uncertain, `1 - TV` is substantially larger than the
probability that two distributions share an argmax: two near-identical
distributions split over a few plausible tokens have `1 - TV` close to 1 while
agreeing on the argmax barely more than chance. Greedy is the harsher bar, and
it is the bar we measured against.

The calibration numbers independently support this. DSpark's confidence head
is trained against `c* = 1 - 0.5 * ||p_draft - p_target||_1`, which is exactly
the temperature-1.0 acceptance rate (ADR-027). Scoring that head against
greedy argmax-equality outcomes should make it look systematically
overconfident, and that is the shape of what we see: ranking stays good
(AUROC 0.80-0.85, inside the paper's reported 0.81-0.90 band) while absolute
calibration is far worse than the paper's 3-8% ECE.

Arithmetic consistency check: at ECE 0.407 with an observed acceptance rate of
0.399, the mean predicted confidence has to sit roughly 0.4 away from the
observed rate, i.e. near 0.8. Combined with good AUROC that implies a head
predicting about twice the acceptance greedy delivers. This is an inference
from the aggregate, not a direct measurement; the script now records mean
predicted confidence so the next run settles the direction outright.

Testing this properly requires temperature > 0 rejection sampling, which is
deferred to Stage D. Until then: **the gap is most plausibly a measurement
difference rather than a port defect**, and the port itself is bit-parity
validated against DeepSpec's own implementation on these exact weights
(ADR-027).

## Per-position survival

Fraction of rounds whose accepted prefix reached past each position
(cumulative survival, not per-position conditional acceptance), templated arm:

| dataset | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.80 | 0.59 | 0.46 | 0.34 | 0.25 | 0.20 | 0.15 |
| humaneval | 0.76 | 0.50 | 0.34 | 0.23 | 0.15 | 0.10 | 0.06 |
| mt-bench | 0.76 | 0.49 | 0.32 | 0.21 | 0.14 | 0.09 | 0.06 |

Position 1 survives at 0.76-0.80, below the 0.88 the paper quotes for DFlash
on math, consistent with our lower overall tau and with the greedy criterion
above.

Dividing successive entries gives per-step conditional acceptance, which is
what the Markov head is meant to keep flat rather than let decay:

- gsm8k: 0.80, 0.73, 0.78, 0.74, 0.74, 0.78, 0.77
- humaneval: 0.76, 0.65, 0.68, 0.68, 0.63, 0.67, 0.61
- mt-bench: 0.76, 0.65, 0.65, 0.67, 0.64, 0.69, 0.65

Flat, in every domain, out to position 7. This reproduces the paper's
qualitative claim for the sequential head: the tail does not collapse the way
it reports for DFlash (0.72 -> 0.63 on chat). Chat and code simply sit lower
throughout rather than degrading with depth.

Practically, cumulative survival still falls below a third past position 4, so
the last two or three slots of a block-7 proposal mostly widen the verify
forward without paying for themselves. That is the waste confidence
scheduling exists to remove.

## Confidence threshold sweep (mt-bench, templated)

| threshold | tau | draft tokens offered/round | accept rate |
|---:|---:|---:|---:|
| off (0.00) | 3.07 | 7.00 | 29.6% |
| 0.30 | 3.05 | 5.73 | 35.7% |
| 0.50 | 2.85 | 3.88 | 47.8% |
| 0.70 | 2.45 | 2.30 | 62.9% |

The qualitative trade the paper describes reproduces, and the mechanism is
visible: raising the threshold discards tokens that were going to be rejected,
so acceptance RATE climbs (29.6% -> 62.9%) while tokens per round falls.

The useful operating point is 0.30: it cuts verified tokens per round by 18%
(7.00 -> 5.73) for a 0.7% reduction in tau (3.07 -> 3.05). That is a nearly
free reduction in verify width, which is confidence scheduling's whole thesis
in miniature. Past 0.50 the threshold starts eating real tokens.

Caveat: tau alone does not identify the optimum. Narrower verify forwards are
cheaper, so the throughput-optimal threshold depends on the target's cost
curve, which is what the paper's batch-global scheduler solves against and
what cannot be evaluated without multi-request batching (Stage D).

## Wall-clock, reported but not load-bearing

Per prompt, because the two arms cover different prompt counts (speculative
runs all 50, the baseline only the first 15, since it exists solely for the
correctness check):

| dataset | arm | spec s/prompt | baseline s/prompt | ratio |
|---|---|---:|---:|---:|
| gsm8k | raw | 3.63 | 10.56 | 2.91x |
| humaneval | raw | 3.88 | 10.23 | 2.64x |
| mt-bench | raw | 4.17 | 10.12 | 2.43x |
| gsm8k | templated | 3.71 | 10.47 | 2.82x |
| humaneval | templated | 4.42 | 10.22 | 2.31x |
| mt-bench | templated | 4.37 | 10.16 | 2.33x |

These 2.3-2.9x numbers are flattering and should not be quoted as DSpark's
speedup. The baseline decodes one token per forward through the
materialized-SDPA torch backend with no CUDA graphs, which is a slow baseline;
the win is mostly "about 3x fewer target forwards" times "a per-forward cost
that barely grows with width on an underutilized L4". A production baseline
with a fused attention kernel would narrow this considerably. The durable
number is the forward count: 6400 tokens in 1710-2134 target forwards.

The two arms also cover different prompt subsets, so the ratio is an
approximation, not a controlled A/B.

## Correctness: bf16 divergence from target-alone, explained

Greedy spec-decode must reproduce target-alone greedy exactly. On GPU it does
not, on 3-12 of the 15 spot-checked prompts per configuration. That is worth
chasing, and it is not a bug in the loop.

1. **Divergence does not scale with speculation.** In the 20-prompt sweep,
   offered tokens per round fell 6x while the divergence count stayed flat. A
   broken accept/reject or cache rollback would diverge more when it
   speculates more.
2. **fp32 is exact on a real model.**
   `tests/unit/test_dspark_speculative_real_model.py` runs Qwen3-0.6B on CPU
   across the all-rejected, partial-accept (2 of 7), and full-accept (7 of 7)
   paths, at prompt lengths chosen to land the cache truncation at different
   offsets inside a block. Every case matches target-alone token-for-token in
   fp32.
3. **bf16 reproduces the divergence on CPU**, prompt-dependently.
4. **Divergences start late.** Mean first-divergence position is 38-65% of the
   way through the generation, and the earliest across all configurations is
   token 1, not token 0. A logic error would diverge at the first verify.

Mechanism: spec-decode verifies at `q_len > 1` while target-alone decodes at
`q_len == 1`. Different matmul shapes take different reduction orders, and
bf16's 8-bit mantissa is narrow enough that two near-tied logits can swap
across the argmax boundary. One flipped token changes the context for
everything after it, so a single tie-break produces a wholly different suffix,
which is why a binary "did the sequences match" flag overstates this so
badly.

This is the same effect ADR-011 recorded for the two-model V1 (visible on A10,
not on H100), and the same stance applies: fp32 parity is the strict oracle,
bf16 parity is a sanity check, matching how vLLM and SGLang treat it.

## Reproduce

```bash
DSPARK_SAMPLES=50 uv run modal run scripts/modal_dspark_bench.py
```

Raw data: `docs/benchmarks/data/dspark-stage-c.json`.

## Caveats

- 50 prompts per dataset, 128 new tokens, greedy, one seed. Enough for
  magnitude and ordering, not for tight intervals on the tau gap.
- Greedy throughout. The paper's Table 1 is temperature 1.0, which is very
  likely most of the remaining gap (see above) and cannot be checked until
  Stage D.
- Single request throughout. The paper's headline is a batch-global scheduler
  under concurrency, which this cannot evaluate at all.
- The threshold sweep was run on chat only.
- Baseline correctness is spot-checked on 15 of 50 prompts per configuration.

## Pointers

- ADR: [ADR-027](../decisions/ADR-027-dspark-drafter-port.md)
- Plan: [dspark-evaluation.md](../plans/dspark-evaluation.md) Stage C
- Implementation: `src/mini_infer/engine/dspark/speculative.py`,
  `src/mini_infer/engine/dspark/proposal.py`
- Prior spec-decode result: [ADR-011](../decisions/ADR-011-speculative-decoding.md),
  [2026-04-29 benchmark](2026-04-29-speculative-decoding.md)
