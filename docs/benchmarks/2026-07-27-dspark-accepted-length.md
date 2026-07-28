# DSpark drafter: accepted length, calibration, and the confidence threshold

Date: 2026-07-27
Hardware: NVIDIA L4 (24 GB), bf16, materialized-SDPA attention backend
Target: `Qwen/Qwen3-4B`
Drafter: `deepseek-ai/dspark_qwen3_4b_block7` (block size 7, 5 layers, 1.39B)
Workload: 50 prompts per dataset from DeepSpec's bundled `eval_datasets/`,
128 new tokens each, greedy, chat template with `enable_thinking=False`
(threshold sweep: mt-bench, same settings)
Engine: mini-infer @ ADR-027 (Stage C)
Script: `scripts/modal_dspark_bench.py`

As far as I can tell this is the first third-party measurement of any DSpark
claim; the paper shipped with no independent replication.

## Headline: the paper reproduces

Accepted length (tau) is mean committed tokens per verification round,
counting the bonus, which is DeepSpec's own definition
(`base_evaluator.py`: `acceptance_lengths.append(accepted_draft_tokens + 1)`).
It is the right headline because it is hardware-independent: ADR-011 found
wall-clock speedup for the two-model V1 collapsing from 1.29x to 0.83x across
prompts on one machine, so a wall-clock number mostly measures the GPU.

| dataset | domain | tau (ours) | paper Table 1 | vs paper | accept | ECE | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| gsm8k | math | 6.24 | ~5.57 | +12% | 74.8% | 0.185 | 0.826 |
| humaneval | code | 5.10 | ~5.12 | -0% | 58.6% | 0.287 | 0.852 |
| mt-bench | chat | 3.90 | ~3.49 | +12% | 41.5% | 0.277 | 0.871 |

Code lands within a rounding error of the published figure; math and chat come
in 12% above it. The domain ordering (math > code > chat) matches. Given a
different prompt subset, greedy rather than temperature 1.0, and 128-token
generations, agreement this close is about as good as this setup can show.

**This is not what the first version of this page reported**, and the
difference was entirely a defect in the harness rather than in the port. That
story is worth keeping, because the wrong numbers were plausible and
self-consistent for three rounds of analysis.

## How three hypotheses were refuted before finding the real cause

The original measurement showed tau at 3.79 / 3.14 / 3.07, roughly a third
below the paper. Three explanations were proposed and tested:

1. **Chat templating** (predicted to help, since the drafter trained on
   templated responses): measured -3% / -12% / -6%. It made things *worse*.
2. **Truncation at 128 tokens** (predicted to bias long-answer domains):
   tripling the budget to 384 bought 3%.
3. **Greedy vs temperature 1.0** (predicted the paper's softer acceptance
   criterion explained it): measured -4% / -1% / -3%. Wrong direction again.

The argument behind the third was also wrong, not merely unconfirmed. It
claimed `1 - TV` (temperature-1.0 acceptance) exceeds `P(argmax match)`
(greedy acceptance) whenever the target is uncertain. That is not generally
true and is false for a well-matched drafter: greedy pays full credit,
acceptance 1, every time the drafter's argmax is right, which here is ~94% of
the time at position 1 on math, while temperature 1.0 discounts those same
cases to `1 - TV < 1`. A strong drafter does better under greedy.

**The real cause: the target was in the wrong reasoning mode.** Qwen3's chat
template defaults to `enable_thinking=True`, so the target opened every answer
with a `<think>` reasoning block. The DSpark drafter is trained on responses
regenerated with `--disable-thinking`, and DeepSpec's own evaluator hardcodes
`enable_thinking=False` (`base_evaluator.py`). Every templated measurement was
scoring the drafter against a distribution it had never been trained on.

One fault, both surprises explained:

- **Why templating appeared to hurt.** Raw dataset text never invokes the chat
  template, so it never triggered thinking mode. The "wrong" raw arm was
  accidentally closer to the drafter's training distribution than the
  "correct" templated one.
- **Why math and code suffered most.** Those are the domains where Qwen3 emits
  the longest reasoning traces.

Effect of the one-flag fix, greedy, same 50 prompts:

| dataset | thinking on | thinking off | change |
|---|---:|---:|---:|
| gsm8k | 3.79 | 6.24 | +65% |
| humaneval | 3.14 | 5.10 | +62% |
| mt-bench | 3.07 | 3.90 | +27% |

The lesson worth carrying: for a measurement comparing two models, prompt
construction is part of the experiment. A template flag that looks cosmetic
silently changed what was being measured, and produced plausible, internally
consistent, entirely wrong numbers three times running.

## Per-position survival

Fraction of rounds whose accepted prefix reached past each position
(cumulative survival, not per-position conditional acceptance):

| dataset | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.94 | 0.86 | 0.80 | 0.74 | 0.68 | 0.63 | 0.58 |
| humaneval | 0.88 | 0.76 | 0.66 | 0.56 | 0.47 | 0.41 | 0.36 |
| mt-bench | 0.79 | 0.61 | 0.45 | 0.35 | 0.27 | 0.23 | 0.19 |

Position 1 survives at 0.94 on math, above the 0.88 the paper quotes for
DFlash and well above the 0.81 it quotes for Eagle3. That is the paper's
central claim for the parallel backbone: it buys capacity at the position
where a rejection costs the entire block.

Per-step conditional acceptance (successive ratios):

- gsm8k: 0.94, 0.91, 0.93, 0.92, 0.92, 0.93, 0.92
- humaneval: 0.88, 0.86, 0.87, 0.85, 0.84, 0.87, 0.88
- mt-bench: 0.79, 0.77, 0.74, 0.78, 0.77, 0.85, 0.83

Flat to position 7 in every domain, with no tail decay whatsoever. This is the
paper's claim for the sequential Markov head, and it reproduces cleanly:
DFlash is reported degrading 0.72 to 0.63 on chat across the block, while this
holds ~0.78 throughout. Chat sits lower than math uniformly, which is a
difference in workload difficulty rather than in the drafter's ability to hold
a block together.

At these survival rates a block of 7 is no longer over-provisioned the way the
earlier (mismeasured) numbers suggested: on math, 58% of rounds still have a
live prefix at position 7.

## Confidence calibration

| dataset | ECE (thinking on) | ECE (thinking off) | AUROC | paper ECE |
|---|---:|---:|---:|---:|
| gsm8k | 0.407 | 0.185 | 0.826 | 3-8% |
| humaneval | 0.425 | 0.287 | 0.852 | 3-8% |
| mt-bench | 0.324 | 0.277 | 0.871 | 3-8% |

Calibration improves sharply with the fix, which is a consequence rather than
a coincidence: the confidence head is trained against
`c* = 1 - 0.5 * ||p_draft - p_target||_1`, so it can only read as calibrated
when the target is in the mode that supervision was collected in.

Ranking is good throughout (AUROC 0.83-0.87, at or above the paper's reported
0.81-0.90 band). Absolute calibration remains worse than the paper's 3-8%,
and two known differences would explain the remainder: the released
checkpoints ship no Sequential Temperature Scaling parameters (their repos
contain only `config.json` and `model.safetensors`), and this scores the head
against greedy argmax-equality outcomes while it is trained against the
temperature-1.0 acceptance rate. Practical consequence unchanged: treat these
values as a ranking signal for prefix truncation, not as probabilities.

## Confidence threshold sweep (mt-bench, greedy, non-thinking)

| threshold | tau | tau cost | offered/round | width cut | accept rate |
|---:|---:|---:|---:|---:|---:|
| off (0.00) | 3.90 | - | 7.00 | - | 41.5% |
| 0.30 | 3.85 | 1.3% | 6.02 | 14.0% | 47.4% |
| 0.50 | 3.67 | 5.9% | 4.37 | 37.6% | 60.9% |
| 0.70 | 3.11 | 20.3% | 2.83 | 59.6% | 74.5% |

Raising the threshold discards tokens that were going to be rejected, so
acceptance rate climbs (41.5% to 74.5%) while tokens per round falls.

**0.50 is the interesting operating point here**: it removes 38% of the
verify width for 6% of the accepted length. Verify cost scales with width
while tau is what the width buys, so trading 6 for 38 is strongly positive
whenever the target's per-forward cost grows with the number of tokens
verified. 0.30 is nearly free (1.3% for 14%) but leaves most of the saving on
the table; 0.70 crosses over into costing real tokens.

These numbers replace an earlier sweep taken before the thinking-mode fix,
which is worth recording because the correction did not simply shift the
curve. Then and now, per unit of tau given up:

| threshold | width cut per 1% tau (thinking, wrong) | (non-thinking, correct) |
|---:|---:|---:|
| 0.30 | 27.8x | 10.9x |
| 0.50 | 6.2x | 6.4x |
| 0.70 | 3.3x | 2.9x |

The low-threshold end changed by more than a factor of two: at 0.30 the old
measurement made truncation look almost free because so many tokens were
being rejected anyway that discarding the least confident ones cost nothing.
With acceptance at 41.5% instead of 29.6%, low-confidence tokens are more
often worth keeping, so cheap truncation buys less. Picking an operating
point from the old numbers would have set the threshold too low.

Caveat: tau alone does not identify the optimum. The right threshold depends
on the target's cost curve, which is what the paper's batch-global scheduler
solves against and what cannot be evaluated without multi-request batching
(Stage D).

## Wall-clock, reported but not load-bearing

Per prompt, from the pre-fix run (speculative covers all 50 prompts, the
baseline only the first 15, so a raw seconds ratio would compare different
workloads):

| dataset | spec s/prompt | baseline s/prompt | ratio |
|---|---:|---:|---:|
| gsm8k | 3.71 | 10.47 | 2.82x |
| humaneval | 4.42 | 10.22 | 2.31x |
| mt-bench | 4.37 | 10.16 | 2.33x |

These are flattering and should not be quoted as DSpark's speedup. The
baseline decodes one token per forward through the materialized-SDPA torch
backend with no CUDA graphs; the win is mostly "about 3x fewer target
forwards" times "a per-forward cost that barely grows with width on an
underutilized L4". A production baseline with a fused attention kernel would
narrow this considerably. With acceptance now much higher, the forward-count
saving is larger than these numbers reflect.

## Correctness: bf16 divergence from target-alone, explained

Greedy spec-decode must reproduce target-alone greedy exactly. On GPU it does
not, on 6-11 of the 15 spot-checked prompts per configuration. That is worth
chasing, and it is not a bug in the loop.

1. **Divergence does not scale with speculation.** In the threshold sweep,
   offered tokens per round fell 6x while the divergence count stayed flat.
2. **fp32 is exact on a real model.**
   `tests/unit/test_dspark_speculative_real_model.py` runs Qwen3-0.6B on CPU
   across the all-rejected, partial-accept (2 of 7) and full-accept (7 of 7)
   paths, at prompt lengths chosen to land the cache truncation at different
   offsets inside a block. Every case matches target-alone token-for-token.
3. **bf16 reproduces the divergence on CPU**, prompt-dependently.
4. **Divergences start late**, 38-65% of the way through the generation on
   average, never at the first token.

Mechanism: spec-decode verifies at `q_len > 1` while target-alone decodes at
`q_len == 1`. Different matmul shapes take different reduction orders, and
bf16's 8-bit mantissa is narrow enough that two near-tied logits can swap
across the argmax boundary. One flipped token changes the context for
everything after it, so a single tie-break produces a wholly different suffix.

Same effect ADR-011 recorded for the two-model V1 (visible on A10, not H100),
and the same stance applies: fp32 parity is the strict oracle, bf16 parity is
a sanity check, as vLLM and SGLang also treat it.

## Reproduce

```bash
DSPARK_SAMPLES=50 DSPARK_TEMPS=0.0 DSPARK_ARMS=templated uv run modal run --detach scripts/modal_dspark_bench.py
```

Raw data: `docs/benchmarks/data/dspark-stage-c.json`.

## Caveats

- 50 prompts per dataset, 128 new tokens, greedy, one seed. Enough for
  magnitude and ordering, not for tight intervals.
- Prompts are the first 50 of each set; DeepSpec shuffles with a seed and
  evaluates more. Some of the +12% on math and chat is likely subset luck.
- Greedy, while the paper's Table 1 is temperature 1.0. Measured at
  temperature 1.0 in thinking mode it cost 1-4%; it has not been re-measured
  in non-thinking mode.
- Single request throughout. The paper's headline is a batch-global scheduler
  under concurrency, which this cannot evaluate at all (Stage D).
- The wall-clock table predates the thinking-mode fix, so its forward-count
  saving understates what current acceptance delivers.
- The threshold sweep covers chat only.
- Baseline correctness is spot-checked on 15 of 50 prompts.

## Pointers

- ADR: [ADR-027](../decisions/ADR-027-dspark-drafter-port.md)
- Plan: [dspark-evaluation.md](../plans/dspark-evaluation.md) Stage C
- Implementation: `src/mini_infer/engine/dspark/speculative.py`,
  `src/mini_infer/engine/dspark/proposal.py`,
  `src/mini_infer/engine/dspark/sampling.py`
- Prior spec-decode result: [ADR-011](../decisions/ADR-011-speculative-decoding.md),
  [2026-04-29 benchmark](2026-04-29-speculative-decoding.md)
