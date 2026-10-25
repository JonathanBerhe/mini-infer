# DSpark under concurrency: throughput, acceptance, and step time

Date: 2026-10-19
Hardware: NVIDIA L4 (24 GB), bf16, `torch` attention backend
Target: `Qwen/Qwen3-4B`
Drafter: `deepseek-ai/dspark_qwen3_4b_block7` (block size 7)
Workload: the first 32 prompts of gsm8k and of mt-bench from DeepSpec's
`eval_datasets/` (commit `005e03b`), 128 new tokens each, greedy, chat
template with `enable_thinking=False`
Engine: mini-infer at `0cb4cb1`, `ContinuousScheduler` with DSpark
([ADR-031](../decisions/ADR-031-multi-request-dspark.md))
Script: `scripts/dspark/modal_dspark_concurrency_bench.py`

## Result

- DSpark speeds up decoding 4.5x on gsm8k and 2.5x on mt-bench with one
  request in flight. With 16 requests in flight, the gain is 3.1 to 3.2x on
  gsm8k and 1.7 to 1.8x on mt-bench. The two values are the steady part and
  the whole run, defined below. Both include prompt prefill. On decode steps
  alone, the gain at 16 requests is about 3.5x and 1.9x.
- Acceptance does not change with concurrency. Tau (committed tokens per
  round, the bonus token included) stays between 5.92 and 6.04 on gsm8k and
  between 3.20 and 3.31 on mt-bench at every level.
- Two costs grow with the batch and explain the smaller gain. The drafter
  runs once per request, so its share of step time grows from 13 to 24-25
  percent. A verify forward of 8 tokens per request costs 1.10x a one-token
  decode forward at one request, and 1.25x at 16.
- A confidence threshold of 0.5 buys no throughput on this backend. On
  mt-bench it cuts the verify width by 46 percent, but throughput does not
  rise, because verify width is almost free. The 1 to 4 percent drop is
  inside the run-to-run variation (see the timed pass).

## How a point is measured

A point submits all prompts of a dataset at once to a scheduler whose
`max_concurrent` is the level under test. While a prompt still waits, the
scheduler refills each freed slot (a closed loop). Every level uses the same
prompts, so a change between levels is a change of concurrency, not of
workload.

After the last admission the batch drains, and the two arms drain
differently: plain requests of one length finish together, speculative ones
finish at different rounds. So each point reports two speeds. The whole run
counts every token. The steady part counts only the steps in which a prompt
still waited. Each point also reports its mean in-flight count, weighted by
time.

Admission runs before the scheduler reaps finished requests, so a freed
slot stays empty for one forward. Plain requests of one length finish
together, so this barely shows for them. DSpark requests finish at different
rounds: at 16 requests, DSpark ran 30 percent (gsm8k) and 18 percent
(mt-bench) of its steady time one or more requests short. Its mean batch in
the steady part was 15.5 and 15.7.

The sweep refuses a point that does not measure what its label says: a
DSpark point with a request that did not speculate, or a point with a
request that admission cancelled. A separate timed pass puts a CUDA
synchronize at every phase boundary and records the width and duration of
every forward. The barriers change the step time: with the same tokens, the
timed points ran from 4 percent slower to 8 percent faster than the untimed
ones. So throughput comes from the untimed points only, and differences
under about 8 percent are inside the run-to-run variation.

## Throughput

Tokens per second over the whole run, with the steady speedup over plain
decoding at the same level:

| dataset | level | plain | DSpark | speedup | DSpark t=0.5 | speedup | in flight (DSpark) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1 | 11.3 | 50.5 | 4.48x | 50.7 | 4.50x | 1.0 |
| gsm8k | 2 | 16.5 | 64.7 | 3.95x | 67.1 | 4.07x | 1.9 |
| gsm8k | 4 | 22.4 | 79.2 | 3.57x | 79.1 | 3.54x | 3.8 |
| gsm8k | 8 | 26.9 | 90.9 | 3.42x | 90.5 | 3.37x | 7.4 |
| gsm8k | 16 | 30.7 | 99.1 | 3.05x | 97.5 | 2.98x | 14.9 |
| mt-bench | 1 | 11.9 | 29.4 | 2.45x | 28.6 | 2.38x | 1.0 |
| mt-bench | 2 | 17.0 | 38.8 | 2.27x | 37.5 | 2.20x | 2.0 |
| mt-bench | 4 | 22.6 | 47.9 | 2.11x | 47.3 | 2.08x | 3.9 |
| mt-bench | 8 | 27.5 | 54.6 | 1.90x | 53.2 | 1.85x | 7.4 |
| mt-bench | 16 | 31.6 | 57.8 | 1.71x | 55.4 | 1.69x | 14.7 |

From one request to 16, plain throughput grows 2.7x and DSpark throughput
grows 2.0x on both datasets. Every gsm8k request ran to 128 tokens. On
mt-bench, 6 to 9 of the 32 requests of each point stopped at end of sequence
first. So an mt-bench point generated 3,713 to 3,817 tokens, not 4,096.
Every request of a DSpark point speculated.

At 16 requests the whole-run speedup is higher than the steady one: 3.22x
against 3.05x on gsm8k, and 1.83x against 1.71x on mt-bench. Prefill
explains the gap. At 16 requests, DSpark makes decode about 3.5x faster on
gsm8k and about 1.9x faster on mt-bench. So the same prompt prefills take a
larger share of its time. Its requests also finish at different rounds, so
its admissions, and their prefills, spread over the whole steady part. In
the steady part, DSpark spends 36 percent (gsm8k) and 21 percent (mt-bench)
of its time in steps that also run a prefill. In the drain it spends 5 to 8
percent, and plain decoding spends 3 to 4 percent in both parts. So both
speedups include prefill and understate the decode gain on gsm8k.

## Acceptance under concurrency

| dataset | level | tau | tau, unclamped | tau, t=0.5 | offered per round, t=0.5 |
|---|---:|---:|---:|---:|---:|
| gsm8k | 1 | 5.95 | 6.12 | 5.97 | 6.65 |
| gsm8k | 16 | 5.92 | 6.05 | 5.94 | 6.64 |
| mt-bench | 1 | 3.26 | 3.31 | 3.02 | 3.67 |
| mt-bench | 16 | 3.20 | 3.24 | 2.99 | 3.63 |

Over all five levels, tau spans 5.92 to 6.04 on gsm8k and 3.20 to 3.31 on
mt-bench. Unclamped tau spans 6.05 to 6.18 and 3.24 to 3.37. Near `max_tokens`
the scheduler offers fewer tokens than the drafter proposed, so those rounds
commit less. About 5 percent of the rounds are shortened this way. "Tau,
unclamped" leaves them out, which makes it the number to compare with the
batch-1 runner, which never shortens a round.

Survival by draft position, threshold 0. Of the rounds that offered a
position, the table gives the fraction whose accepted prefix reached it:

| dataset | level | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1 | 0.93 | 0.85 | 0.78 | 0.72 | 0.67 | 0.61 | 0.55 |
| gsm8k | 16 | 0.93 | 0.85 | 0.77 | 0.71 | 0.66 | 0.59 | 0.54 |
| gsm8k, batch-1 runner | 1 | 0.94 | 0.86 | 0.80 | 0.74 | 0.68 | 0.63 | 0.58 |
| mt-bench | 1 | 0.74 | 0.53 | 0.36 | 0.25 | 0.18 | 0.14 | 0.11 |
| mt-bench | 16 | 0.73 | 0.51 | 0.36 | 0.24 | 0.17 | 0.13 | 0.10 |

The batch-1 row comes from the first 50 prompts in
[the batch-1 benchmark](2026-07-27-dspark-accepted-length.md). On gsm8k with
one request in flight, the scheduler matches it within 0.03 at every
position. Its unclamped tau (6.12) is 2 percent under the batch-1 value
(6.24). With 16 requests the gap grows to 0.05 at the last positions. The gsm8k prompts are
all of one kind, so this comparison shows that the multi-request scheduler
drafts and verifies like the batch-1 runner.

On mt-bench, tau is lower than the batch-1 value of 3.90 because the prompt
set differs, not because of the scheduler. mt-bench lists its prompts by
category. The first 32 are writing (10), role-play (10), reasoning (10), and
math (2). With one request in flight, the tau of each category is:
role-play about 2.7, writing 3.0, reasoning 4.0, and the two math prompts
6.6. The first 50, which the batch-1 benchmark used, also include 10
math and 10 coding prompts.

## Where the step time goes

Share of step time in the timed pass, threshold 0:

| dataset | level | arm | forward | draft | verify | sample |
|---|---:|---|---:|---:|---:|---:|
| gsm8k | 1 | plain | 99.3% | - | - | 0.6% |
| gsm8k | 1 | DSpark | 86.6% | 12.6% | 0.8% | 0.0% |
| gsm8k | 4 | DSpark | 79.5% | 19.7% | 0.8% | 0.0% |
| gsm8k | 16 | DSpark | 75.4% | 23.8% | 0.7% | 0.0% |
| mt-bench | 1 | DSpark | 86.1% | 13.0% | 0.8% | 0.0% |
| mt-bench | 4 | DSpark | 78.3% | 20.8% | 0.9% | 0.0% |
| mt-bench | 16 | DSpark | 74.2% | 25.1% | 0.7% | 0.0% |

"Draft" is the drafter, which runs once per request in a loop. Its share
almost doubles from one request to 16. "Verify" is the acceptance logic
after the forward, and it stays under 1 percent.

## Forward cost against batch size

Median forward time in the timed pass. A plain step decodes one token per
request. A verify step carries 1 + 7 tokens per request:

| batch | plain decode | verify | tokens in verify | verify / decode |
|---:|---:|---:|---:|---:|
| 1 | 82.3 ms | 90.6 ms | 7.8 | 1.10 |
| 4 | 165.1 ms | 194.9 ms | 31.3 | 1.18 |
| 16 | 486.3 ms | 607.1 ms | 127.2 | 1.25 |

These are the gsm8k numbers. mt-bench gives the same shape: 1.11, 1.19, and
1.25, plus the batch sizes between them in the data file. On this backend,
the cost of a forward follows the number of requests, not the number of
tokens. A forward with 16 requests costs 5.9x a forward with one, and 8x the
tokens per request cost only 10 to 25 percent more.

## What the results mean

- Speculation wins this much partly because the backend makes verify width
  almost free. The `torch` backend rebuilds the packed K/V of every request
  for every layer of every forward. So its decode is slow, and its cost
  grows with the batch. A fused decode kernel (FlashInfer) makes a one-token
  decode much cheaper, so the speedup over it will be smaller.
- The capacity curve that the batch-global confidence scheduler needs is
  flat in width and steep in batch size on this backend. Such a scheduler
  has little to gain from shorter verify windows here, which matches the
  result of the 0.5 threshold.
- The per-request drafter loop is the next cost to remove: 24 to 25 percent
  of step time at 16 requests. A batched drafter forward is the deferred item
  in ADR-031 that this measurement now justifies.
- A freed slot stays empty for one forward, because admission runs before
  the reap. At 16 requests, DSpark ran 18 to 30 percent of its steady time
  with fewer requests than allowed. A reap at the start of the step, before
  admission, can close that gap. It needs its own measurement.

## Output agreement

Greedy DSpark output must match greedy plain output. At fp32 it does, and
the unit tests pin that on a small model. In bf16 on the GPU, 9 to 20 of the
32 requests match exactly at each point. The mean common prefix is 55 to 79
percent of the output. The agreement does not trend with concurrency. The
verify forward runs at more than one query token per request. So its
matmuls reduce in a different order, and near-tied bf16 logits can flip. The
[batch-1 benchmark](2026-07-27-dspark-accepted-length.md) explains the same
effect in detail.

## Reproduce

```bash
uv run modal run --detach scripts/dspark/modal_dspark_concurrency_bench.py --run-tag main
```

The script writes the full results to
`docs/benchmarks/data/dspark-concurrency-main.json`. The summary in
[`data/dspark-concurrency.json`](data/dspark-concurrency.json) leaves out the
per-request tokens and the per-step records, and adds the capacity table.

## Caveats

- One backend: `torch`, the only one that the DSpark path ran on with a GPU.
  It favors speculation (see above).
- 32 prompts per dataset, 128 new tokens, greedy, one run. Enough for
  magnitude and trend, not for tight intervals.
- The mt-bench subset is mostly writing, role-play, and reasoning, so its
  tau is not comparable with a full mt-bench run.
- One GPU (L4). A GPU with more memory bandwidth changes the decode cost and
  so the speedup.
- The timed pass covers levels 1, 4, and 16 only.

## Pointers

- Design: [ADR-031](../decisions/ADR-031-multi-request-dspark.md)
- Drafter port: [ADR-027](../decisions/ADR-027-dspark-drafter-port.md)
- Batch-1 result: [2026-07-27](2026-07-27-dspark-accepted-length.md)
- Driver: `src/mini_infer/bench/dspark_sweep.py`
