# Open-loop rate sweep: A10 + Qwen-0.5B and H100 + Qwen-7B

Date: 2026-04-30
Hardware: NVIDIA A10G (Ampere, 24 GB) and NVIDIA H100 80GB HBM3 (Hopper)
Engine: mini-infer @ this slice (continuous batching, paged KV, HTTP `/v1/completions` with SSE streaming)
Driver: [scripts/http_openloop_bench.py](../../scripts/http_openloop_bench.py)
Modal entrypoint: [scripts/modal_openloop_bench.py](../../scripts/modal_openloop_bench.py)
Reproducer:

```
# A10 + Qwen-0.5B
MINI_INFER_BENCH_GPU=A10 uv run modal run scripts/modal_openloop_bench.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --rates 0.1,0.2,0.4,0.6,1.0 --duration 30 --warmup 1 \
    --max-tokens 128 --num-blocks 4096

# H100 + Qwen-7B
MINI_INFER_BENCH_GPU=H100 uv run modal run scripts/modal_openloop_bench.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --rates 0.5,1,2,4 --duration 30 --warmup 1 \
    --max-tokens 128 --num-blocks 4096
```

## What this measures

Every existing mini-infer bench is closed-loop: submit N concurrent
requests, drain to completion, divide tokens by wall-clock. That answers
"what is the peak throughput if the engine is hammered." It does not
answer "what does p99 TTFT look like at 0.4 RPS." For a serving engine
the second question is the more interesting one, and it requires
**open-loop** load: clients fire requests at a fixed cadence regardless
of how many are still pending on the server. As the offered rate
approaches the engine's service rate the in-flight count grows and
latency diverges; the curve of latency vs offered RPS is the headline
artifact.

The driver follows the methodology used by Modal's LLM Almanac and Neural
Magic's `guidellm`:

- Requests are dispatched on a fixed cadence (`1 / target_rate` seconds
  between submissions). Asynchronous `httpx` task per request; no client-
  side cap on in-flight.
- Each request opens an SSE stream against `POST /v1/completions` and
  the driver records `submit_t`, `first_token_t` (TTFT), and `finish_t`
  per request, plus the per-step token count to compute inter-token
  latency (ITL).
- `n` requests per rate point depends on `rate * duration` plus a small
  warmup discard.
- Percentiles are linear-interpolated order statistics; throughput is
  the achieved RPS (`n_ok / (last_finish - first_submit)`).

## Workload

- Real long prompt: [scripts/data/technical_passage.md](../../scripts/data/technical_passage.md),
  10989 chars (~2800 tokens). Same corpus the packed benches use, so
  numbers carry across harnesses.
- `max_tokens = 128`, greedy (`temperature = 0`).
- Block pool sized at `num_blocks = 4096` (vs the 1024 default) so the
  admission watermark sits well below the offered load.

## Results

### A10 + Qwen-0.5B-Instruct

5 rate points, 30 s of measurement per rate, latencies in milliseconds:

| target RPS | achieved | n_ok | TTFT p50 | TTFT p90 | TTFT p99 | ITL p50 | ITL p90 | ITL p99 | total p50 | total p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.11 |   4 |   1616 |   1617 |   1618 |  33 |  33 |  33 |   5791 |   5795 |
| 0.2 | 0.19 |   7 |   1700 |   1706 |   1707 |  47 |  47 |  48 |   7670 |   7713 |
| 0.4 | 0.33 |  13 |   1958 |   2055 |   2092 | 117 | 148 | 154 |  16829 |  20730 |
| 0.6 | 0.37 |  19 |   3603 |  10668 |  12093 | 216 | 270 | 275 |  31089 |  36240 |
| 1.0 | 0.40 |  31 |  13261 |  35880 |  36083 | 324 | 444 | 495 |  56886 |  63082 |

Sustained capacity: ~**0.40 RPS**.

### H100 + Qwen-7B-Instruct

4 rate points, 30 s of measurement per rate, latencies in milliseconds:

| target RPS | achieved | n_ok | TTFT p50 | TTFT p90 | TTFT p99 | ITL p50 | ITL p90 | ITL p99 | total p50 | total p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.40 |  16 |   1678 |   1813 |   1886 | 126 | 156 | 161 |  17688 |  21411 |
| 1.0 | 0.48 |  31 |   7949 |  24089 |  24569 | 200 | 229 | 235 |  33269 |  44719 |
| 2.0 | 0.49 |  61 |  38847 |  84196 |  85251 | 315 | 591 | 684 |  83871 |  98538 |
| 4.0 | 0.47 | 121 | 105927 | 196305 | 219656 | 853 |1645 |1699 | 216060 | 228025 |

Sustained capacity: ~**0.47 RPS**.

Errors: zero. 303 successful streamed completions across the two configs
(74 on A10, 229 on H100).

## Reading the data

### Three regimes, visible in both curves

- **Unsaturated** (A10: 0.1, 0.2; H100: 0.5): achieved RPS tracks target
  RPS, TTFT is flat and equal to the intrinsic single-request prefill
  cost (~1.6-1.7 s for an 11 K-char prompt on either config), ITL is the
  raw decode rate (~33 ms on A10+0.5B, ~126 ms on H100+7B).
- **Knee** (A10: 0.4-0.6; H100: 1.0): achieved RPS starts to plateau,
  TTFT p90 begins to diverge from TTFT p50 as queue depth grows.
- **Saturation** (A10: 1.0; H100: 2.0-4.0): achieved RPS pegs at the
  service rate, in-flight count grows unboundedly with elapsed time,
  TTFT scales linearly with offered-rate excess.

This is textbook M/M/1-ish queue behaviour, and it is the signature that
closed-loop benches cannot show.

### The cross-config finding

Sustained capacity is roughly the same on both configs:

- A10 + Qwen-0.5B: ~0.40 RPS
- H100 + Qwen-7B:  ~0.47 RPS

despite the H100+7B config being a 14x larger model on a 4x faster
GPU. The reason is that at this workload (~2800-token prompts + 128-
token outputs), the engine is **prefill-dominated**, and the prefill
compute cost scales with both axes in roughly equal measure. H100 has
~10x the FLOPS of A10, but Qwen-7B has ~14x the parameters of
Qwen-0.5B; the ratio nearly cancels.

The *decode* path tells a different story. Unsaturated ITL is ~33 ms on
A10+0.5B and ~126 ms on H100+7B; the H100 is doing 4x more work per
token (14x weights / 5x bandwidth ≈ 3-4x). Per-request decode tokens/sec
is ~30 on the A10 config and ~8 on the H100 config. The H100 recovers
the per-request gap with much higher concurrent-batch capacity, which is
why the *aggregate* sustained throughput ends up similar.

### Why TTFT explodes 8x for 2x more offered rate

At the knee (A10, 0.6 RPS), TTFT p90 jumps from 2.1 s to 10.7 s. This is
the queue-depth amplification: when offered rate is just barely above
service rate, each second of overage adds (offered - service) requests
to the queue, and every queued request waits for the entire queue ahead
of it. The numbers fit roughly the M/M/1 formula `W = 1 / (μ - λ)` where
the queue blows up as `λ → μ`.

## Engine fix: OOM no longer kills the scheduler thread

A first attempt to run this bench at `num_blocks = 1024` (the default)
revealed a real correctness issue: the continuous scheduler reserves
`prompt_blocks + 8 decode_headroom` at admission, but those blocks are
not physically allocated, only watermarked. With several long-prompt
requests in flight at once, a mid-step `BlockPool.allocate()` can OOM.
The engine thread then re-raised, terminated, and every subsequent
`submit()` failed with "scheduler not running" until the server was
restarted (i.e., a single OOM took down all in-flight requests and all
future ones).

[scheduler/continuous_scheduler.py](../../src/mini_infer/scheduler/continuous_scheduler.py)
now catches `OutOfMemoryError` at the engine-loop boundary and runs a
small preemption: pick the youngest in-flight request (preferring
prefillers over decoders so a request with output already streamed is
preserved), finish it with `cancelled`, reap its blocks, and continue
the loop. The cancelled client sees `finish_reason="cancelled"` rather
than a 503 across the board.

All 10 existing scheduler unit tests still pass, plus a dedicated test for
the preemption path itself
([tests/unit/test_scheduler_oom.py](../../tests/unit/test_scheduler_oom.py)):
it forces a mid-step `OutOfMemoryError` and asserts the engine thread
survives, cancels the youngest prefiller, and the retried step succeeds.
The path did not trigger during the calibrated sweeps above (errs=0)
because `num_blocks=4096` keeps the pool comfortably ahead of offered load.

## What this validates

- The open-loop harness measures TTFT and ITL correctly via SSE; 303
  successful completions across two configs with deterministic
  per-request numbers and zero protocol errors.
- mini-infer's continuous-batching scheduler degrades gracefully under
  open-loop overload: the queue grows, latency rises, no requests are
  dropped or corrupted.
- The OOM-preemption refactor of the engine loop does not regress
  existing scheduler behaviour (`tests/unit/test_scheduler.py`, 10/10).
- Cross-hardware comparisons can use the same driver against the same
  prompt corpus, producing comparable curves.

## What this does NOT validate

- Sustained capacity on shorter workloads (decode-dominated). At
  ~2800-token prompts the engine is prefill-bound; a 400-token prompt
  workload would shift the picture toward H100+7B's decode advantage.
- Tail behaviour beyond p99 with this sample size. Each rate point has
  4-121 samples; bootstrap confidence intervals would tighten the
  reporting and are an obvious next step.
- Multi-rate-replay variance. Each rate was run once; cross-run variance
  is not measured here.
- Speculative decoding, INT8 weights, TurboQuant KV, or any other
  optimization path. These are orthogonal axes that should each get
  their own calibrated sweep.
- The OOM-preemption path under live overload. The mechanism is
  unit-tested (`tests/unit/test_scheduler_oom.py`), but a deliberately
  oversubscribed Modal run would exercise it end-to-end on real GPU memory.
