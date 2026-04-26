# Continuous batching throughput sweep

**Date:** 2026-04-27
**Hardware:** Modal A10 (NVIDIA A10), via Modal default image
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`, bf16
**Workload:** identical 40-token prompt × 32 max_tokens × concurrency C ∈ {1, 2, 4, 8}, one warmup run per concurrency before timing.
**Reproducer:** `uv run modal run scripts/modal_concurrent_bench.py`.

## Numbers

End-to-end wall-clock for `C` simultaneously-submitted requests (lower elapsed = better for the workload; higher tok/s = better aggregate throughput):

| Concurrency | Elapsed (s) | Total output tokens | Throughput (tok/s) | Per-request latency (s) |
|---|---|---|---|---|
| 1 | 0.700 | 32 | 45.7 | 0.700 |
| 2 | 0.805 | 64 | 79.5 | 0.805 |
| 4 | 1.002 | 128 | 127.7 | 1.002 |
| 8 | 1.365 | 256 | 187.5 | 1.365 |

Throughput scaling vs C=1:

| Concurrency | Throughput multiplier |
|---|---|
| 2 | 1.74× |
| 4 | 2.79× |
| 8 | 4.10× |

## Reading the numbers

- **Throughput scales sub-linearly, as expected.** The forward-pass time grows with batch size (more attention rows, larger K/V gather), but slower than linearly because the linear projections and MLP — both batched matmuls — amortize across requests. At C=8 we're getting 4.1× the tok/s of C=1; the gap between 4.1× and the 8.0× ideal is the per-step compute that does scale with B.
- **Per-request latency degrades gracefully.** C=8 is only 1.95× slower per request than C=1, while serving 8× the requests in that time. This is the throughput-vs-latency trade that continuous batching is for: at saturation, you take some hit on individual TTFT/ITL to avoid serializing requests behind each other entirely.
- **The B=1 path didn't regress.** 45.7 tok/s at C=1 is in line with the per-request throughput we'd expect from the same kernel-only-on-decode setup measured in 2.2's bench (the prior bench was decode-step μs, not end-to-end tok/s, so not directly comparable, but order-of-magnitude consistent given prefill ~10× decode for a ~40-token prompt + 32 decode steps).

## What this validates

- **Correctness on real hardware**: the Modal smoke (`scripts/modal_concurrent_smoke.py`) confirmed the batched scheduler produces token-for-token identical output to a serial reference at C=4. The benchmark above ran on the same code path.
- **The batched Triton kernel is exercised end-to-end**: every decode step at C ≥ 2 fires the multi-request kernel grid `(B * num_q_heads,)` with per-request `block_tables` and `seq_lens`.
- **The scheduler keeps the batch full**: throughput would scale much worse if requests were serializing inside the engine thread.

## What this does NOT validate

- **Larger batches (C ≥ 16).** A10 has plenty of headroom for a 0.5B model; a follow-up sweep at C=16, 32 on a bigger card would show where the kernel saturates.
- **Long contexts.** Workload was 40-token prompt × 32 decode tokens. Effects of batching at 1k+ contexts (where the K/V gather becomes a bigger fraction of step cost) are not measured here.
- **Mixed-length workload.** All requests use the same prompt and max_tokens, so the batch size stays constant across the run. Real serving has requests joining and leaving mid-batch (`batch_idx` drift), which is exercised by the unit tests but not measured here as a throughput number.
- **Comparison vs vLLM / SGLang.** Phase 4 polish does that head-to-head with proper hardware (likely H100) and a richer workload mix.
