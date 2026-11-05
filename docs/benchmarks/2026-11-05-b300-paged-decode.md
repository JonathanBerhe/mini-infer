# The paged decode path on one B300 and one B200

Date: 2026-11-05
Hardware: one NVIDIA B300 SXM6 AC (Blackwell Ultra, compute capability 10.3,
148 SMs, 275,040 MiB) and one NVIDIA B200 (compute capability 10.0, 148 SMs),
on Modal
Software on both GPUs: CUDA 13.2, torch 2.14.1+cu132, Triton 3.8.0,
FlashInfer 0.6.10
Model: `Qwen/Qwen2.5-7B` in bf16, and the same weights as weight-only int8
Workload: the eight bench prompts, greedy decode, 32 new tokens, concurrency
1, 8 and 32, through `ContinuousScheduler` on `PagedKVCache`, with the
FlashInfer backend and the decode and prefill CUDA graphs
Engine: mini-infer at `7c6c7da`
Script: `scripts/bench/modal_paged_graph_bench.py`, one run per GPU, both
runs at the same time

## Question

The B300 has 288 GB of HBM, against 192 GB on the B200. Its memory
bandwidth (8 TB/s) and its bf16 tensor-core rate are the same as on the
B200. How fast does the paged decode path run on a B300, against a B200
with the same software?

## Result

| Configuration | B200 | B300 | B300 / B200 |
|---|---:|---:|---:|
| bf16, 1 request, tokens per second | 292 | 290 | 0.99 |
| bf16, 8 requests, tokens per second | 2221 | 2215 | 1.00 |
| bf16, 32 requests, tokens per second | 4455 | 4504 | 1.01 |
| int8, 1 request, tokens per second | 317 | 320 | 1.01 |
| int8, 8 requests, tokens per second | 2401 | 2421 | 1.01 |
| int8, 32 requests, tokens per second | 3669 | 3712 | 1.01 |
| bf16, GPU ms per decode step at 8 requests | 3.37 | 3.34 | 0.99 |
| int8, GPU ms per decode step at 8 requests | 3.06 | 2.98 | 0.97 |

On this 7B model the B300 is as fast as the B200, within 1% in every row.
The decode step reads the weights from memory, and both GPUs have the same
bandwidth. The B300 of Modal also has the same 148 SMs as the B200, not
the 160 SMs of the full Blackwell Ultra chip.

The decode step is GPU-bound on both GPUs. At 8 requests the B300 step
takes 3.34 ms of GPU time in 3.36 ms of wall time. A bf16 step reads about
14 GB of weights (6.53B parameters without the embeddings, plus the
`lm_head`), so the whole step averages about 4.2 TB/s, 53% of the peak
bandwidth.

The B200 in this software stack matches the report of 2026-11-02 with
torch 2.10.0, Triton 3.6.0 and CUDA 12.8
([2026-11-02-paged-decode-chain.md](2026-11-02-paged-decode-chain.md),
the "Both graphs" table). At 8 requests bf16 gives 2221 against 2168
tokens per second, and int8 gives 2401 against 2347. That report gives
about 6% of run-to-run variance, so the move to CUDA 13.2 costs nothing.

int8 is 9% faster than bf16 at 8 requests, but 18% slower at 32 requests,
on both GPUs. The split-K int8 kernel serves up to 64 rows, and its tiles
come from sweeps at 1 to 16 rows. This run did not profile the 32-request
case.

int8 is lossy. Against bf16, 1 of 1, 3 of 8 and 20 of 32 requests decoded
the same 32 tokens.

## How the runs were made

Modal starts a B300 only with CUDA 13.1 or newer in the image. The image
starts from `nvidia/cuda:13.2.1-devel-ubuntu22.04`, because FlashInfer
compiles its kernels at first use with the `nvcc` of the image. It
installs `torch==2.14.1+cu132`, the torch release that pins Triton 3.8.0.
The torch wheel has no `sm_103` code. Its kernels run on the B300 from the
`sm_100` code, which CUDA allows within one major version. Triton and
Gluon compile every kernel for `sm_103a` at run time.

For each configuration, the script builds the runner, which captures the
graphs. Then it runs one warm-up sweep and three timed sweeps per
concurrency level, and it reports the median. The three sweeps differed by
1.4% or less in every row. Last, one sweep at 8 requests runs under the
step profiler (`profile_scheduler_steps`). The table gives the median GPU
busy time over the 32 decode steps of that sweep.

Before this run, the kernel and engine unit tests ran on the B300 with the
same image (`scripts/smoke/modal_b300_check.py`, #77). They include the
Gluon MoE kernels, the CUDA graph tests and the FlashInfer backend.

## What is next

- A model that does not fit one B200 shows what the B300 adds. DeepSeek-V4-Flash
  (158 GB on disk) runs with tensor parallelism on two B200s today. On one
  B300 it fits with room for the KV cache and the graphs.
- Qwen3.8-27B at 128k and 256k tokens of context fits on one B300, dense
  and with LSA.
- A profile of int8 at 32 requests, to find what makes it slower than bf16.
