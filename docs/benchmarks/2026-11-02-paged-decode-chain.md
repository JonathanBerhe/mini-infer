# The paged decode path on one H100 and one B200: one day of changes

Date: 2026-11-02
Hardware: one NVIDIA H100 80GB and one NVIDIA B200, CUDA 12.8 runtime,
torch 2.10.0+cu128, FlashInfer 0.6.10
Model: `Qwen/Qwen2.5-7B` in bf16, and the same weights as weight-only int8
Workload: the eight bench prompts, greedy decode, 32 new tokens, concurrency
1 and 8, through `ContinuousScheduler` on `PagedKVCache`
Engine: mini-infer from `1aed800` (the `--profile` flag) to `4a97186` (the
split-K int8 kernel), plus the follow-up of 2026-11-03 at the end, decisions in
[ADR-050](../decisions/ADR-050-paged-decode-cuda-graph.md) and
[ADR-051](../decisions/ADR-051-paged-decode-fewer-kernels.md)
Scripts: `scripts/bench/modal_bench_all.py --profile` for the first profile,
then one same-container A/B script per change (two techniques in one
container, same prompts, tokens compared)

## Question

The paged path ran the decode step eagerly and the host was its limit. This
report records what each change did to the step, on one H100 where the
changes were made, and on one B200 where the result was measured at the end.

## Result

![Throughput per step and GPU time per decode step](data/paged-decode-chain.png)

The chain on the H100 at concurrency 8, tokens per second, with the "after"
row of each A/B:

| Step | H100 tok/s | What changed |
|---|---:|---|
| Eager path, host sampling | 414 | the baseline of ADR-050 |
| Decode graph | 812 | one graph replay per step, ADR-050 |
| Fused RMS norm | 818 | one kernel per norm, ADR-051 |
| FlashInfer plan without host waits | not measured apart | host syncs per step 6 to 3, ADR-051 |
| Fused RoPE and K/V write | 926 | one Triton kernel per layer, ADR-051 |
| Merged q, k and v projection | 979 | one GEMM per layer, ADR-051 |
| Async sampling | 1154 | the ids feed the next step on the device, the host reads them one step late (#65) |
| Int8 weights, split-K kernel | 1310 | lossy, the W8A16 decode kernel of #66 |

Every bf16 step decoded identical tokens to the step before it. The int8
step is lossy: the worst last-token logits cosine against bf16 over the eight
prompts was 0.9972, all eight kept their top token, and in the 32-token greedy
runs four of eight requests matched bf16 to the end.

The B200 ran three of the configurations in one container:

| Configuration | B200 tok/s at 8 | B200 tok/s at 1 | GPU ms per decode step at 8 | Wall ms per step at 8 |
|---|---:|---:|---:|---:|
| Eager path, host sampling | 1310 | 171 | 3.62 | 13.3 |
| Graph, async sampling, bf16 | 2062 | 256 | 3.54 | 3.5 |
| Graph, async sampling, int8 with the H100 tile | 2035 | 246 | 3.52 | 3.4 |
| Graph, async sampling, int8 with the Blackwell tiles (its own run, bf16 row 1933) | 1884 | 234 | 3.35 | 3.0 |

The wall times come from the profiled run, where the profiler slows the host.
On the B200 the eager step took 13.3 ms of wall time for 3.6 ms of GPU work,
so the host was the limit by a factor of nearly four. The graph with async
sampling brought the step to its GPU time. Int8 then cut the GPU time of the
step by 7 percent with the Blackwell tiles, but the throughput did not move
within the run-to-run variance of about 6 percent, because a 3.4 ms step is
host-bound again: the scheduler work plus the launch of a 685-node graph,
against 461 nodes for bf16.

## GPU time per decode step

Each pair comes from one A/B in one container, at the profiled level.

| Change | GPU | Before, ms | After, ms | Level |
|---|---|---:|---:|---:|
| Fused RMS norm | H100 | 7.75 | 7.20 | 1 |
| Fused RoPE and K/V write | H100 | 7.31 | 6.47 | 1 |
| Merged q, k and v projection | H100 | 6.16 | 5.79 | 1 |
| Async sampling (wall, not GPU) | H100 | 8.1 | 6.5 | 8 |
| Int8 weights, split-K kernel | H100 | 6.27 | 4.64 | 8 |
| Int8 weights, Blackwell tiles | B200 | 3.60 | 3.35 | 8 |

## The int8 kernel per call

GPU time per call with the calls captured in a CUDA graph, 20 calls per
graph. An eager loop of Triton launches measures the Python launch cost
instead, which is larger than the kernel at these shapes.

| Shape (N by K) | H100 cuBLAS bf16, us | H100 split-K int8, us | B200 cuBLAS bf16, us | B200 split-K int8, us |
|---|---:|---:|---:|---:|
| gate and up, 18944 by 3584 | 53.6 | 32.6 | 22.7 | 18.2 |
| down, 3584 by 18944 | 50.4 | 32.5 | 27.7 | 22.7 |
| q, k and v, 4608 by 3584 | 9.4 | 10.3 | 7.0 | 8.2 |
| o, 3584 by 3584 | 7.9 | 9.5 | 10.3 | 9.2 |

The two small shapes stay in the L2 cache inside a graph of 20 calls, so
their numbers flatter both sides. On the H100 the int8 kernel reads the MLP
weights at 2.1 to 2.25 TB/s. On the B200 cuBLAS reaches 6 TB/s on bf16 and
the Triton kernel about 4 TB/s of int8 bytes, so int8 wins by 1.2 to 1.25
times there against 1.6 to 1.7 on the H100.

## Follow-up on 2026-11-03: one launch per int8 linear

The reduce of the split-K kernel moved into the kernel. Each program stores
its partial sums and adds one to a counter of its n-tile. The program that
arrives last sums the splits in split order, applies the scale and the bias,
casts the result and resets the counter. The sum runs in the same order as
in the reduce kernel, so the result is bit for bit the same. The linears
without a bias also stop filling a one-element dummy bias per call, which
was 112 fills per step. Two host changes came with it: the sampled ids skip
the gather kernel when the batch is the first rows of the buffer, and the
scheduler decodes the text of a request one id at a time instead of all its
ids per step.

One B200, one container, the three configurations in turn, concurrency 8
then 1, 32 new tokens. The kernel test ran first: on four shapes and three
row counts, 0 of 320 calls differed from the two-kernel path. The two int8
configurations decoded identical tokens.

| Configuration | B200 tok/s at 8 | B200 tok/s at 1 | GPU ms per decode step at 8 | Graph nodes | Prefill launches |
|---|---:|---:|---:|---:|---:|
| Graph, async sampling, bf16 | 2040 | 254 | 3.60 | 460 | 405 |
| Int8, split-K kernel and reduce kernel | 2163 | 265 | 3.31 | 684 | 657 |
| Int8, the reduce fused into the kernel | 2184 | 294 | 3.23 | 432 | 405 |

The int8 graph now has fewer nodes than the bf16 graph. The GPU time of the
decode step fell by 0.08 ms. The fused kernel takes 2.25 ms per step against
1.81 plus 0.39 ms for the pair, so the fusion itself costs 0.05 ms, and the
112 dummy fills that went away gave 0.13 ms back. At concurrency 8 the two
int8 rows are within the run-to-run variance of each other. At concurrency
1 the step is shorter by 11 percent, because the launch of a graph of 432
nodes costs less than one of 684. In this run int8 beat bf16 by 7 percent
at concurrency 8 and by 16 percent at concurrency 1, where the chain run
above saw a wash. The two runs differ by the variance of about 6 percent,
not by a change in the kernel.

The profile corrects the guess above that a 3.4 ms step is host-bound. In
every configuration the host waits about 2 ms per decode step for the GPU,
so with the graph the decode step on the B200 is GPU-bound and the host has
slack. The prefill step runs eagerly, and its launches set its host time.
The int8 prefill now launches 405 kernels, the same count as bf16, against
657 with the separate reduce. The profiler gives one prefill sample per
configuration, and those samples did not follow the launch count, so this
run does not show the prefill gain.

## Follow-up on 2026-11-03: TMA loads for the int8 kernel

One B200 timed three variants of the int8 kernel against the kernel of the
repo, every call captured in a CUDA graph, with a tile sweep per variant.
The TMA variants load through tensor descriptors. The swapped one computes
`w @ x^T`, so the weight tile is the wide side of the tensor-core MMA. This
container ran torch 2.15 and Triton 3.9, newer than the runs above, so
compare its rows only with each other.

| Shape at 8 rows | cuBLAS bf16, us | Repo kernel, us | TMA, us | TMA swapped, us |
|---|---:|---:|---:|---:|
| gate and up, 18944 by 3584 | 22.2 | 15.3 | 14.8 | 13.9 |
| down, 3584 by 18944 | 24.8 | 22.8 | 21.6 | 20.8 |
| q, k and v, 4608 by 3584 | 6.5 | 8.4 | 7.4 | 7.9 |
| o, 3584 by 3584 | 9.5 | 7.9 | 7.2 | 7.9 |

The best variant per shape saves about 6.5 us per layer, or about 0.18 ms
of a decode step. That is below the run-to-run variance of the throughput,
so the repo keeps the pointer kernel. The kernel of the repo already copies
its weights with 16-byte asynchronous copies in two stages, and it converts
int8 to bf16 in about one instruction per weight. Nsight Compute shows few
memory stalls. Each int8 call has about 1.5 us of fixed cost and reads 5.0
to 5.4 TB/s at the margin, where cuBLAS bf16 reads 6.1 TB/s. Short calls
pay the fixed cost most.

## What is next

- The H100 did not run with the fused reduce. By the same count its int8
  graph goes from 657 to about 433 nodes, and its step loses the 0.32 ms of
  the reduce kernel less the cost of the fusion.
- One GEMM for gate and up saves 1.8 us per layer with int8, because it
  pays the fixed cost of a call once. It saves nothing with bf16.
- The eager prefill step pays about 20 to 50 us of host time per launch. A
  graph per prefill bucket, or fewer launches per layer, would cut it.
