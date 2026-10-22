# Grouped expert matmuls for MoE layers on H100 and B200

Date: 2026-10-21
Hardware: one NVIDIA H100 80GB HBM3 and one NVIDIA B200, bf16, PyTorch
2.14.1 with CUDA 13.0, Triton 3.8.0
Workloads: one MoE layer with random weights at the sizes of Nemotron 3.5
Lightning 30B-A3B, Mixtral 8x7B and GLM-5.3 (with bf16 and with block-FP8
experts). The real `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
(revision `a9904d2`), with prefill on `scripts/data/technical_passage.md`
Engine: mini-infer with the grouped expert paths of
[ADR-036](../decisions/ADR-036-grouped-expert-matmuls.md) (bf16),
[ADR-037](../decisions/ADR-037-block-fp8-grouped-kernel.md) (block FP8) and
[ADR-038](../decisions/ADR-038-gluon-fp8-grouped-hopper.md) (block FP8 in
Gluon on the H100)
Script: `scripts/modal_moe_grouped_bench.py` (`--fp8-only`, `--fp8-kernel`,
`--fp8-tiles`, `--fp8-gluon` and `--fp8-expert-tiles` for the FP8 runs)
Raw output: bf16 on the [H100](data/moe-grouped-gemm-h100.txt) and the
[B200](data/moe-grouped-gemm-b200.txt), FP8 on the
[H100](data/moe-grouped-fp8-h100.txt) and the
[B200](data/moe-grouped-fp8-b200.txt), the Gluon kernel on the
[H100](data/moe-grouped-fp8-gluon-h100.txt), the expert tiles on the
[H100](data/moe-grouped-fp8-expert-tiles-h100.txt) and the
[B200](data/moe-grouped-fp8-expert-tiles-b200.txt)

## Result

- Prefill of the real Nemotron 30B-A3B is 8.5x (H100) and 8.0x (B200)
  faster at 512 tokens. At 2048 tokens it is 4.9x faster on the H100 and
  6.7x faster on the B200.
- Decode with the CUDA graph is 1.52x faster at batch 4 on the H100 and
  1.21x faster on the B200. At batch 16 the old path did not fit in a CUDA
  graph. The new path runs in one: 214 ms eager before, 19.7 ms after
  (H100), and 203 ms before, 14.2 ms after (B200).
- One Nemotron MoE layer at 2048 tokens ran 1959 GPU operations and 130
  host syncs per forward before, with the GPU idle 73 to 75 percent of the
  time. After, it runs 66 GPU operations, no host syncs, and the GPU is
  idle 15 to 16 percent of the time.
- Layers with many experts gain the most. GLM-5.3 (256 experts) is 6 to 20
  times faster up to 1024 tokens. Mixtral (8 experts) is 1.4 to 2.2 times
  faster up to 256 tokens, and equal or slower from 4096 tokens.
- With block-FP8 experts, the first version of the Triton kernel made the
  GLM-5.3 layer 13 to 30 times faster than the per-expert loop up to 1024
  tokens. The second version is another 1.1 to 2.4 times faster in a CUDA
  graph at decode sizes, and 1.05 to 1.39 times faster at prefill sizes.
  Up to 256 tokens the FP8 layer is now faster than the same layer with
  bf16 experts. From 4096 tokens it is 1.26 to 1.46 times slower. The
  section "Block-FP8 experts" has the details.
- On the H100, the Gluon kernel makes the FP8 layer 1.41 to 1.61 times
  faster than the Triton kernel from 1024 tokens. The FP8 layer is then
  1.26 times faster than the bf16 layer at 4096 tokens and equal at 16384
  tokens. The section "Gluon kernel for Hopper" has the details.
- In the Triton kernel, tiles of one expert each make the FP8 layer 1.08
  to 1.42 times faster on the B200 and 1.06 to 1.31 times faster on the
  H100 than row tiles that span experts. The section "Expert tiles in the
  Triton kernel" has the details.
- The grouped path is slower in three places: batch-1 graph decode on the
  B200 (3.6 percent), eager decode at small batches, and Mixtral at 16384
  tokens on the B200 (0.88x). The section "Where the grouped path is
  slower" explains each one.

## How a point is measured

The old and the new path are both in the code.
`grouped_experts._DISABLED_FOR_BENCH` switches between them, so each GPU
measures both paths in one process, with the same weights and inputs.
"Before" is the old path: the per-expert loop for Mixtral and GLM. For
Nemotron-H it is the expert-weight gather up to 64 (token, expert) pairs
and the loop above that. "After" is the grouped path.

A layer point is the median wall time of one forward. The timer syncs the
GPU before and after each call, so launch cost and host syncs count, as they
do in serving. The layer benchmark uses random routing, which spreads the
tokens about evenly over the experts. The real-model points use the model's
own routing.

A graph point captures one forward in a CUDA graph and times its replay.
"Not capturable" means that the path syncs with the host, so a CUDA graph
cannot hold it.

## One MoE layer, eager

Milliseconds per forward.

Nemotron 3.5 Lightning 30B-A3B: hidden size 2688, expert width 1856, 128
experts, top-6.

| tokens | H100 before | H100 after | speedup | B200 before | B200 after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.467 | 0.518 | 0.90x | 0.545 | 0.569 | 0.96x |
| 4 | 0.825 | 0.618 | 1.34x | 0.529 | 0.581 | 0.91x |
| 16 | 8.935 | 0.990 | 9.02x | 9.782 | 0.738 | 13.26x |
| 64 | 16.285 | 1.379 | 11.81x | 16.522 | 0.945 | 17.48x |
| 256 | 16.169 | 1.474 | 10.97x | 17.718 | 1.033 | 17.15x |
| 1024 | 16.235 | 1.775 | 9.15x | 18.679 | 1.236 | 15.11x |
| 4096 | 16.716 | 2.899 | 5.77x | 19.168 | 1.867 | 10.27x |
| 16384 | 19.746 | 9.291 | 2.13x | 19.816 | 5.790 | 3.42x |

Mixtral 8x7B: hidden size 4096, expert width 14336, 8 experts, top-2.

| tokens | H100 before | H100 after | speedup | B200 before | B200 after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.967 | 0.594 | 1.63x | 1.150 | 0.639 | 1.80x |
| 4 | 1.907 | 1.377 | 1.38x | 1.730 | 0.865 | 2.00x |
| 16 | 2.142 | 1.483 | 1.44x | 2.299 | 1.058 | 2.17x |
| 64 | 2.149 | 1.432 | 1.50x | 2.401 | 1.074 | 2.24x |
| 256 | 2.192 | 1.432 | 1.53x | 2.389 | 1.098 | 2.18x |
| 1024 | 2.499 | 2.175 | 1.15x | 2.303 | 1.274 | 1.81x |
| 4096 | 5.725 | 5.734 | 1.00x | 3.328 | 3.364 | 0.99x |
| 16384 | 20.734 | 20.892 | 0.99x | 10.813 | 12.343 | 0.88x |

GLM-5.3: hidden size 6144, expert width 2048, 256 experts, top-8, one
shared expert.

| tokens | H100 before | H100 after | speedup | B200 before | B200 after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.560 | 0.818 | 15.36x | 16.076 | 0.796 | 20.21x |
| 4 | 15.843 | 1.406 | 11.27x | 17.540 | 1.106 | 15.85x |
| 16 | 26.342 | 3.529 | 7.47x | 27.396 | 2.070 | 13.24x |
| 64 | 45.911 | 6.568 | 6.99x | 59.145 | 4.309 | 13.73x |
| 256 | 47.742 | 7.636 | 6.25x | 53.109 | 4.822 | 11.01x |
| 1024 | 51.289 | 8.250 | 6.22x | 52.955 | 5.247 | 10.09x |
| 4096 | 48.553 | 12.962 | 3.75x | 53.018 | 6.834 | 7.76x |
| 16384 | 55.325 | 29.694 | 1.86x | 58.450 | 18.736 | 3.12x |

## One MoE layer, CUDA graph replay

Milliseconds per replay at decode sizes.

| layer | tokens | H100 before | H100 after | B200 before | B200 after |
|---|---:|---:|---:|---:|---:|
| Nemotron | 1 | 0.221 | 0.203 | 0.147 | 0.163 |
| Nemotron | 4 | 0.634 | 0.351 | 0.333 | 0.247 |
| Nemotron | 16 | not capturable | 0.681 | not capturable | 0.428 |
| Nemotron | 64 | not capturable | 1.108 | not capturable | 0.658 |
| Mixtral | 1 | not capturable | 0.366 | not capturable | 0.293 |
| Mixtral | 4 | not capturable | 1.024 | not capturable | 0.544 |
| Mixtral | 16 | not capturable | 1.233 | not capturable | 0.695 |
| Mixtral | 64 | not capturable | 1.142 | not capturable | 0.705 |
| GLM-5.3 | 1 | not capturable | 0.424 | not capturable | 0.285 |
| GLM-5.3 | 4 | not capturable | 1.001 | not capturable | 0.635 |
| GLM-5.3 | 16 | not capturable | 3.097 | not capturable | 1.809 |
| GLM-5.3 | 64 | not capturable | 6.175 | not capturable | 3.918 |

## Profile of the Nemotron layer

`torch.profiler` over 5 forwards, per forward. A GPU operation is a kernel,
a copy or a memset. The wall time comes from a separate timed run at the
same size. GPU idle is the share of the wall time with no GPU operation
running.

| GPU | tokens | path | GPU ops | GPU busy (ms) | wall (ms) | GPU idle | host syncs |
|---|---:|---|---:|---:|---:|---:|---:|
| H100 | 4 | before | 40 | 0.592 | 0.825 | 28% | 0 |
| H100 | 4 | after | 50 | 0.310 | 0.618 | 50% | 0 |
| H100 | 2048 | before | 1959 | 6.125 | 24.484 | 75% | 130 |
| H100 | 2048 | after | 66 | 1.894 | 2.264 | 16% | 0 |
| B200 | 4 | before | 38 | 0.341 | 0.529 | 36% | 0 |
| B200 | 4 | after | 50 | 0.245 | 0.581 | 58% | 0 |
| B200 | 2048 | before | 1959 | 6.674 | 24.360 | 73% | 130 |
| B200 | 2048 | after | 66 | 1.288 | 1.523 | 15% | 0 |

At 4 tokens, the old path spends about half of its GPU time in one gather
kernel, which copies the chosen experts' weights: 0.304 ms on the H100 and
0.148 ms on the B200. The grouped path reads the weights in place. At 2048
tokens, the old path syncs with the host about once per expert, and the GPU
waits for each sync. In the grouped path, the CUTLASS grouped GEMM kernel
for the GPU family (`sm9x` on the H100, `sm10` on the B200) takes 1.04 ms
of 1.89 ms (H100) and 0.58 ms of 1.29 ms (B200).

## Real model: Nemotron 3.5 Lightning 30B-A3B

Prefill of the real passage, milliseconds:

| tokens | H100 before | H100 after | speedup | B200 before | B200 after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 528.2 | 62.1 | 8.51x | 497.5 | 62.4 | 7.97x |
| 2048 | 624.4 | 127.7 | 4.89x | 566.3 | 84.5 | 6.70x |

Decode, milliseconds per step. Each row of the batch has a 64-token random
prompt, as in `scripts/modal_nemotron_h_smoke.py`.

| GPU | batch | before eager | before graph | after eager | after graph |
|---|---:|---:|---:|---:|---:|
| H100 | 1 | 45.2 | 9.97 | 41.9 | 9.23 |
| H100 | 4 | 46.4 | 19.48 | 56.1 | 12.82 |
| H100 | 16 | 213.9 | not capturable | 48.5 | 19.66 |
| B200 | 1 | 36.7 | 7.47 | 40.1 | 7.74 |
| B200 | 4 | 37.6 | 12.36 | 41.2 | 10.24 |
| B200 | 16 | 203.5 | not capturable | 41.2 | 14.20 |

## Agreement of the two paths

- One layer: the relative difference of the outputs is at most 8e-4 for
  Nemotron and about 3e-4 for Mixtral. For GLM-5.3 it is 3.3e-3, because
  the old loop adds the 8 expert outputs into a bf16 tensor one at a time.
  The grouped path rounds that sum once instead of eight times.
- Real model, 512-token prefill: logit cosine 0.99940 (H100) and 0.99952
  (B200), the same argmax at 96.3 and 97.3 percent of the positions. The
  port against transformers gave a cosine of 0.9949 to 0.9989
  ([ADR-035](../decisions/ADR-035-nemotron-h-port.md)). Both gaps come from
  bf16 rounding that reaches a top-6 routing near-tie and picks another
  expert.
- Greedy decode: all 52 tokens are the same at batch 1 on both GPUs. At
  batch 4, 137 (H100) and 143 (B200) of 208 positions are the same, and at
  batch 16, 646 and 627 of 832. The prompts are random tokens, so near-ties
  are frequent. After one token differs, the rest of that row differs.

## Where the grouped path is slower

- Batch-1 decode with the CUDA graph on the B200: 7.74 ms against 7.47 ms.
  At one token, the old path gathers only 6 experts, and the grouped path
  runs 10 to 12 more small GPU operations per layer (the sort, the group
  offsets, the row gather and the un-sort). On the H100 the grouped path is faster at
  batch 1 (9.23 ms against 9.97 ms). In serving, the decode graph runs all
  `max_batch_size` rows (8 by default), so one row per step occurs only at
  `max_batch_size=1`.
- Eager decode at batch 1 and 4: 9 to 21 percent slower, except batch 1 on
  the H100. Eager decode on this model is bound by the host's launch time,
  so 10 to 12 more launches per MoE layer cost more than the gather saves. Serving
  on CUDA replays the decode step from the CUDA graph, where batch 4 is
  faster on both GPUs.
- Mixtral at 4096 tokens and more: 0.99 to 1.00x, and 0.88x at 16384
  tokens on the B200. With 8 experts, each expert gets about 4096 rows at
  16384 tokens. These are large matmuls, and cuBLAS runs them faster than
  the CUTLASS grouped kernel on the B200.

## Block-FP8 experts

The GLM-5.3 layer again, with the routed experts in e4m3 and one fp32 scale
per 128x128 block, as the GLM-5 FP8 checkpoints ship them. "Before" is the
per-expert loop of `Fp8Expert`, which dequantizes each expert's weights
through full-size fp32 temporaries. "After" is the first version of the
Triton grouped kernel of
[ADR-037](../decisions/ADR-037-block-fp8-grouped-kernel.md), which
dequantizes in registers. The next section measures its second version.
Before the timings, each run passed the 27 grouped-expert and kernel unit
tests on its GPU.

Milliseconds per forward, eager:

| tokens | H100 before | H100 after | speedup | B200 before | B200 after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.179 | 1.047 | 14.50x | 9.124 | 0.706 | 12.93x |
| 4 | 29.535 | 1.532 | 19.27x | 17.556 | 1.169 | 15.01x |
| 16 | 68.958 | 2.275 | 30.31x | 44.153 | 1.684 | 26.22x |
| 64 | 145.822 | 6.105 | 23.89x | 88.091 | 4.931 | 17.87x |
| 256 | 164.458 | 6.961 | 23.62x | 98.953 | 4.739 | 20.88x |
| 1024 | 167.323 | 9.710 | 17.23x | 99.886 | 5.964 | 16.75x |
| 4096 | 168.129 | 19.615 | 8.57x | 100.812 | 12.984 | 7.76x |
| 16384 | 182.455 | 56.962 | 3.20x | 109.045 | 40.617 | 2.68x |

The loop is not capturable. The grouped path in a CUDA graph, milliseconds
per replay: 0.552, 1.057, 1.824 and 5.703 at 1, 4, 16 and 64 tokens on the
H100, and 0.574, 1.049, 1.557 and 4.819 on the B200.

Profile, per forward:

| GPU | tokens | path | GPU ops | GPU busy (ms) | wall (ms) | GPU idle | host syncs |
|---|---:|---|---:|---:|---:|---:|---:|
| H100 | 4 | before | 2124 | 17.763 | 29.535 | 40% | 256 |
| H100 | 4 | after | 49 | 1.037 | 1.532 | 32% | 0 |
| H100 | 2048 | before | 8441 | 137.182 | 173.776 | 21% | 256 |
| H100 | 2048 | after | 65 | 13.414 | 13.831 | 3% | 0 |
| B200 | 4 | before | 2124 | 13.803 | 17.556 | 21% | 256 |
| B200 | 4 | after | 49 | 1.077 | 1.169 | 8% | 0 |
| B200 | 2048 | before | 8317 | 96.219 | 102.177 | 6% | 256 |
| B200 | 2048 | after | 63 | 8.850 | 8.471 | 0% | 0 |

Before, most GPU time goes to the elementwise kernels of the dequantization
(the fp32 copies, the scale multiply and the cast to bf16), and the loop
syncs with the host once for each of the 256 experts. After, the Triton
kernel takes 0.89 to 0.90 ms of the GPU time at 4 tokens, and 11.9 ms (H100)
and 7.4 ms (B200) at 2048 tokens.

### Second version of the kernel

In the first version, one program computed a row tile and walked the
experts of its rows one after another. At decode, each row of a tile has
its own expert, so a few programs did long serial walks and most of the GPU
stayed idle. The second version changes three things:

- Up to 256 rows, one program per expert segment, so the experts of a batch
  load in parallel.
- The dequantized weights are the left operand of the dot. Hopper's `wgmma`
  takes its left operand from registers, where the dequantized tile is.
- One tile table per GPU family, from the sweeps below.

"Previous" is the first version and "new" the second, in one process on
each GPU (`--fp8-tiles`, raw output for the
[H100](data/moe-grouped-fp8-tiles-h100.txt) and the
[B200](data/moe-grouped-fp8-tiles-b200.txt)). Each run first passed 39 unit
tests on its GPU.

CUDA graph replay, milliseconds:

| tokens | H100 previous | H100 new | speedup | B200 previous | B200 new | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.553 | 0.279 | 1.98x | 0.582 | 0.239 | 2.44x |
| 4 | 1.067 | 0.605 | 1.76x | 1.071 | 0.484 | 2.21x |
| 16 | 1.828 | 1.537 | 1.19x | 1.656 | 1.208 | 1.37x |
| 64 | 4.996 | 4.237 | 1.18x | 4.195 | 3.822 | 1.10x |

Eager, milliseconds per forward:

| tokens | H100 previous | H100 new | speedup | B200 previous | B200 new | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 6.077 | 5.240 | 1.16x | 4.692 | 4.452 | 1.05x |
| 1024 | 9.028 | 8.030 | 1.12x | 5.865 | 5.578 | 1.05x |
| 4096 | 18.782 | 16.249 | 1.16x | 12.898 | 10.150 | 1.27x |
| 16384 | 54.105 | 42.704 | 1.27x | 40.047 | 28.896 | 1.39x |

The two versions differ by a relative 2e-4 to 6e-4 from 1 to 16 tokens,
where the per-segment split adds in another order, and not at all from 64
tokens. At 4 tokens the kernel itself went from 0.90 to 0.50 ms (H100) and
from 0.92 to 0.40 ms (B200), in the profiles of the `--fp8-kernel` runs
(raw output for the [H100](data/moe-grouped-fp8-kernel-h100.txt) and the
[B200](data/moe-grouped-fp8-kernel-b200.txt)).

The same runs timed each default tile with and without the swapped
operands. With medium batches (64 to 1024 tokens) the swap saves 1 to 12
percent of the time on the H100 and 6 to 8 percent on the B200. With large
batches it saves 2 to 6 percent. At decode sizes the two orders differ by
less than eager timing varies between runs (up to about 20 percent at one
token on the B200), so the decode gain comes from the per-segment split.

### Against the bf16 experts

The same layer with bf16 experts on `grouped_mm` (from the `--fp8-kernel`
runs) against the FP8 layer on the second version of the kernel. Both are
on the same GPU type, in different runs.

| tokens | timing | H100 bf16 | H100 FP8 | B200 bf16 | B200 FP8 |
|---:|---|---:|---:|---:|---:|
| 1 | graph | 0.438 | 0.279 | 0.290 | 0.239 |
| 4 | graph | 1.011 | 0.605 | 0.638 | 0.484 |
| 16 | graph | 3.106 | 1.537 | 1.795 | 1.208 |
| 64 | graph | 6.163 | 4.237 | 4.004 | 3.822 |
| 256 | eager | 7.766 | 5.240 | 4.897 | 4.452 |
| 1024 | eager | 8.385 | 8.030 | 5.211 | 5.578 |
| 4096 | eager | 12.941 | 16.249 | 6.938 | 10.150 |
| 16384 | eager | 30.665 | 42.704 | 19.803 | 28.896 |

Up to 256 tokens the FP8 layer is now 1.05 to 2.0 times faster than the
bf16 one on both GPUs, because it reads half the bytes. At 1024 tokens the
two are about equal. From 4096 tokens the FP8 layer is 1.26 to 1.46 times
slower, against 1.51 to 2.17 times with the first version. The kernel
dequantizes on the CUDA cores and multiplies in bf16, without TMA or warp
specialization, and the remaining gap is there.

### Kernel tiles

The `--fp8-kernel` runs swept tiles of the second version, as (BLOCK_M,
BLOCK_N, BLOCK_K, warps, stages), with swapped operands. Eager
milliseconds per forward. Per-segment tiles at decode sizes:

| tile | H100 1 tok | H100 4 | H100 16 | B200 1 tok | B200 4 | B200 16 |
|---|---:|---:|---:|---:|---:|---:|
| 16, 32, 128, 4, 3 | 1.109 | 1.282 | 2.256 | 0.437 | 0.725 | 1.638 |
| 16, 64, 64, 4, 3 | 1.152 | 1.345 | 2.477 | 0.446 | 0.737 | 1.644 |
| 16, 64, 128, 4, 3 | 1.162 | 1.246 | 2.199 | 0.462 | 0.801 | 1.861 |
| 16, 128, 64, 4, 3 | 1.136 | 1.320 | 2.476 | 0.431 | 0.677 | 1.397 |
| 16, 128, 128, 8, 3 | 1.161 | 1.244 | 2.210 | 0.465 | 0.792 | 1.787 |
| 32, 64, 128, 4, 3 | 1.136 | 1.261 | 2.223 | 0.470 | 0.845 | 1.948 |

Row-tile tiles at prefill sizes:

| tile | H100 256 tok | H100 1024 | H100 4096 | H100 16384 | B200 256 tok | B200 1024 | B200 4096 | B200 16384 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64, 128, 64, 4, 4 | 6.375 | 9.323 | 18.663 | 59.086 | 4.271 | 5.215 | 12.005 | 37.173 |
| 64, 128, 128, 4, 3 | 5.469 | 8.364 | 17.409 | 50.945 | 4.902 | 6.235 | 13.043 | 40.765 |
| 128, 64, 64, 4, 4 | 8.310 | 10.234 | 24.808 | 76.811 | 6.645 | 7.332 | 14.567 | 42.455 |
| 128, 128, 64, 4, 4 | 7.305 | 8.808 | 18.763 | 49.793 | 4.976 | 5.435 | 10.916 | 31.014 |
| 128, 128, 64, 8, 3 | 8.103 | 10.493 | 20.622 | 56.962 | 5.403 | 5.743 | 11.270 | 32.302 |
| 128, 128, 128, 8, 3 | 6.946 | 8.553 | 17.650 | 48.955 | 6.561 | 7.157 | 12.427 | 35.112 |
| 64, 256, 64, 8, 3 | 6.988 | 9.628 | 18.058 | 58.299 | 4.260 | 5.190 | 11.159 | 34.189 |
| 128, 256, 64, 8, 3 | 7.465 | 8.911 | 16.444 | 45.064 | 5.206 | 5.637 | 10.224 | 28.676 |

The kernel picks a tile by GPU family and by the number of (token, expert)
rows. Each default is the fastest tile of the sweep at its sizes, or within
5 percent of it:

| rows (tokens at top-8) | H100 (Hopper) | B200 (Blackwell) |
|---|---|---|
| up to 256 (32) | 16, 64, 128, 4, 3 | 16, 128, 64, 4, 3 |
| up to 8192 (1024) | 64, 128, 128, 4, 3 | 64, 128, 64, 4, 4 |
| above | 128, 256, 64, 8, 3 | 128, 256, 64, 8, 3 |

The sweep of the first version (9 tiles) is in the raw output of the first
FP8 runs.

### W8A8, tried and removed

A W8A8 mode of the kernel quantized the activations per token and per 128
columns, as DeepSeek, vLLM and transformers do, and multiplied e4m3 by e4m3
on the FP8 tensor cores. Both scales then multiplied each block's fp32
partial sum. It was measured on 2026-10-22 against the W8A16 kernel in one
process per GPU (raw output for the
[H100](data/moe-grouped-fp8-w8a8-h100.txt) and the
[B200](data/moe-grouped-fp8-w8a8-b200.txt)), and then removed.

Milliseconds per forward, W8A8 with the default tiles of W8A16:

| tokens | timing | H100 W8A16 | H100 W8A8 | B200 W8A16 | B200 W8A8 |
|---:|---|---:|---:|---:|---:|
| 1 | graph | 0.265 | 0.307 | 0.228 | 0.258 |
| 64 | graph | 4.374 | 3.878 | 3.544 | 13.118 |
| 1024 | eager | 8.111 | 8.464 | 5.153 | 15.032 |
| 4096 | eager | 16.193 | 38.588 | 9.801 | 97.643 |
| 16384 | eager | 42.864 | 107.149 | 27.680 | 251.333 |

With its own best tile from a sweep of 7, W8A8 took 17.1 and 55.4 ms at
4096 and 16384 tokens on the H100, and 19.0 and 55.0 ms on the B200. That
is still slower than W8A16 on both GPUs. Against an fp32 computation with
the same routing (unrounded dequantized weights, unquantized bf16 inputs),
the routed-expert output of W8A16 has a relative error of 5.4e-3 to 5.5e-3,
and that of W8A8 4.3e-2 to 4.4e-2, on both GPUs and at 1, 64 and 2048
tokens.

### Gluon kernel for Hopper

The Gluon kernel of
[ADR-038](../decisions/ADR-038-gluon-fp8-grouped-hopper.md) against the
Triton kernel of the sections above, in one process on one H100 on
2026-10-22 (`--fp8-gluon`, [raw output](data/moe-grouped-fp8-gluon-h100.txt)).
Before the timings, the run passed 84 unit tests on the GPU. They include
the Gluon kernel against fp32 math, against the Triton kernel, in a CUDA
graph, and under Triton's concurrency sanitizer.

Eager milliseconds per forward. "Gluon" uses the default tile for the size
and the copy warp (warp specialization). "No copy warp" is the same tile
without it. Each Gluon run is forced on, also at 32 tokens, where the
default path keeps the Triton kernel:

| tokens | Triton | Gluon | no copy warp | Gluon tile | speedup |
|---:|---:|---:|---:|---|---:|
| 32 | 2.737 | 3.117 | 3.255 | 128, 64, 64, 4, 4 | 0.88x |
| 64 | 4.468 | 4.173 | 4.325 | 128, 64, 64, 4, 4 | 1.07x |
| 256 | 5.162 | 4.848 | 5.016 | 128, 64, 64, 4, 4 | 1.06x |
| 1024 | 8.066 | 5.393 | 5.704 | 128, 64, 64, 4, 4 | 1.50x |
| 2048 | 11.132 | 7.230 | 7.845 | 128, 64, 64, 4, 4 | 1.54x |
| 4096 | 16.162 | 10.025 | 10.819 | 128, 128, 64, 8, 4 | 1.61x |
| 8192 | 24.774 | 16.859 | 17.515 | 128, 256, 64, 8, 3 | 1.47x |
| 16384 | 42.378 | 30.052 | 31.507 | 128, 256, 64, 8, 3 | 1.41x |

A tile is (weight rows, token rows, K, warps, stages). The outputs of the
two kernels are bit-identical at every size. At 32 tokens (256 rows) the
Gluon kernel is slower than the per-segment Triton split, so the default
path calls it only above 256 rows. The copy warp makes the layer 3.5 to 8.5
percent faster at every size.

Against the same layer with bf16 experts on `grouped_mm`, in the same run:

| tokens | bf16 | FP8 Triton | FP8 Gluon |
|---:|---:|---:|---:|
| 1024 | 8.214 | 8.066 | 5.393 |
| 4096 | 12.622 | 16.162 | 10.025 |
| 16384 | 30.285 | 42.378 | 30.052 |

Each projection alone, with random routing over the 256 experts. The
median milliseconds of `triton.testing.do_bench`, and the useful TFLOPS
(`2 * rows * N * K` per call):

| tokens | projection | Triton | Gluon | speedup | Triton TFLOPS | Gluon TFLOPS |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | gate-up | 4.607 | 2.894 | 1.59x | 90 | 142 |
| 1024 | down | 2.357 | 1.473 | 1.60x | 87 | 140 |
| 4096 | gate-up | 8.864 | 5.872 | 1.51x | 186 | 281 |
| 4096 | down | 4.859 | 2.639 | 1.84x | 170 | 312 |
| 16384 | gate-up | 21.015 | 13.336 | 1.58x | 314 | 495 |
| 16384 | down | 10.652 | 7.457 | 1.43x | 310 | 442 |

At 1024 tokens there are 32 rows per expert, so the layer reads all 9.7 GB
of e4m3 expert weights for few products. The TFLOPS stay low there for both
kernels.

Profile at 4096 tokens: the Triton kernel takes 13.113 ms of 16.071 ms GPU
time per forward, the Gluon kernel 7.025 ms of 10.200 ms. The other
kernels of the layer (routing, gather, scatter, shared expert) take about
the same time in both runs: 3.0 and 3.2 ms.

Gluon tiles, eager milliseconds per forward:

| tile | copy warp | 256 tok | 1024 | 4096 | 16384 |
|---|---|---:|---:|---:|---:|
| 128, 32, 64, 4, 4 | yes | 4.810 | 6.257 | 15.909 | 57.063 |
| 128, 64, 64, 4, 4 | yes | 5.034 | 5.537 | 11.861 | 39.715 |
| 128, 64, 128, 4, 3 | yes | 4.921 | 5.533 | 11.678 | 38.889 |
| 64, 128, 64, 4, 4 | yes | 7.321 | 7.751 | 12.738 | 38.766 |
| 128, 128, 64, 8, 4 | yes | 6.282 | 6.413 | 10.108 | 32.998 |
| 128, 128, 128, 8, 3 | yes | 5.783 | 6.235 | 10.991 | 31.968 |
| 128, 256, 64, 8, 3 | yes | 8.964 | 9.241 | 10.804 | 30.821 |
| 128, 64, 64, 4, 4 | no | 5.144 | 5.768 | 12.642 | 42.453 |
| 128, 128, 64, 8, 4 | no | 6.439 | 6.883 | 11.215 | 35.474 |
| 128, 256, 64, 8, 3 | no | 9.247 | 9.614 | 11.482 | 31.441 |

The kernel picks the token rows of a tile from the mean rows per expert:
64 up to 64 rows (1024 and 2048 tokens), 128 up to 192 rows (4096 tokens),
and 256 above. At 1024, 4096 and 16384 tokens, each default is the
fastest tile of the sweep, or within 0.1 percent of it. At 256 tokens the
32-row tile is 4.5 percent faster than the default tile in the sweep, but
only 0.8 percent faster than the same default in the first table. That is
within the noise of eager timing, so the kernel has no 32-row tier.

Two earlier runs on the same day failed in the unit tests. The first found
wrong results in the warp-specialized version: the next K step wrote the
registers that the previous `wgmma` was still reading. ADR-038 describes
the fault and the SASS check that now finds it without a GPU. The second
run failed on a test with an odd number of K steps, which the kernel hands
to the Triton kernel.

### Expert tiles in the Triton kernel

The Triton kernel's split for many rows, with row tiles that span experts
against tiles of one expert each (the update of
[ADR-037](../decisions/ADR-037-block-fp8-grouped-kernel.md)). One process
per GPU on 2026-10-22 (`--fp8-expert-tiles`, raw output for the
[B200](data/moe-grouped-fp8-expert-tiles-b200.txt) and the
[H100](data/moe-grouped-fp8-expert-tiles-h100.txt)). The B200 run first
passed 63 unit tests (the 27 Gluon tests skip there), the H100 run 90. Both
splits use the default tiles of their GPU, so on the B200 the expert tiles
up to 1024 tokens ran with the earlier medium tile (64, 128, 64, 4, 4).

Eager milliseconds per forward. On the H100, "Gluon" is the kernel of
ADR-038:

| tokens | B200 row tiles | B200 expert tiles | speedup | H100 row tiles | H100 expert tiles | H100 Gluon |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4.372 | 3.078 | 1.42x | 4.145 | 3.819 | 3.945 |
| 256 | 4.187 | 3.406 | 1.23x | 5.225 | 4.741 | 4.823 |
| 1024 | 5.188 | 4.019 | 1.29x | 8.097 | 6.402 | 5.382 |
| 2048 | 7.009 | 5.249 | 1.34x | 10.991 | 8.389 | 7.203 |
| 4096 | 10.107 | 8.428 | 1.20x | 16.612 | 12.662 | 10.105 |
| 8192 | 16.467 | 14.370 | 1.15x | 24.852 | 22.098 | 17.058 |
| 16384 | 28.888 | 26.763 | 1.08x | 42.911 | 40.452 | 32.116 |

The bf16 layer on `grouped_mm` took 5.116, 7.026 and 19.146 ms at 1024,
4096 and 16384 tokens on the B200, and 8.126, 12.853 and 31.010 ms on the
H100. The outputs of the two splits are bit-identical on both GPUs.

On the H100, the expert tiles give 60 to 70 percent of the Gluon kernel's
gain over the row tiles at 1024 to 4096 tokens, and 23 percent at 16384
tokens. At 64 and 256 tokens they are 3 and 2 percent faster than the
Gluon kernel, which is within the noise of eager timing between runs (the
Gluon kernel took 4.173 and 3.945 ms at 64 tokens in two runs).

Each projection alone at 16384 tokens, useful TFLOPS:

| GPU | projection | row tiles | expert tiles | Gluon |
|---|---|---:|---:|---:|
| B200 | gate-up | 480 | 548 | - |
| B200 | down | 462 | 510 | - |
| H100 | gate-up | 299 | 328 | 518 |
| H100 | down | 299 | 308 | 486 |

Profile at 4096 tokens: on the B200 the kernel took 7.587 ms with row tiles
and 6.018 ms with expert tiles, of 10.023 and 8.600 ms GPU time per
forward. The tile count adds small GPU operations: 87 against 63 per
forward.

The expert-tile sweep on the B200 (eager milliseconds per forward, all with
swapped operands):

| tile | 256 tok | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|
| 64, 128, 64, 4, 4 | 3.491 | 3.880 | 10.460 | 35.872 |
| 64, 256, 64, 8, 3 | 3.402 | 3.739 | 9.657 | 32.903 |
| 128, 128, 64, 4, 4 | 4.130 | 4.494 | 9.010 | 29.911 |
| 128, 256, 64, 8, 3 | 4.211 | 4.590 | 8.442 | 26.879 |
| 32, 256, 64, 8, 3 | 3.147 | 4.543 | 13.780 | 50.336 |

The raw output has all 10 tiles of each GPU. On the B200, (64, 256, 64, 8,
3) is now the tile for 257 to 8192 rows: the fastest at 1024 tokens, and
within 9 percent of the fastest at 256 tokens. The tile for more rows,
(128, 256, 64, 8, 3), stays the fastest at 4096 and 16384 tokens. On the
H100 the sweep confirms the existing Hopper tiles.

Even with expert tiles, the B200 kernel reaches only about 550 useful
TFLOPS at 16384 tokens, a fraction of the B200's bf16 tensor-core peak.
In the sm100 code that Triton makes for this kernel, the dequantized
weight tile goes into one tensor-memory buffer, which `tcgen05` reads. With
one buffer, the dequantization of a K step waits until the previous
multiply is done. The row tile goes through registers into shared memory,
without TMA, and no warp is specialized.

