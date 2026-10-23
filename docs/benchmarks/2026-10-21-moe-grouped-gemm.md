# Grouped expert matmuls for MoE layers on H100 and B200

Date: 2026-10-21
Hardware: one NVIDIA H100 80GB HBM3 and one NVIDIA B200, bf16, PyTorch
2.14.1 with CUDA 13.0, Triton 3.8.0
Workloads: one MoE layer with random weights at the sizes of Nemotron 3.5
Lightning 30B-A3B, Mixtral 8x7B, GLM-5.3 (with bf16 and with block-FP8
experts) and DeepSeek-V4-Flash (with bf16 and with FP4 experts). The real `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
(revision `a9904d2`), with prefill on `scripts/data/technical_passage.md`
Engine: mini-infer with the grouped expert paths of
[ADR-036](../decisions/ADR-036-grouped-expert-matmuls.md) (bf16),
[ADR-037](../decisions/ADR-037-block-fp8-grouped-kernel.md) (block FP8),
[ADR-038](../decisions/ADR-038-gluon-fp8-grouped-hopper.md) (block FP8 in
Gluon on the H100) and
[ADR-039](../decisions/ADR-039-gluon-fp8-grouped-blackwell.md) (block FP8
in Gluon on the B200),
[ADR-040](../decisions/ADR-040-fp4-grouped-kernel.md) (FP4) and
[ADR-041](../decisions/ADR-041-gluon-fp4-grouped.md) (FP4 in Gluon)
Script: `scripts/modal_moe_grouped_bench.py` (`--fp8-only`, `--fp8-kernel`,
`--fp8-tiles`, `--fp8-gluon` and `--fp8-expert-tiles` for the FP8 runs,
`--fp4` and `--fp4-gluon` for the FP4 runs)

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
  faster than the Triton row tiles from 1024 tokens. A row tile can hold
  the rows of several experts. The FP8 layer is then 1.26 times faster
  than the bf16 layer at 4096 tokens and equal at 16384 tokens. The
  section "Gluon kernel for Hopper" has the details.
- In the Triton kernel, tiles of one expert each (expert tiles) make the
  FP8 layer 1.08 to 1.42 times faster than row tiles on the B200. On the
  H100 they make it 1.06 to 1.31 times faster. The section "Expert tiles
  in the Triton kernel" has the details.
- `tcgen05` is the tensor-core instruction of Blackwell GPUs such as the
  B200. On the B200, a Gluon kernel for `tcgen05` makes the FP8 layer 1.22
  to 1.26 times faster than the Triton expert tiles from 4096 tokens. The
  FP8 layer is then 1.04 times faster than the bf16 layer at 4096 tokens
  and 1.15 times slower at 16384 tokens. The section "Gluon kernel for the
  B200" has the details.
- DeepSeek-V4-Flash keeps its experts in FP4. A Triton kernel makes the
  V4-Flash layer 20 to 66 times faster than the per-expert loop up to 1024
  tokens on the H100. On the B200 it is 28 to 50 times faster. The FP4 layer
  is then faster than the bf16 layer up to 1024 tokens on both GPUs. It is
  1.28 (H100) and 1.55 (B200) times slower at 16384 tokens. On the real
  V4-Flash with two B200 GPUs, prefill of 2021 tokens is 7.1 times faster.
  The section "FP4 experts" has the details.
- Gluon kernels for the FP4 experts decode one weight tile while the
  tensor cores multiply the previous one. At 16384 tokens the FP4 layer is
  then 1.07 (H100) and 1.13 (B200) times slower than the bf16 layer. At
  2048 and 4096 tokens it is 1.09 to 1.24 times faster. The section "Gluon
  kernels for FP4" has the details.
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
each GPU (`--fp8-tiles`). Each run first passed 39 unit tests on its GPU.

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
from 0.92 to 0.40 ms (B200), in the profiles of the `--fp8-kernel` runs.

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
rows. The table gives the tiles of the current code. Each default is the
fastest tile of the sweeps above at its sizes, or within 5 percent of it:

| rows (tokens at top-8) | H100 (Hopper) | B200 (Blackwell) |
|---|---|---|
| up to 256 (32) | 16, 64, 128, 4, 3 | 16, 128, 64, 4, 3 |
| up to 8192 (1024) | 64, 128, 128, 4, 3 | 64, 256, 64, 8, 3 |
| above | 128, 256, 64, 8, 3 | 128, 256, 64, 8, 3 |

At first the B200 tile for 257 to 8192 rows was (64, 128, 64, 4, 4).
After the expert-tile sweep in "Expert tiles in the Triton kernel", we
replaced it with (64, 256, 64, 8, 3).

### W8A8, tried and removed

A W8A8 mode of the kernel quantized the activations per token and per 128
columns, as DeepSeek, vLLM and transformers do, and multiplied e4m3 by e4m3
on the FP8 tensor cores. Both scales then multiplied each block's fp32
partial sum. It was measured on 2026-10-22 against the W8A16 kernel in one
process per GPU, and then removed.

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

This section compares the Gluon kernel of
[ADR-038](../decisions/ADR-038-gluon-fp8-grouped-hopper.md) with the
Triton kernel of the sections above (row tiles for many rows). We ran one
process on one H100 on 2026-10-22 (`--fp8-gluon`). Before the timings,
the run passed 84 unit tests on the GPU. They include the Gluon kernel
against fp32 math, against the Triton kernel, in a CUDA graph, and under
Triton's concurrency sanitizer. The sanitizer stops on a shared-memory
race or a deadlock.

Eager milliseconds per forward. "Gluon" uses the default tile for the size
and a copy warp. A copy warp is one extra warp that only issues the TMA
copies (warp specialization). "No copy warp" is the same tile without it.
The bench script forces the Gluon kernel on at every size, also at 32
tokens, where the default path keeps the Triton kernel:

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

The next table times each projection alone, with random routing over the
256 experts. It gives the median milliseconds of `triton.testing.do_bench`
and the useful TFLOPS (`2 * rows * N * K` per call):

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

In the profile at 4096 tokens, the Triton kernel takes 13.113 ms of
16.071 ms GPU time per forward. The Gluon kernel takes 7.025 ms of
10.200 ms. The other kernels of the layer (routing, gather, scatter,
shared expert) take about the same time in both runs: 3.0 and 3.2 ms.

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

The kernel picks the token rows of a tile from the mean rows per expert. A
tile has 64 token rows up to 64 mean rows (1024 and 2048 tokens), 128 up
to 192 (4096 tokens), and 256 above. At 1024, 4096 and 16384 tokens, each
default is the fastest tile of the sweep, or within 0.1 percent of it. At
256 tokens the tile with 32 token rows is 4.5 percent faster than the
default tile in the sweep. But it is only 0.8 percent faster than the same
default in the first table. That is within the noise of eager timing, so
the kernel has no tier with 32 token rows.

Two earlier runs on the same day failed in the unit tests. The first found
wrong results in the warp-specialized version. A K step is one tile of the
shared dimension K of the matmul. The next K step wrote registers while
the previous `wgmma` still read them. ADR-038 describes the fault and a
check of the SASS (the machine code of the GPU) that now finds it without
a GPU. The second run failed on a test with an odd number of K steps, a
shape that the kernel hands to the Triton kernel.

### Expert tiles in the Triton kernel

This section compares two splits of the Triton kernel for many rows. A
row tile holds sorted rows that can span several experts. An expert tile
holds the rows of one expert (the update of
[ADR-037](../decisions/ADR-037-block-fp8-grouped-kernel.md)). We ran one
process per GPU on 2026-10-22 (`--fp8-expert-tiles`).

The B200 run first passed 63 unit tests (the 27 Gluon tests skip there),
and the H100 run passed 90. Both splits use the default tiles of their
GPU. So on the B200, the expert tiles up to 1024 tokens ran with the
earlier tile (64, 128, 64, 4, 4).

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

On the B200, the bf16 layer on `grouped_mm` took 5.116, 7.026 and
19.146 ms at 1024, 4096 and 16384 tokens. On the H100 it took 8.126,
12.853 and 31.010 ms. The outputs of the two splits are bit-identical on
both GPUs.

On the H100 at 1024 to 4096 tokens, the expert tiles give 60 to 70 percent
of the Gluon gain over the row tiles. At 16384 tokens they give 23
percent. In this `--fp8-expert-tiles` run, the expert tiles are 3 and 2
percent faster than the Gluon kernel at 64 and 256 tokens. That is within
the noise of eager timing between runs. At 64 tokens the Gluon kernel took
4.173 ms in the first `--fp8-gluon` run and 3.945 ms in this run. A later
`--fp8-gluon` run measured 4.285 ms.

That later run on the H100 used the expert tiles as the Triton default
(99 GPU tests passed). In that run, the Gluon kernel was 1.17 to 1.33 times faster
than the Triton kernel from 1024 tokens. The Gluon kernel took 5.552 ms
against 6.645 ms at 1024 tokens, 9.911 against 12.683 at 4096, and 29.881
against 39.789 at 16384.

Each projection alone at 16384 tokens, useful TFLOPS:

| GPU | projection | row tiles | expert tiles | Gluon |
|---|---|---:|---:|---:|
| B200 | gate-up | 480 | 548 | - |
| B200 | down | 462 | 510 | - |
| H100 | gate-up | 299 | 328 | 518 |
| H100 | down | 299 | 308 | 486 |

In the profile at 4096 tokens on the B200, the kernel took 7.587 ms with
row tiles and 6.018 ms with expert tiles. The GPU time per forward was
10.023 and 8.600 ms. The tile count adds small GPU operations: 87 against
63 per forward.

The expert-tile sweep on the B200 (eager milliseconds per forward, all with
swapped operands):

| tile | 256 tok | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|
| 64, 128, 64, 4, 4 | 3.491 | 3.880 | 10.460 | 35.872 |
| 64, 256, 64, 8, 3 | 3.402 | 3.739 | 9.657 | 32.903 |
| 128, 128, 64, 4, 4 | 4.130 | 4.494 | 9.010 | 29.911 |
| 128, 256, 64, 8, 3 | 4.211 | 4.590 | 8.442 | 26.879 |
| 32, 256, 64, 8, 3 | 3.147 | 4.543 | 13.780 | 50.336 |

On the B200, (64, 256, 64, 8,
3) is now the tile for 257 to 8192 rows. It is the fastest at 1024 tokens
and within 9 percent of the fastest at 256 tokens. The tile for more rows,
(128, 256, 64, 8, 3), stays the fastest at 4096 and 16384 tokens. On the
H100 the existing Hopper tiles stay the fastest in the sweep.

Even with expert tiles, the B200 kernel reaches only about 550 useful
TFLOPS at 16384 tokens. That is a fraction of the bf16 tensor-core peak of
the B200. In the sm100 code that Triton makes for this kernel, the
dequantized weight tile goes into one buffer in tensor memory. Tensor
memory is a memory next to the tensor cores, and `tcgen05` reads its left
operand from there.

With one buffer, the dequantization of a K step waits until the previous
multiply is done. The tile of token rows goes through registers into
shared memory, without TMA. Triton also gives all warps the same work.

### Gluon kernel for the B200

This section compares the `tcgen05` kernel of
[ADR-039](../decisions/ADR-039-gluon-fp8-grouped-blackwell.md) with the
Triton kernel and its expert tiles. We ran one process on one B200 on
2026-10-22 (`--fp8-gluon`).
The run first passed 82 unit tests on the GPU (the 26 sm90 tests skip
there). They include the Gluon kernel against fp32 math, against the
Triton kernel, in a CUDA graph, and under Triton's concurrency sanitizer.

Eager milliseconds per forward. The bench script forces the Gluon kernel on
at every size. The default path calls it only above 64 mean rows per
expert (from 4096 tokens here):

| tokens | Triton expert tiles | Gluon | Gluon tile | speedup | bf16 |
|---:|---:|---:|---|---:|---:|
| 32 | 2.232 | 2.539 | 128, 64, 64, 8, 4 | 0.88x | |
| 64 | 3.212 | 3.112 | 128, 64, 64, 8, 4 | 1.03x | |
| 256 | 3.671 | 3.615 | 128, 64, 64, 8, 4 | 1.02x | |
| 1024 | 4.020 | 4.045 | 128, 64, 64, 8, 4 | 0.99x | 5.108 |
| 2048 | 5.457 | 5.659 | 128, 64, 64, 8, 4 | 0.96x | |
| 4096 | 8.266 | 6.583 | 128, 128, 128, 4, 3 | 1.26x | 6.870 |
| 8192 | 14.330 | 11.762 | 128, 128, 128, 4, 3 | 1.22x | |
| 16384 | 26.549 | 21.113 | 128, 128, 128, 4, 3 | 1.26x | 18.413 |

The "bf16" column is the same layer with bf16 experts on `grouped_mm`, in
the same run. On sm100 the fourth number of a tile counts the
dequantization warps. The outputs of the two kernels are bit-identical at
every size.

The next table times each projection alone. It gives the median
milliseconds of `do_bench` and the useful TFLOPS:

| tokens | projection | Triton | Gluon | speedup | Triton TFLOPS | Gluon TFLOPS |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | gate-up | 1.957 | 1.997 | 0.98x | 211 | 206 |
| 1024 | down | 1.047 | 1.037 | 1.01x | 197 | 199 |
| 4096 | gate-up | 3.951 | 3.055 | 1.29x | 417 | 540 |
| 4096 | down | 2.205 | 1.548 | 1.42x | 374 | 533 |
| 16384 | gate-up | 11.803 | 8.580 | 1.38x | 559 | 769 |
| 16384 | down | 6.335 | 4.345 | 1.46x | 521 | 759 |

In the profile at 4096 tokens, the Triton kernel takes 5.665 ms of
8.206 ms GPU time per forward. The Gluon kernel takes 4.099 ms of
6.746 ms. The Gluon path also skips the GPU operations that count the
expert tiles (63 against 87 per forward).

Gluon tiles, eager milliseconds per forward:

| tile | 256 tok | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|
| 128, 64, 64, 4, 4 | 3.792 | 4.220 | 9.348 | 31.587 |
| 128, 64, 64, 8, 4 | 3.729 | 4.179 | 9.155 | 30.695 |
| 128, 128, 64, 4, 4 | 4.068 | 4.550 | 7.401 | 23.180 |
| 128, 128, 128, 4, 3 | 3.809 | 4.250 | 6.906 | 21.104 |
| 128, 128, 128, 8, 3 | 3.744 | 4.199 | 6.865 | 21.317 |
| 128, 256, 64, 4, 3 | 5.574 | 6.166 | 7.494 | 20.783 |
| 128, 256, 128, 4, 2 | 5.986 | 6.630 | 7.906 | 21.764 |

The default above 64 mean rows was (128, 128, 128, 4, 3), and it has 4
stages since the profile below. It is within 1
percent of the fastest tile at 4096 tokens and within 2 percent at 16384
tokens. The tile with 256 token rows has one accumulator, so the store of
a tile stalls the next multiply. It is the fastest at 16384 tokens but 9
percent slower at 4096.

An earlier run on the same day used 2-byte stores and other default tiles. It also passed all GPU tests. With the
same tile, the 16-byte stores of the final version are 1 to 6 percent
faster from 4096 tokens.

### Rerun after the review fixes

A code review changed the Gluon kernels in two ways that reach the GPU.
The sm90 multiply warps and the sm100 dequantization warps now run a fence
before they free a stage. The item walk, the store and the setup moved
into shared helpers. We ran `--fp8-gluon` again on 2026-10-22, on one B200
and on one H100.

- The B200 passed 87 GPU tests and the H100 passed 116. New tests give
  each program many items, also under the concurrency sanitizer.
- On both GPUs, the outputs of the Gluon and Triton kernels are
  bit-identical at every size.
- On the B200, the Gluon layer took 6.902, 11.800 and 21.385 ms at 4096,
  8192 and 16384 tokens. Before, it took 6.583, 11.762 and 21.113 ms. In
  the profile at 4096 tokens, the kernel took 4.059 ms, against 4.099 ms
  before.
- Alone at 16384 tokens, the B200 projections reached 762 and 737 TFLOPS,
  against 769 and 759 before. The down projection is 3 percent slower.
  This can come from the fence or from the noise between two runs.
- On the H100, the Gluon layer took 10.326 and 30.708 ms at 4096 and 16384
  tokens. The later run before took 9.911 and 29.881 ms. Alone, the gate-up
  projection at 16384 tokens reached 521 TFLOPS, against 495 in the first
  run.
- No fix touched the bf16 layer, and it still moved by up to 3 percent
  between the same runs. For example, it took 18.989 against 18.413 ms on
  the B200 at 16384 tokens.

### Profile of the B200 kernel and a fourth stage

On 2026-10-22 we profiled both projections at 16384 tokens with Nsight
Compute 2025.3.1 on one B200. The container cannot
lock the GPU clocks, so the profile ran with `--clock-control none`. The
Gluon kernel had the tile (128, 128, 128, 4, 3):

| gate-up projection | Gluon | Triton expert tiles | bf16 `grouped_mm` |
|---|---:|---:|---:|
| duration (ms) | 7.44 | 10.68 | 6.18 |
| SM clock (GHz) | 1.66 | 1.83 | 1.12 |
| tensor cores busy | 49% | 31% | 96% |
| issue slots busy | 40% | 37% | 4% |
| DRAM busy | 16% | 41% | 60% |
| L2 hit rate | 80% | 33% | 44% |

The stall samples of the Gluon kernel show where its warps wait. The copy
warp waited for a free stage in about 90 percent of its time. The
dequantization warps waited for a copy in about 18 percent of their time.
The MMA warp waited for a dequantized weight tile in about half of its
time. A stage stays busy from its copy through the dequantization and the
multiply, so 3 stages were too few.

On 2026-10-23 we tried two fixes on one B200. The first is a fourth
stage. The second gives the weight tiles and the tiles of token rows rings
of their own. Nsight
Compute counted the cycles of each projection, which do not depend on the
clock. A fourth stage saved 12 to 14 percent of the cycles. The best split
rings saved less than 1 percent more, so only the fourth stage stays
([ADR-039](../decisions/ADR-039-gluon-fp8-grouped-blackwell.md), update).

Then a `--fp8-gluon` run with the new default tile passed all 87 GPU tests.
Eager
milliseconds per forward:

| tokens | Triton expert tiles | Gluon, 4 stages | speedup | bf16 |
|---:|---:|---:|---:|---:|
| 4096 | 8.229 | 6.601 | 1.25x | 6.958 |
| 8192 | 14.336 | 11.024 | 1.30x | |
| 16384 | 26.566 | 20.656 | 1.29x | 19.160 |

The FP8 layer is 1.05 times faster than the bf16 layer at 4096 tokens and
1.08 times slower at 16384 tokens. Before the fourth stage it was 1.15
times slower. The tile sweep of the same run compares 3 and 4 stages. The
fourth stage made the layer 7.7 percent faster at 4096 tokens (6.962 to
6.428 ms). At 16384 tokens it made the layer 3.6 percent faster (20.926 to
20.166 ms). The outputs stayed bit-identical to the Triton kernel.

More tensor-core work also lowers the clock. In the cycle counts, the clock
fell from 1.66 to about 1.50 GHz, so fewer cycles gave a smaller gain in
time. bf16 `grouped_mm` keeps its tensor cores busy 96 to 97 percent of the
time at 1.1 to 1.3 GHz. Both kernels multiply in bf16, so the power limit
holds the FP8 layer near the speed of the bf16 layer.

## FP4 experts

DeepSeek-V4-Flash keeps its routed experts in FP4: e2m1 values, two per
byte, with one scale per row and 32 columns. The layer has hidden size
4096, expert width 2048, 256 experts, top-6 routing with sqrt-softplus
scores, and one shared expert. The FP4 weights are random, with
power-of-two scales as in the checkpoint.

"Loop" is the per-expert loop of the V4 block. It dequantizes each expert
that got tokens to bf16. "Grouped" is the Triton kernel of
[ADR-040](../decisions/ADR-040-fp4-grouped-kernel.md). "bf16" is the same
layer with bf16 experts on `grouped_mm`. We ran one process per GPU on
2026-10-23 (`--fp4`). The H100 run first passed 144 GPU tests, and the B200
run passed 115.

H100, milliseconds per forward:

| tokens | loop | grouped | speedup | grouped, graph | bf16 | bf16, graph |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.779 | 0.891 | 19.95x | 0.199 | 0.981 | 0.293 |
| 4 | 33.982 | 0.918 | 37.04x | 0.403 | 1.058 | 0.632 |
| 16 | 96.809 | 1.474 | 65.68x | 1.073 | 2.110 | 1.669 |
| 64 | 208.922 | 3.607 | 57.93x | 3.052 | 4.117 | 3.740 |
| 256 | 264.826 | 4.434 | 59.73x | | 5.246 | |
| 1024 | 266.679 | 4.992 | 53.42x | | 5.734 | |
| 4096 | 266.385 | 7.391 | 36.04x | | 7.621 | |
| 16384 | 272.767 | 25.582 | 10.66x | | 20.012 | |

B200, milliseconds per forward:

| tokens | loop | grouped | speedup | grouped, graph | bf16 | bf16, graph |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.533 | 0.340 | 28.05x | 0.218 | 0.347 | 0.223 |
| 4 | 17.395 | 0.449 | 38.73x | 0.328 | 0.529 | 0.389 |
| 16 | 45.322 | 0.898 | 50.48x | 0.775 | 1.096 | 0.998 |
| 64 | 103.182 | 2.185 | 47.21x | 2.026 | 2.323 | 2.180 |
| 256 | 132.314 | 2.736 | 48.36x | | 3.265 | |
| 1024 | 133.651 | 3.067 | 43.58x | | 3.595 | |
| 4096 | 134.342 | 5.370 | 25.02x | | 4.901 | |
| 16384 | 139.285 | 18.568 | 7.50x | | 11.981 | |

The loop does not fit in a CUDA graph. The FP4 layer is faster than the
bf16 layer up to 1024 tokens on both GPUs, in a CUDA graph by 1.02 to 1.57
times. At 4096 tokens it is 1.03 times faster on the H100 and 1.10 times
slower on the B200. At 16384 tokens it is 1.28 (H100) and 1.55 (B200)
times slower. The outputs of the two paths differ by a relative 2.6e-3 to
2.8e-3.

These runs used the earlier tile tables. The section "FP4 splits and
tiles" gives the two tiles that changed after the sweep. In the sweep, the
new tiles are 1.4 to 5.1 percent faster.

In the profile at 4 tokens, the loop ran 2434 to 2529 GPU operations and
325 to 328 host syncs per forward. The grouped layer ran 46 operations and
no host syncs. At 2048 tokens the loop ran about 14,370 operations and 1024
host syncs, and the grouped layer 84 or 85 operations and none. There the
kernel took 4.64 ms of 6.01 ms of GPU time on the H100, and 3.34 ms of
4.49 ms on the B200.

### First and second FP4 decode

The first version of the kernel computed each bf16 value with a compare
and select for the small e2m1 codes. It also loaded the scale of each
weight on its own. In a run earlier on 2026-10-23 with the same tiles, it
made the FP4 layer 3 to 10 times slower than the bf16 layer. The second
version (ADR-040, decision 2) loads one scale per row and 32 columns, and
it decodes with bit moves into an fp32.

Milliseconds per forward, in a CUDA graph up to 64 tokens and eager from
256 tokens:

| tokens | H100 first | H100 second | speedup | B200 first | B200 second | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.931 | 0.199 | 4.68x | 1.158 | 0.218 | 5.31x |
| 4 | 3.050 | 0.403 | 7.57x | 3.255 | 0.328 | 9.92x |
| 16 | 11.188 | 1.073 | 10.43x | 9.551 | 0.775 | 12.32x |
| 64 | 40.034 | 3.052 | 13.12x | 26.112 | 2.026 | 12.89x |
| 256 | 51.571 | 4.434 | 11.63x | 34.351 | 2.736 | 12.55x |
| 1024 | 52.295 | 4.992 | 10.48x | 34.766 | 3.067 | 11.34x |
| 4096 | 47.004 | 7.391 | 6.36x | 32.518 | 5.370 | 6.06x |
| 16384 | 160.159 | 25.582 | 6.26x | 112.896 | 18.568 | 6.08x |

We compiled both versions for sm90 without a GPU, with the argument facts
of a V4 launch (unit strides, 16-byte alignment). Then we counted the
instructions of the K loop, per thread and K step:

| tile | version | instructions | registers | spill reloads | global loads |
|---|---|---:|---:|---:|---:|
| 128, 256, 64, 8, 3 | first | 1450 | 255 | 62 | 72 |
| 128, 256, 64, 8, 3 | second | 532 | 219 | 0 | 8 |
| 16, 64, 128, 4, 3 | first | 1319 | 254 | 0 | 70 |
| 16, 64, 128, 4, 3 | second | 533 | 80 | 0 | 6 |

64 of the first version's global loads per step are scale loads, one per
weight. The time fell more than the instruction count, up to 13 times
against 2.7 times. The decode tile did not spill in the first version. It
still got 4.7 to 13 times faster, so the spills are not the main cause. We
did not profile the share of the scale loads.

### FP4 splits and tiles

Prefill sweep, eager milliseconds per forward. "expert" is one tile of one
expert's rows, "row" a row tile across experts, "segment" the decode split:

| tile, split | H100 256 tok | 1024 | 4096 | 16384 | B200 256 tok | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64, 128, 64, 4, 4, expert | 4.454 | 4.973 | 9.898 | 32.836 | 2.779 | 3.122 | 6.603 | 22.444 |
| 64, 128, 128, 4, 3, expert | 4.522 | 5.119 | 10.216 | 33.334 | 3.176 | 3.525 | 7.418 | 24.480 |
| 128, 128, 64, 4, 4, expert | 5.331 | 5.832 | 8.349 | 27.629 | 3.642 | 3.961 | 5.243 | 19.156 |
| 128, 128, 128, 8, 3, expert | 6.159 | 6.648 | 8.626 | 28.747 | 4.086 | 4.411 | 5.727 | 19.861 |
| 64, 256, 64, 8, 3, expert | 4.442 | 4.998 | 9.858 | 31.716 | 2.970 | 3.284 | 6.951 | 22.993 |
| 64, 256, 128, 8, 3, expert | 4.855 | 5.472 | 10.864 | 34.583 | 3.511 | 3.847 | 8.054 | 26.344 |
| 128, 256, 64, 8, 3, expert | 5.385 | 5.979 | 7.515 | 25.450 | 3.756 | 4.079 | 5.370 | 18.528 |
| 128, 256, 128, 8, 3, expert | 5.859 | 6.486 | 8.158 | 27.354 | 3.603 | 3.929 | 5.286 | 18.157 |
| 128, 256, 64, 8, 3, row | 7.423 | 6.814 | 10.502 | 27.742 | 5.557 | 4.904 | 7.643 | 19.926 |
| 64, 128, 64, 4, 4, segment | 4.413 | 5.508 | 13.180 | 45.555 | 2.911 | 4.882 | 15.897 | 60.478 |

With the same tile, the expert tiles are 1.09 to 1.40 times faster than
the row tiles on the H100. On the B200 they are 1.08 to 1.48 times faster.

Two tiles changed. On the H100, the tile for 257 to 8192 rows is now
(64, 128, 64, 4, 4). At 256 and 1024 tokens it is 1.5 and 2.9 percent
faster than (64, 128, 128, 4, 3). The H100 and the B200 now share it. On
the B200, the decode tile is now (16, 64, 64, 4, 3). At 1, 4 and 16 tokens
it is 1.4 to 5.1 percent faster than (16, 128, 64, 4, 3). The tile for more than 8192
rows, (128, 256, 64, 8, 3), stays on both GPUs. It is the fastest at 4096
and 16384 tokens on the H100 and within 2.5 percent of the fastest on the
B200.

### Real model: DeepSeek-V4-Flash on 2x B200

On 2026-10-23 we loaded the real `deepseek-ai/DeepSeek-V4-Flash` on two
B200 GPUs with tensor parallelism, 128 experts per GPU
(`scripts/modal_v4_flash_generate.py --compare-paths`). That image had
PyTorch 2.10.0 and Triton 3.6.0. The script loads the model once and runs
both MoE paths on it. All 43 MoE layers ran on the FP4 kernel, and the two
GPUs generated the same tokens.

Eager milliseconds, without a CUDA graph:

| workload | loop | grouped | speedup |
|---|---:|---:|---:|
| prefill of 2021 tokens of `scripts/data/technical_passage.md` | 3286.9 | 463.7 | 7.09x |
| prompt "The capital of France is", then 32 greedy tokens | 12912.4 | 4176.6 | 3.09x |

The second row includes the prefill of the 6-token prompt.

Last-token logits, grouped path against the loop:

| prompt | cosine | max abs diff | same top token |
|---|---:|---:|---|
| "The capital of France is" (6 tokens) | 0.99976 | 0.875 | yes |
| the passage (2021 tokens) | 0.99935 | 1.6875 | yes |

The greedy outputs match for 20 tokens. Both write "Paris. It is one of
the most famous cities in the world. Paris is known for its beautiful".
Then the grouped path writes "architecture, art, and culture" and the loop
writes "landmarks like the Eiffel Tower". The two paths add in a different
order, so their bf16 roundings differ. We did not measure the logit gap
at the token where they part. On Nemotron, the bf16 paths of ADR-036 had a
prefill logit cosine of 0.9994 to 0.9995.

### Gluon kernels for FP4

On 2026-10-23 we profiled the FP4 Triton kernel at 16384 tokens with
Nsight Compute, on one H100 and one B200. The profile timed both
projections with random routing, as the layer runs them. Then we built the
Gluon kernels of [ADR-041](../decisions/ADR-041-gluon-fp4-grouped.md) and
ran `--fp4-gluon` three times on the B200 and twice on the H100.

Both projections alone at 16384 tokens, before the Gluon kernels.
Milliseconds are CUDA-event times. Cycles and busy shares come from Nsight
Compute. The B200 lowers its clock under tensor load, so cycles compare
kernels better than milliseconds:

| GPU | kernel | ms | M cycles | tensor cores busy | issue slots busy |
|---|---|---:|---:|---:|---:|
| B200 | Triton, default tile | 11.60 | 21.1 | 22 to 23% | 44 to 46% |
| B200 | Triton, tile 256, 128, 64, 8, 4 | 9.16 | 15.8 | 33 to 36% | 44 to 46% |
| B200 | bf16 `grouped_mm` | 4.79 | 6.5 | 95 to 97% | 5 to 8% |
| H100 | Triton, default tile | 15.63 | 28.2 | 37 to 39% | 41% |
| H100 | bf16 `grouped_mm` | 9.85 | 14.6 | 83 to 86% | 11 to 12% |

The Triton kernel decodes a weight tile and then multiplies it, and one
program fills an SM. On the H100 no tile of the sweep beat the default.

The first Gluon version made the layer 1.25 (B200) and 1.17 (H100) times
faster than Triton at 16384 tokens. Its profile showed one limit on each
GPU:

| GPU | first Gluon kernel | M cycles | tensor cores busy | main wait |
|---|---|---:|---:|---|
| B200 | one dequantization warpgroup | 13.1 to 14.2 | 34 to 43% | the MMA warp waits for decoded tiles |
| H100 | decode into the `wgmma` register operand | 19.9 | 61 to 64% | single-byte shared-memory loads |

On the B200 the dequantization warps almost never waited, so one warpgroup
decoded too slowly. On the H100 the operand layout scatters a thread's
bytes, so the decode issued 33 single-byte loads per K step. The second
version added a second dequantization warpgroup on the B200. On the H100
it decodes 16 bytes per thread into shared memory, and the `wgmma` reads
the tile from there. Its outputs stayed bit-identical to the Triton kernel
on both GPUs. The final runs passed 197 GPU tests on the H100 and 167 on
the B200.

The layer with the final kernels, eager milliseconds per forward:

| tokens | H100 Triton | H100 Gluon | H100 bf16 | B200 Triton | B200 Gluon | B200 bf16 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4.178 | 4.287 | 4.996 | 2.985 | 2.622 | 3.197 |
| 1024 | 4.592 | 4.798 | 5.528 | 3.344 | 2.950 | 3.638 |
| 2048 | 6.111 | 5.334 | 6.184 | 4.654 | 3.315 | 4.124 |
| 4096 | 7.332 | 6.901 | 7.521 | 5.424 | 4.041 | 4.949 |
| 8192 | 14.105 | 13.234 | 10.471 | 10.258 | 7.759 | 6.433 |
| 16384 | 25.783 | 20.743 | 19.343 | 18.386 | 13.651 | 12.040 |

The default path takes the Gluon kernel on the B200 at all these sizes. On
the H100 it keeps the Triton kernel up to 32 mean rows per expert, so at
256 and 1024 tokens. At 16384 tokens the FP4 layer is now 1.07 (H100) and
1.13 (B200) times slower than the bf16 layer. With the Triton kernel it was
1.32 and 1.57 times slower.

Both projections alone at 16384 tokens with the final kernels (CUDA-event
milliseconds):

| kernel | H100 | B200 |
|---|---:|---:|
| Triton, default tile | 15.967 | 11.622 |
| Triton, tile 256, 128, 64, 8, 4 | 16.444 | 9.316 |
| Gluon, default tile | 11.222 | 7.004 |
| bf16 `grouped_mm` | 9.153 | 4.709 |

The Gluon tile sweep, eager milliseconds per forward. On the B200 `warps`
counts the dequantization warps (4 is one warpgroup, 8 is two):

| GPU | tile | 1024 tok | 4096 | 16384 |
|---|---|---:|---:|---:|
| H100 | 128, 64, 64, 4, 4 | 4.863 | 9.923 | 32.872 |
| H100 | 128, 128, 64, 8, 4 | 5.233 | 6.867 | 23.879 |
| H100 | 128, 128, 128, 8, 3 | 5.285 | 6.923 | 24.243 |
| H100 | 128, 256, 64, 8, 3 | 6.908 | 8.362 | 20.815 |
| H100 | 128, 256, 64, 8, 4 | 6.591 | 8.137 | 21.005 |
| B200 | 128, 128, 128, 4, 4 | 3.323 | 4.396 | 14.690 |
| B200 | 128, 128, 64, 8, 4 | 3.550 | 4.690 | 15.492 |
| B200 | 128, 128, 128, 8, 4 | 3.069 | 4.175 | 14.068 |
| B200 | 128, 256, 64, 8, 3 | 4.394 | 5.589 | 14.105 |
| B200 | 128, 256, 64, 8, 4 | 4.246 | 5.528 | 13.835 |

On the B200, two dequantization warpgroups are 1.04 to 1.08 times faster
than one with the tile (128, 128, 128). The B200 table now takes that tile
up to 192 mean rows per expert and (128, 256, 64, 8, 4) above. The H100
table keeps its tiles.

At 16384 tokens the layer is compute-bound. Both layers multiply in bf16,
and the FP4 layer also decodes its weights. So these kernels can approach
the bf16 layer at large batches but not pass it. On the B200, bf16
`grouped_mm` keeps its tensor cores busy 98 percent of the time.

