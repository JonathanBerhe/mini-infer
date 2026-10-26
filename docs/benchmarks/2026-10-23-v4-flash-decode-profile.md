# Where one DeepSeek-V4-Flash decode step spends its time on 2x B200

Date: 2026-10-23
Hardware: 2x NVIDIA B200, tensor parallel over 2 ranks, CUDA 12.8 runtime,
NCCL 2.27.5
Model: `deepseek-ai/DeepSeek-V4-Flash` (43 layers: 21 CSA, 20 HCA, 2 SWA),
routed experts in FP4 on the grouped path of
[ADR-040](../decisions/ADR-040-fp4-grouped-kernel.md)
Workload: batch 1, greedy decode, eager (no CUDA graph). Two contexts: the
6-token prompt "The capital of France is", and the first 2,021 tokens of
`scripts/data/technical_passage.md`
Engine: mini-infer at `d9587f0`
Script: `scripts/deepseek_v4/modal_v4_flash_generate.py --profile-decode`

## Question

The grouped FP4 experts made the real model's MoE layers fast. A decode
token still took about 127 ms, and the MoE layers explain only 10 to 15 ms
of that. This report measures where the rest of the step goes, and ranks
what to optimize first.

## Result

- The decode step is bound by the host, not by the GPU. A step takes 120 ms
  of wall time at the 6-token context and 142 ms at the 2,021-token context.
  The GPU is busy for about 32 ms of it at both contexts.
- One step runs about 8,400 kernels and 344 copies. The host spends 13 to
  16 µs on each GPU operation (Python, the PyTorch dispatcher and the
  launch). The GPU waits for the host for about 75 percent of the step.
- The GPU time itself is mostly tiny kernels. The median kernel takes 2.4 µs.
  About 7,600 kernels per step take less than 5 µs each, and together they
  take 19 ms of the 32 ms.
- Attention takes 43 percent of the GPU time and 4,400 kernels per step. The
  routed experts take 21 percent, hyper-connections 15 percent, and NCCL
  3 percent.
- The FP4 grouped kernel takes 3.6 ms per step, which is 3 percent of the
  wall time. It is not the bottleneck of decode today.
- A CUDA graph of the full decode step is the first thing to do. It removes
  the host time from the step, which takes it from 120 ms to about 33 to
  36 ms (an estimate).
- Measured on 2026-10-24 ([ADR-042](../decisions/ADR-042-cuda-graph-v4-decode.md)):
  the step replays from a CUDA graph in 34.2 ms (6 tokens) and 34.7 ms
  (2,021 tokens), with the same tokens as the eager step. The section "CUDA
  graph of the decode step" has the details.

## Per-step time

| Measure | 6-token context | 2,021-token context |
|---|---:|---:|
| Wall time, no profiler (median of 16 steps) | 120.0 ms | 142.0 ms |
| Wall time, no profiler (min to max) | 112.7 to 136.0 ms | 136.7 to 160.3 ms |
| GPU busy | 31.2 ms | 31.6 ms |
| GPU busy on a CSA flush step | 34.2 ms | 34.6 ms |
| Kernels | 8,362 | 8,423 |
| Kernels on a CSA flush step | 9,454 | 9,515 |
| Copies (memcpy) | 344 | 344 |
| Host syncs | 3 | 3 |
| Host syncs on a CSA flush step | 24 | 24 |

A CSA layer compresses every 4 tokens into one block. On the step that closes
a block (a flush step), every CSA layer does extra work. One step in 4 is a
flush step. An HCA layer flushes every 128 tokens, and such a step has 44
host syncs.

Both ranks measure the same wall time to within 0.6 ms, because every step
all-reduces across them.

## GPU time by part of the model

Milliseconds per step. The kernel counts include copies and are averaged over
4 profiled steps, one of them a flush step. The host time is the time the
host spends inside each part, measured under the profiler. The profiler
slows the host by about 1.8 times (223 ms against 120 ms per step), so only
the proportions of that column are useful.

| Part | GPU ms, 6 tokens | GPU ms, 2,021 tokens | Share | Kernels | Host ms under profiler |
|---|---:|---:|---:|---:|---:|
| Attention (CSA, HCA, SWA, indexer, compressors) | 13.79 | 14.28 | 43% | 4,401 | 104.7 |
| MoE routed experts | 6.78 | 6.69 | 21% | 1,290 | 28.5 |
| Hyper-connections | 4.74 | 4.74 | 15% | 1,477 | 34.5 |
| MoE gate | 1.81 | 1.79 | 6% | 556 | 9.6 |
| MoE shared expert | 1.70 | 1.70 | 5% | 301 | 11.6 |
| Norms (input and post-attention RMSNorm) | 1.62 | 1.62 | 5% | 696 | 10.7 |
| NCCL (108 all-reduces, 1 other collective) | 1.03 | 1.05 | 3% | 109 | 0.0 |
| Rest (lm_head, RoPE, sampling, embedding, MoE glue) | 0.49 | 0.50 | 2% | 197 | 23.1 |
| Total | 31.97 | 32.37 | | 9,026 | 222.7 |

The norms inside attention (`q_a_layernorm`, `kv_norm`) count as attention.
In the CSA and HCA layers they add about 1.5 ms and 660 kernels per step.

The routed experts split into two parts:

- The FP4 grouped kernel: 3.57 ms per step, 86 launches of 41.5 µs (two per
  layer).
- The routing glue around it (sort, clamp, index, scatter, fills): 3.1 ms
  per step over about 1,200 kernels.

## Top kernels

Per step, 6-token context. Each name is the kernel without its template
arguments, with the PyTorch op that launched it.

| Kernel (op) | Part | GPU ms | Launches |
|---|---|---:|---:|
| `elementwise_kernel` (`aten::mul`) | attention | 3.78 | 1,281 |
| `_block_fp4_grouped_mm_kernel` | routed experts | 3.60 | 86 |
| `unrolled_elementwise_kernel` (`aten::copy_`) | attention | 2.83 | 835 |
| `nvjet_tst_8x64_64x16` (`aten::mm`) | attention | 2.07 | 231 |
| `vectorized_elementwise_kernel` (`aten::add`) | attention | 1.44 | 728 |
| `reduce_kernel` (`aten::mean`) | attention | 1.13 | 314 |
| `vectorized_elementwise_kernel` (`aten::copy_`) | attention | 1.05 | 554 |
| `reduce_kernel` (`aten::sum`) | hyper-connections | 1.03 | 291 |
| `nvjet_tst_32x64_64x16` (`aten::mm`) | attention | 1.00 | 107 |
| `nvjet_tst_16x64_64x16` (`aten::mm`) | shared expert | 0.98 | 86 |

## Host syncs

A host sync makes the host wait until the GPU finishes all queued work. The
CUDA sync debug mode found these sources, per step:

| Source | Syncs per step | What it does |
|---|---:|---|
| the decode loop (as in `StateCacheGenerator`) | 1 | copies the new token id to the GPU |
| `models/deepseek_v4/deepseek_v4.py:1070` | 1 | copies the token position to the GPU |
| `engine/sampler.py:26` | 1 | `.item()` of the greedy argmax |
| `models/deepseek_v4/deepseek_v4.py:1096` | 5.25 (short), 10.25 (long) | copies the position of a flushed block to the GPU, once per flushing layer |

The copies to the GPU sync because they read from pageable host memory. The
syncs cost almost nothing today (0.0 to 0.3 ms per step), because the GPU is
idle when the host reaches them. They matter for a CUDA graph, which cannot
contain a sync.

## What to optimize first

The savings are estimates from this profile, not measurements.

1. A CUDA graph of the full decode step. A graph replays all kernels of the
   step with one launch, so the step takes about its GPU-busy time. That is
   120 to 142 ms down to about 33 to 36 ms per token, about 3.5 to 4 times
   more tokens per second. `DecodeCudaGraph` in
   `src/mini_infer/engine/cuda_graph_decode.py` already does this for the
   Qwen hybrid (110 ms to 39.6 ms per step). V4 needs three changes:
   - The token position and its RoPE values come from a tensor on the GPU,
     instead of the two host copies above.
   - The compressed history and the indexer top-k keep fixed shapes. They
     read the full buffer with a mask, instead of a slice that grows with
     each flush.
   - One graph per flush pattern: no flush, CSA flush, and CSA plus HCA
     flush. Three graphs are enough, because 128 is a multiple of 4.
2. Fewer and larger kernels, after the graph. With a graph, the GPU-busy
   time is the step time, and most of it is kernels under 5 µs. Examples:
   - RMSNorm as one kernel instead of 8 (about 3.2 ms per step with the
     norms inside attention).
   - Compiled or fused glue code for attention (about 100 kernels per
     layer), hyper-connections (34 per layer) and the gate (13 per layer).
   - The sparse attention core as one kernel instead of about 12.

   The estimate is 32 ms of GPU time down to about 12 to 18 ms.
3. A decode path for the routed experts, after the graph. The FP4 kernel
   reads about 38 MB per layer per GPU at decode: 3 of the 6 active experts,
   12.6 MB each in FP4. At the B200's 8 TB/s, that takes about 5 µs, against
   83 µs measured for the two launches. A kernel for 1 to 8 rows, plus
   routing without the sort and index glue, saves about 5 to 6 ms per step.
4. The shared expert and NCCL. The shared expert is bf16 and every GPU holds
   a full copy (1.7 ms). NCCL takes 1.0 ms. Each offers about 1 ms.

Two kernel options for the FP4 experts do not help decode today:

- A Gluon decode kernel saves at most 3.4 ms of a 120 ms step. After the
  graph it is part of item 3.
- e8m0 (1-byte) scales reduce the bytes that the kernel reads. The kernel
  runs about 18 times slower than its memory-bandwidth limit (83 µs against
  about 5 µs per layer), so fewer bytes do not change its time.

A fused V4 decode attention kernel gained about 0 percent end to end in an
earlier test. That test ran in this same host-bound eager mode. After the
graph, attention GPU time is the largest part of the step, so that result
needs a new measurement.

## CUDA graph of the decode step

Date: 2026-10-24, engine at `6b39ffb` (branch `feat/v4-decode-cuda-graph`).
[ADR-042](../decisions/ADR-042-cuda-graph-v4-decode.md) makes the ragged
decode step capturable. The profile mode with `--cuda-graph` captures that
step in a `DecodeCudaGraph` at batch 1 on both ranks, copies the prefilled
state into the graph's cache, and replays the same number of steps as the
eager run. Both modes ran in the same container.

| Measure | 6-token context | 2,021-token context |
|---|---:|---:|
| Eager step, no profiler (median of 16) | 67.8 ms | 78.5 ms |
| CUDA graph step, no profiler (median of 16) | 34.2 ms | 34.7 ms |
| Speedup | 1.98x | 2.26x |
| Graph step, min to max | 34.2 to 34.3 ms | 34.6 to 34.7 ms |
| Graph step, GPU busy (profiled) | 34.4 to 34.7 ms | 34.6 to 34.8 ms |
| GPU operations per step, eager / graph | 9,026 / 11,881 | 9,087 / 11,882 |
| Tokens equal to the eager step | 33 of 33 | 33 of 33 |

- The eager step took 67.8 ms here and 120.0 ms in the first run of this
  report, with the same code. The host time under the profiler halved
  between the two containers, so the host CPU sets the eager speed. The
  graph step does not depend on it.
- The graph step is now bound by the GPU work. Without the profiler it takes
  34.2 ms, and under the profiler the GPU is busy for 34.4 to 34.7 ms of it.
- The graph runs the ragged decode step, which does more GPU work than the
  eager single-request step: about 2,850 more operations and 6 ms more GPU
  time. Row-indexed writes (`index_elementwise_kernel`, 442 against 89 per
  step) and the masks add most of it. The full `torch.cat` of the sliding
  window and the compressed history in each layer costs little: about
  0.14 ms per step at batch 16 with a 1,024-token cache (see "Partial RoPE"
  below). An earlier version of this report gave it 2.5 ms by mistake.
- At batch 1 the graph decodes about 29 tokens per second, against 15 for
  the eager step of the same run.

The two-GPU continuous-batching server with the graph
(`TensorParallelStateCacheContinuousServer(cuda_graph=True)`), measured with
`scripts/deepseek_v4/modal_v4_flash_cb_bench.py --cuda-graph` in one run. Each batch
size times 63 greedy decode steps after 4 warm-up steps:

| Batch | Eager ms/step | CUDA graph ms/step | Speedup | Tokens per second |
|---:|---:|---:|---:|---:|
| 1 | 172.8 | 35.7 | 4.84x | 5.8 to 28.0 |
| 8 | 170.3 | 48.7 | 3.50x | 47 to 164 |
| 16 | 169.8 | 57.3 | 2.96x | 94 to 279 |

The graph tokens equal the eager tokens at every batch size. The eager step
costs about 170 ms at every batch size, so the host sets it. The graph step
grows by about 1.4 ms per extra request.

## Graph step at batch 16

`scripts/deepseek_v4/modal_v4_flash_generate.py --profile-batch --batch-sizes 1,16` runs
the ragged decode step of the server at batch 1 and 16, with a cache of 1,024
tokens. Row `b` prefills its own slice of the passage, of `24 + 8 * b` tokens.
A graph replay is one launch, so its trace cannot say which part of the model
a kernel belongs to. The graph replays the kernels of the eager step, though,
so the eager step runs with every module labelled to split the GPU time.

| Batch | Graph ms/step | Graph GPU busy | GPU operations per step | Tokens equal to eager |
|---:|---:|---:|---:|---|
| 1 | 35.4 | 35.5 ms | 11,860 | yes |
| 16 | 53.8 | 53.7 ms | 11,882 | yes |

GPU ms per step of the eager step, by part of the model:

| Part | Batch 1 | Batch 16 | Growth |
|---|---:|---:|---:|
| MoE routed experts | 6.41 | 17.43 | +11.03 |
| Unlabelled (ragged attention and layer glue) | 15.32 | 18.66 | +3.34 |
| NCCL collectives | 1.01 | 2.68 | +1.67 |
| Hyper-connections | 4.40 | 5.92 | +1.52 |
| Norms | 1.33 | 2.40 | +1.07 |
| Attention (labelled submodules) | 5.93 | 6.37 | +0.44 |
| MoE gate | 1.60 | 2.02 | +0.42 |
| MoE shared expert | 1.69 | 1.34 | -0.35 |
| Rest | 0.37 | 0.42 | +0.05 |
| Total | 38.06 | 57.24 | +19.18 |

This profile labelled the batched attention methods (`forward_decode_ragged`
and `forward_decode_step_ragged`) only partly, so most attention kernels and
the layer glue count as unlabelled. The script now labels them.

- The routed experts explain 11 ms of the 18 ms growth. The FP4 grouped
  kernel alone goes from 3.6 to 14.3 ms per step (86 launches). At batch 16
  the 96 expert picks of a layer touch at most 48 experts per GPU, about
  600 MB in FP4. At 8 TB/s, 43 layers of that take at most 3.3 ms, so the
  kernel runs at least 4.4 times slower than its memory-bandwidth limit.
  The kernel is now the largest single item of a serving step.
- The NCCL all-reduces grow from 1.0 to 2.7 ms. The MoE output is reduced in
  fp32, so its message grows with the batch.
- The softmax of the HCA compressors costs 2.3 ms per step at every batch
  size: 20 calls of about 116 µs each. The batched step computes the
  compressed block at every step, while an HCA layer closes a block only once
  every 128 tokens. The softmax also runs over the slot dimension, which is
  not the last one, so PyTorch runs it with very few thread blocks.

For serving at batch 8 to 16, these are the largest items in order: the FP4
kernel at a few rows per expert (up to about 11 ms), the partial RoPE
helper (5.2 ms, see "Partial RoPE" below), the HCA compressor softmax
(about 2.3 ms), and the fp32 MoE all-reduce (about 1 ms to gain at batch
16). e8m0 scales still do not help, because the kernel is not bound by its
memory reads.

## FP8 expert inputs at batch 16

[ADR-043](../decisions/ADR-043-w4a8-fp4-experts.md) adds a mode that
quantizes the expert inputs to e4m3 and runs the FP4 matmuls on the
Blackwell tensor cores without the software decode.
`scripts/deepseek_v4/modal_v4_flash_generate.py --compare-fp8-activations` runs both
modes on one load:

| Measure | bf16 inputs | e4m3 inputs |
|---|---:|---:|
| Passage perplexity (2,048 tokens) | 7.925 | 7.845 |
| Graph step, batch 1 | 34.0 ms | 33.2 ms |
| Graph step, batch 16 | 51.7 ms | 45.8 ms |

The two modes give the same 33 greedy tokens and the same next token at
95.1% of the passage positions. The bf16 step at batch 16 is 2.1 ms faster
than in the section above. Between the two runs, the only change to the
bf16 path is the HCA compressor softmax, which now runs over the last
dimension. Above, that softmax cost 2.3 ms per step. The mode is now on by
default on sm100 GPUs such as the B200.

## Partial RoPE

The batch traces of "Graph step at batch 16" also show where the small
kernels of the step come from. `apply_partial_rope_last_n_dims`, which
rotates the last 64 features of each head, runs as separate PyTorch ops: 212
calls and 1,908 kernels per step. They take 3.6 ms of GPU time per step at
batch 1 and 5.2 ms at batch 16. The full-width cat of the window and the
compressed history, by contrast, takes about 0.14 ms at batch 16 with the
1,024-token cache.

[ADR-044](../decisions/ADR-044-fused-partial-rope.md) does each call in
one Triton kernel with the same rounding.
`scripts/deepseek_v4/modal_v4_flash_generate.py --compare-fused-rope` runs both on one
load, with the FP4 experts on W4A8:

| Measure | PyTorch partial RoPE | Fused kernel |
|---|---:|---:|
| Passage perplexity (2,048 tokens) | 7.845 | 7.845 |
| Graph step, batch 1 | 33.5 ms | 30.5 ms |
| Graph step, batch 16 | 46.1 ms | 41.9 ms |

The outputs are bit-identical: the same NLL, the same last logits and the
same 33 greedy tokens.

## RMS norms

The same traces show 1,916 copy kernels per step at batch 16 (5.3 ms),
almost all dtype casts, and about half of them inside RMS norms. A step
runs 235 `RMSNorm` calls of 8 kernels each (3.5 ms at batch 1, 5.2 ms at
batch 16) and 43 per-head q-norms of 7 kernels each (about 0.7 ms).

[ADR-045](../decisions/ADR-045-fused-rms-norm.md) runs each norm as one
Triton kernel. Its sum runs in another order than PyTorch's, so it is not
bit-identical. `scripts/deepseek_v4/modal_v4_flash_generate.py --compare-fused-norm`
runs both on one load, with W4A8 and the fused RoPE:

| Measure | PyTorch norms | Fused kernel |
|---|---:|---:|
| Passage perplexity (2,048 tokens) | 7.845 | 7.874 |
| Graph step, batch 1 | 30.2 ms | 27.1 ms |
| Graph step, batch 16 | 41.9 ms | 37.0 ms |

The perplexity is 0.37% higher, within the 1% bar of ADR-043. The 33
greedy tokens are the same, and the next token agrees at 95.0% of the
passage positions.

## How it is measured

For each context, each rank prefills the context and runs 8 warm-up decode
steps. Then it times 16 decode steps with `time.perf_counter`. A step is one
forward, the greedy sample and the cache advance, as in
`StateCacheGenerator`. The `.item()` of the sample ends each step with a
sync.

Then rank 0 runs 4 steps with `torch.cuda.set_sync_debug_mode("warn")` and
counts the warnings by source line. It profiles 4 more steps with
`torch.profiler` (CPU and CUDA activities, no shapes, no stacks). Rank 1
runs the same steps without the profiler, because every step all-reduces
across both ranks. Each set of 4 steps covers one CSA flush.

For the profile, forward hooks put each module call in its own profiler
range. Wrappers do the same for the decode methods that run outside
`Module.__call__` (`forward_decode`, `forward_decode_step`, `hc_pre`,
`hc_post`, the grouped expert matmuls). Each kernel and copy belongs to the
host call that launched it, through its correlation id. That host call gives
the step, the innermost range (the part of the model) and the innermost
PyTorch op. NCCL kernels count as NCCL whatever range launched them. GPU busy
is the length of the union of all kernel and copy intervals of a step.

Both contexts decode coherent text. The 6-token prompt continues with "Paris.
It is one of the most famous cities in the world." All 43 MoE layers ran on
the packed FP4 experts.

## Limits

- Batch 1 only. Larger batches have the same number of kernels per step, so
  the host share falls as the batch grows.
- The 2,021-token context takes 22 ms more per step without the profiler.
  Under the profiler the two contexts differ by only 0.4 ms of GPU time and
  3.3 ms of wall time. This report does not explain that gap. A CUDA graph
  removes host time in both cases.
- Only rank 0 is profiled. The profiler makes rank 0 the slower rank, so
  rank 1 waits in each all-reduce. The NCCL times of rank 0 show the
  transfer, not the wait.
- The CUDA sync debug mode is a prototype and does not detect every sync.
  The trace agrees with it: 3 sync calls per step, 24 on a flush step.

## Reproduce

```bash
uv run modal run scripts/deepseek_v4/modal_v4_flash_generate.py --profile-decode
```

```bash
uv run modal run scripts/deepseek_v4/modal_v4_flash_generate.py --profile-decode --cuda-graph
```

```bash
uv run modal run scripts/deepseek_v4/modal_v4_flash_cb_bench.py --cuda-graph --batch-sizes 1,8,16
```

```bash
uv run modal run scripts/deepseek_v4/modal_v4_flash_generate.py --profile-batch --batch-sizes 1,16
```

```bash
uv run modal run scripts/deepseek_v4/modal_v4_flash_generate.py --compare-fp8-activations
```

```bash
uv run modal run scripts/deepseek_v4/modal_v4_flash_generate.py --compare-fused-rope
```

```bash
uv run modal run scripts/deepseek_v4/modal_v4_flash_generate.py --compare-fused-norm
```

`--trace-dir <dir>` also saves the raw Chrome traces (about 80 MB each).
Keep them out of the repository.
