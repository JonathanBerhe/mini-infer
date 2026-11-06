# Kimi Linear decode profile and the absorbed MLA decode

- Date: 2026-11-06
- Hardware: one H100 80GB on Modal (CUDA 13.2, torch 2.14.1), bf16. Also an
  Apple M1 Pro CPU (8 threads, 16 GB), fp32.
- Model: Kimi-Linear-48B-A3B layer widths with random weights (hidden 2304,
  32 MLA heads, `kv_lora_rank` 512, 32 KDA heads of 128, experts of 1024).
  The step profile uses a 4-layer slice: 3 KDA layers, 1 MLA layer, layer 0
  dense, 3 MoE layers, a vocabulary of 32,000. The slice has 256 experts on
  the H100 and 32 on the CPU, so that it fits 16 GB.
- Workload: one decode step after a prefill, greedy, batch 1 unless the
  table says otherwise
- Engine: main at 4ab26ed against the branch of
  [ADR-057](../decisions/ADR-057-kimi-mla-absorbed-decode.md)
- Scripts: `scripts/kimi_linear/profile_kimi_decode.py`, and
  `scripts/kimi_linear/modal_kimi_mla_decode.py` for the H100

## Question

Kimi Linear 48B decodes at 2.1 tokens per second on one B200
([model metrics](2026-11-05-model-metrics.md)). How does a decode step split
between the MLA layers, the KDA layers and the MoE? Does the MLA cost grow
with the capacity of the state buffer (`max_seq_len`) or with the history?
How much does the absorbed MLA decode save?

## Result

- The MLA decode cost grew with the history, not with the buffer capacity.
  The old step read the buffer up to the end of its longest row.
- On the H100, the absorbed decode makes one MLA layer 3.0x faster at
  16,384 history tokens (1.42 to 0.48 ms) and 7.3x faster for a ragged batch
  of 8 with a longest row of 4,096 tokens (4.61 to 0.63 ms). At 1,024
  tokens and below it is 0.05 to 0.09 ms slower per layer, because it
  launches 4 more kernels and the layer is then bound by launches, not by
  work.
- On the CPU, where the work sets the time, the absorbed layer is 2.7x
  faster at 1,024 tokens (13.1 to 4.9 ms) and 20.7x faster at 16,384 (141.9
  to 6.9 ms).
- The eager decode step on the H100 is host-bound. The 4-layer slice takes
  30 to 35 ms per step for 8 ms of GPU work. 768 of its 769 host syncs per
  step come from the MoE: one `aten::nonzero` per expert and layer.
- For the real model, 26 MoE layers of 256 experts, that is about 6,700
  syncs per step (an estimate scaled from the slice). The MoE expert loop is
  the main cost of the 2.1 tokens per second, not the MLA layers.

## One MLA layer on the H100

One `_KimiMlaAttention` decode call in bf16, median of 32 calls.
"Decompressed" and "absorbed" are the two forms of this branch on one set of
weights. The decompressed form is the main path without its norm of the
history at every read. The last column is the largest difference between
the outputs of the two forms.

| Batch | History | Capacity | Decompressed (ms) | Absorbed (ms) | Largest difference |
|---|---:|---:|---:|---:|---:|
| 1 | 256 | 32,768 | 0.46 | 0.55 | 2.4e-4 |
| 1 | 1,024 | 32,768 | 0.46 | 0.51 | 2.4e-4 |
| 1 | 4,096 | 32,768 | 0.59 | 0.49 | 1.2e-4 |
| 1 | 16,384 | 32,768 | 1.42 | 0.48 | 1.2e-4 |
| 1 | 1,024 | 2,048 | 0.39 | 0.48 | 2.4e-4 |
| 1 | 1,024 | 8,192 | 0.41 | 0.48 | 2.4e-4 |
| 1 | 1,024 | 32,768 | 0.41 | 0.48 | 2.4e-4 |
| 8, ragged | longest 1,024 | 8,192 | 1.40 | 0.50 | 4.9e-4 |
| 8, ragged | longest 4,096 | 8,192 | 4.61 | 0.63 | 2.4e-4 |

In the ragged rows, row `b` of 8 holds `history * (b + 1) / 8` tokens.

## One MLA layer on the CPU

The same call in fp32, median of 16 calls. "Before" is main.

| Batch | History | Capacity | Before (ms) | Decompressed (ms) | Absorbed (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 256 | 32,768 | 6.16 | 6.57 | 4.86 |
| 1 | 1,024 | 32,768 | 13.12 | 12.24 | 4.94 |
| 1 | 4,096 | 32,768 | 37.23 | 38.69 | 5.51 |
| 1 | 16,384 | 32,768 | 141.91 | 131.84 | 6.86 |
| 1 | 1,024 | 2,048 | 12.57 | 12.35 | 4.74 |
| 1 | 1,024 | 8,192 | 12.56 | 11.94 | 4.70 |
| 1 | 1,024 | 32,768 | 12.40 | 12.05 | 4.85 |
| 8, ragged | longest 1,024 | 8,192 | 112.20 | 115.20 | 8.14 |
| 8, ragged | longest 4,096 | 8,192 | 507.42 | 647.09 | 14.57 |

In fp32 the two forms differ by at most 6e-8.

## Decode step on the H100

The 4-layer slice with 256 experts, bf16. The step column is the median of
16 steps without the profiler. The other columns come from 4 profiled steps:
GPU busy is the union of the kernel times, and the parts are GPU time.

| History | MLA form | Step (ms) | GPU busy (ms) | Kernels | Host syncs | MLA (ms) | KDA, 3 layers (ms) | MoE, 3 layers (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | decompressed | 30.0 | 8.0 | 3,788 | 769 | 0.16 | 0.68 | 6.87 |
| 1,024 | absorbed | 31.1 | 7.9 | 3,792 | 769 | 0.12 | 0.68 | 6.87 |
| 8,192 | decompressed | 35.4 | 8.4 | 3,788 | 769 | 0.66 | 0.68 | 6.86 |
| 8,192 | absorbed | 35.1 | 7.9 | 3,791 | 769 | 0.16 | 0.68 | 6.85 |

The sync sites per step are the same in all four runs: 768 in
`block_sparse_moe` (`aten::nonzero`) and 1 in the MLA layer
(`aten::_local_scalar_dense`, the end of the longest row). The step changes
by about 1 ms between the forms, which is the noise of a host-bound step.

## Decode step on the CPU

The 4-layer slice with 32 experts, fp32. The parts are host time per step,
the mean of 4 profiled steps.

| History | Code | Step (ms) | KDA, 3 layers (ms) | MLA, 1 layer (ms) | MoE, 3 layers (ms) |
|---:|---|---:|---:|---:|---:|
| 1,024 | main | 1,625 | 1,680.6 | 12.7 | 30.8 |
| 1,024 | absorbed | 2,219 | 2,101.9 | 6.9 | 41.7 |
| 8,192 | main | 1,751 | 1,783.3 | 70.4 | 28.8 |
| 8,192 | absorbed | 1,228 | 1,379.4 | 6.3 | 29.2 |

On the CPU, PyTorch runs the depthwise convolution of the KDA layers as one
small convolution per channel (12,288 calls per layer and step). That takes
more than 95 percent of the step and changes by up to 40 percent between
runs, so the step column does not show the MLA change. The MLA column does.
The MLA layer reads 1 value back to the host per step, against 2 on main.

## What to optimize first

1. The MoE of the Kimi family. `_KimiMoe.forward` loops over all experts
   and calls `torch.where` on the mask of each expert, which syncs on a GPU.
   On the H100 the slice spends 30 ms per step for 8 ms of GPU work, with
   768 MoE syncs. The grouped expert path of `MoEFFN`
   (`PackedSwigluExperts`) has no per-expert sync.
2. A decode CUDA graph for the family, after the MoE change. Under a graph
   the launch cost of the absorbed MLA form goes away, and its smaller GPU
   work stays.
3. The KDA layers on the CPU only. The decode step of the short convolution
   can use `causal_conv1d_step` from `blocks/kda.py` instead of a depthwise
   `conv1d`.
4. An MLA decode kernel that reads only the filled positions of each row,
   for example the FlashInfer MLA decode, whose head sizes (512 and 64)
   match Kimi.

## How it is measured

- `mla`: one MLA layer with random weights and a random state buffer. The
  call writes the same slot at every step, so the history stays fixed. Two
  warm-up calls, then the median of the timed calls, with a device sync
  after each call on the GPU.
- `split`: a prefill of the history in chunks of 1,024 tokens, 4 warm-up
  steps, 16 timed steps, then 4 steps under `torch.profiler` with every
  module labelled (`mini_infer.bench.step_profile`). On the GPU each kernel
  belongs to the labelled range that launched it. On the CPU the time of a
  part is the host time inside its ranges, and a host read is one
  `aten::_local_scalar_dense` or `aten::nonzero` op.
- Both forms run in one process on one set of weights, switched with
  `_KimiMlaAttention.absorbed_decode`.

## Limits

- One H100, eager mode, batch 1 for the step. No B200, H200 or B300 run, and
  no run with the real checkpoint.
- The step profile is a 4-layer slice. The numbers for the full 48B are
  estimates scaled from it.
- Random weights. The time does not depend on the values. The unit and
  parity tests compare the outputs.

## Reproduce

```bash
uv run modal run scripts/kimi_linear/modal_kimi_mla_decode.py
```

```bash
uv run python scripts/kimi_linear/profile_kimi_decode.py mla --mla-path absorbed
```

```bash
uv run python scripts/kimi_linear/profile_kimi_decode.py split --mla-path absorbed --prompt 8192 --max-seq-len 16384
```

For the "before" column, put a checkout of 4ab26ed at `/tmp/main` and run
the script of this branch against its package, with the default
`--mla-path current`:

```bash
PYTHONPATH=/tmp/main/src uv run python scripts/kimi_linear/profile_kimi_decode.py mla
```
