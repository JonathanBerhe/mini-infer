# Kimi Linear decode profile and the absorbed MLA decode

- Date: 2026-11-06
- Hardware: Apple M1 Pro, 10 CPU cores (8 threads used), 16 GB, PyTorch CPU
- Model: Kimi-Linear-48B-A3B layer widths with random weights, fp32 (hidden
  2304, 32 MLA heads, `kv_lora_rank` 512, 32 KDA heads of 128, experts of
  1024). The step profile uses a 4-layer slice (3 KDA layers, 1 MLA layer,
  layer 0 dense, 32 experts, a vocabulary of 32,000) so that it fits 16 GB.
- Workload: one decode step after a prefill, greedy, batch 1 unless the
  table says otherwise
- Engine: main at 4ab26ed against the branch of
  [ADR-057](../decisions/ADR-057-kimi-mla-absorbed-decode.md)
- Script: `scripts/kimi_linear/profile_kimi_decode.py`

## Question

Kimi Linear 48B decodes at 2.1 tokens per second on one B200
([model metrics](2026-11-05-model-metrics.md)). How does a decode step split
between the MLA layers, the KDA layers and the MoE? Does the MLA cost grow
with the capacity of the state buffer (`max_seq_len`) or with the history?
How much does the absorbed MLA decode save?

## Result

- The MLA decode cost grew with the history, not with the buffer capacity.
  The old step read the buffer up to the end of its longest row. At 1,024
  history tokens one layer took 12.6, 12.6 and 12.4 ms at capacities of
  2,048, 8,192 and 32,768.
- The absorbed decode makes one MLA layer 2.7x faster at 1,024 history
  tokens (13.1 to 4.9 ms) and 20.7x faster at 16,384 (141.9 to 6.9 ms). A
  ragged batch of 8 with a longest row of 4,096 tokens goes from 507.4 to
  14.6 ms.
- After the change the layer time is almost flat in the history. The fixed
  projections (`q_proj`, `kv_a_proj_with_mqa`, `o_proj`) take most of it.
  Inside a full decode step at 8,192 history tokens, the MLA layer goes
  from 70.4 to 6.3 ms, and it reads 1 value back to the host instead of 2.
- On the CPU the KDA layers take more than 95 percent of the whole step,
  because PyTorch runs their depthwise convolution as one small convolution
  per channel (12,288 calls per layer and step). This is a CPU-only effect.
- On a GPU the largest item is very likely the MoE, not the MLA. The MoE
  block runs one `torch.where`, a host sync on a GPU, for each of the 256
  experts in each of the 26 MoE layers. That is about 6,700 syncs per step
  (an estimate from the op count, not measured on a GPU).

## One MLA layer

One `_KimiMlaAttention` decode call, median of 16 calls. "Before" is main.
"Decompressed" and "absorbed" are the two forms of this branch on one set of
weights.

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

In the ragged rows, row `b` of 8 holds `history * (b + 1) / 8` tokens.

## Decode step by part of the model

The 4-layer slice: 3 KDA layers, 1 MLA layer, 3 MoE layers of 32 experts
and 1 dense MLP. The parts are host time per step under the profiler, the
mean of 4 profiled steps. The step column is the median of 16 steps without
the profiler.

| History | Code | Step (ms) | KDA, 3 layers (ms) | MLA, 1 layer (ms) | MoE, 3 layers (ms) |
|---:|---|---:|---:|---:|---:|
| 1,024 | main | 1,625 | 1,680.6 | 12.7 | 30.8 |
| 1,024 | absorbed | 2,219 | 2,101.9 | 6.9 | 41.7 |
| 8,192 | main | 1,751 | 1,783.3 | 70.4 | 28.8 |
| 8,192 | absorbed | 1,228 | 1,379.4 | 6.3 | 29.2 |

The KDA time changes by up to 40 percent between runs of the same code, so
the step column does not show the MLA change. The MLA column does.

Operations and host reads per step, from the same profile:

| Part | ATen ops | Host reads |
|---|---:|---:|
| KDA, 3 layers | 390 | 0 |
| MLA, 1 layer, main | 48 | 2 |
| MLA, 1 layer, absorbed | 49 | 1 |
| MoE, 3 layers of 32 experts | 564 | 102 |

Each MoE layer reads 34 values back to the host: one `aten::nonzero` per
expert and 2 more. For the real model, 26 MoE layers of 256 experts, that is
about 6,700 host reads per step (an estimate from this count).

## What to optimize first

1. The MoE of the Kimi family (estimate). `_KimiMoe.forward` loops over all
   experts and calls `torch.where` on each expert's mask. On a GPU each call
   waits for the device. With 256 experts and 26 MoE layers that is about
   6,700 syncs per step. At 20 to 60 microseconds each, that is 130 to
   400 ms of the 476 ms step that the B200 measured (an estimate). The
   grouped expert path of `MoEFFN` (`PackedSwigluExperts`) has no per-expert
   sync.
2. The KDA layers on the CPU only. The decode step of the short convolution
   can use `causal_conv1d_step` from `blocks/kda.py` instead of a depthwise
   `conv1d`. This matters for CPU runs, not for the GPU targets.
3. The MLA decode on a GPU. After this change, the next step is a kernel
   that reads only the filled positions of each row, for example the
   FlashInfer MLA decode, whose head sizes (512 and 64) match Kimi.

## How it is measured

- `mla`: one MLA layer with random weights and a random state buffer. The
  call writes the same slot at every step, so the history stays fixed.
  Two warm-up calls, then the median of 16 timed calls.
- `split`: a prefill of the history in chunks of 1,024 tokens, 4 warm-up
  steps, 16 timed steps, then 4
  steps under `torch.profiler` with every module labelled
  (`mini_infer.bench.step_profile`). The host time of a part is the time
  inside its labelled ranges. The op count is the number of top-level ATen
  ops, which stands in for the kernel launches of a GPU run. A host read is
  one `aten::_local_scalar_dense` or `aten::nonzero` op. On a GPU each one
  is a sync.

## Limits

- CPU only. The CPU is compute-bound, while a GPU decode step of this model
  is bound by kernel launches and host syncs. So the CPU speedups of one
  layer do not predict the GPU speedup of a step.
- The whole-step wall time on the CPU varies by up to 2x between runs,
  because the slow KDA convolution dominates it. Only the per-part times and
  the single-layer table compare well.
- Random weights. The time does not depend on the values. The unit and
  parity tests compare the outputs.

## Reproduce

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
