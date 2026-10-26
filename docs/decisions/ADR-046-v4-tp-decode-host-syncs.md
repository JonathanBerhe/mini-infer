# ADR-046: Fewer host syncs in the V4 tensor-parallel decode loop

Date: 2026-10-24
Status: Accepted

## Context

The two-GPU DeepSeek-V4-Flash server
(`TensorParallelStateCacheContinuousServer`) replays its decode step from a
CUDA graph ([ADR-042](ADR-042-cuda-graph-v4-decode.md)). The host work
between two replays had many host syncs. A host sync is a point where the
CPU waits for the GPU to finish its queued work:

- The scheduler called `sample()` once per row, and each call ended in
  `.item()`: 16 syncs per step at batch 16.
- The leader sent each op with `dist.broadcast_object_list` on the NCCL
  group: about 2 syncs on the leader and 3 on the follower.
- The graph copied its inputs from pageable host memory: 2 more syncs per
  rank.

DeepSeek's DeepInfer engine samples all rows in one GPU kernel and keeps
control traffic off NCCL. This ADR ports the cheap parts.

## Decision

1. Measure first. `DecodeStepTimer` records CUDA events around each replay
   and reads them later, so it adds no sync. The host gap is the device
   time from the end of one replay to the start of the next.
2. Sample the batch at once. `sample_batch(logits, params)` samples every
   row on the device and copies the ids to the host once. A greedy row
   takes the argmax, as `sample()` does.
3. Send each op as one fixed-shape int64 message on a gloo group
   (`get_cpu_group()`), so no GPU stream waits for it.
4. Copy the inputs from pinned host memory with `non_blocking=True`.

## Alternatives Considered

- A fused Triton sampling kernel, as in DeepInfer. Greedy `sample_batch`
  runs one argmax kernel, so the gain is small.
- Fixed-shape tensor broadcasts on NCCL. The follower must still read the
  op back from the GPU (a sync).
- Keep the sampled tokens on the GPU. The scheduler needs them on the host
  to stream text and stop at EOS.

## Consequences

Measured on 2x B200, real V4-Flash, in one container
(`scripts/deepseek_v4/modal_v4_flash_cb_bench.py --host-gap`): first commit e18778b,
then this change. Each batch size serves greedy requests of 64 tokens. The
worst gap is the largest of the 63 decode steps, and the host gap is their
median.

| Batch | Host gap (ms) | Worst gap (ms) |
|---:|---:|---:|
| 1 | 0.65 → 0.41 | 4.61 → 0.59 |
| 8 | 1.19 → 0.56 | 4.41 → 0.76 |
| 16 | 1.65 → 0.65 | 2.13 → 0.99 |

- The graph replay was already 96% of the step at batch 16. There the
  change takes 1.0 ms off the gap, so the step is 2.5% faster (388.8 to
  398.7 tokens per second).
- The replay time does not change. It is not comparable with ADR-045,
  which timed another script and workload.
- A steady decode step now has one host sync on the leader and none on the
  follower. Prefill and the eager step keep their syncs.
- Greedy tokens do not change. The two-rank gloo tests compare the server
  with the single-process generator.
- A sampled row uses another random stream than `sample()`, so a seed gives
  other tokens. `tests/unit/engine/test_sampler.py` tests the filters and the
  draw distribution.
- The follower now waits under the gloo timeout (30 minutes by default),
  not the NCCL timeout (10 minutes). In both cases, a server that is idle
  for longer can lose its follower.
