# ADR-046: Fewer host syncs in the V4 tensor-parallel decode loop

Date: 2026-10-24
Status: Accepted

## Context

The two-GPU DeepSeek-V4-Flash server
(`TensorParallelStateCacheContinuousServer`) replays its decode step from a
CUDA graph ([ADR-042](ADR-042-cuda-graph-v4-decode.md)). The graph removed
most of the launch cost inside a step. Nobody measured the host work between
two replays, and that work had many host syncs. A host sync is a point where
the CPU waits until the GPU finishes its queued work:

- The scheduler sampled each row with its own `sample()` call, and each call
  ended in `.item()`. At batch 16, that is 16 syncs per step.
- The leader sent each op with `dist.broadcast_object_list` on the NCCL
  group. That call pickles the op and copies it to the GPU. The follower
  reads the size and the bytes back to the host. That is about 2 syncs on
  the leader and 3 on the follower per step.
- The graph copied its inputs from pageable host memory. PyTorch waits for
  the stream after each such copy, so every rank had 2 more syncs per step.

DeepSeek's DeepInfer engine samples all rows in one GPU kernel, keeps
control traffic off NCCL, and times graph replays. This ADR ports the cheap
parts of that design.

## Decision

1. **Measure first.** `DecodeStepTimer` records a CUDA event before and after
   each replay, and reads the events later, so it adds no sync per step. For
   each step it reports the replay time, the host gap, and the wall time
   between the starts of two steps. The host gap is the device time from the
   end of one replay to the start of the next. The server takes
   `time_decode=True`. `scripts/modal_v4_flash_cb_bench.py --host-gap` times
   the real serving loop: greedy requests through
   `TensorParallelStateCacheContinuousScheduler`.
2. **Sample the batch at once.** `sample_batch(logits, params)` samples every
   row on the device and copies the ids to the host once. A greedy row takes
   the argmax, the same id as `sample()`. A sampling row applies the
   temperature, top-k and top-p filters of `sample()` per row, in float32.
   `torch.multinomial` draws one sample as the argmax of `probs / q`, with q
   from an exponential distribution, but it first checks its input with two
   host syncs. `sample_batch` does the same draw without the checks.
3. **Send control messages on a CPU group.** Each op is one int64 message of
   a fixed shape: `[op, slot, prompt length, tokens (B), positions (B)]`. The
   leader broadcasts it on a gloo group (`get_cpu_group()`), so the message
   stays in host memory and no GPU stream waits for it. A prefill sends the
   prompt ids in a second broadcast.
4. **Copy the inputs from pinned memory.** On a CUDA device the message is in
   pinned (page-locked) host memory, and the graph copies its inputs with
   `non_blocking=True`. The pinned-memory allocator of PyTorch does not give
   a buffer out again before the copy from it is done.

## Alternatives Considered

- **A fused Triton sampling kernel**, as in DeepInfer. With greedy decoding,
  `sample_batch` runs one argmax kernel, so a fused kernel saves little. It
  becomes useful if many requests sample with top-k or top-p.
- **Fixed-shape tensor broadcasts on NCCL.** They also remove the pickling.
  But the follower must read the op back from the GPU, which is a sync, and
  the NCCL broadcast waits behind the GPU stream of each rank.
- **Keep the sampled tokens on the GPU** and feed them to the next replay
  without the host. That removes the last sync per step. But the scheduler
  needs the tokens on the host to stream text and to stop at EOS, so this is
  a larger change.

## Consequences

Measured on 2x B200, real V4-Flash, in one container
(`scripts/modal_v4_flash_cb_bench.py --host-gap`). The container ran the
same benchmark twice: first with the package of commit e18778b, which has the
probe but the old sampling and broadcast, then with this change. Each batch
size serves greedy requests of 64 tokens through the scheduler, after an
untimed round. The worst gap is the largest of the 63 decode steps. Each
other value is their median.

| Batch | Host gap, before | Host gap, after | Worst gap, before | Worst gap, after | Step, before | Step, after |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.65 ms | 0.41 ms | 4.61 ms | 0.59 ms | 27.53 ms | 27.37 ms |
| 8 | 1.19 ms | 0.56 ms | 4.41 ms | 0.76 ms | 35.89 ms | 35.39 ms |
| 16 | 1.65 ms | 0.65 ms | 2.13 ms | 0.99 ms | 41.15 ms | 40.13 ms |

- The host gap was already small: the graph replay was 96% of the step at
  batch 16. The change takes 1.0 ms off the gap at batch 16, so the step is
  2.5% faster (388.8 to 398.7 tokens per second). At batch 1 the step is
  0.6% faster.
- The worst gap goes from 4.6 ms to below 1 ms.
- The replay time does not change (39.46 and 39.49 ms at batch 16). The GPU
  work of the step is now the only large part. The replay times are not
  comparable with ADR-045, which timed another script and workload.
- The digest of all generated tokens is the same before and after at every
  batch size.
- This run does not split the remaining 0.4 to 0.65 ms of gap. It holds the
  one host sync, the bookkeeping of the scheduler (it decodes the text of
  each request every step), the gloo broadcast and the input copies.
- A steady decode step now has one host sync on the leader: the copy of the
  sampled ids. The follower has none. The prefill of a new request and the
  eager (no graph) step keep their syncs.
- Greedy decoding gives the same tokens as before, because a greedy row takes
  the same argmax. The two-rank gloo tests compare the server, the scheduler
  and the graph path against the single-process generator.
- A sampled row uses another random stream than `sample()`, so a seed gives
  other tokens than before. `tests/unit/test_sampler.py` checks that the
  top-k and top-p filters equal the filters of `sample()`, and that the draws
  follow the softmax.
- The follower now waits for the next op under the timeout of the gloo
  group, 30 minutes by default. Before, it waited in an NCCL broadcast under
  the NCCL timeout, 10 minutes by default. In both cases, a server that is
  idle for longer than the timeout can lose its follower.
- The single-GPU `StateCacheContinuousScheduler` still samples one row at a
  time. `sample_batch` can replace that loop too.
