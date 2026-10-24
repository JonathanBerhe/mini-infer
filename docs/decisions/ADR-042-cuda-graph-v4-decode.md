# ADR-042: CUDA-graph decode for DeepSeek-V4

Date: 2026-10-24
Status: Accepted

## Context

A profile of the real DeepSeek-V4-Flash on 2x B200
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md)) showed that
the batch-1 decode step is bound by the host. A step ran about 8,400 small
kernels. The GPU was busy for about 32 ms of a 120 to 142 ms step, and it
waited for the host for the rest.

[ADR-033](ADR-033-cuda-graph-state-cache-decode.md) replays the decode step
of a state-cache model from a CUDA graph, but a graph capture of the V4
ragged step failed. It had these problems:

- Host syncs: `bool(flush_mask.any())` and `int(write_slots.max())` before each
  block write, `int(positions.max())` for the width of the compressed
  history, a boolean-mask copy in the compressor, and `torch.tensor(-1,
  device=...)` copies in the index builders.
- Shapes that change from step to step: each layer sliced its compressed
  history to the widest valid prefix, and the indexer took
  `min(top_k, n_compressed_max)` blocks.
- On GPUs without grouped matmuls, the MoE blocks run their experts one at a
  time, and that loop syncs to find the tokens of each expert.

## Decision

1. **Make the ragged decode step static.** Each CSA and HCA layer attends over
   its whole compressed history buffer, and masks the blocks that a row has
   not reached. The indexer scores the whole buffer and takes
   `min(top_k, buffer size)` blocks. `write_flushed_blocks` writes the blocks
   of a step with a masked select: every row writes, a row that does not
   flush writes back the old entry, and the slot is clamped to the buffer.
   The compressor slides its overlap state with `torch.where`. The index
   builders use a scalar `-1`. The block RoPE is computed once per
   compression ratio instead of once per layer.
2. **Split the checks from the step.** `decode_step_static` has no host sync
   and no data-dependent shape. `forward_decode_with_cache_ragged` checks the
   shapes and the capacity on the host (`StateCache.check_capacity`, one sync
   per step), then calls `decode_step_static`. The two are the same step, so
   the eager scheduler and the graph run the same kernels.
3. **Opt in where the step is capture-safe.** V4 sets
   `SUPPORTS_DECODE_CUDA_GRAPH`. A new optional method,
   `can_capture_decode(device)`, is true only when every MoE block runs its
   experts as grouped matmuls (`HashRoutedMoEFFN.runs_grouped`).
   `supports_decode_cuda_graph(model, device)` checks it, so the HTTP server
   graphs V4 on Hopper and newer GPUs and keeps the eager step elsewhere.
4. **Keep the single-request path.** `forward_decode_with_cache`, which
   `StateCacheGenerator` drives, does not change. It still slices the history
   to the valid prefix.
5. **Graph the two-GPU server too.**
   `TensorParallelStateCacheContinuousServer(cuda_graph=True)` captures the
   step on every rank when it is built, before any prefill, and replays it for
   each decode op. The constructor runs on every rank in the same order, so
   the all-reduces inside the capture line up. Shutdown drops the graph on
   every rank, because a live graph that holds NCCL work can hang the process
   group teardown. The flag is off by default.

## Alternatives Considered

- **One graph per flush pattern on the single-request path.** CSA layers
  flush every 4 tokens and HCA layers every 128, so three graphs cover every
  step. But the compressed history still grows by one block per flush,
  so the step still needs a fixed width. The schedulers also run the ragged
  step, where rows flush at different steps. One static ragged step serves
  every batch size and both paths.
- **A graph per history width.** It keeps the attention narrow, but the width
  changes every 4 tokens, so it needs many captures and their memory.
- **`torch.compile(mode="reduce-overhead")`.** The reasons of ADR-033 hold:
  a long compile for 43 layers, and the syncs split the graph anyway.

## Consequences

Measured on 2x B200, real V4-Flash, batch 1, greedy, in one run
(`scripts/modal_v4_flash_generate.py --profile-decode --cuda-graph`):

| Context | Eager ms/step | CUDA graph ms/step | Speedup |
|---|---:|---:|---:|
| 6 tokens | 67.8 | 34.2 | 1.98x |
| 2,021 tokens | 78.5 | 34.7 | 2.26x |

- The graph tokens equal the eager tokens for all 33 tokens at both
  contexts. `tests/unit/test_v4_cuda_graph_decode.py` pins graph-equals-eager
  and the absence of host syncs on CUDA, with dense, bf16-expert and
  FP4-expert models.
- The eager step depends on the host CPU of the machine. In the first
  profile run it took 120.0 and 142.0 ms. The graph step does not depend on
  it: it varies by 0.1 ms over 16 steps.
- At batch 1 the graph decodes about 29 tokens per second, against 15 for
  the eager step of the same run.
- The static step runs about 11,900 GPU operations (kernels and copies) and
  35 ms of GPU work, against about 9,000 operations and 29 ms for the eager
  single-request step. The full
  `torch.cat` of the window and the history in each layer adds about 2.5 ms.
  The row-indexed writes and the masks add the rest. One buffer per layer for
  the window and the history, and fewer index kernels, are the next levers,
  then the kernel fusion of the report.
- The eager ragged step now does the full-width work as well. With a large
  `max_seq_len` and short requests, it spends more GPU time than before. At
  batch 1 the eager step is bound by the host, so the extra work hides there.
- A captured graph holds NCCL work. If a process destroys its process group
  while such a graph is alive, the teardown can hang. Free the graphs first.

The two-GPU server, measured on 2x B200 in one run
(`scripts/modal_v4_flash_cb_bench.py --cuda-graph`), 63 timed greedy decode
steps per batch size:

| Batch | Eager ms/step | CUDA graph ms/step | Speedup | Tokens per second |
|---:|---:|---:|---:|---:|
| 1 | 172.8 | 35.7 | 4.84x | 5.8 to 28.0 |
| 8 | 170.3 | 48.7 | 3.50x | 47 to 164 |
| 16 | 169.8 | 57.3 | 2.96x | 94 to 279 |

- The graph tokens equal the eager tokens at every batch size.
- The eager step of the server costs about 170 ms at every batch size: the
  host sets it. It is slower than the eager single-request step, because the
  ragged step runs more kernels and every step also broadcasts its inputs to
  the follower rank.
- The graph step grows by about 1.4 ms per extra request. Each request picks
  6 experts, so a larger batch reads more expert weights. This is an
  inference from the numbers, not a profile.
