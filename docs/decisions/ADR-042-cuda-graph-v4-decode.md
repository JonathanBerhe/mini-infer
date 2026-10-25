# ADR-042: CUDA-graph decode for DeepSeek-V4

Date: 2026-10-24
Status: Accepted

## Context

A profile of the real DeepSeek-V4-Flash on 2x B200
([report](../benchmarks/2026-10-23-v4-flash-decode-profile.md)) showed that
the host bounds the batch-1 decode step. A step ran about 8,400 small
kernels, and the GPU was busy for about 32 ms of a 120 to 142 ms step.

[ADR-033](ADR-033-cuda-graph-state-cache-decode.md) replays a state-cache
decode step from a CUDA graph. The V4 ragged step failed to capture,
because it had host syncs such as `bool(flush_mask.any())` and shapes that
changed per step. Without grouped matmuls, its MoE loop also syncs once per
expert.

## Decision

1. Make the ragged decode step static. Each CSA and HCA layer attends over
   its whole compressed history buffer and masks the blocks that a row did
   not reach. `write_flushed_blocks` writes with a masked select.
2. Move the host tests out of the step. `decode_step_static` has no host
   sync and no data-dependent shape. `forward_decode_with_cache_ragged`
   tests the capacity on the host (`StateCache.check_capacity`, one sync
   per step), then calls it. So the eager scheduler and the graph run the
   same kernels.
3. Opt in where the step is capture-safe. V4 sets
   `SUPPORTS_DECODE_CUDA_GRAPH`, and `can_capture_decode(device)` is true
   only when every MoE block runs grouped matmuls
   (`HashRoutedMoEFFN.runs_grouped`). So the HTTP server graphs V4 only on
   Hopper and newer GPUs.
4. The single-request path (`forward_decode_with_cache`) does not change.
5. `TensorParallelStateCacheContinuousServer(cuda_graph=True)` captures the
   step on every rank at build time, before any prefill, so the all-reduces
   line up. The flag is off by default.

## Alternatives Considered

- One graph per flush pattern or per history width. The width changes
  every 4 tokens, so these still need a fixed width or many captures.
- `torch.compile(mode="reduce-overhead")`. As in ADR-033, the compile is
  long for 43 layers, and the syncs split the graph anyway.

## Consequences

Real V4-Flash on 2x B200, batch 1, greedy
(`scripts/modal_v4_flash_generate.py --profile-decode --cuda-graph`):

| Context | Eager ms/step | CUDA graph ms/step | Speedup |
|---|---:|---:|---:|
| 6 tokens | 67.8 | 34.2 | 1.98x |
| 2,021 tokens | 78.5 | 34.7 | 2.26x |

- Graph and eager tokens are equal for all 33 tokens.
  `tests/unit/test_v4_cuda_graph_decode.py` pins graph-equals-eager and no
  host syncs on CUDA.
- The static step does more GPU work than the eager single-request step
  (about 35 against 29 ms).
- A first version of this ADR wrongly gave about 2.5 ms to the full
  `torch.cat` of the window and the history in each layer.
  [ADR-044](ADR-044-fused-partial-rope.md) measured the cat at about
  0.14 ms per step at batch 16. The partial RoPE helper, at 3.6 to 5.2 ms
  per step, was the larger lever.
- Shutdown drops the graph on every rank first, because a live graph that
  holds NCCL work can hang the process group teardown.

The two-GPU server (`scripts/modal_v4_flash_cb_bench.py --cuda-graph`), 63
steps per batch size:

| Batch | Eager ms/step | CUDA graph ms/step | Speedup |
|---:|---:|---:|---:|
| 1 | 172.8 | 35.7 | 4.84x |
| 8 | 170.3 | 48.7 | 3.50x |
| 16 | 169.8 | 57.3 | 2.96x |

- Graph and eager tokens are equal at every batch size.
- The graph step grows by about 1.4 ms per extra request.
- `scripts/modal_v4_flash_cb_serve.py` turns the graph on, and `--eager`
  turns it off. 6 concurrent HTTP requests of 48 tokens each finished in
  2.75 s, all with HTTP 200.
