# ADR-052: Async sampling and the int8 decode kernel

Date: 2026-11-03
Status: Accepted

## Context

After [ADR-051](ADR-051-paged-decode-fewer-kernels.md) the paged decode step
of Qwen2.5-7B on one H100 at concurrency 8 took 8.1 ms of wall time for 6.3
ms of GPU work. The host read the sampled ids at the end of each step and
built the next step while the GPU idled. The MLP GEMMs (matrix multiplies)
then took about 4 ms of the step at the memory roofline. Int8 weights halve
their bytes, but the old W8A16 kernel (int8 weights, bf16 activations) was
slower than cuBLAS bf16. The numbers are in
[the campaign report](../benchmarks/2026-11-02-paged-decode-chain.md).

## Decision

1. The scheduler samples on the device after each forward and starts the
   copy to the host without a wait. A plain decode step takes its ids from
   that tensor through `DeviceTokens`, launches the forward, and only then
   reads the previous ids and applies the finish conditions. The mode is on
   for CUDA runners, and `async_sampling` turns it off.
2. Up to `_DECODE_ROWS` (64) rows, `fused_w8a16_linear` runs one Triton
   kernel with one program per (column tile, K split). Each program
   dequantizes int8 tiles in registers and stores fp32 partial sums. The
   last program of a column tile sums the splits in a fixed order, applies
   the scale and the bias, and casts. Above 64 rows the layer dequantizes
   the weight and calls cuBLAS, because a prefill chunk is compute-bound.
   `decode_tile_for` picks the tile per GPU architecture.

## Alternatives Considered

- Prepare step t+1 on the host while replay t runs. It hides about 0.4 ms
  of the 2 ms gap. Reading the ids one step late hides all of it.
- One program per column tile in the int8 kernel. The down projection of
  the 7B model gets 56 programs on 132 streaming multiprocessors, so the
  memory system idles.
- Dequantize per call and run cuBLAS for every shape. The round trip
  through HBM burns the byte saving.
- Sum the splits in a second kernel. It adds one launch per linear: 0.32 ms
  of GPU time per step on the H100 and 224 graph nodes.

## Consequences

Measured on Qwen2.5-7B, FlashInfer backend, decode graph on, 32 new tokens,
concurrency 8, each change in its own same-container A/B.

| Change | GPU | Per decode step | tok/s |
|---|---|---|---:|
| Async sampling | H100 | wall 8.1 to 6.5 ms at 6.3 ms of GPU time | 1004 to 1154 |
| Int8, split-K kernel | H100 | GPU 6.27 to 4.64 ms | 1123 to 1310 |
| Int8, Blackwell tiles | B200 | GPU 3.60 to 3.35 ms | 1933 to 1884 |
| Reduce fused into the kernel | B200 | GPU 3.31 to 3.23 ms, graph nodes 684 to 432 | 2163 to 2184 |

- From the eager path of ADR-050 to this ADR, concurrency 8 on the H100
  went from 414 to 1310 tok/s. Async sampling kept the tokens identical to
  host sampling.
- Int8 is lossy. The worst last-token logits cosine against bf16 over the
  eight bench prompts was 0.9972, and four of eight 32-token greedy runs
  matched bf16 to the end. `quant="int8"` stays opt-in.
- The host learns a finished request one step late, so one extra forward
  runs for the last request of a batch. A step that needs the ids on the
  host first, such as a prefill chunk that joins, takes the old path.
- On the B200 the throughput at concurrency 8 moves within the run-to-run
  variance of about 6 percent. The int8 kernel reads about 4 TB/s there
  against 6 TB/s for cuBLAS bf16, so TMA loads are its next step.
- The arrival counters of the fused reduce (`tile_counters`) must exist
  before a graph capture. The eager warm-up of the decode graph allocates
  them.
