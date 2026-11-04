# ADR-054: Run the Gated DeltaNet prefill with its scalar gate

Date: 2026-11-04
Status: Accepted

## Context

The Gated DeltaNet layers of the Qwen hybrid ran their prefill through
`kda_chunkwise`, the chunk form of Kimi Delta Attention
([ADR-032](ADR-032-qwen38-hybrid-port.md)). Kimi decays each key channel on
its own. So inside each 64-token chunk the function loops over the 64
positions and builds each coupling column from small element-wise
operations.

For 4096 tokens one layer call dispatches 78,003 operations, and about
20,000 of them are GPU kernels. The 27B has 48 such layers, so one
4096-token prefill chunk launched about 1M kernels. On an H100 that chunk
took 15.5 s of CPU time against 4.3 s of GPU time, and a 32k prompt took
88 s.

## Decision

1. `gdn_chunkwise` in `qwen/gdn.py` runs the same chunk algorithm for a
   scalar gate. Gated DeltaNet decays a whole head by one number per
   position. So the decay between positions `i` and `c` of a chunk is
   `exp(G_c - G_i)`, with `G` the cumulative log-decay inside the chunk. The
   key-key and query-key couplings are then one matmul times a `(64, 64)`
   decay mask, for all chunks at once.
2. The mask is set to minus infinity above the diagonal before the `exp`,
   where the positive differences can overflow.
3. Only the state hand-off from one chunk to the next stays sequential, at
   about nine kernels per chunk.
4. `_QwenGatedDeltaNet` calls `gdn_chunkwise` for every span longer than one
   token. The one-token step keeps `kda_recurrent` or the fused decode kernel
   of [ADR-034](ADR-034-fused-gdn-decode-kernel.md). `kda.py` does not
   change, so Kimi Linear and Kimi K3 keep their per-channel form. This
   narrows the reuse of ADR-032: the one-token recurrence stays shared, and
   the chunk form does not.
5. A unit test counts the dispatched operations per chunk and fails above 45.

## Alternatives Considered

- A Triton kernel for the whole chunk algorithm, like the
  `chunk_gated_delta_rule` kernels of flash-linear-attention. It also
  removes the launches of the chunk hand-off, but it is a new kernel to test
  on two GPU generations. The PyTorch change removes 97 percent of the
  dispatches first. A profile of the new prefill decides if a kernel is
  worth it.
- A fast path inside `kda_chunkwise` that detects equal channels. It hides a
  model-specific shortcut in a shared function and costs a check on every
  call.
- The product form `(q * exp(G)) @ (k * exp(-G))^T`, which needs no
  `(64, 64)` mask. `exp(-G)` overflows in fp32 when a chunk decays strongly,
  so the mask form stays.

## Consequences

Measured on one H100, Qwen3.8-27B in bf16, prefill in 4096-token chunks of
real prose, with the old and the new form in the same container:

| Tokens | Old | New | Speedup |
|---|---:|---:|---:|
| 4,096 | 10.9 s | 0.8 s | 13.0x |
| 32,768 | 87.9 s | 8.4 s | 10.5x |

- Dispatched operations per layer call for 4096 tokens: 78,003 before,
  2,448 after.
- The top-1 token is the same at all 16 compared positions. The logits
  cosine is 0.99929 at 4,096 tokens and 0.99734 at 32,768. In fp32 on CPU
  the two forms agree to about 1e-7. On the GPU each layer rounds its output
  to bf16, and the other operation order changes some of those roundings.
- The unit tests pin `gdn_chunkwise` to the transformers chunk rule and to
  the one-token recurrence. The real Qwen3.5-4B weights test still passes.
- The function holds four `(chunks, 64, 64)` fp32 matrices per head at once.
  For a 4096-token span of the 27B that is about 50 MB each.
