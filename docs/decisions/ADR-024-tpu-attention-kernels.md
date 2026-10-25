# ADR-024: TPU attention kernels (Pallas): paging via scalar prefetch, online softmax on the sequential grid

Date: 2026-07-05
Status: Accepted

## Context

ADR-023 made the TPU (JAX/Pallas) a first-class backend. Attention raises
questions that apply to every TPU kernel:

- Paged attention reads the page of a token from
  `block_table[seq, logical_page]` at run time, a gather that plain XLA
  cannot express. This motivates the Ragged Paged Attention (RPA) paper,
  arXiv 2604.15464.
- FlashAttention carries softmax state across the KV loop, but a TPU runs
  its grid in sequence, not as concurrent thread blocks.
- Models use grouped-query attention (GQA) and ragged batches (one context
  length per sequence).

This ADR covers M0 row-wise softmax (scaffold), M1 dense flash-attention, M2
paged + ragged decode attention, a paged prefill / multi-query kernel, and a
dispatcher.

## Decision

1. Paging via scalar prefetch. `pltpu.PrefetchScalarGridSpec` loads the
   block table and lengths into SMEM. Each KV `BlockSpec.index_map` reads it
   and returns the physical page. Only here does the TPU toolchain accept a
   runtime page index.
2. Online softmax in VMEM across the sequential grid. The page axis is the
   innermost grid axis, so each grid step is one FlashAttention iteration.
   The running state lives in `pltpu.VMEM` scratch, reset at
   `pl.when(step == 0)` and finalized as `acc / l`. The grid is the loop.
3. GQA by index_map. Query head `h` reads kv head
   `h // (num_heads // num_kv_heads)`, with no copy.
4. Ragged masking by absolute position. A key at `j` is valid only if
   `j < length[seq]`. Invalid scores get a large finite sentinel, not
   `-inf`, so all values stay finite.
5. Vendor-first, per ADR-023. We hand-write only the paged gather. M1 exists
   as the online-softmax base for M2.
6. Develop in interpret mode, a CPU emulation of Pallas
   (`pallas_call(..., interpret=True)`). Each kernel must match a NumPy
   reference and a PyTorch golden (cosine > 0.99 and allclose, the ADR-023
   bar). A shuffled-physical-pages test catches a kernel that ignores the
   block table. `scripts/run_tpu_pallas_kernels.py` runs the kernels on a
   real TPU.
7. `dispatch_attention` routes to the dense or paged kernel.
   `tpu_backend_available(require_device=...)` gates on JAX and a TPU.

## Alternatives Considered

1. Manual double-buffered DMA (`pltpu.make_async_copy`), as the JAX
   `paged_attention` kernel does. Rejected: far more code.
2. Materialize the KV of each sequence in XLA. Rejected: that is not paging.
3. Copy KV heads up to num_heads for GQA. Rejected: it wastes HBM.
4. Wait for TPU access. Rejected: interpret mode needs no hardware.

## Consequences

- The RPA-style gather works on the JAX/XLA path, which also covers
  Trainium.
- The kernels are not tuned: conservative block sizes, no DMA pipelining.
- The PyTorch runner does not use the kernels yet. That work is out of scope
  under the ADR-023 backend-isolation rule.

## Amendment (2026-07-09): first real-TPU run

On a Colab TPU v5e, the dense kernels passed: `dense causal=False` at cosine
0.9999949 and `causal=True` at 0.9999963 against fp32 NumPy. The paged
kernels broke the Mosaic block-shape rule, which interpret mode does not
test. The last two dims of a block must be divisible by (8, 128) or equal
the array dims, and `(1, page_size, 1, head_dim)` is neither.

The fix is heads-first page pools,
`(num_kv_heads, num_pages, page_size, head_dim)`, so every KV block ends in
`(page_size, head_dim)`. Decode queries get a singleton axis, and prefill
queries are transposed to `(num_seqs, num_heads, q_len, head_dim)`. Colab's
preinstalled libtpu can also reject valid Mosaic ("expected <= 7 but got
8"), so the notebook installs a matched `jax[tpu]` and restarts once.

With this layout, the full suite passed the same day (commit 83321f8).
Dense, paged decode and paged prefill (MHA and GQA) all reached cosine >=
0.999995 against NumPy. Results:
docs/benchmarks/2026-07-09-tpu-v5e-pallas-kernels.md.

## Amendment (2026-07-09): mixed prefill/decode batches in one launch

`pallas_paged_mixed_attention` runs decoding and prefilling sequences in one
kernel launch, the case RPA targets. Without it, continuous batching needs
two launches per step.

- Queries are padded to `max_q_len`, and `q_lens` is a third
  scalar-prefetch input. Row `t` of sequence `s` is real only if
  `t < q_lens[s]`. The causal mask is `k_pos <= lengths[s] - q_lens[s] + t`,
  and `q_lens[s] == 0` marks an inactive slot.
- A fully masked padding row keeps its running max at the sentinel. Then
  `exp(score - max)` is `exp(0) == 1`, and the row returns an average of
  unrelated values. The kernel zeroes masked probabilities and divides
  safely, so padding rows are exact zeros.

The decode and prefill kernels stay. They skip the padding waste and are the
tested baselines in tests/unit/test_tpu_pallas_mixed.py. Interpret-mode
parity is complete, and scripts/run_tpu_pallas_kernels.py has two mixed
cases for the next Colab run.
