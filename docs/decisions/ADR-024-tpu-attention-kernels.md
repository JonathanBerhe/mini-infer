# ADR-024: TPU attention kernels (Pallas): paging via scalar prefetch, online softmax on the sequential grid

Date: 2026-07-05
Status: Accepted

## Context

ADR-023 opened the cross-accelerator scope and made the TPU (JAX/Pallas) a
first-class backend. Attention is the first non-trivial thing the backend needs,
and it raised a cluster of design questions that recur for every TPU kernel we
will write:

- The signature TPU-hard operation is paged attention's KV gather: the physical
  page for a token is `block_table[seq, logical_page]`, known only at run time.
  Plain XLA wants statically shaped, contiguous access, so this is the exact
  pattern it cannot express (the motivation in the Ragged Paged Attention paper,
  arXiv 2604.15464).
- FlashAttention keeps running softmax state (max, denominator, accumulator)
  across the KV loop. On a TPU the grid runs sequentially, unlike CUDA where the
  KV loop is inside a concurrent thread block. Where does the state live and how
  is the loop expressed?
- Real models use grouped-query attention (num_kv_heads < num_heads) and ragged
  batches (per-sequence context lengths).
- We have no TPU on the development box. How do we develop and hold the kernels
  to the same parity bar as the CUDA path?

The kernels built under this ADR: M0 row-wise softmax (scaffold), M1 dense
flash-attention, M2 paged + ragged decode attention, plus grouped-query support
and a backend dispatcher.

## Decision

1. **Paging via scalar prefetch.** The block table and per-sequence lengths are
   prefetched into SMEM with `pltpu.PrefetchScalarGridSpec`, and each KV
   `BlockSpec.index_map` reads the block table to return the physical page for
   the current grid step (`block_table[seq, page]`). This is the one place the
   TPU toolchain accepts a runtime-computed page index; it is the readable
   expression of RPA's "dynamic slicing over ragged memory."

2. **Online softmax carried in VMEM across the sequential grid.** The KV/page
   axis is the innermost grid axis, so successive lexicographic grid steps are
   successive FlashAttention iterations. Running max/denominator/accumulator live
   in `pltpu.VMEM` scratch that Pallas threads across steps; we reset on the
   first step (`pl.when(step == 0)`) and finalize (`acc / l`) on the last. The
   grid IS the loop.

3. **Grouped-query attention by index_map mapping.** Query head `h` reads kv
   head `h // (num_heads // num_kv_heads)`; no kv replication, so no wasted
   memory or compute.

4. **Ragged masking by absolute position.** A key at absolute position `j` is
   valid for a sequence only if `j < length[seq]`; invalid scores get a large
   finite negative sentinel (not `-inf`, to keep interpret-mode intermediates
   finite). This covers partial final pages and page slots past a sequence's
   real pages (which gather an in-bounds dummy page and are fully masked).

5. **Vendor-first, per ADR-023.** We hand-write only where XLA cannot express
   the memory pattern (the paged gather). The dense M1 kernel could ride XLA,
   but we wrote it in Pallas deliberately to build the online-softmax foundation
   that M2's paged kernel extends; it is the stepping stone, not a claim that
   dense attention needs a custom kernel.

6. **Develop and validate in interpret mode; confirm on hardware separately.**
   Plain `jax` runs Pallas kernels in interpret mode on CPU
   (`pallas_call(..., interpret=True)`). Every kernel is parity-tested there
   against a NumPy reference and a PyTorch golden (cosine > 0.99 and allclose,
   the ADR-023 bar), including a shuffled-physical-pages test that fails for any
   kernel that ignores the block table. A standalone script
   (`scripts/run_tpu_pallas_kernels.py`) runs the same kernels on a real TPU for
   the on-hardware confirmation.

7. **A backend dispatcher as the engine seam.** `dispatch_attention` routes to
   the dense or paged kernel; `tpu_backend_available(require_device=...)` gates
   on JAX and optionally a physical TPU. Nothing in the PyTorch runner routes to
   it yet; that integration is deferred (see Consequences) and, per ADR-023,
   must keep backend code isolated under `backends/`.

## Alternatives Considered

1. **Manual double-buffered DMA** (as JAX's production `paged_attention` kernel
   does, with `pltpu.make_async_copy` and semaphores). Rejected for this kernel
   family: far more code and complexity for a readable reference. The index_map
   indirection is clear, correct, and sufficient; DMA/prefetch tuning is a later
   performance concern, not a correctness one.
2. **Materialize each sequence's KV in XLA, then run dense attention.** Rejected:
   that is not paging. It defeats the memory-efficiency point and does not
   demonstrate the gather that motivates the whole exercise.
3. **Implement GQA by replicating kv heads up to num_heads.** Rejected: wastes
   HBM and bandwidth; the index_map head mapping is free.
4. **Wait for real TPU access to develop.** Rejected: interpret mode lets us
   develop and parity-test with zero hardware and zero cost; the on-TPU run is a
   separate confirmation, not a prerequisite for correctness work.

## Consequences

Positive:

- The RPA-style dynamic KV gather is expressed and validated, on the JAX/XLA
  through-line that also covers Trainium.
- Readable reference kernels a paper reader can follow, held to the same parity
  bar as the CUDA kernels.
- Zero-cost, zero-hardware development and CI via interpret mode; grouped-query
  and ragged batches supported.

Costs and risks:

- Interpret mode does NOT exercise real Mosaic lowering, tiling, or padding, so
  on-TPU execution is a real, still-open confirmation step (run the script).
- These are correctness-first reference kernels, not performance-tuned; block
  sizes are conservative and there is no DMA pipelining.
- The paged kernel is decode-oriented (one query per sequence). Prefill and
  mixed prefill/decode batches (variable query length, RPA's distribution-aware
  case) are a planned extension.
- Nothing is wired into the PyTorch runner yet. A JAX execution path in the
  runner is a larger cross-framework effort, intentionally out of scope here to
  respect ADR-023's backend-isolation rule; the dispatcher is the seam it will
  plug into.
