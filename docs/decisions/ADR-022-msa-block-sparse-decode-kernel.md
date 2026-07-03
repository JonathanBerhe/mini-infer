# ADR-022: MSA block-sparse paged decode kernel (opt-in)

Date: 2026-07-03
Status: Accepted

## Context

MiniMax-M3's serving path is pinned to the `torch` attention backend: the MSA
block mask needs a materialized SDPA. At decode that path pays O(context) per
layer per step: materialize the full K/V history, build a dense additive mask,
run attention over every cached token, then throw away everything the selection
masked out. MSA's whole premise is that each query attends to at most
`topk_blocks + local` 128-token blocks, so the memory traffic the torch path
pays is mostly wasted at long context.

The reference `MiniMax-AI/MSA` repo does not help directly: its sparse decode
entry point is a stub (`NotImplementedError`), so the only authority is our own
dense-mask oracle. ADR-020 (the V4 decode-kernel rejection) sets the bar: a
kernel ships default-on only if the END-TO-END A/B on the real model shows a
win AND the greedy tokens match; per-primitive speedups do not count.

## Decision

Ship `msa_paged_decode` (`cache/msa_paged_attention.py`), a Triton decode
kernel that reads ONLY the indexer-selected blocks straight from the paged
pool, **opt-in and off by default** (`MiniMaxM3ForCausalLM.set_decode_kernel`).

Design:

- One program per `(request, KV head)`: a `(16, 128)` output tile covers the
  whole GQA group, so each selected block is read once per group (not once per
  q head) and QK / PV are real `tl.dot` MMAs. The selection is one set per
  request shared across all heads (the HF-verified property), which removes
  the per-head selection plumbing the reference carries.
- Selection comes from `MiniMaxM3Indexer.select_cached`, the SAME routine that
  builds the oracle's dense mask; the kernel and oracle cannot disagree about
  which blocks are attended, only about float rounding.
- Host-side, selected index blocks expand to (pool block id, base position)
  pairs via the block table (`index_block_size` must be a multiple of the pool
  block size), ascending-sorted for deterministic accumulation order; `-1`
  selection padding becomes dead entries the kernel skips outside the softmax
  update (running them through would produce `exp(-inf - -inf) = nan`).
- fp32 scores, fp32 online softmax, fp32 accumulator, ieee-precision PV, one
  bf16 cast at the store, mirroring the oracle's dtype discipline. bf16
  matmul reduction order still differs, so bit-exactness vs the oracle is not
  claimed (the ADR-020 trap); parity is cosine 1.000 at max_abs ~1e-4 bf16.
- Dense layers (0-2) reuse the existing `paged_attention_decode_batched`
  kernel on the same opt-in flag.
- Pure decode steps only (every request contributes one token), single rank;
  prefill, mixed batches, TP, and CPU always take the materialized oracle.

A pure-torch reference (`msa_paged_decode_torch`) is CPU-validated against the
dense-mask oracle; the Triton kernel is GPU-validated against that reference,
plus a sparsity probe: perturbing K/V outside the selected blocks leaves the
kernel output bit-identical.

## Alternatives Considered

- **Port the reference kernel's machinery** (split-KV with up to 256 splits,
  LSE combine pass, schedule kernel, exp2-space softmax). Rejected: all of it
  serves dense 1M-token decode; MSA decode reads at most ~2.2K tokens, one
  program covers that in <= 17 block iterations, and a combine pass would add
  an HBM round-trip for nothing.
- **One program per (request, q head)** (the shape of our existing dense paged
  kernel). Rejected after review: it re-reads every K/V block 16x (once per
  head in the group); the group-tile design reads each block once.
- **Keeping selection inside the kernel** (score + top-k on device, the
  reference's fused ambition). Out of scope: the indexer's O(context) scoring
  is shared with the oracle path and stays in torch; the kernel replaces only
  the main-attention traffic.

## Consequences

- Attention-op decode cost becomes O(topk * 128) instead of O(context): 2.3x
  at 1K context to 105.9x at 64K (A10, M3 head geometry; see
  docs/benchmarks/2026-07-03-msa-decode-kernel.md).
- End-to-end on a 114M-param synthetic M3-shaped model at 32K context the
  kernel arm is 0.95x (a slight LOSS): per-step host overhead and the
  indexer's O(context) re-scoring dominate at toy scale. This is exactly why
  the flag stays off: the ship decision belongs to the end-to-end A/B on the
  real 428B checkpoint, where 57 sparse layers of 64-head attention at long
  context are a materially larger share of each step.
- The remaining O(context) decode term is the index branch (re-scores the full
  cached index-K history every step); a future incremental-scoring or fused
  selection kernel is the next lever if the real-model A/B shows the indexer
  dominating.

## References

- ADR-020 (V4 decode-kernel rejection; the end-to-end-or-nothing rule).
- ADR-021 (MiniMax-M3 port; the dense-mask oracle this kernel is validated
  against).
- `MiniMax-AI/MSA` (reference repo; sparse decode stubbed, contract only).
- Benchmark: docs/benchmarks/2026-07-03-msa-decode-kernel.md.
