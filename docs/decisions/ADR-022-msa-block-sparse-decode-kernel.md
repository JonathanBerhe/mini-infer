# ADR-022: MSA block-sparse paged decode kernel (opt-in)

Date: 2026-07-03
Status: Accepted

## Context

The MiniMax-M3 serving path uses the `torch` attention backend, because the
MSA block mask needs a materialized SDPA. At decode it attends over every
cached token, at O(context) cost per layer per step. But each MSA query
attends to at most `topk_blocks + local` 128-token blocks.

The `MiniMax-AI/MSA` reference has no sparse decode (`NotImplementedError`),
so our dense-mask oracle (ADR-021) is the only authority. ADR-020 sets the
bar: a kernel is on by default only if the real-model end-to-end A/B shows a
gain and the greedy tokens match.

## Decision

Ship `msa_paged_decode` (`cache/msa_paged_attention.py`), a Triton decode
kernel that reads only the selected blocks from the paged pool. It is off by
default (`MiniMaxM3ForCausalLM.set_decode_kernel`).

- One program per `(request, KV head)` reads each selected block once for the
  whole GQA group.
- `MiniMaxM3Indexer.select_cached` picks the blocks and also builds the
  oracle mask, so the two paths differ only in float rounding.
- The math is fp32, with one bf16 cast at the store. Parity with the oracle
  is cosine 1.000 at max_abs ~1e-4, not bit-exact.
- Prefill, mixed batches, TP and CPU keep the oracle.

A pure-torch reference (`msa_paged_decode_torch`) matches the oracle on CPU,
and the kernel matches that reference on GPU. Changes to K/V outside the
selected blocks leave the output bit-identical.

## Alternatives Considered

- Port the split-KV machinery of the reference (up to 256 splits and a
  combine pass). Rejected: MSA decode reads at most ~2.2K tokens, so the
  combine pass only adds an HBM round trip.
- One program per (request, q head). Rejected: it reads every K/V block 16x.
- Selection inside the kernel. Out of scope: the oracle shares the indexer
  scoring, which stays in torch.

## Consequences

- The attention op at decode is 2.3x faster at 1K context and 105.9x at 64K
  (A10, see docs/benchmarks/2026-07-03-msa-decode-kernel.md).
- The ship gate on the real 428B checkpoint (4x H200, 16K context) gave 0.98x
  with token identity PASS (see
  docs/benchmarks/2026-07-03-minimax-m3-428b-gate.md). MoE weight traffic
  (~13 GB/step) is much larger than the K/V traffic that the kernel removes
  (~2 GB/step). So the kernel stays off by default, as in ADR-020. The arms
  are token-identical, so the flag is a safe opt-in.
- The next lever is the indexer, which still re-scores the full index-K
  history every step.
