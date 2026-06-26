# ADR-020: Cross-request prefix sharing for V4, and a rejected fused decode kernel

Date: 2026-06-25
Status: Accepted. Prefix sharing implemented and wired into serving. A fused
decode-attention kernel was prototyped, GPU-measured, and rejected.

## Context

ADR-019 shipped continuous batching for DeepSeek-V4 and deferred two "Phase 3"
optimizations: *"the fast fused-kernel backend and cross-request prefix
sharing."* This ADR resolves both: it ships prefix sharing, and it records why a
fused decode-attention kernel was prototyped and then dropped after measurement.

V4 keeps per-request attention state in a `StateCache` (not a paged pool). When
two prompts share a prefix (a system prompt, few-shot examples, a RAG document),
the cache state after that prefix is a deterministic function of the prefix
tokens and positions, so it is identical and recomputing it is wasted work.

## Decision: cross-request prefix sharing (shipped)

`StatePrefixCache` (`cache/state_prefix_cache.py`) snapshots the full per-request
`StateCache` after a prompt is prefilled (per-layer SWA window + compressed
history + in-flight compressor state, the scalar counters, the CSA indexer
sub-state, `start_pos`, and the post-prefix logits), keyed by the prompt
token-ids. A later prompt that extends a cached one restores the snapshot and
replays only the new suffix token by token, skipping the shared prefill.

The reuse logic is one shared function (`prefill_with_prefix_cache`) called by
both the single-request generator (`generate_ids_prefix_cached`, the readable
reference) and the continuous-batching scheduler, which already prefills each
request into a temporary B=1 cache before copying the row into its slot, so the
same B=1 restore + suffix-replay drops straight in. The HTTP server enables it by
default for the StateCache path (`MINI_INFER_PREFIX_SHARING=0` to disable).

This is **bit-exact**: it replays the identical computation, so output is
token-for-token identical to a fresh prefill (the parity tests assert this in
both the generator and the scheduler). v1 caches at the full-prompt boundary and
copies snapshots (FIFO-capped); the eventual paged-pointer design (shared
prefixes pointed at, not copied) is deferred, as ADR-019 noted.

## Rejected: a fused decode-attention Triton kernel

V4's SWA/CSA/HCA decode attention all route through one primitive,
`hca_mqa_with_sink`: gather a per-query set of KV positions from a shared K=V
buffer, score against Q, softmax with a per-head attention-sink logit in the
denominator, weighted sum. We prototyped a fused Triton kernel for the decode
case (one launch per `(request, head)`, fp32 online softmax seeded by the sink,
no materialized gather) and benchmarked it end to end before committing to it.

**Per-primitive (A10):** parity cosine 1.0 vs the PyTorch path across fp32/bf16,
padding, and head-dim 192; 1.35-1.97x faster for one attention call.

**End-to-end (real V4-Flash, 2x B200, continuous batching, kernel on vs off):**

| path | decode | throughput |
|---|---|---|
| kernel off (PyTorch) | 85.17 s | 12.0 tok/s |
| kernel on (fused) | 85.38 s | 12.0 tok/s |

- **Speedup: 0.997x (none).** V4-Flash is a ~158GB MoE; per-token decode is
  dominated by the expert FFNs and weight movement, and attention is a thin
  slice. A 1.35-1.97x speedup on a thin slice does not move the total.
- **Output diverged.** The kernel is numerically close (cosine 1.0) but not
  bit-exact to the PyTorch path: its fp32 online-softmax accumulates in a
  different order, so the bf16 result differs in the last bits. Under greedy
  decode a sub-epsilon logit difference eventually flips an argmax, and the
  sequences drift apart over ~1024 autoregressive tokens.

mini-infer's thesis is "we're allowed to be slower; we're not allowed to be
different." A default decode path that diverges token-for-token from the
bit-parity-validated PyTorch path, for ~0% end-to-end gain on the only model it
applies to, fails that bar. So the kernel was dropped; the PyTorch sparse-gather
path remains the sole decode-attention implementation.

## Alternatives Considered

- **Ship the kernel off-by-default (opt-in).** Keeps it as a readable textbook
  implementation. Rejected for now: it gives ~0% on the target model and carries
  a divergent path to maintain. Revisit if attention becomes a larger share of
  decode (e.g. after the expert path is optimized, or at much longer context).
- **Make the kernel bit-exact to the PyTorch path.** Would require replicating
  the batched-softmax accumulation order, which defeats the fusion. Not viable.
- **Prefix sharing via paged-pointer instead of snapshot-copy.** More memory
  efficient and the eventual target, but needs the compressed history moved into
  a paged pool first. Deferred; v1 copies.

## Consequences

- Prefix sharing removes the shared prefill for repeated / extended prompts and
  is active in serving (on by default, opt-out via env). It is bit-exact, so it
  does not affect the golden / parity guarantees.
- No fused attention kernel ships. The end-to-end measurement (not the
  per-primitive one) is what informed this: it caught a divergent, zero-benefit
  default path before it shipped. Recorded here so the experiment is not repeated
  without new motivation.
- Snapshotting every prefill costs a bounded (FIFO-capped) snapshot pool, the
  trade for skipping re-prefills under prefix-sharing workloads.

## References

- ADRs this builds on: ADR-014 (V4 hybrid attention), ADR-018 (HC Sinkhorn Triton
  port), ADR-019 (V4 continuous batching; deferred these two items).
- Code: [state_prefix_cache.py](../../src/mini_infer/cache/state_prefix_cache.py),
  [state_cache_generator.py](../../src/mini_infer/engine/state_cache_generator.py)
  (`prefill_with_prefix_cache`),
  [state_cache_continuous_scheduler.py](../../src/mini_infer/scheduler/state_cache_continuous_scheduler.py)
  (per-slot prefill), [server.py](../../src/mini_infer/api/server.py) (the flag).
- Validation: [test_state_prefix_cache.py](../../tests/unit/test_state_prefix_cache.py),
  [test_state_cache_continuous_scheduler.py](../../tests/unit/test_state_cache_continuous_scheduler.py)
  (prefix sharing in the scheduler).
