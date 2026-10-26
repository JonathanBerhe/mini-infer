# ADR-020: Cross-request prefix sharing for V4, and a rejected fused decode kernel

Date: 2026-06-25
Status: Accepted. Prefix sharing implemented and wired into serving. A fused
decode-attention kernel was prototyped, GPU-measured, and rejected.

## Context

ADR-019 (continuous batching for DeepSeek-V4) deferred a fused kernel and
cross-request prefix sharing. V4 keeps attention state per request in a
`StateCache`. Prompts with a shared prefix, such as a system prompt, have
the same state after it.

## Decision: cross-request prefix sharing (shipped)

`StatePrefixCache` (`cache/state_prefix_cache.py`) stores a snapshot of the
full `StateCache` and the logits after prefill, keyed by the prompt token
ids. A later prompt that extends a cached one restores the snapshot and
replays only the new suffix. The generator and the continuous-batching
scheduler both call `prefill_with_prefix_cache`. The server turns it on by
default (`MINI_INFER_PREFIX_SHARING=0` turns it off). Output is
bit-exact, token-for-token equal to a fresh prefill, so the golden and
parity guarantees hold. v1 copies snapshots into a FIFO-capped pool.

## Rejected: a fused decode-attention Triton kernel

The prototype fused `hca_mqa_with_sink`, the V4 decode-attention primitive,
into one launch per `(request, head)`. On an A10, one call was 1.35-1.97x
faster, with parity cosine 1.0. End to end on real V4-Flash (2x B200,
continuous batching), decode took 85.17 s without the kernel and 85.38 s
with it. Both ran at 12.0 tok/s (speedup 0.997x).

V4-Flash is a ~158GB MoE, and attention is a thin slice of its decode time.
The output also diverged. The fp32 online softmax adds in a different
order, so the bf16 result differs in the last bits. Under greedy decode
this flips an argmax, and sequences drift apart over ~1024 tokens.

mini-infer can be slower, but its output must not change. A default path
that diverges for ~0% end-to-end gain fails that rule, so the PyTorch
sparse-gather path stays.

## Alternatives Considered

- Ship the kernel off by default. It gives ~0% and adds a divergent path.
  Revisit it if attention becomes a larger share of decode.
- Make the kernel bit-exact. That needs the accumulation order of the
  batched softmax, which defeats the fusion.
- Share prefixes by paged pointer, not by copy. This uses less memory, but
  the compressed history must move into a paged pool first. Deferred, as
  ADR-019 noted.

## Consequences

- Prefix sharing skips the shared prefill for repeated or extended prompts.
- No fused attention kernel ships. Only the end-to-end measurement caught
  the problem, so a per-primitive speedup is not enough. Do not repeat this
  experiment without a new reason.
- Tests: [test_state_prefix_cache.py](../../tests/unit/cache/test_state_prefix_cache.py),
  [test_state_cache_continuous_scheduler.py](../../tests/unit/scheduler/test_state_cache_continuous_scheduler.py).
