# ADR-031: Multi-request DSpark speculative decoding in ContinuousScheduler

Date: 2026-10-18
Status: Accepted

## Context

[ADR-027](ADR-027-dspark-drafter-port.md) ported the DSpark drafter for one
request at a time. The batch-global confidence scheduler of the paper sets the
verification lengths of all requests in flight, so it needs speculative
decoding inside `ContinuousScheduler` first.

`ContinuousScheduler` assumed one token per request per step: it sampled one
token, appended one position, read only the last logits, and reserved headroom
for one token. It also sampled at the start of a step from the logits of the
step before. Speculative decoding drafts and verifies in the same step, so the
engine knows what to emit only after the verify forward.

Terms used below: a round is one drafted block plus its verification. The
anchor is the last committed token. The offered tokens are the drafted tokens
that a round sends for verification. Taps are the hidden states of the target
at fixed layers, and the drafter receives them as injected context.

## Decision

Every rule that touches cache lengths keeps one invariant true. The drafter
rotates its keys against an unsliced position table, so at each proposal:

```
target_slot_seq_len == draft_cache_len + context_rows          (INV)
```

A break of (INV) does not fail cleanly: it can crash the engine thread or
rotate the context with wrong positions. The code asserts (INV) in the branch
that drafts. It is a precondition of proposing, not of the draft phase,
because a proposal that already exists folds the context into the drafter
cache.

1. Opt-in. `ContinuousScheduler(..., dspark=DSparkSpec(...))` turns it on.
   With `None`, every path is unchanged. The constructor refuses a prefix
   cache (its hits skip the forwards that produce taps) and a target whose
   forward cannot return taps.
2. One optional field per request, `RunningRequest.spec`, holds the drafter
   cache, the context, the anchor, the proposal, and the statistics.
3. Eligibility is decided once at admission. Greedy and temperature sampling
   speculate, with rejection sampling at verify. Three kinds of request run
   the plain path in the same batch: one with top-k or top-p, one with a
   prompt over `max_spec_prompt_tokens`, and one with no room for the extra
   headroom. Speculation never turns away a
   request that the plain path can serve.
4. The step order: cancel, admit, draft (or sample plain decoders), reap, one
   packed forward with all-position logits, then verify. The verify phase
   selects requests by `spec.offered is not None`, never by state, because a
   request that just finished prefill is DECODING with no proposal.
5. Each request takes its own slice of the packed forward: its tap rows and
   its verify logits, at its `cu_seqlens_q` offset. The tap rows follow the
   configured layer order and are cloned, so they do not pin the whole packed
   activation in memory.
6. A mid-step OOM rolls every target slot back to its length before the
   forward. Each speculator keeps its anchor, its proposal, and its drafter
   cache, and the retry reuses the proposal. Only the forward can raise
   `OutOfMemoryError`, and it allocates before it writes, so the rollback is
   complete.
7. Emission matches the plain path: the commit is clamped to the remaining
   `max_tokens`, EOS is not emitted, and the slot is truncated to the emitted
   length. The draft is also clamped to the room left, so the verify forward
   never appends positions that cannot commit.
8. Admission adds `ceil((gamma + 1) / block_size) + 1` blocks to the plain
   headroom. A round appends `1 + offered` positions but truncates back to
   what it commits, so the blocks per emitted token do not change. Only the
   window in flight needs room.
9. A verify step that commits several tokens arrives as one `GenerationStep`.
   `GenerationStep.num_tokens` says how many tokens it carries, and
   `GenerationResult.spec_summary` keeps a fixed-size summary of the rounds.

The benchmark (`mini_infer.bench.dspark_sweep`) runs inside the engine
process as a closed loop: `max_concurrent` is the level, and all prompts are
submitted at once. It reports the whole-run and the steady-state throughput,
the mean in-flight count, and `tau` with and without rounds that the room
clamp shortened.

## Alternatives Considered

- Roll the drafter cache back on OOM and draft again. That is consistent too,
  but it costs a second drafter forward. A mix of the two (clear the
  proposal, keep the drafter cache) breaks (INV).
- Multiply the headroom by `gamma + 1`. It reserves a large constant per
  request and cuts the concurrency that the engine can reach, while the
  truncation already returns the unused blocks.
- Reject a request that has no room for the speculative headroom. That makes
  an optimization fail work that the plain path serves.
- Inject only the suffix after a prefix-cache hit. It keeps (INV), but the
  drafter then sees less context than in training. That is a measurement
  for later, not a fallback.
- A transactional K/V append. It fixes the two-phase append at its root, but
  the rollback costs far less.
- Measure on the HTTP open-loop benchmark. The server cannot enable dspark,
  its default prompt is too long to speculate, its client counts chunks as
  tokens, and its arrival rate does not control the number in flight.

## Consequences

- Greedy output does not depend on the drafter, so token equality cannot see
  a corrupted context or a missing cache rollback. The tests assert the state
  itself: tap rows, slot lengths, and (INV). A context bug shows up only as a
  lower `tau`.
- The test gate has three parts. Bookkeeping is exact against scripted
  doubles. Numerics are exact on a small CPU fp32 model. GPU bf16 is
  statistical: the logits of a request depend on the shape of the batch.
  That shape depends on the accepted counts of the other requests.
- The tap buffer lives outside the block pool, so its OOM is not
  recoverable. `max_spec_prompt_tokens` bounds it.
- The fused TurboQuant attention path needs one query token per request, so
  a `turbo3` run uses the materialized path on every verify step.
- Deferred items:
  - The batch-global confidence scheduler and its capacity curve.
  - Batched drafter forwards. The drafter now runs once per request.
  - Top-k and top-p under speculation.
  - `PDScheduler` support.
