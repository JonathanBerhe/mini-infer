# ADR-031: Multi-request DSpark speculative decoding in ContinuousScheduler

Date: 2026-10-18
Status: Accepted

## Context

[ADR-027](ADR-027-dspark-drafter-port.md) ported the DSpark drafter for one
request at a time. The batch-global confidence scheduler of the paper sets
the verify lengths of all requests in flight, so it needs speculative
decoding inside `ContinuousScheduler`. That scheduler assumed one token per
request per step.

Terms: a round is one drafted block of `gamma` tokens plus its
verification. The anchor is the last committed token. The offered tokens
are the drafted tokens that a round sends to verify. Taps are target hidden
states at fixed layers, which the drafter receives as injected context.

## Decision

The drafter rotates its keys against an unsliced position table, so every
cache-length rule keeps this invariant at each proposal:

```
target_slot_seq_len == draft_cache_len + context_rows          (INV)
```

A break of (INV) can crash the engine thread or rotate the context with
wrong positions.

1. Opt-in: `ContinuousScheduler(..., dspark=DSparkSpec(...))`.
   `RunningRequest.spec` holds the per-request state. The constructor
   refuses a prefix cache, because its hits skip the forwards that produce
   taps.
2. Admission decides once per request. Greedy and temperature sampling
   speculate, with rejection sampling at verify. Top-k, top-p, a prompt over
   `max_spec_prompt_tokens` or no room for the extra headroom keep a request
   on the plain path in the same batch.
3. Step order: cancel, draft, reap, admit, reap again, one packed forward,
   verify. With admission before the reap, at 16 requests DSpark ran 18 to
   30 percent of its steady time with fewer requests than allowed
   ([benchmark](../benchmarks/2026-10-19-dspark-concurrency.md)).
4. Each verify replaces the context with the tap rows of the committed
   tokens: at most the drafted block plus one bonus token from the target.
5. A mid-step OOM rolls every target slot back to its length before the
   forward. Each speculator keeps its proposal and drafter cache for the
   retry.
6. Admission adds `ceil((gamma + 1) / block_size) + 1` blocks to the plain
   headroom. A round truncates back to what it commits, so only the verify
   window in flight needs room.
7. A multi-token verify step arrives as one `GenerationStep`, with
   `GenerationStep.num_tokens`.

The benchmark (`mini_infer.bench.dspark_sweep`) runs in the engine process
as a closed loop: `max_concurrent` sets the level, and all prompts arrive at
once.

## Alternatives Considered

- On OOM, roll the drafter cache back and draft again. That costs a second
  drafter forward. A mix of the two choices breaks (INV).
- Multiply the headroom by `gamma + 1`. That cuts concurrency, while
  truncation already returns the unused blocks.
- Reject a request with no room for the headroom. Then an optimization
  fails work that the plain path serves.
- Measure with the HTTP open-loop benchmark. The server cannot enable
  dspark, and its arrival rate does not control the number in flight.

## Consequences

- Greedy output does not depend on the drafter, so token equality cannot
  see a corrupt context. The tests assert tap rows, slot lengths and (INV).
- Bookkeeping and CPU fp32 numerics tests are exact. GPU bf16 tests are
  statistical, because the batch shape depends on the accepted counts.
- The tap buffer is outside the block pool, so its OOM is not recoverable.
- A `turbo3` run uses the materialized attention path on every verify step.
- Deferred: the confidence scheduler, batched drafter forwards, top-k and
  top-p under speculation, and `PDScheduler` support.
