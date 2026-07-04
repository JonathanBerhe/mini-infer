# ADR-024: Recover from mid-step OOM by preemption, not by crashing the engine

Date: 2026-07-04
Status: Accepted. The `ContinuousScheduler` engine loop now catches
`OutOfMemoryError` mid-step and preempts one in-flight request instead of
tearing down the thread.

## Context

`ContinuousScheduler` admits a request only if the block pool has room for its
prompt plus a fixed decode headroom (`prompt_blocks + decode_headroom_blocks`).
That admission check watermarks capacity; it does not physically allocate the
blocks. The blocks are handed out lazily during the forward pass, as
`PagedKVCache.append_kv_packed` calls `BlockPool.allocate()` for the tokens each
step actually produces.

Two things make the watermark an imperfect guarantee, so a `BlockPool.allocate()`
inside a forward can still raise `OutOfMemoryError` after admission already
accepted everyone in the batch:

- **Decode growth.** A decoding request crosses a block boundary and needs a new
  block that was not reserved when it was admitted (headroom is finite).
- **Concurrent pressure.** Several long-prompt requests admitted across
  successive steps can, together, out-run the pool mid-step even though each
  passed admission on its own.

Before this change the engine loop wrapped the whole `while` in a single
`try/except Exception: raise`. Any `OutOfMemoryError` propagated out of the
engine thread, killed it, and left every subsequent `submit()` failing with
"scheduler not running". A single mid-step OOM therefore took down not just the
request that triggered it but every in-flight request and every future one, until
the server was restarted. The open-loop rate sweep
([docs/benchmarks/2026-04-30-open-loop-rate-sweep.md](../benchmarks/2026-04-30-open-loop-rate-sweep.md))
surfaced this at `num_blocks=1024` under sustained load.

## Decision

Catch `OutOfMemoryError` at the engine-loop boundary and run a small preemption
(`_preempt_on_oom`), then let the loop retry the step. Everything else still
propagates and crashes loudly, because any other exception signals a real bug,
not a recoverable resource condition.

Victim selection:

1. Prefer a **prefiller** (`PREFILLING` / `CHUNKED_PREFILLING`) over a decoder.
   A prefiller has not emitted any user-visible output yet, so cancelling it does
   the least user-facing damage; a decoder mid-stream has already sent tokens the
   client is consuming.
2. Among the candidates, cancel the **youngest** (last-admitted). `_running`
   preserves admission order, so the tail is the most recent admission, which is
   the one most likely to have pushed the batch past the watermark.
3. If no request is in flight at all (the OOM came from something the preemption
   cannot help), it is a no-op and the loop simply retries; the error was not
   ours to fix by shedding a request.

The victim finishes with `finish_reason="cancelled"` (already a valid
`FinishReason`), so its client gets a clean terminal step rather than a hung
stream or a blanket 503. Its blocks are reclaimed by the normal `_reap_done`
pass, freeing room for the retry.

## Alternatives Considered

- **Physically pre-allocate at admission (never over-commit).** Reserving
  `prompt + max_new_tokens` blocks up front removes mid-step OOM entirely, but at
  the cost of admitting far fewer requests: almost no request uses its full
  `max_tokens`, so the pool sits mostly idle. Continuous-batching engines
  deliberately over-commit and reclaim on overflow; this ADR keeps that model and
  makes the overflow path graceful.
- **Keep crashing (treat OOM as fatal).** Simplest, but a single unlucky
  interleaving of admitted requests kills the whole server. Unacceptable for a
  serving loop whose entire job is to stay up under variable load.
- **Cancel the oldest / a decoder first.** Rejected: the oldest requests are
  closest to finishing (freeing blocks soon on their own), and a decoder has
  already produced visible output. Shedding the youngest prefiller reclaims the
  most blocks for the least user-visible cost.

## Consequences

- The engine thread survives mid-step OOM and keeps serving; overload degrades to
  "the newest request is cancelled" instead of "the server stops".
- Under a correctly-sized pool the path never fires (the calibrated sweeps ran at
  `num_blocks=4096` with zero errors); it is a safety net, not a routine path.
- A cancelled request is lost work, not corrupted output: the client sees
  `finish_reason="cancelled"` and can retry. There is no partial or wrong result.
- Covered by [tests/unit/test_scheduler_oom.py](../../tests/unit/test_scheduler_oom.py):
  victim-selection unit tests (prefiller-over-decoder, youngest-first, empty
  no-op) plus an end-to-end test that forces a mid-step OOM through the real
  engine thread and asserts it recovers and keeps serving.
