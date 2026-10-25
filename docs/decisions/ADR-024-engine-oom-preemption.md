# ADR-024: Recover from mid-step OOM by preemption, not by crashing the engine

Date: 2026-07-04
Status: Accepted. The `ContinuousScheduler` engine loop now catches
`OutOfMemoryError` mid-step and preempts one in-flight request instead of
tearing down the thread.

## Context

`ContinuousScheduler` admits a request only if the pool has room for its
prompt plus a fixed decode headroom
(`prompt_blocks + decode_headroom_blocks`). But `BlockPool.allocate()` runs
during the forward, so `OutOfMemoryError` can still occur. A decoder can
cross a block boundary past its headroom, or several long prompts can
together exceed the pool.

Before this change, an OOM killed the engine thread. Every later `submit()`
failed with "scheduler not running" until a restart. The open-loop rate
sweep
([docs/benchmarks/2026-04-30-open-loop-rate-sweep.md](../benchmarks/2026-04-30-open-loop-rate-sweep.md))
found this at `num_blocks=1024`.

## Decision

The engine loop catches `OutOfMemoryError`, runs `_preempt_on_oom` and
retries the step. Other exceptions still crash the thread, because they
signal a bug. Victim selection:

1. Prefer a prefiller (`PREFILLING` / `CHUNKED_PREFILLING`) over a decoder,
   because a prefiller sent no output yet.
2. Cancel the youngest candidate (last admitted). `_running` keeps
   admission order, and the newest request most likely pushed the batch
   past the limit.
3. If no request is in flight, do nothing and retry.

The victim ends with `finish_reason="cancelled"`, a clean terminal step,
not a hung stream or a 503. `_reap_done` frees its blocks for the retry.

## Alternatives Considered

- Allocate `prompt + max_new_tokens` blocks at admission. Almost no request
  uses its full `max_tokens`, so the pool sits mostly idle.
- Keep crashing on OOM. One unlucky mix of requests stops the server.
- Cancel the oldest request or a decoder first. The oldest requests free
  blocks soon, and a decoder already shows output.

## Consequences

- Overload cancels the newest request instead of stopping the server. The
  client can retry, and no output is wrong.
- With a correctly sized pool the path never runs. The calibrated sweeps
  ran at `num_blocks=4096` with zero errors.
- [tests/unit/test_scheduler_oom.py](../../tests/unit/test_scheduler_oom.py)
  covers victim selection. An end-to-end test forces a mid-step OOM
  through the real engine thread, and the engine recovers and keeps
  serving.
