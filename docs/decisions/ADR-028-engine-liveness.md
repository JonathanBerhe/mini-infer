# ADR-028: Fail requests when the engine thread dies

Date: 2026-10-18
Status: Accepted

## Context

Each scheduler runs its engine on one thread (`PDScheduler` in parallel mode
uses two). `ContinuousScheduler` recovers from `OutOfMemoryError`
([ADR-024](ADR-024-engine-oom-preemption.md)). Any other exception is a bug
and ends the thread. No request in flight then receives its terminal step.

`RequestHandle.get_step()`, `steps()` and `wait()` without a timeout then
blocked forever. The HTTP server also ran `scheduler.run()` inside the async
endpoint, which blocked the event loop (the one thread that serves every
HTTP request). So one dead engine froze the whole server.

Five schedulers hand out a `RequestHandle`, with no shared base class:
`ContinuousScheduler`, `StateCacheContinuousScheduler`, the two
tensor-parallel StateCache schedulers, and `PDScheduler`.

## Decision

The handle detects the dead thread itself.

1. If the engine thread is not running, `submit()` raises a new
   `EngineNotRunningError(RuntimeError)` from `mini_infer.exceptions`.
   `PDScheduler.submit()` now also rejects a request before `start()` and
   once `stop()` begins.
2. `RequestHandle` takes an `engine_alive` callable. The handle reads it
   when the output queue stays empty for one poll slice
   (`LIVENESS_POLL_SECONDS`, 0.5 s). The order is fixed: the queue first,
   then liveness, then the caller's deadline. So buffered steps always
   arrive before the error, and a dead engine never looks like a slow one.
   After liveness reads False, one last non-blocking read catches a final
   step.
3. Each scheduler binds `engine_alive` at `submit()` to its thread at that
   moment. After a crash and a quick `start()`, the new thread can lack the
   request, so the handle must not wait on it. `PDScheduler` counts as
   alive only while every thread of its mode lives.
4. A clean `stop()` finishes each running or waiting request as `cancelled`
   after the join. `ContinuousScheduler` also frees their KV blocks, so a
   later `start()` begins empty. `engine_alive` reads True while `stop()`
   runs, so the handles wait for these steps. If the join times out,
   `stop()` cancels nothing, because the thread can still touch the state.
5. A crash stays a crash. The engine loop sets a `_crashed` flag before it
   re-raises. A later `stop()` or `start()` frees the state but sends no
   terminal step, so the handles raise `EngineNotRunningError`. Without the
   flag, the result depended on timing.
6. If `submit()` raises, the endpoint returns a 503 with the error type
   `engine_unavailable`. If the engine dies after a stream starts, the
   stream ends with an SSE error event and then `[DONE]`, as vLLM and SGLang
   do. The non-streaming
   path reads step by step on the event loop
   ([ADR-029](ADR-029-event-loop-output-reads.md)). The cause stays in the
   server log.

## Alternatives Considered

- Push the error from the crash handler of the engine. This misses a
  request between the waiting queue and a slot, and it runs on state that
  just failed.
- Poll `scheduler.is_running` in the server. Every other consumer then
  repeats the loop.
- Bind liveness to the current thread (`lambda: self.is_running`). After a
  restart, an old handle sees the new thread alive and waits forever.
- Run one blocking `wait()` per request in a thread. Each one holds a
  worker of the default executor for the whole generation.
- A heartbeat. One legitimate step, for example a long V4 prefill, can take
  tens of seconds, so a heartbeat gives false errors.

## Consequences

- A dead engine fails each request in flight within about 0.5 s. Streaming
  clients still get the output from before the crash.
- A clean `stop()` ends requests as `cancelled`, and a crash ends them with
  `EngineNotRunningError`, on every scheduler.
- If the generation of `TensorParallelStateCacheScheduler` outlasts the
  join timeout of `stop()`, the waiting requests end with
  `EngineNotRunningError`, not `cancelled`.
- The liveness check sees a thread that exited, not one that is stuck. The
  process-group timeout bounds a tensor-parallel leader that waits on a dead
  follower.
- Do not restart a scheduler until the thread of a timed-out `stop()`
  exits. This change does not guard that case.
- The process stays up after its engine dies, and every later request gets
  the 503. No health endpoint exists.
- Concurrent non-streaming requests now overlap. Before, they ran one at a
  time, so the 2026-06-24 HTTP gate of
  [ADR-019](ADR-019-v4-continuous-batching.md) never admitted two requests
  at once.
