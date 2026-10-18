# ADR-028: Fail requests when the engine thread dies

Date: 2026-10-18
Status: Accepted

## Context

Each scheduler runs its engine on one thread (`PDScheduler` in parallel mode
uses two). `ContinuousScheduler` recovers from `OutOfMemoryError`
([ADR-024](ADR-024-engine-oom-preemption.md)). Every other exception ends the
thread, because it signals a bug. After that, no request in flight receives
its terminal step.

The consumers did not notice. `RequestHandle.get_step()`, `steps()`, and
`wait()` without a timeout blocked on the output queue forever. The HTTP server
streamed with `asyncio.to_thread(handle.get_step)`. It served a non-streaming
request with `scheduler.run()` directly inside the async endpoint. That call
blocks the event loop (the one thread that serves every HTTP request) for the
whole generation. So one dead engine froze the whole server, not only the requests
in flight.

Five schedulers hand out a `RequestHandle`: `ContinuousScheduler`,
`StateCacheContinuousScheduler`, the two tensor-parallel StateCache schedulers,
and `PDScheduler`. No shared base class exists, and each one copies its own
`start`, `stop`, and `submit`.

## Decision

The handle detects the dead thread itself. The engine does not have to report
its own death.

1. A new `EngineNotRunningError(RuntimeError)` in `mini_infer.exceptions`.
   If the engine thread is not running, `submit()` raises it. It subclasses
   `RuntimeError` because four of the five schedulers raised a plain
   `RuntimeError` from `submit()` before. `PDScheduler.submit()` did not
   check at all: it queued the request, and the engine served it after a
   later `start()`. It now rejects the request before `start()` and once
   `stop()` begins.
2. `RequestHandle` takes an `engine_alive` callable. When the output queue
   stays empty for one poll slice (`LIVENESS_POLL_SECONDS`, 0.5 s), the handle
   reads it. The order of the checks is fixed: the queue first, then
   liveness, then the caller's deadline. Buffered steps therefore always
   arrive before the error, and a dead engine is never reported as a slow
   one. After liveness reads False, one more non-blocking read catches a step
   that the thread put just before it exited. Then the handle cancels its
   request and raises `EngineNotRunningError`.
3. Each scheduler binds `engine_alive` at `submit()` to the thread object that
   it has at that moment, not to the thread that it has later. A crash and a
   quick `start()` can leave the request unknown to the new thread, because
   the StateCache schedulers reset their slots. The handle must not wait on
   that thread.
   `PDScheduler` counts as alive only while every thread of its mode lives.
4. A clean `stop()` is a boundary on every scheduler. After the join, it
   finishes each request still running or waiting as `cancelled`.
   `ContinuousScheduler` also returns their KV blocks to the pool, so a later
   `start()` begins empty. The terminal steps go out after the thread exits,
   so `engine_alive` also reads True while `stop()` runs. The thread is read
   first and the flag second. Dead threads and a cleared flag then mean a
   crash or a finished stop. In both cases the queue already holds every step
   that will come. If the join times out, `stop()` cancels nothing, because
   the thread can still touch the state.
5. The endpoint maps the error. If `submit()` raises, the response is a 503
   with the error type `engine_unavailable`. After a stream starts, the 200
   status is already sent. So the stream ends with an SSE error event and
   then `[DONE]`, as vLLM and SGLang do. The non-streaming path reads step by
   step, like the stream, through `asyncio.to_thread(handle.get_step, 0.25)`.
   It never blocks the event loop. The error message is fixed, and the cause
   stays in the server log.

## Alternatives Considered

- Push the error from the engine. The engine loop's crash handler sends a
  terminal step to every request before it re-raises. As the only mechanism,
  this misses a request that the thread took off the waiting queue but did not
  yet put into a slot. It also runs on scheduler state that just failed.
  `FinishReason` has no error value, and adding one changes the
  OpenAI-compatible surface. The pull check covers every case with one rule.
- Poll `scheduler.is_running` in the server. Every other consumer
  (`bench/harness.py`, `run_point`, the Modal scripts) then repeats the same
  loop. `run_point` already had its own copy, which this change removes.
- Bind liveness to the scheduler's current thread (`lambda: self.is_running`).
  After a crash and a restart, an old handle sees the new thread alive and
  waits forever. The StateCache schedulers reset their slots in `start()`, so
  that request never finishes.
- Leave `stop()` as it was and let handles fail. `ContinuousScheduler` keeps
  its requests across a restart, so the outcome of such a request then
  depends on timing: it completes on the new thread, unless one gap between
  its steps reaches a poll slice.
- Run one blocking `wait()` per non-streaming request in a thread. It holds a
  worker of the default executor (min(32, cpus + 4) threads) for the whole
  generation. The output queues of other requests then go unread, and a full
  queue drops steps.
- Time out a thread that is alive but makes no progress (a heartbeat). One
  legitimate step, for example a first compile or a long V4 prefill, can take
  tens of seconds, so a heartbeat gives false errors.

## Consequences

- A dead engine fails each request in flight within about 0.5 s. A
  streaming client and a caller of `get_step()` or `steps()` still receive
  the output that the engine produced before it died. A non-streaming
  request and a caller of `wait()` get only the error.
- On every scheduler, a request that a clean `stop()` ends finishes as
  `cancelled`. Before this change, only `PDScheduler.stop()` sent terminal
  steps, and a waiter on any other scheduler blocked forever.
- A `submit()` on another thread that races `stop()` can still land after
  the drain. Its handle then fails through the liveness check. The HTTP
  server calls both on its one event-loop thread, so it never races them.
- The check covers a thread that exited, not one that is stuck. A
  tensor-parallel leader blocked in a collective whose follower died still
  looks alive. The process-group timeout bounds that case.
- The executor pool is still shared by every request in flight. When more
  requests are in flight than the pool has threads, unread output queues can
  still fill and drop steps. Streaming requests had this limit before this
  change. A dedicated executor, or a read of every buffered step per thread
  hop, is the follow-up.
- The process stays up after its engine dies, and every later request gets
  the 503 until the server restarts. No health endpoint exists.
- Concurrent non-streaming requests now overlap. Before, the blocked event loop
  served them one at a time. So the 2026-06-24 HTTP gate of
  [ADR-019](ADR-019-v4-continuous-batching.md) never admitted two requests at
  once, and an elapsed time from `scripts/modal_v4_flash_cb_serve.py` is not
  comparable with earlier runs.
