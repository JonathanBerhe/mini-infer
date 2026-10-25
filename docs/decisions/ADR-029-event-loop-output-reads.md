# ADR-029: Read engine output on the event loop, not on worker threads

Date: 2026-10-18
Status: Accepted

## Context

The engine thread puts the output steps of each request into a bounded queue
(256 steps by default). The HTTP server ran each blocking read in
`asyncio.to_thread`, which shares the default executor of min(32, cpus + 4)
threads.

A blocked read holds its thread until a step arrives, even while its request
waits for admission. When the waiting readers outnumber the threads, an
admitted request rarely gets a thread. Its queue fills, the scheduler drops
the oldest step, and the response text has gaps. `PDScheduler` instead
blocks its engine thread for 10 s and then cancels the request.

A probe sent 40 concurrent requests of 300 tokens, with 16 admitted at a
time. With thread reads of 0.25 s, 16 of the 40 responses lost text. With
this decision, all 40 came back complete in about the same time.

## Decision

The queue wakes the event loop, and the server waits on the loop with no
thread.

1. `StepQueue`, a `queue.Queue` subclass in `scheduler/request_state.py`,
   calls a listener after each put, outside the lock of the queue.
2. `RequestHandle.get_step_async()` sets the listener to
   `loop.call_soon_threadsafe(ready.set)`. It reads the queue without
   blocking. If the queue is empty, it awaits `ready` for one liveness
   slice. The liveness rules of `get_step()` apply unchanged.
3. The streaming and the non-streaming paths of `/v1/completions` both await
   `get_step_async()`.
4. A listener that raises `RuntimeError` does not fail the put of the
   engine. The step stays in the queue.

## Alternatives Considered

- A dedicated thread pool above the maximum concurrency. Requests in flight
  have no fixed bound, and each app that mounts the `completions` route
  without the server's lifespan must set up the pool.
- Batch the wakeups per event loop. It adds a lock and a registry, and on
  uvloop the plain listener costs nothing measurable.
- Poll the queue with `asyncio.sleep`. Every step then waits for the next
  poll, and a short interval wakes the loop for every idle request.
- Take the final text from the engine side (`last_text`). Streams still lose
  steps.

## Consequences

- A waiting request costs no thread, so the pool size no longer limits how
  many requests the server serves without loss. A test pins this with one
  executor thread and two waiting requests.
- The listener runs on the engine thread, so it must stay a single
  `call_soon_threadsafe` call. On the stdlib loop, 128 streams with a 10 ms
  forward grew the wall time from 2.8 to 3.3 s. On uvloop, which
  `uvicorn[standard]` installs, it stayed at 2.8 to 2.9 s.
- Each handle allows one reader. A second concurrent `get_step_async()` call
  replaces the listener.
- A slow client can still fill the queue of its own stream.
- In-process callers (`wait()`, `get_step()`, `steps()`) keep the blocking
  reads.
