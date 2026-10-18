# ADR-029: Read engine output on the event loop, not on worker threads

Date: 2026-10-18
Status: Accepted

## Context

The engine thread puts each request's output steps into a bounded queue (256
steps by default). The HTTP server reads those queues from its asyncio event
loop. A `queue.Queue` read blocks, so the server ran each read in
`asyncio.to_thread`. That call uses the default executor of the loop: a pool
of min(32, cpus + 4) threads for every request in flight. That is 14 threads
on a 10-core machine and 6 to 8 on a CI runner.

Each blocked read holds one thread until a step arrives. A request that
waits in the admission queue gets no step for a long time, so it holds its
thread for that whole time. `ContinuousScheduler` and `PDScheduler` admit 16
requests by default, and more can wait. When the waiting readers outnumber
the threads, an admitted request gets a thread only now and then. Its queue
fills, and the schedulers drop the oldest step. The response then has gaps
in its text while `usage.completion_tokens` stays correct. `PDScheduler`
instead blocks its engine thread for 10 s on the full queue and then
cancels the request.

Short timed reads ([ADR-028](ADR-028-engine-liveness.md) used 0.25 s) only
spread the damage. A probe sent 40 concurrent non-streaming requests of 300
tokens each to the real endpoint. The scheduler admitted 16 at a time, with
a 2 ms engine step on a 10-core machine. With 0.25 s thread reads, 16 of the 40
responses came back with 261 of their 300 characters. With the reads of this
decision, all 40 came back complete, in about the same time (3.1 s against
3.3 s).

## Decision

The queue wakes the event loop, and the server waits on the loop with no
thread.

1. `StepQueue`, a `queue.Queue` subclass in `scheduler/request_state.py`,
   calls a listener after each put. The engine thread keeps the ordinary
   `put` calls, so the schedulers change only the class of the queue that
   they build.
2. `RequestHandle.get_step_async()` sets the listener to
   `loop.call_soon_threadsafe(ready.set)`, the one asyncio call that is safe
   from another thread. It clears `ready` and reads the queue without
   blocking. If the queue is empty, it awaits `ready` for one liveness slice.
   The liveness rules of `get_step()` apply unchanged: buffered steps first,
   then liveness, then the error. The call removes the listener before it
   returns.
3. The streaming and the non-streaming paths of `/v1/completions` both await
   `get_step_async()`. The server no longer calls `asyncio.to_thread`.
4. A listener that raises `RuntimeError` (the reader's loop closed under it)
   does not fail the engine's put. The step stays in the queue.

## Alternatives Considered

- Read every buffered step per thread hop. This cuts the number of hops,
  but a request still waits for a free thread before each turn. With enough
  waiting readers, its queue still overflows.
- A dedicated thread pool, sized above the maximum concurrency. It needs a
  size, and the number of requests in flight has no fixed bound. Some apps
  mount the `completions` route without the server's lifespan: the
  tensor-parallel Modal scripts and two test modules. Each of them must then
  set the pool up too.
- Poll the queue from the loop with `asyncio.sleep`. It needs no listener,
  but every step then waits for the next poll. A short poll interval wakes
  the loop for every idle request all the time.
- Take the final text of a non-streaming response from the engine side
  (`last_text`). This fixes only the non-streaming text. Streams still lose
  steps, and `PDScheduler` still stalls on a full queue.

## Consequences

- A request in flight costs no thread while it waits, so the pool size no
  longer bounds the number of requests that the server can serve without
  loss. A test pins this: one executor thread, one admitted request, and two
  waiting requests. Every response keeps all its text.
- A step reaches the reader as soon as the engine puts it. Before, a read
  first had to wait for a free thread.
- The listener runs on the engine thread while the queue's lock is held. It
  must stay a single `call_soon_threadsafe` call.
- One reader per handle. Two concurrent `get_step_async()` calls on the same
  handle replace each other's listener.
- A slow client can still fill its own stream's queue, because the
  generator yields only as fast as the client reads. That limit belongs to
  the client, not to the server.
- In-process callers (`wait()`, `get_step()`, `steps()`) keep the blocking
  reads. They do not run on an event loop.
