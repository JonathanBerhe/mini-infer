# ADR-030: Report a failed request as an error, without its cause

Date: 2026-10-18
Status: Accepted

## Context

Two schedulers catch an exception for one request and keep running.
`PDScheduler` catches a failed `prefill_batch` and ends every request of the
batch. `TensorParallelStateCacheScheduler` catches a failed generation and
ends that request. Both ended the request with `finish_reason="cancelled"`.
A client saw no difference between an engine failure and a cancellation.

`PDScheduler` also put `f"engine error: {exc}"` into the text of the last
step. The streaming endpoint sends the text of every step to the client, the
last one included. So a streaming client received the raw exception message
(it can name paths, shapes, or devices) with a 200 status. The server
promises the opposite: the global exception handler sends only a fixed
message, and the cause stays in the log.

If the output queue stayed full for 5 s, `PDScheduler` also dropped the last
step. The engine was alive, so the liveness check of
[ADR-028](ADR-028-engine-liveness.md) never fired, and the handle waited
forever.

## Decision

1. `GenerationStep` gets a `failed` flag. A scheduler that fails a request
   sends a last step with `failed=True`, no text, and
   `finish_reason="cancelled"` for its own bookkeeping.
2. `RequestHandle` checks every step that it reads. On a failed step it
   raises `RequestFailedError(RuntimeError)` with a fixed message. A caller
   never receives that step, so it cannot mistake it for a cancellation.
   Steps buffered before it still arrive first.
3. The endpoint answers a non-streaming failure with a 500 and the error
   type `server_error`. A stream gets an SSE error event of that type and
   then `[DONE]`. A 503 `engine_unavailable` keeps its meaning: the engine is
   down for every request.
4. `PDScheduler._emit_terminal` still waits up to `TERMINAL_PUT_TIMEOUT_S`
   (5 s) for a full queue. After that, it drops one buffered step and puts
   the last step, as `ContinuousScheduler._finish` does.
5. A full KV pool in prefill is not a failure. `PDScheduler` has no memory
   check at admission, so `prefill_batch` can raise `OutOfMemoryError`. Both
   prefill handlers catch it first and end the batch as `cancelled`, with no
   `failed` flag, as the OOM preemption of `ContinuousScheduler` does. An
   OOM in decode (`DecodeSession.step` or `add_handoff`) has no handler. It
   still kills the engine thread, and the requests end with
   `EngineNotRunningError`.

## Alternatives Considered

- Remove the text and keep `cancelled`. The leak stops, but a client still
  cannot tell a failure from a cancellation, and a benchmark counts a
  failure as a normal end.
- Add `"error"` to `FinishReason`. The OpenAI-compatible schema has no such
  value, so clients that check the field can break.
- An error field on `RunningRequest` that the handle reads after the last
  step. It works, but the flag on the step travels with the queue put,
  which already orders it after every earlier step.
- Send the cause to the client. The cause comes from inside the engine. The
  log has it, with the request id of the completion.

## Consequences

- A client sees a real error for a failed request, and the cause stays in
  the server log.
- An in-process caller of `wait()`, `get_step()`, or `steps()` now gets
  `RequestFailedError` where it got a `cancelled` result before. No caller
  in the repository depended on the old result.
- A slow reader on `PDScheduler` can lose one buffered step, but its request
  always ends.
- An OOM ends as `cancelled` where the engine handles it:
  `ContinuousScheduler` preempts one request
  ([ADR-024](ADR-024-engine-oom-preemption.md)), and `PDScheduler` cancels
  the prefill batch that did not fit. That is a resource decision of the
  engine, not a failure. A decode-side OOM on `PDScheduler` is not handled
  and still kills its engine thread.
