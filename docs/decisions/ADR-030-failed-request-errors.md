# ADR-030: Report a failed request as an error, without its cause

Date: 2026-10-18
Status: Accepted

## Context

`PDScheduler` and `TensorParallelStateCacheScheduler` ended a failed request
with `finish_reason="cancelled"`, so a client saw a failure as a
cancellation. `PDScheduler` also streamed `f"engine error: {exc}"` to the
client with a 200 status, but the cause must stay in the server log.

If the output queue stayed full for 5 s, `PDScheduler` dropped the last step.
The live engine never tripped [ADR-028](ADR-028-engine-liveness.md), so the
handle waited forever.

## Decision

1. `GenerationStep` gets a `failed` flag. The last step of a failed request
   has `failed=True` and no text.
2. On that step, `RequestHandle` raises `RequestFailedError(RuntimeError)`
   with a fixed message. Earlier buffered steps arrive first.
3. The endpoint answers with a 500 and the error type `server_error`. A
   stream gets an SSE error event, then `[DONE]`.
4. After `TERMINAL_PUT_TIMEOUT_S` (5 s) on a full queue,
   `PDScheduler._emit_terminal` drops one buffered step for the last step.
5. A full KV pool in prefill is not a failure. The `PDScheduler` prefill
   handlers catch `OutOfMemoryError` and end the batch as `cancelled`. An OOM
   in decode (`DecodeSession.step` or `add_handoff`) still kills the engine
   thread.

## Alternatives Considered

- Remove the text but keep `cancelled`. A benchmark still counts a failure
  as a normal end.
- Add `"error"` to `FinishReason`. The OpenAI-compatible schema has no such
  value.
- An error field on `RunningRequest`. A flag on the step stays in queue
  order.
- Send the cause to the client. The log keeps it.

## Consequences

- In-process callers of `wait()`, `get_step()`, or `steps()` now get
  `RequestFailedError`, not a `cancelled` result.
- A slow reader on `PDScheduler` can lose one buffered step, but its request
  always ends.
- An OOM that the engine handles still ends as `cancelled`, as in the
  preemption of [ADR-024](ADR-024-engine-oom-preemption.md).
