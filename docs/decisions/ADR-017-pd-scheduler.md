# ADR-017: Continuous-batching scheduler for PD (PDScheduler)

Date: 2026-05-17
Status: Accepted

## Context

ADR-016 shipped the disaggregated PD pipeline (prefill and decode on
separate workers): `PrefillWorker`, `DecodeWorker`, `KVHandoff`, a
single-request `Orchestrator` and a thin HTTP adapter
(`PDStreamingScheduler`). The HTTP server ran requests one at a time,
although batched primitives existed.
`PrefillWorker.prefill_batch([requests]) -> [handoffs]` runs N prompts
through one packed-varlen forward, and
`DecodeWorker.decode_batch([handoffs]) -> [[int]]` runs the decode loop
over a shared session. But nothing drove them with concurrent requests.

The scheduler must:

1. Admit multiple in-flight requests at the same time.
2. Batch prefill through `prefill_batch`.
3. Batch decode in a long-lived session that adds and removes requests
   during decode.
4. Stream tokens per request, with EOS, `max_tokens` and cancellation
   handled per request.
5. Keep the `start / stop / submit / run` surface of the HTTP server, so
   the swap is one import change.

`ContinuousScheduler` already has this shape for the non-PD path: one
engine loop that admits, batches both phases, emits tokens and reaps. PD
adds one choice: one engine thread for both phases, or one thread per phase
for real cross-GPU overlap.

## Decision

Ship `PDScheduler` in `mini_infer/workers/pd_scheduler.py` with the
`start / stop / submit / run` surface.

1. A new per-step primitive on `DecodeWorker`: `DecodeSession`. It owns the
   paged cache and the bookkeeping per slot (one decoding request):
   - `add_handoff(handoff) -> slot_id` loads the handoff KV into a new
     slot and returns a stable slot id.
   - `step() -> {slot_id: new_token}` runs one batched decode forward.
   - `remove_slot(slot_id)` frees the blocks. The cache compacts, and the
     session maps the stable slot ids to the new batch indices.
2. `DecodeWorker.decode(handoff)` and `decode_batch([handoffs])` become
   wrappers over the session. Existing tests do not change.
3. Two threading variants in one class, selected by `mode`:
   - `mode="serial"`: one engine thread runs `_engine_loop`. Each iteration
     admits, runs batched prefill, runs one batched decode step and reaps.
   - `mode="parallel"`: a prefill thread admits, runs `prefill_batch` and
     puts `(request, handoff)` into a bounded handoff queue
     (`maxsize = max_concurrent`). A decode thread moves handoffs from the
     queue into the session and runs `session.step()` per iteration.
4. Output is identical between modes. Greedy decoding is deterministic, so
   the threading changes only timing. A cross-mode parity test proves it.

`MINI_INFER_PD_MODE=serial|parallel` picks the variant at server start.
The default is `parallel`, because on multi-GPU hardware the two phases run
on different devices. `serial` is for debugging and comparison.

`decode_batch(handoffs)` runs the whole decode loop in one call. The
scheduler must instead add a slot when a prefill completes, remove one when
a request hits EOS, and own the loop. So `decode_batch` now calls
`add_handoff`, `step` and `remove_slot`.

In parallel mode, each piece of state has one owner thread, so no locks are
needed. The prefill thread owns `_prefilling`. The decode thread owns
`_decoding`, `_handoffs` and `_session`. Two thread-safe queues connect the
threads: `_waiting` (API to prefill) and `_handoff_queue` (prefill to
decode). The API thread sets `_stop_event`, and both engine threads read
it.

Cancellation: `handle.cancel()` sets `running.cancel_event`. The prefill
thread reads it after admit. The decode thread reads it when it takes a
handoff from the queue, and per step in `_reap_cancelled`. Each path emits a
terminal step. On shutdown, the API thread joins both engine threads. Then
`stop()` frees the active session slots and ends the pending handoffs and
the prefill batch with a terminal step, so no blocks leak.

Backpressure: prefill can produce handoffs faster than decode consumes
them. The bound is `_handoff_queue = queue.Queue(maxsize=max_concurrent)`.
When decode is full, the prefill thread blocks on its `put`, but it still
polls the stop event. Worst case, roughly `2 × max_concurrent` requests are
in flight. For `max_concurrent = 16` (the default), that is 32.

## Why serial and parallel both

Serial only is simple, but it misses the PD throughput gain on multi-GPU
hardware. Parallel only adds thread overhead with no benefit on a
single-GPU development host (M1 or single H100). Both modes in one class
cost ~60 LoC of duplicated loop logic (`_engine_loop` against
`_prefill_thread_loop` + `_decode_thread_loop`).

## Consequences

- The PD HTTP path supports many concurrent users. On a multi-GPU host,
  `MINI_INFER_USE_PD=1` gives the throughput shape of the disaggregated
  modes of vLLM and SGLang.
- `PDScheduler` replaces `PDStreamingScheduler` (ADR-016). The surface is
  identical, so no caller breaks.
- Two threading paths need maintenance. They share the class, the state and
  the API, and the parity tests catch drift.
- Prefill and decode still share one `ModelRunner` block pool. As in
  ADR-016, the cross-device coordination is at the worker level.
- Admission is FIFO with no prefix-cache awareness, as in
  `ContinuousScheduler`.
- PD has no chunked prefill. Chunked prefill needs a partial-KV handoff,
  which does not exist yet.
- Reverting is clean. The default API server still uses
  `ContinuousScheduler`, and `MINI_INFER_USE_PD` is opt-in.

The test contract is in `tests/unit/workers/test_pd_scheduler_multi.py`. Both modes
give the same tokens as `ContinuousScheduler` under greedy decoding, for
N=1 and N=3 concurrent requests, and give identical tokens to each other.
After each run, the block pool is fully free. Requests with `max_tokens` of
3, 5 and 8 terminate independently. A `handle.cancel()` during a stream
emits a terminal step, and the two other requests complete. `start()` and
`stop()` are idempotent in both modes.

The `test_api.py` suite passes unchanged. A manual smoke with
`MINI_INFER_USE_PD=1 MINI_INFER_PD_MODE=...` serves `/v1/completions`
through both modes. `scripts/bench/bench_pd_scheduler.py` compares the
schedulers on CPU. The 2× H100 throughput run of parallel against serial
mode did not run yet.
