# ADR-017: Continuous-batching scheduler for PD (PDScheduler)

Date: 2026-05-17
Status: Accepted

## Context

ADR-016 shipped the disaggregated PD pipeline as a worker-level
abstraction: `PrefillWorker`, `DecodeWorker`, `KVHandoff`, plus a
single-request `Orchestrator` and a thin HTTP adapter
(`PDStreamingScheduler`). The HTTP server processed requests strictly
sequentially — first request prefill + decode end-to-end, second
request waits until the first is done. Same shape as a one-window food
truck.

Meanwhile the batched primitives existed:

- `PrefillWorker.prefill_batch([requests]) -> [handoffs]` runs N
  prompts through one packed-varlen forward.
- `DecodeWorker.decode_batch([handoffs]) -> [[int]]` runs the decode
  loop over a shared session.

But nothing drove them with multiple concurrent requests. The HTTP path
backed by `PDStreamingScheduler` invoked the single-request
`Orchestrator.run_stream`; concurrent users serialised at the worker.

We need a real scheduler that:

1. Admits multiple in-flight requests concurrently.
2. Batches prefill via `prefill_batch`.
3. Batches decode via a long-lived shared session, adding new requests
   mid-decode and removing terminated ones.
4. Streams tokens per request with EOS / `max_tokens` / cancellation
   handled independently.
5. Matches the HTTP server's existing
   `start / stop / submit / run` surface so swapping it in is one
   import change.

The shape that achieves all of this is the same shape
`ContinuousScheduler` uses for the non-PD path: one engine loop that
admits, batches both phases, emits tokens, reaps. On the PD path we
have one extra design choice — whether the two phases share an engine
thread (simpler) or run on separate threads (real cross-GPU overlap).

## Decision

Ship a `PDScheduler` class in `mini_infer/workers/pd_scheduler.py`
with the `start / stop / submit / run` surface. Internally:

1. **A new per-step primitive on `DecodeWorker`: `DecodeSession`.**
   Owns the paged cache + per-slot bookkeeping. Three methods:
   - `add_handoff(handoff) -> slot_id`: materialize handoff KV into
     a fresh slot; return a stable slot id.
   - `step() -> {slot_id: new_token}`: one batched decode forward;
     return new tokens per slot.
   - `remove_slot(slot_id)`: free blocks; cache compacts and
     subsequent slot_ids stay stable (session re-maps to the new
     batch_idx layout internally).
2. **`DecodeWorker.decode(handoff)` and `decode_batch([handoffs])`
   refactored as session wrappers.** Existing tests untouched. The
   session is the primitive; the single-call APIs are conveniences.
3. **Two threading variants in one class, selected by `mode`:**
   - **`mode="serial"`** (default): one engine thread runs
     `_engine_loop` — admit + batched prefill + batched decode step +
     reap, in sequence per iteration. Simplest correctness story.
   - **`mode="parallel"`**: two engine threads. Prefill thread admits
     + runs `prefill_batch`, puts `(request, handoff)` into a
     bounded handoff queue (`maxsize = max_concurrent`); decode
     thread drains the queue into the long-lived session and runs
     `session.step()` per iteration. The queue's bound provides
     backpressure: when decode is saturated, the prefill thread
     blocks on its `put`, which naturally bounds total in-flight.
4. **Output is identical between modes.** Greedy decoding is
   deterministic, so the threading shape only affects timing, not
   tokens. Verified by a cross-mode parity test.

The `MINI_INFER_PD_MODE=serial|parallel` env var picks the variant at
server start. Default `parallel` because that's where the PD
throughput win is (on multi-GPU hardware the two phases run on
different devices); `serial` exists for debugging + comparison.

## Why per-step `DecodeSession` (not just refactoring `decode_batch`)

`decode_batch(handoffs)` runs the whole decode loop in one call: it
materializes every handoff, loops over `(step → emit → check
termination)` until everyone's done, then frees all slots. The
scheduler can't use this because it needs to:

- **Add a new handoff mid-decode** — a new prefill completes while
  request A is on decode step 5; A and the new request now share
  the next forward.
- **Remove a slot mid-decode** — request B hits EOS at step 3; the
  cache compacts; the next forward runs only over the survivors.
- **Drive the loop externally** — the scheduler owns the engine
  loop, not the worker.

Refactoring without breaking existing tests: the session is the new
primitive; `decode_batch(handoffs)` becomes a convenience wrapper
that calls `add_handoff` N times, drives `step` in a loop with
per-slot alive tracking, then `remove_slot` on each. Same external
behaviour, same test results — but now the same primitives are
exposed for the scheduler to drive.

## Why serial and parallel both

Three options were considered:

- **Serial only**: one engine thread, both phases sequential. Simple
  to reason about; bit-equivalent tests; ships fast. Misses the PD
  throughput win on multi-GPU hardware where the two phases run on
  different devices.
- **Parallel only**: two engine threads always. Cleaner story for the
  "real PD value" case. But for the single-GPU dev path (M1 / single
  H100), the threading adds overhead with no benefit.
- **Both, configurable** (this decision): one class, `mode` parameter,
  same API. The parity tests assert both produce identical tokens.
  Production picks parallel on multi-GPU hosts; debugging / dev
  picks serial.

We picked the third. Cost: ~60 LoC of duplicated loop logic
(`_engine_loop` vs `_prefill_thread_loop` + `_decode_thread_loop`).
Benefit: real PD throughput is reachable, single-thread debugging is
still available, and one env-var toggle exposes both at the API
layer.

## Threading model (parallel mode)

State partitioning, no locks needed because each piece is owned by
exactly one thread:

| State | Owner |
|---|---|
| `_waiting: Queue[RunningRequest]` | API thread (producer) + prefill thread (consumer); `queue.Queue` is thread-safe |
| `_prefilling: list[RunningRequest]` | Prefill thread only |
| `_handoff_queue: Queue[tuple[RunningRequest, KVHandoff]]` | Prefill thread (producer) + decode thread (consumer); thread-safe |
| `_decoding: dict[int, RunningRequest]` | Decode thread only |
| `_handoffs: dict[int, KVHandoff]` | Decode thread only |
| `_session: DecodeSession` | Decode thread only |
| `_stop_event: threading.Event` | API thread sets; both engine threads check |

Cancellation flow (asynchronous):

- API thread calls `handle.cancel()` → sets `running.cancel_event`
- For a request in `_waiting`: prefill thread checks
  `cancel_event` after admit; skips + emits terminal.
- For a request prefilled but not yet absorbed: decode thread
  checks `cancel_event` when reading the handoff queue; emits
  terminal without adding to session.
- For a request decoding: decode thread checks `cancel_event` per
  step (in `_reap_cancelled`); calls `_terminate_slot`.

Shutdown flow:

- API thread sets `_stop_event` and joins both engine threads
  (each polls the event in its loop and exits at the next
  iteration boundary).
- After join: `stop()` drains any remaining state — active session
  slots (frees blocks), pending handoffs in the handoff queue
  (terminal step), in-progress prefilling batch (terminal step).
  No leaked blocks, no orphaned handles.

## Backpressure

In parallel mode the prefill thread can produce handoffs faster than
the decode thread can consume them. Without bounds, the decode
session would grow unboundedly and exhaust HBM.

The bound: `_handoff_queue = queue.Queue(maxsize=max_concurrent)`.
When decode is saturated, the prefill thread blocks on its `put`. It
keeps polling the stop event so shutdown is responsive, but it
otherwise waits for capacity. The decode thread continues consuming
at its natural pace; once a slot terminates and frees, the queue has
room and the prefill thread unblocks.

Worst-case total in-flight: roughly `2 × max_concurrent` (prefill
holding one batch + handoff queue full + session full). For
`max_concurrent = 16` (the default), that's 32 worst case, well
inside any reasonable KV pool.

## Numerical correctness

Two parity contracts:

1. **Both PD modes produce the same tokens as `ContinuousScheduler`**
   on the same prompt + same model under greedy decoding. Validated
   in `tests/unit/test_pd_scheduler_multi.py` for N=1 and N=3
   concurrent requests.
2. **Serial mode and parallel mode produce identical tokens** on the
   same workload. Threading is invisible at the output level under
   greedy decoding. Validated by an explicit cross-mode test.

These are bit-parity contracts; any drift indicates a bug.

## Consequences

**Positive**:

- The HTTP path backed by PD now supports many concurrent users.
  `MINI_INFER_USE_PD=1` on a real multi-GPU host gives the PD
  throughput shape vLLM and SGLang's disaggregated modes deliver.
- The `DecodeSession` primitive is composable beyond the
  scheduler: a future caller (e.g. a benchmark harness that wants
  to interleave operations) can drive it directly.
- The single-thread variant stays simple — useful for unit tests
  + debugging the engine without thread interleaving complications.
- The `PDStreamingScheduler` (single-request adapter from ADR-016)
  is deleted: subsumed by `PDScheduler`'s degenerate-batch case,
  and the surface is identical so no caller breakage.

**Negative**:

- Two threading paths to maintain. Mitigated by: same class, same
  state, same API surface; the parity tests catch divergence.
- Block-pool ownership is still shared between prefill and decode
  (same underlying `ModelRunner`). On a 2-GPU host where each phase
  should own its own GPU, ADR-016's note still applies: the
  scheduler orchestrates the calls; the cross-device coordination
  is at the worker level (each worker has its own paged cache, KV
  travels through the handoff). The PD throughput shape works;
  GPU placement is a deployment concern.
- No prefix-cache-aware admission. The waiting queue is FIFO; if
  multiple requests share a prefix, the engine doesn't take
  advantage. This is a known follow-up (`ContinuousScheduler`
  doesn't do this either today).
- No chunked-prefill within PD. The whole prompt prefills in one
  forward per request. Long prompts will be slow on the prefill
  side; chunked prefill in PD is a non-trivial follow-up because
  partial KV mid-prefill is not yet a supported handoff shape.

**Reversibility**: clean. The default API server still uses
`ContinuousScheduler`; `MINI_INFER_USE_PD` is opt-in. Removing
`PDScheduler` (if we ever wanted to) restores the pre-Slice-1
behaviour with one revert.

## Validation

- **Unit (CPU, single-thread)**: serial mode parity vs
  `ContinuousScheduler` on N=1 and N=3 concurrent requests. Block
  pool returns to fully free.
- **Unit (CPU, two-thread)**: same workloads in parallel mode.
  Cross-mode parity test asserts serial and parallel produce
  identical tokens.
- **Per-request termination**: heterogeneous `max_tokens` per request
  (3 / 5 / 8) test verifies each slot terminates independently and
  freed slots don't block the survivors.
- **Cancellation**: mid-stream `handle.cancel()` with two other
  requests still streaming; cancelled emits terminal, survivors
  complete cleanly, block pool returns to fully free.
- **Lifecycle**: `start()` and `stop()` idempotent in both modes.
- **HTTP integration**: existing `test_api.py` suite passes
  unchanged (default backend is still `ContinuousScheduler`);
  manual smoke with `MINI_INFER_USE_PD=1 MINI_INFER_PD_MODE=...`
  serves `/v1/completions` correctly through both PD modes.

Tests under `tests/unit/`. Real-hardware throughput validation
(parallel-mode 2× speedup on a 2-GPU host where the phases overlap
on different devices) is gated on Modal budget; the bench script
`scripts/bench_pd_scheduler.py` runs the comparison locally on
CPU (informative trend, not the headline number).

## Pointers

- `src/mini_infer/workers/pd_scheduler.py` — `PDScheduler` class.
- `src/mini_infer/workers/decode_worker.py` — `DecodeSession` + the
  refactored `decode` / `decode_batch` wrappers.
- `src/mini_infer/api/server.py` — `MINI_INFER_USE_PD` /
  `MINI_INFER_PD_MODE` env-var dispatch.
- `tests/unit/test_pd_scheduler_multi.py` — parametrized parity
  tests (serial + parallel) + cross-mode parity.
- `scripts/bench_pd_scheduler.py` — concurrency-sweep bench harness.

## Follow-ups

- **Prefix-cache-aware admission.** Sort the waiting queue by
  prefix overlap with the running set so shared-prefix requests
  batch together. Same opportunity exists in `ContinuousScheduler`.
- **Chunked prefill in PD.** Long prompts currently prefill whole-
  shot; chunked-prefill would let other requests' decode steps
  interleave with the long prompt's prefill chunks. Requires
  partial-KV handoff (the decode side has to know that the prefill
  isn't done yet); non-trivial.
- **Real-hardware throughput bench.** 2× H100 Modal run comparing
  parallel mode vs serial mode on a concurrent workload. Gated on
  Modal budget; the script + harness ship now.
- **Mid-batch slot compaction across cache abstractions.** Today
  `cache.remove_request(batch_idx)` shifts later indices down;
  the session re-maps. If the cache layout ever changes (e.g.
  to a vLLM-style block table with stable indices), the session's
  re-mapping logic can simplify.
