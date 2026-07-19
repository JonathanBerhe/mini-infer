# Plan: PDScheduler (continuous batching for PD)

> Status: implemented (Slices 1-4 shipped; see ADR-017).
> Date: 2026-05-16. Implemented: 2026-05-16 through 2026-05-17.
> Replaced `PDStreamingScheduler`'s single-request serial path with a
> multi-request scheduler that batches prefill + decode across
> concurrent requests. `PDStreamingScheduler` no longer exists in the
> codebase; `PDScheduler` is the only PD-path scheduler.

## Context (as of 2026-05-16, before this plan)

The PD path (post-ADR-016) shipped:

- `PrefillWorker.prefill_batch([requests]) -> [handoffs]` (batched prefill).
- `DecodeWorker.decode_batch([handoffs]) -> [[int]]` (batched decode).
- `Orchestrator.run(request) / run_stream(request)` (single-request).
- `PDStreamingScheduler` (HTTP-shaped adapter; pulls requests off a
  queue, runs one through `Orchestrator.run_stream` at a time).

The batched primitives existed but nothing drove them with multiple
concurrent requests. The HTTP server backed by `PDStreamingScheduler`
processed requests strictly sequentially: first request ran
prefill + decode end-to-end, second request waited until the first
was done.

This is the same throughput shape as a one-window food truck: one
order, one cook, one plate at a time. The improvement path is
exactly what `ContinuousScheduler` does for non-PD: an admission
queue + an engine loop that batches prefill chunks + batched decode
forwards over all in-flight requests.

## Goal (shipped)

1. **`PDScheduler`** (`src/mini_infer/workers/pd_scheduler.py`) exposes
   the same `start / stop / submit / run` surface as
   `ContinuousScheduler`. `PDStreamingScheduler` was removed entirely
   rather than kept alongside it. Internally `PDScheduler`:
   - Admits multiple requests concurrently.
   - Batches prefill via `PrefillWorker.prefill_batch`.
   - Batches decode via a long-lived `DecodeSession` (one forward per
     step, all alive decoders share it).
   - Streams tokens per-request to each handle's output queue.
   - Handles EOS / `max_tokens` / cancellation independently per
     request.
2. **Greedy parity vs `ContinuousScheduler`** on N concurrent requests:
   token-for-token identical output distributions. Covered by
   `tests/unit/test_pd_scheduler_multi.py`.
3. **HTTP toggle**: `MINI_INFER_USE_PD=1` switches the server to
   `PDScheduler` (`src/mini_infer/api/server.py`); `MINI_INFER_PD_MODE`
   picks `serial` or `parallel` (default `parallel`).
4. **Bench**: `scripts/bench_pd_scheduler.py` covers the CPU
   correctness/relative-speedup comparison. The 2x H100 Modal run
   comparing real cross-GPU phase overlap is still deferred until
   budget allows (see `roadmap-2026.md`); that part of this goal is
   not yet closed out.

## Approach

Mirrors `ContinuousScheduler`'s structure (one engine thread, one
running list, one waiting queue), with the prefill and decode phases
distinguished so each iteration handles both phases of different
requests in parallel.

| Phase | What runs | What stays the same |
|---|---|---|
| Waiting | request sits in the admission queue | same `Queue[RunningRequest]` shape |
| Prefilling | request is in the batch passed to `prefill_batch` next step | tokens not yet emitted; handle waits |
| Decoding | handoff is in the decode pool; one decode forward per step adds a token | tokens flow to handle's queue per step |
| Done | all tokens emitted (EOS / max_tokens / cancel) | handle drains terminal step, frees |

Two variants of the engine loop are realistic:

- **Variant A (one-thread, mixed step)**: a single engine thread runs
  one iteration that does prefill batch + decode batch in sequence.
  Same shape as `ContinuousScheduler` (which mixes chunked-prefill
  q-tokens and decode q-tokens in one packed forward). Simpler to
  reason about; bit-equivalent test surface.
- **Variant B (two-thread, parallel phases)**: a prefill thread + a
  decode thread, connected by a handoff queue. The prefill thread
  produces handoffs as fast as it can; the decode thread consumes
  them. The phases overlap. Closer to "real" PD value where two
  different GPUs do two different things concurrently.

We ship A first (cleaner correctness story) and B as the natural
follow-on (where the PD throughput win actually shows up).

## Phased execution

### Slice 1 (shipped): single-thread PDScheduler, mode="serial"

Drop-in replacement for `PDStreamingScheduler`. Same engine-thread
pattern, each step:

1. **Admit** waiting requests (up to `max_concurrent` total in-flight).
2. **Prefill batch**: collect requests in `PREFILLING` state into a
   list; call `PrefillWorker.prefill_batch(...)`; the resulting
   handoffs transition those requests to `DECODING`.
3. **Decode step**: for all requests in `DECODING` state, call
   `DecodeSession.step()`, which runs ONE batched decode forward over
   the existing pool and returns `slot_id -> new_token`. Each request
   gets one new token; push to its handle's queue.
4. **Reap** requests that hit EOS / `max_tokens` / cancellation; emit
   terminal step.

Per-request state is stored in `mini_infer.scheduler.request_state.RunningRequest`,
reused as planned.

The per-step decode primitive landed as a `DecodeSession` class
(`src/mini_infer/workers/decode_worker.py`), not as bare methods added
to `DecodeWorker`:

```python
class DecodeSession:
    def add_handoff(self, handoff: KVHandoff) -> int:
        """Materialize handoff into the cache; return slot_id."""

    def step(self) -> dict[int, int]:
        """Run one batched decode forward; return slot_id -> new_token."""

    def remove_slot(self, slot_id: int) -> None:
        """Free the slot (cancellation, EOS, max_tokens)."""
```

`DecodeWorker.decode_batch(handoffs)` stayed as a convenience
composition over a `DecodeSession` for callers that still want the
whole-loop entry point (existing tests kept passing through the
refactor, as planned).

**Delivered**: `src/mini_infer/workers/pd_scheduler.py` +
`DecodeSession` in `decode_worker.py`, plus
`tests/unit/test_pd_scheduler_multi.py`.

**Test contract met**: greedy parity vs `ContinuousScheduler` on
concurrent requests with different `max_tokens` budgets and mid-stream
cancellation.

### Slice 2 (shipped): two-thread PDScheduler, mode="parallel"

Prefill and decode each run on their own thread. A bounded handoff
queue sits between them (`maxsize=max_concurrent`, giving
backpressure). The decode pool grows when handoffs arrive, shrinks on
EOS / `max_tokens` / cancel.

Two-thread mode matters for real PD: on a 2-GPU host (one prefill, one
decode), the GPUs idle in `mode="serial"` whenever the engine thread is
in the other phase. Two-thread lets both GPUs run continuously.

**Delivered**: `pd_scheduler.py` gained the `mode: Literal["serial",
"parallel"]` constructor parameter; both modes live in the same file
and share the same test contract.

**Test contract met**: same greedy parity as Slice 1 on the same
workload, with `mode="parallel"`. Output distribution does not depend
on the threading mode.

### Slice 3 (shipped): server wiring

`PDStreamingScheduler` was removed from `api/server.py` (not kept as a
fallback); `PDScheduler` backs `/v1/completions` when
`MINI_INFER_USE_PD=1`. `MINI_INFER_PD_MODE` picks `serial` or
`parallel` (default `parallel`, matching this plan's stated default).

**Delivered**: `src/mini_infer/api/server.py` swap; API-level coverage
in the existing HTTP test suite.

### Slice 4 (partially shipped): bench + ADR

ADR-017 (`docs/decisions/ADR-017-pd-scheduler.md`, Accepted) documents
the design, both modes, the `DecodeSession` decomposition, and the
test contract.

`scripts/bench_pd_scheduler.py` runs concurrent requests through both
`ContinuousScheduler` and `PDScheduler` (both modes) and reports
tok/s + per-request latency on CPU. The CPU bench is in; the 2x H100
Modal run comparing real cross-GPU phase overlap has not been run yet
(still gated on Modal budget, see `roadmap-2026.md`).

CPU bench is informative (`mode="serial"` vs single-request) and free.
GPU bench (`mode="parallel"`'s phase overlap) requires Modal hardware
when budget allows.

## Decisions made

1. **One thread vs two thread.** Both shipped: `mode="serial"`
   (Slice 1) and `mode="parallel"` (Slice 2), selected by a
   constructor parameter. Server default is `parallel`
   (`MINI_INFER_PD_MODE`).
2. **`max_concurrent`**: bounded by KV pool capacity.
   `PDScheduler.DEFAULT_MAX_CONCURRENT = 16`, matching
   `ContinuousScheduler`. In `mode="parallel"` it also sizes the
   handoff queue (`maxsize=max_concurrent`), which is what provides
   backpressure between the two threads.
3. **Admission policy**: FIFO from the waiting queue, same as
   `ContinuousScheduler`. No prefix-cache-aware admission; still an
   open follow-up.
4. **Prefill chunking inside a batch**: shipped as whole-prompt, as
   planned. Chunked-prefill in PD remains an open follow-up for the
   same reason: per-request prefill chunks would each produce a
   partial KV that the decode session doesn't yet know how to consume
   mid-prefill.
5. **What happens when the KV pool fills**: as planned, no
   preemption. `PrefillWorker.prefill_batch` raising propagates as a
   batch failure (see decision 6); `ContinuousScheduler`'s separate
   OOM-preemption path (ADR-024) was not carried over to `PDScheduler`.
6. **Failure mode for one request**: shipped narrower than planned. If
   `PrefillWorker.prefill_batch` raises, `_run_prefill_batch` fails the
   whole batch and emits a terminal `cancelled` step for every request
   in it (`src/mini_infer/workers/pd_scheduler.py`, `_run_prefill_batch`).
   The plan's "re-enqueue the survivors" was not implemented; a batch
   failure currently drops every request in that batch rather than
   retrying any of them. Worth a follow-up if partial-batch prefill
   failures turn out to be common in practice.

## Modal cost

- **Slices 1, 2, 3**: shipped at zero Modal cost, as planned. CPU
  multi-process + small models on M1 covered correctness.
- **Slice 4 bench on CPU**: shipped at zero Modal cost. Validates
  relative speedups between single-request and batched.
- **Slice 4 bench on GPU** (still deferred): ~$3-5 for an end-to-end
  throughput run on 2x H100 with Qwen2.5-7B. Not yet run; see
  `roadmap-2026.md` for when Modal budget opens up for this.

Total spent to ship Slices 1-4 (excluding the still-deferred GPU
bench): zero Modal spend, as planned.

## Risks (as identified during planning; see ADR-017 for how each landed)

1. **Per-step decode-batch primitive doesn't exist yet.** Resolved:
   landed as the `DecodeSession` class (`add_handoff` / `step` /
   `remove_slot`) in `decode_worker.py`, with `decode_batch(handoffs)`
   kept as a convenience wrapper. Existing batched-parity tests stayed
   green through the refactor.
2. **Cancellation correctness across slots.** Resolved: mid-batch
   cancellation marks the slot dead and keeps the forward shape
   constant, generalizing the pattern already exercised by
   `_decode_loop_batch`'s heterogeneous-max-tokens test.
3. **Slot recycling at the KV pool level.** Resolved: the scheduler's
   `slot_id -> RunningRequest` mapping tracks the index shift from
   `remove_request(batch_idx)`, following the pattern already used by
   `ContinuousScheduler`.
4. **Parallel mode's two-thread synchronization.** Resolved: a bounded
   producer/consumer queue (`_handoff_queue`, `maxsize=max_concurrent`)
   connects the prefill and decode threads; the bound is also what
   provides backpressure.

## Files created / modified (as shipped)

Source:

- `src/mini_infer/workers/pd_scheduler.py` (new): `PDScheduler`.
  Slices 1 + 2 landed in the same file with the
  `mode="serial" | "parallel"` parameter.
- `src/mini_infer/workers/decode_worker.py` (edited): added the
  `DecodeSession` class (`add_handoff`, `step`, `remove_slot`).
  `decode_batch(handoffs)` kept as a convenience wrapper.
- `src/mini_infer/workers/__init__.py` (edited): exports `PDScheduler`.
- `src/mini_infer/api/server.py` (edited): `PDStreamingScheduler`
  removed and replaced by `PDScheduler`. Added `MINI_INFER_PD_MODE`.
- `README.md` (edited): documents `MINI_INFER_USE_PD` and
  `MINI_INFER_PD_MODE` in the HTTP server section.

Docs:

- `docs/decisions/ADR-017-pd-scheduler.md` (new): the design ADR,
  Accepted.

Scripts:

- `scripts/bench_pd_scheduler.py` (new): CPU bench, shipped. Modal GPU
  bench still deferred.

Tests:

- `tests/unit/test_pd_scheduler_multi.py` (new): the scheduler's test
  file (named `_multi` rather than the plan's `test_pd_scheduler.py`).

## Verification (results)

1. **Slice 1**: concurrent requests through `PDScheduler` in
   `mode="serial"` produce greedy output token-for-token identical to
   running each through `ContinuousScheduler` separately. Block-pool
   fully frees after each run. Covered by
   `tests/unit/test_pd_scheduler_multi.py`.
2. **Slice 2**: same workload, same output, with `mode="parallel"`.
   Cross-thread correctness holds (no dropped handoffs, no double
   emit).
3. **Slice 3**: the server's HTTP test coverage passes with
   `MINI_INFER_USE_PD=1 MINI_INFER_PD_MODE=parallel`; concurrent HTTP
   streaming requests complete with the expected tokens.
4. **Slice 4**: the CPU bench prints throughput tables comparing
   `PDScheduler` against `ContinuousScheduler` and against
   single-request PD. ADR-017 records the design. The GPU bench is
   still outstanding.

## What this gets us

The PD path supports the same multi-request concurrency as
`ContinuousScheduler`. The HTTP server backed by PD is a real
concurrent serving target, not a single-request demo. Internally, the
engine demonstrates the production pattern of "separate prefill and
decode workers with continuous batching across both": the actual
technique vLLM and SGLang use in their disaggregated modes.

Aligned with the research-paper-engine niche under the "production
techniques in readable code" axis: the PD scheduler is the
textbook implementation of the continuous-batching-over-disaggregated-
pipeline pattern, ready for a reader who wants to follow how it
actually works.
