# Plan: PDScheduler (continuous batching for PD)

> Status: proposed.
> Date: 2026-05-16.
> Replaces `PDStreamingScheduler`'s single-request serial path with a
> multi-request scheduler that batches prefill + decode across
> concurrent requests.

## Context

Today's PD path (post-ADR-016) ships:

- `PrefillWorker.prefill_batch([requests]) -> [handoffs]` (batched prefill).
- `DecodeWorker.decode_batch([handoffs]) -> [[int]]` (batched decode).
- `Orchestrator.run(request) / run_stream(request)` (single-request).
- `PDStreamingScheduler` (HTTP-shaped adapter; pulls requests off a
  queue, runs one through `Orchestrator.run_stream` at a time).

The batched primitives exist but nothing drives them with multiple
concurrent requests. The HTTP server backed by `PDStreamingScheduler`
processes requests strictly sequentially — first request runs
prefill + decode end-to-end, second request waits until the first
is done.

This is the same throughput shape as a one-window food truck: one
order, one cook, one plate at a time. The improvement path is
exactly what `ContinuousScheduler` does for non-PD: an admission
queue + an engine loop that batches prefill chunks + batched decode
forwards over all in-flight requests.

## Goal

After this plan completes:

1. **`PDScheduler`** exposes the same `start / stop / submit / run`
   surface as `ContinuousScheduler` and `PDStreamingScheduler`, but
   internally:
   - Admits multiple requests concurrently.
   - Batches prefill via `PrefillWorker.prefill_batch`.
   - Batches decode via `DecodeWorker.decode_batch` (one forward per
     step, all alive decoders share it).
   - Streams tokens per-request to each handle's output queue.
   - Handles EOS / `max_tokens` / cancellation independently per
     request.
2. **Greedy parity vs `ContinuousScheduler`** on N concurrent
   requests: token-for-token identical output distributions.
3. **HTTP toggle**: `MINI_INFER_USE_PD=1` continues to switch the
   server to PD; the new scheduler replaces `PDStreamingScheduler`.
4. **Bench** comparing PD vs single-request PD throughput on real
   hardware (deferred until budget allows).

## Approach

Mirror `ContinuousScheduler`'s structure (one engine thread, one
running list, one waiting queue) but with the prefill and decode
phases distinguished — so each iteration handles both phases of
different requests in parallel.

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

### Slice 1 — Single-thread PDScheduler

Drop-in replacement for `PDStreamingScheduler`. Same engine-thread
pattern, but each step:

1. **Admit** waiting requests (up to `max_concurrent` total in-flight).
2. **Prefill batch**: collect requests in `PREFILLING` state into a
   list; call `PrefillWorker.prefill_batch(...)`; the resulting
   handoffs transition those requests to `DECODING`.
3. **Decode step**: for all requests in `DECODING` state, build the
   per-request `last_token` list; call `DecodeWorker.decode_batch_step`
   (a new lower-level entry point that runs ONE step over the existing
   decode pool; doesn't loop internally). Each request gets one new
   token; push to its handle's queue.
4. **Reap** requests that hit EOS / `max_tokens` / cancellation; emit
   terminal step.

Per-request state stored in a `RunningRequest`-shaped struct (reuse
`mini_infer.scheduler.request_state.RunningRequest` if it generalizes
cleanly; new dataclass otherwise).

**Critical new primitive**: `DecodeWorker.decode_batch_step`. Today's
`decode_batch` runs the full per-request loop internally. We need a
per-step variant the scheduler drives:

```python
class DecodeWorker:
    def add_handoff(self, handoff: KVHandoff) -> int:
        """Materialize handoff into cache; return slot_idx."""
        ...

    def step(self, slot_to_last_token: dict[int, int]) -> dict[int, int]:
        """Run one batched decode forward; return slot_idx -> new_token."""
        ...

    def remove_slot(self, slot_idx: int) -> None:
        """Free the slot (cancellation, EOS, max_tokens)."""
        ...
```

The current `decode_batch(handoffs)` becomes a convenience composition
over `add_handoff` + repeated `step` + `remove_slot`.

**Deliverable**: ~350 LoC source (`workers/pd_scheduler.py` + small
addition to `decode_worker.py`) + ~250 LoC tests. Single commit.

**Test contract**: greedy parity vs `ContinuousScheduler` on N=4
concurrent requests with different `max_tokens` budgets and one
mid-stream cancel.

### Slice 2 — Two-thread PDScheduler

Prefill and decode each run on their own thread. A handoff queue
sits between them. Decode pool grows when handoffs arrive, shrinks
on EOS / `max_tokens` / cancel.

Why two-thread matters for real PD: on a 2× GPU host (one prefill, one
decode), the GPUs idle in Variant A whenever the engine thread is
in the other phase. Two-thread lets the GPUs run continuously.

**Deliverable**: extends `pd_scheduler.py` with a thread mode; ~200
LoC additional + ~150 LoC tests. Single commit.

**Test contract**: same greedy parity vs Slice 1 on the same
workload. The output distribution must not depend on the threading
mode.

### Slice 3 — Server wiring

Replace `PDStreamingScheduler` in `api/server.py` with `PDScheduler`
when `MINI_INFER_USE_PD=1`. Add a `MINI_INFER_PD_MODE` env var to
pick between `single` / `serial` / `parallel` (`serial` = Variant A,
`parallel` = Variant B); default `parallel`.

**Deliverable**: ~50 LoC server edits + ~50 LoC API-level tests.
Single commit.

### Slice 4 — Bench + ADR

ADR-017 documents the design, the variants A / B, the
`decode_batch_step` decomposition, and the test contract.

Bench script `scripts/bench_pd_scheduler.py` runs N=1, 2, 4, 8, 16
concurrent requests through both `ContinuousScheduler` and
`PDScheduler` (Variants A and B); reports tok/s + per-request
latency.

CPU bench is informative (Variant A vs single-request) and free.
GPU bench (Variant B's parallel phases) requires Modal hardware
when budget allows.

**Deliverable**: ~300 LoC ADR-017 + ~200 LoC bench script. Single
commit.

## Decisions to confirm before implementation

1. **One thread vs two thread.** Slice 1 is single-thread; Slice 2
   adds the two-thread variant. We ship both. The default mode
   (Variant A vs B) is configurable.
2. **`max_concurrent`**: how many requests can be in-flight at once?
   Bounded by KV pool capacity. Default 16 (matches
   `ContinuousScheduler`).
3. **Admission policy**: FIFO from the waiting queue. Same as
   `ContinuousScheduler`. No prefix-cache-aware admission yet (that's
   a follow-up).
4. **Prefill chunking inside a batch**: do we chunk long prompts (per
   `ContinuousScheduler`'s chunked-prefill) or run them whole-prompt?
   For the first ship: whole-prompt. Chunked-prefill in PD is a
   non-trivial follow-up because the per-request prefill chunks would
   each produce a partial KV that the decode worker doesn't yet know
   how to consume mid-prefill.
5. **What happens when the KV pool fills**: today
   `ContinuousScheduler` raises `OOMError` and the API server returns
   503. Same behaviour here; no preemption.
6. **Failure mode for one request**: if `PrefillWorker.prefill_batch`
   raises (e.g., tokenizer error on one prompt), do we fail the whole
   batch or just that request? For the first ship: fail the whole
   batch and re-enqueue the survivors. Same recovery shape as a
   transient model error.

## Modal cost

- **Slice 1, 2, 3**: zero Modal cost. CPU multi-process + small models
  on M1 are enough for correctness.
- **Slice 4 bench on CPU**: zero Modal cost. Validates relative
  speedups between single-request and batched.
- **Slice 4 bench on GPU** (deferred until Modal spend cycle): ~$3-5
  for an end-to-end throughput run on 2× H100 with Qwen2.5-7B.

Total to ship Slices 1-4 (excluding the GPU bench): zero Modal
spend. The throughput claims that require real hardware get added
post-budget-reset.

## Risks

1. **Per-step decode-batch primitive doesn't exist yet.** Today's
   `decode_batch` runs the whole decode loop internally; we need
   `add_handoff` / `step` / `remove_slot`. Refactoring the existing
   path while keeping `decode_batch(handoffs)` working as a
   convenience wrapper is the cleanest approach but does touch
   `decode_worker.py`. Risk: the existing batched-parity tests need
   to keep passing during the refactor.
2. **Cancellation correctness across slots.** Today
   `Orchestrator.run_stream` cancels by stopping the loop; in a
   batched scheduler, mid-batch cancellation needs to mark the slot
   dead AND keep the forward shape constant (feed a no-op token,
   discard output). The pattern is already in
   `_decode_loop_batch`'s heterogeneous-max-tokens test — generalising
   it for explicit `cancel_event` is the actual work.
3. **Slot recycling at the KV pool level.** Each EOS / max-tokens
   frees the slot, but the cache's `remove_request(batch_idx)` shifts
   later indices down by 1. The scheduler's mapping `request_id ->
   slot_idx` has to track that shift. `ContinuousScheduler` already
   handles it; we copy the pattern.
4. **Variant B's two-thread synchronisation.** Producer/consumer queue
   between the prefill thread and the decode thread. Standard
   pattern; risk is correctness (no dropped handoffs on shutdown,
   no double-add).

## Files to create / modify

NEW source:

- `src/mini_infer/workers/pd_scheduler.py` — the new class. Slices 1
  + 2 land in the same file with a `mode="serial" | "parallel"`
  parameter.

EDIT source:

- `src/mini_infer/workers/decode_worker.py` — add `add_handoff`,
  `step`, `remove_slot`. Keep `decode_batch(handoffs)` as a
  convenience wrapper over the new primitives so existing tests
  stay green.
- `src/mini_infer/workers/__init__.py` — export `PDScheduler`.
- `src/mini_infer/api/server.py` — swap `PDStreamingScheduler` for
  `PDScheduler`. Add `MINI_INFER_PD_MODE` env var.
- `README.md` — update the HTTP server section to mention the new
  mode env var.

NEW docs:

- `docs/decisions/ADR-017-pd-scheduler.md` — the design ADR.

NEW scripts:

- `scripts/bench_pd_scheduler.py` — CPU bench (free) + Modal bench
  (deferred).

NEW tests:

- `tests/unit/test_pd_scheduler.py` — new test file for the
  scheduler. Per-slice test additions in the same file.

## Verification

Slice-by-slice:

1. **Slice 1**: 4 concurrent requests run through `PDScheduler`;
   greedy output token-for-token identical to running each through
   `ContinuousScheduler` separately. Block-pool fully free after
   each run.
2. **Slice 2**: same workload, same output, but with `mode="parallel"`.
   Asserts cross-thread correctness (no dropped handoffs, no double-
   emit). Performance is NOT asserted at this stage (just correctness).
3. **Slice 3**: existing API tests (`test_api.py`) pass with
   `MINI_INFER_USE_PD=1 MINI_INFER_PD_MODE=parallel`. 4 concurrent
   HTTP streaming requests all complete with the expected tokens.
4. **Slice 4**: CPU bench prints throughput tables; user-readable
   comparison vs single-request PD. ADR-017 records the design.

## What this gets us

After Slice 4: the PD path supports the same multi-request
concurrency as `ContinuousScheduler`. The HTTP server backed by PD
becomes a real concurrent serving target, not a single-request
demo. Internally, the engine demonstrates the production pattern of
"separate prefill + decode workers with continuous batching across
both" — the actual technique vLLM and SGLang use in their
disaggregated modes.

Aligned with the research-paper-engine niche under the "production
techniques in readable code" axis: the PD scheduler is the
textbook implementation of the continuous-batching-over-disaggregated-
pipeline pattern, ready for a reader who wants to follow how it
actually works.
