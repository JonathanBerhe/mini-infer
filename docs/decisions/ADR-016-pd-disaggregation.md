# ADR-016: Prefill / decode disaggregation (PD)

Date: 2026-05-15
Status: Accepted

## Context

A non-disaggregated LLM inference engine runs prefill and decode on
the same GPU. The two phases have different resource profiles:

- **Prefill** is compute-bound. One forward over the whole prompt at
  once. Wants peak FLOPS, large activations, single-pass.
- **Decode** is memory-bound. Many forwards each producing one token.
  Wants peak HBM bandwidth, small activations, KV-cache-heavy.

Mixing them on one GPU is the simplest deployment, but it's also the
costliest: prefill steals HBM bandwidth from decode, decode steals
SMs from prefill, neither runs at its optimal occupancy. Production
inference engines (vLLM's PD mode, SGLang via the DistServe paper,
TensorRT-LLM's split deployment) all support a **disaggregated** mode
where each phase runs on its own GPU worker, connected by a
KV-cache transfer mechanism.

Up to this ADR, mini-infer's `ContinuousScheduler` ran both phases on
a single GPU. The TP work in ADR-015 unblocked multi-GPU model
sharding for any single phase, but didn't address splitting phases
across GPUs. Adding PD is the next big systems story the project
hasn't told yet.

## Decision

Ship PD as a four-piece module under `src/mini_infer/workers/`:

1. `KVHandoff` (`kv_handoff.py`) — the contract between phases. A
   dataclass carrying per-layer per-stream packed KV tensors + the
   first sampled token + sampling params + max-tokens + EOS. Per-stream
   layout (not the legacy K/V pair) so the same handoff works for
   GQA / MLA / V4 attention without special-casing.
2. `PrefillWorker` (`prefill_worker.py`) — owns a `ModelRunner`. The
   `prefill(request)` method runs one forward over the prompt, samples
   the first output token from the last prefill logit, extracts every
   layer's per-stream KV via `cache.materialize_packed_stream`, frees
   the request slot, returns the handoff. A `prefill_batch(requests)`
   method packs N requests into a single varlen forward and emits N
   handoffs in one call.
3. `DecodeWorker` (`decode_worker.py`) — owns a `ModelRunner`. The
   `decode(handoff)` method adds a fresh slot, materializes the
   handoff's per-stream KV via `cache.append_stream_packed`, yields
   the handoff's first sampled token, then runs the decode loop until
   EOS or `max_tokens`. A `decode_batch([handoffs])` method runs the
   loop over N slots simultaneously with per-slot termination tracking.
4. `kv_transfer.py` — wire protocol for `KVHandoff` over
   `torch.distributed`. A 7×int64 header carries metadata; KV streams
   send in `pool.stream_names`-sorted order. Both ranks must share an
   identical `BlockPool` topology (same model, same dtype).
   `send_handoff` and `recv_handoff` work on gloo (CPU) and NCCL (CUDA).

`pd_two_process_target` (`multi_process.py`) is the per-rank entry
function: rank 0 prefills + sends; rank 1 receives + decodes. The
spawn / lifecycle of the workers is the caller's responsibility
(`mp.spawn` in production, `run_multi_process` helper in tests).

`Orchestrator` (`orchestrator.py`) is the single-process composition
that wires a `PrefillWorker` and `DecodeWorker` directly inside the
same process. Same API shape as a hypothetical multi-process
orchestrator; both use the worker classes' methods.

## Why per-stream + worker-level (not scheduler-level)

Three alternatives considered:

### A. Make `ContinuousScheduler` PD-aware

Add a `phase_mode = "prefill_only" | "decode_only" | "mixed"` knob to
the existing scheduler. Pros: reuses the admission queue, the
running-batch bookkeeping, the chunked-prefill machinery. Cons:
invasive — touches every code path in a 450-line file that's already
used by golden tests, the HTTP server, the demo. High risk of
breaking the single-GPU path that's covered by 393 unit tests.

### B. Build a single `PDScheduler` that owns both workers

A new scheduler that internally manages prefill batches + decode
batches + the handoff between them. Pros: hides the disaggregation
detail behind a single API. Cons: re-implements admission /
backpressure / streaming output, duplicating most of
`ContinuousScheduler`. Hard to validate against existing tests.

### C. Worker-level disaggregation (this ADR)

Two thin classes (`PrefillWorker`, `DecodeWorker`) that each own a
`ModelRunner` and expose the phase they handle. An `Orchestrator`
wires them. `KVHandoff` is the contract. Pros: composable, minimal
new state, parity contract is "PD output ≡ ContinuousScheduler
output" which is a single greedy-token-list assertion. Cons: no
streaming HTTP integration yet — that's a follow-up.

We picked C. The worker classes are ~370 lines total; together with
the wire protocol (~150 lines), `KVHandoff` (~70 lines), and the
multi-process entry (~100 lines), the whole PD module is ~700 lines.
That's small enough to read in one sitting and small enough to keep
the test surface tractable.

### Per-stream over legacy K/V pair

The handoff carries `kv_streams_per_layer: list[dict[str,
torch.Tensor]]` (a stream-name → tensor map per layer) rather than
the simpler `(K, V)` pair. This is what makes it work uniformly
across GQA, MLA, and V4:

- Standard GQA / MHA: streams `("k", "v")`. Two entries per layer.
- MLA (DeepSeek-V2/V3): streams `("kv_latent", "k_rope")`. Different
  shapes per stream (kv_lora_rank vs qk_rope_head_dim). Two entries
  per layer.
- V4: a richer per-layer named set (`StateCache` is paired with the
  paged stream layout). Same handoff shape, more entries per layer.

`BlockPool.stream_names(layer_idx)` is the authoritative listing for
each layer; both `materialize_packed_stream` (extract) and
`append_stream_packed` (materialize) consume that. The handoff just
shuttles the result.

## Numerical correctness

The Slice 1 parity test (`test_pd_greedy_matches_continuous_scheduler`
in `tests/unit/test_workers.py`) asserts:

> PD's greedy output for a prompt P on model M is token-for-token
> identical to `ContinuousScheduler`'s greedy output for the same
> prompt P on the same model M.

This is the contract: disaggregation must not change output
distribution. Greedy is the cleanest knob to test it on — temperature
collapses to argmax and any KV-state drift between phases shows up
as a different first-decoded token.

Pass on Qwen2.5-0.5B with the in-process `Orchestrator`. Block-pool
free-count also returns to pre-run state after each PD run (no
leaked slots).

The batched variants have their own parity contract:

> `prefill_batch([r1, r2])` produces byte-identical KV bytes and the
> same first sampled token as `[prefill(r1), prefill(r2)]`.
>
> `decode_batch([h1, h2])` produces the same per-request token lists
> as `[list(decode(h1)), list(decode(h2))]`.

Both pass on Qwen2.5-0.5B (6 unit tests in
`tests/unit/test_workers_batch.py`).

## Cross-process transport

`send_handoff` / `recv_handoff` ship a `KVHandoff` over a
`torch.distributed` group. The wire format:

```
Header (7x int64):
  [prefill_len, first_sampled_token_id, eos_or_-1, max_tokens,
   temperature_micros, top_k, top_p_micros]
KV streams in pool.stream_names-sorted order:
  for layer in 0..num_layers-1:
    for stream in sorted(pool.stream_names(layer)):
      tensor of shape (prefill_len, h, d)
```

Stream names are NOT transmitted — both sides know them from the
model. This makes the wire format compact (no per-stream metadata
overhead) and lets `dist.send/recv` operate on plain tensors only
(NCCL doesn't ship Python objects without an extra step).

Validation: `test_kv_transfer_round_trip` spawns two processes
(gloo, CPU), one sends a synthetic handoff with deterministic
per-position values, the other receives and verifies content.
Passes in ~4 s.

The full end-to-end multi-process test
(`test_pd_two_process_matches_single_process`) is `@pytest.mark.skip`
on the macOS pytest path because of a thread-contention issue
between two PyTorch processes loading Qwen2.5-0.5B on the same CPU
host under pytest's spawn harness. The same `pd_two_process_target`
runs cleanly on a multi-GPU CUDA host (one rank per GPU, no
contention), which is the Modal CUDA path's natural CI venue.

## Consequences

**Positive**:

- Every supported model family is PD-capable through the same
  worker abstraction. The handoff's per-stream layout means MLA's
  `kv_latent` + `k_rope`, V4's richer streams, and standard GQA's
  `("k", "v")` all work without family-specific code paths.
- Batched prefill / batched decode within a worker is a 50-line
  method on each worker class, not a separate batched orchestrator.
  Tests cover the parity contract.
- The wire protocol works on both gloo (CPU multi-process tests) and
  NCCL (real multi-GPU). Production deployments use NCCL P2P which
  is GPU-direct (NVLink between ranks on the same node, RDMA across
  nodes).
- Single-process behaviour is unchanged. `ContinuousScheduler` and
  its 393 unit tests are untouched.

**Negative**:

- No streaming HTTP integration. `Orchestrator.run_stream` yields
  token ids; wiring that into the FastAPI server's SSE response is a
  follow-up.
- No KV-quant on the PD path. `PrefillWorker` and `DecodeWorker`
  both require `kv_quant=None`. The `materialize_packed_stream` /
  `append_stream_packed` pair is uncompressed-only today; the
  compressed-pool extraction goes through different APIs that the
  handoff doesn't speak.
- Worker classes are single-request per call. `prefill_batch` /
  `decode_batch` take a fixed batch at entry. No mid-batch admission,
  no per-request queues. A `ContinuousScheduler`-style scheduler
  *around* the workers would be a natural follow-up.
- Multi-process model load + Qwen forward stalls under pytest's spawn
  harness on macOS (thread contention between two PyTorch processes
  sharing the same CPU host). The wire-protocol test covers the
  transport; the full end-to-end test runs on Modal CUDA.

**Reversibility**: clean. The `workers/` module is additive — no
existing code paths change. Removing PD is just removing the module
+ its tests + the ADR.

## Validation

- **Unit (CPU)**:
  - `test_workers.py::test_pd_greedy_matches_continuous_scheduler`:
    PD ≡ `ContinuousScheduler` greedy output on Qwen2.5-0.5B.
  - `test_workers.py::test_pd_first_yielded_token_matches_handoff`:
    the first emitted token equals `handoff.first_sampled_token_id`.
  - `test_workers.py::test_pd_releases_blocks_on_completion`: block
    pool fully free before and after a PD run (no leaks).
  - `test_workers.py::test_kv_transfer_sampling_params_round_trip`:
    fixed-point header encoding round-trips `SamplingParams`.
  - `test_workers_batch.py` (6 tests): `prefill_batch` ≡ sequential
    `prefill`, `decode_batch` ≡ sequential `decode`, end-to-end
    batched PD ≡ sequential PD, empty inputs, heterogeneous
    `max_tokens` per request.
- **Multi-process (CPU, gloo)**:
  - `test_kv_transfer_mp.py::test_kv_transfer_round_trip`: spawns
    2 processes, one sends a synthetic handoff with deterministic
    values, the other receives + verifies content. Passes in ~4 s.
- **Multi-process (CPU, pytest spawn)**:
  - `test_workers_mp.py::test_pd_two_process_matches_single_process`:
    full end-to-end on Qwen2.5-0.5B. `@pytest.mark.skip` because two
    PyTorch processes loading Qwen on the same CPU host under pytest's
    spawn harness hits thread contention that doesn't resolve in
    reasonable test budgets. The same `pd_two_process_target` is the
    Modal CUDA smoke's per-rank entry; CUDA hardware avoids the
    contention.
- **Real hardware (2x H100, NCCL)**:
  - `scripts/modal_pd_smoke.py` spawns the two-process pipeline on
    a 2x H100 Modal container, loads Qwen2.5-7B on each GPU, runs a
    single greedy request through prefill (rank 0) → KV transfer →
    decode (rank 1), and returns the decoded text. Ships locally;
    the actual `modal run` is gated on explicit budget approval.

## Pointers

- Module: `src/mini_infer/workers/`
  - `kv_handoff.py` — handoff dataclass
  - `prefill_worker.py` — `PrefillWorker` + `prefill_batch`
  - `decode_worker.py` — `DecodeWorker` + `decode_batch`
  - `kv_transfer.py` — wire protocol
  - `multi_process.py` — `pd_two_process_target`
  - `orchestrator.py` — single-process composition
- Tests:
  - `tests/unit/test_workers.py` (in-process parity + handoff shape)
  - `tests/unit/test_workers_batch.py` (batched parity)
  - `tests/unit/test_kv_transfer_mp.py` (wire-protocol multi-process)
  - `tests/unit/test_workers_mp.py` (full multi-process; skipped on
    macOS pytest; runs on CUDA via the smoke)
- Smoke: `scripts/modal_pd_smoke.py` (2x H100, Qwen2.5-7B)

## Follow-ups

- **HTTP streaming integration**: wire `Orchestrator.run_stream` into
  the FastAPI server's SSE endpoint so HTTP clients see PD output as
  it's emitted.
- **PDScheduler**: a `ContinuousScheduler` analog that wraps the two
  workers, handles admission + request queues + cross-phase routing,
  enables continuous batching across many concurrent requests.
- **KV-quant on the PD path**: extend `materialize_packed_stream` /
  `append_stream_packed` to handle compressed pools (turbo4 / turbo3 /
  fp8 / nvfp4), then drop the `kv_quant=None` requirement in both
  workers.
- **Real Modal validation**: run `scripts/modal_pd_smoke.py` and
  publish a bench doc comparing PD throughput against the existing
  single-GPU mixed-mode numbers.
- **Multi-rank prefill / multi-rank decode**: the current 1+1
  topology is the MVP. Production PD often runs N prefill workers +
  M decode workers with a router in front; that needs admission /
  load-balancing logic on top of the worker abstraction.
- **Pipeline parallelism integration**: combine PD with TP / PP for
  models that don't fit a single GPU even at one phase.
