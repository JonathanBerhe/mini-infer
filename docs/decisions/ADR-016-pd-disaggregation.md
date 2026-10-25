# ADR-016: Prefill / decode disaggregation (PD)

Date: 2026-05-15
Status: Accepted

## Context

Prefill is compute-bound: one forward over the whole prompt. Decode is
memory-bound: many forwards of one token each, limited by HBM bandwidth.
On one GPU, prefill takes HBM bandwidth from decode, and decode takes SMs
from prefill. vLLM (PD mode), SGLang (after the DistServe paper) and
TensorRT-LLM (split deployment) have a disaggregated mode. Each phase runs
on its own GPU worker, and a transfer moves the KV cache between them.

Before this ADR, `ContinuousScheduler` ran both phases on one GPU. The
tensor parallelism of ADR-015 splits one phase across GPUs, but not the two
phases.

## Decision

Add PD as a module of four parts under `src/mini_infer/workers/`:

1. `KVHandoff` (`kv_handoff.py`) is the contract between the phases. This
   dataclass carries the packed KV tensors per layer and per stream, the
   first sampled token, the sampling parameters, max-tokens and EOS.
2. `PrefillWorker` (`prefill_worker.py`) owns a `ModelRunner`.
   `prefill(request)` runs one forward over the prompt and samples the first
   token. It extracts the KV of each layer with
   `cache.materialize_packed_stream`, frees the slot and returns the
   handoff. `prefill_batch(requests)` packs N requests into one forward.
3. `DecodeWorker` (`decode_worker.py`) owns a `ModelRunner`.
   `decode(handoff)` adds a slot, writes the KV with
   `cache.append_stream_packed`, yields the first sampled token and decodes
   until EOS or `max_tokens`. `decode_batch([handoffs])` runs N slots at
   once and ends each slot on its own.
4. `kv_transfer.py` sends a `KVHandoff` over `torch.distributed`.
   `send_handoff` and `recv_handoff` work on gloo (CPU) and NCCL (CUDA).

`pd_two_process_target` (`multi_process.py`) is the entry function per
rank: rank 0 prefills and sends, and rank 1 receives and decodes.
`Orchestrator` (`orchestrator.py`) connects the two workers inside one
process.

The handoff carries `kv_streams_per_layer: list[dict[str, torch.Tensor]]`,
a map from stream name to tensor for each layer, not a `(K, V)` pair. So
one handoff serves each attention type. GQA and MHA use `("k", "v")`. MLA
(DeepSeek-V2/V3) uses `("kv_latent", "k_rope")`, with different shapes per
stream. V4 uses a larger named set per layer.
`BlockPool.stream_names(layer_idx)` is the authoritative list of streams
for each layer.

## Why per-stream + worker-level (not scheduler-level)

- A. A `phase_mode = "prefill_only" | "decode_only" | "mixed"` knob in
  `ContinuousScheduler`. It reuses admission and chunked prefill. But it
  touches every path in a 450-line file that the golden tests, the HTTP
  server and the demo use. It puts at risk the single-GPU path that 393
  unit tests cover.
- B. A new `PDScheduler` that owns both workers. It repeats the admission,
  backpressure and streaming output of `ContinuousScheduler`, and it is
  hard to test against the existing tests.
- C. Worker-level disaggregation (chosen). It needs only two thin worker
  classes, an `Orchestrator` and the `KVHandoff` contract. The parity
  contract is one assertion on a greedy token list. The whole PD module is
  about 700 lines.

## Numerical correctness

Disaggregation must not change the output.
`test_pd_greedy_matches_continuous_scheduler` (in
`tests/unit/workers/test_workers.py`) asserts that PD greedy output is
token-for-token identical to the greedy output of `ContinuousScheduler`
for the same prompt and model. Any drift of KV state between the phases
shows as a different first decoded token. The test passes on Qwen2.5-0.5B
with the in-process `Orchestrator`. After each PD run, the block pool is
fully free again, so no slot leaks.

The 6 tests in `tests/unit/workers/test_workers_batch.py` pass on Qwen2.5-0.5B.
`prefill_batch([r1, r2])` gives byte-identical KV and the same first token
as `[prefill(r1), prefill(r2)]`. `decode_batch([h1, h2])` gives the same
token lists as `[list(decode(h1)), list(decode(h2))]`.

## Cross-process transport

Both ranks must share the same `BlockPool` topology (same model, same
dtype). The wire format:

```
Header (7x int64):
  [prefill_len, first_sampled_token_id, eos_or_-1, max_tokens,
   temperature_micros, top_k, top_p_micros]
KV streams in pool.stream_names-sorted order:
  for layer in 0..num_layers-1:
    for stream in sorted(pool.stream_names(layer)):
      tensor of shape (prefill_len, h, d)
```

Both sides know the stream names from the model, so the format does not
send them. `dist.send/recv` then moves only plain tensors. This matters,
because NCCL does not send Python objects without an extra step.

`test_kv_transfer_mp.py::test_kv_transfer_round_trip` spawns two gloo
processes. One sends a synthetic handoff, and the other compares the
content. It passes in ~4 s.

`test_workers_mp.py::test_pd_two_process_matches_single_process`, the full
test on Qwen2.5-0.5B, is `@pytest.mark.skip` on macOS. Two PyTorch
processes on one CPU host stall on thread contention under the pytest spawn
harness. On a CUDA host each rank has its own GPU.
`scripts/smoke/modal_pd_smoke.py` runs `pd_two_process_target` on 2x H100 with
Qwen2.5-7B. This ADR does not include a run of that smoke.

## Consequences

- Every supported model family can use PD with no family-specific code.
- Batched prefill and batched decode are a 50-line method on each worker.
- The wire protocol works on gloo and NCCL. NCCL P2P is GPU-direct: NVLink
  on one node, RDMA across nodes.
- `ContinuousScheduler` and its 393 unit tests are untouched. The
  `workers/` module is additive, so removing PD means removing the module,
  its tests and this ADR.
- No streaming HTTP. `Orchestrator.run_stream` yields token ids, but the
  SSE response of the FastAPI server does not use it yet.
- No KV quantization. Both workers require `kv_quant=None`, because
  `materialize_packed_stream` and `append_stream_packed` handle only
  uncompressed pools (not turbo4 / turbo3 / fp8 / nvfp4).
- Each worker call takes a fixed batch, with no admission during the batch.
  The follow-up is a `PDScheduler` around the workers.
- Other follow-ups: a benchmark of PD against single-GPU mixed mode, and PD
  with TP or PP. A router for N prefill and M decode workers can replace
  the 1+1 topology of today.
