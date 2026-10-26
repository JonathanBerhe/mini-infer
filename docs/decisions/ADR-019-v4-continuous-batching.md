# ADR-019: Continuous batching for DeepSeek-V4 via a paged state pool

Date: 2026-06-23
Status: Accepted. Lockstep cohort + full ragged continuous batching implemented and GPU-benchmarked 2026-06-24. Cross-request prefix sharing later shipped; a fused decode-attention kernel was prototyped and rejected. See ADR-020.

## Context

V4 served one request at a time through `StateCacheScheduler`. The open
throughput item was continuous batching for V4: many requests in one forward
pass, as `ContinuousScheduler` and `PagedKVCache` do for every other family.
ADR-014 deferred it until cross-request prefix sharing became a workload.
This ADR is that follow-up.

The V4 decode path keys off one scalar `start_pos`
([state_cache.py](../../src/mini_infer/cache/state_cache.py)), plus
per-layer `n_compressed_blocks` and `swa_count`. The decode write puts the
whole batch in one window slot (`swa_kv[:, start_pos % n_win]`), and the
compression flush is one batch-wide condition (`(start_pos + 1) % m == 0`).
Continuous batching needs one position per request, which breaks all three.

The vendored DeepSeek reference
([model.py](../../third_party/deepseek_v4_reference/model.py)) batches up to
`max_batch_size=4`, but in lockstep (all rows at one position). So no
bit-parity anchor exists for ragged decode (each request at its own
position).

vLLM and SGLang shipped ragged V4 in April 2026 and give the design. vLLM
uses 256-token blocks in three page-size buckets. SGLang uses "ShadowRadix",
a radix tree over virtual token slots with separate SWA / C4 / C128 pools.
Both fold V4 state into a paged pool with per-request virtual indexing.
Their FP8 indexer keys and custom kernels are not bit-identical to the
reference. mini-sglang has no V4 model, so it is only a template for the
pool and scheduler.

## Proposed approach

Move the window, compressed and indexer state of V4 out of `StateCache` into
per-request streams on the paged `BlockPool`. Positions then become
per-request page-table lookups, and V4 plugs into `ContinuousScheduler`.
Build a ragged decode path for CSA and HCA.

## Kernel decision (where it lands)

Ragged sparse attention (a per-request top-k gather, then MQA with a sink
at per-request seqlens) needs a kernel:

- (a) FlashMLA-sparse / DeepGEMM indexer kernels, as vLLM and SGLang use.
  They are fast, but tied to the Hopper/Blackwell stack and hard to read.
- (b) A readable torch/Triton ragged variant of `hca_mqa_with_sink`
  ([hca_attention.py](../../src/mini_infer/models/deepseek_v4/hca_attention.py)) that
  takes per-request seqlens and top-k indices.

The decision: (b) as the reference, with (a) as an optional fast backend
behind the same interface, as the `flashinfer_backend` toggle does.

## Parity decision (where it lands)

The oracle is our single-request `StateCache` path, which ADR-014 tested
against the reference to cosine > 0.9999. For requests at ragged positions,
the batched ragged decode must be token-identical (greedy) to each request
run alone through `StateCacheGenerator`. The lockstep reference is then a
transitive oracle. The decode-parity tests already anchor the batched
attention math to the reference at B=2. A B=4 run of the full reference
Transformer added no coverage and was skipped.

## Effort and phasing (proposed)

Correctness work needs no GPU (gloo or CPU, synthetic). The GPU gives only
throughput numbers.

- Phase 0 (DONE 2026-06-23): lockstep cohort batching.
  `generate_ids_batched` and `iter_generate_ids_batched` on
  `StateCacheGenerator`, and `StateCacheCohortScheduler`, which batches
  equal-length requests in lockstep.
- Phases 1-2 (DONE 2026-06-24): ragged continuous batching, all modes. The
  paged `BlockPool` streams were not needed. The per-request `StateCache`
  already has a batch dim, and every counter derives from position. So
  ragged decode needed only per-request `positions`, per-row scatter/gather
  and a per-row masked indexer top-k. Shipped: `forward_decode_ragged` on
  SWA / HCA / CSA, `DeepseekV4ForCausalLM.forward_decode_with_cache_ragged`,
  `StateCacheContinuousScheduler` (its own admit/evict loop, not
  `ContinuousScheduler`) and `TensorParallelStateCacheContinuousServer`.
  Parity holds token-for-token against single-stream runs on CPU and gloo.
  Admission prefills a whole prompt into a free slot, so chunked prefill was
  not needed.
- Phase 3 (resolved, ADR-020): the fast fused-kernel backend and
  cross-request prefix sharing. Prefix sharing shipped and is bit-exact. A
  fused decode-attention kernel gave ~0% end-to-end on V4-Flash, because
  the MoE dominates decode. It also diverged token-for-token from the parity
  path, so it was rejected. Prefix sharing v1 copies snapshots. The target
  is still a paged pool, where shared blocks are pointed at, not copied.

## Benchmark (2026-06-24, real V4-Flash, 2x B200, TP)

`scripts/deepseek_v4/modal_v4_flash_cb_bench.py`, 16 requests x 64 tokens, identical
output:

| path | wall | throughput |
|---|---|---|
| one-at-a-time | 298.5 s | 3.4 tok/s |
| ragged continuous batching | 82.9 s | 12.3 tok/s |

Throughput rises 3.60x. The path is unfused PyTorch on a ~158 GB MoE, so
absolute tok/s is low. The relative speedup is the result.

## Serving wiring (2026-06-24)

`/v1/completions` serves V4 with `StateCacheContinuousScheduler` on one
GPU. A model too big for one GPU uses
`TensorParallelStateCacheContinuousScheduler`, a rank-0 leader over
`TensorParallelStateCacheContinuousServer`. `StateCacheScheduler` and
`StateCacheCohortScheduler` are gone. The GPU gate served real V4-Flash on
2x B200 with 6 concurrent requests: all HTTP 200, coherent output, clean
follower teardown (`scripts/deepseek_v4/modal_v4_flash_cb_serve.py`).

Correction (2026-10-18): that HTTP gate served its 6 requests one at a time.
The endpoint ran the blocking `scheduler.run()` inside the async handler,
which held the event loop for each whole request (see
[ADR-028](ADR-028-engine-liveness.md)). So the gate did not exercise
multi-slot admission on the real checkpoint. Ragged batched decode did run
on the real checkpoint through `generate_cohort` in the benchmark. A rerun
of `scripts/deepseek_v4/modal_v4_flash_cb_serve.py` is the pending gate for that path.

## Alternatives Considered

- Status quo, one request at a time. No risk and no throughput.
- Lockstep cohort only (Phase 0). It is cheap and anchored to the reference,
  but it waits for the slowest member and cannot admit mid-flight.
- Depend on vLLM/SGLang kernels. They are fast, but hard to read and not
  bit-identical to the reference.
- Full ragged with a readable kernel (this ADR). Most work, with
  self-tested parity.

## Consequences

- V4 gets continuous-batching throughput. The shipped path kept
  `StateCache`, so the `PagedKVCache` unification that ADR-014 named is
  still open.
- The ragged decode has no direct reference. Its parity is self-tested
  only.
- The readable path is slower than vLLM/SGLang, so we do not claim
  competitive throughput.
- Two code paths remain: the single-stream oracle and the ragged path.

## References

- ADRs: ADR-005 (continuous batching), ADR-006 (chunked prefill), ADR-008
  (paged FA varlen), ADR-009 (prefix caching), ADR-013 (TurboQuant KV, a
  different kind of "compressed"), ADR-014 (V4 attention), ADR-015 (tensor
  parallelism).
- Design anchors (not bit-parity): vLLM DeepSeek-V4 blog
  (`vllm.ai/blog/2026-04-24-deepseek-v4`) and SGLang/LMSYS DeepSeek-V4 blog
  (`lmsys.org/blog/2026-04-25-deepseek-v4/`).
- Readable machinery template: mini-sglang `kvcache/mha_pool.py`,
  `kvcache/base.py`, `scheduler/{prefill,decode,table}.py`.
