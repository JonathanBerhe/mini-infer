# ADR-019: Continuous batching for DeepSeek-V4 via a paged state pool

Date: 2026-06-23
Status: Accepted. Lockstep cohort + full ragged continuous batching implemented and GPU-benchmarked 2026-06-24 (Phase 3 kernel/prefix-sharing optimizations not done).

## Context

V4 now serves over HTTP: one request at a time through `StateCacheScheduler`,
and across GPUs through the tensor-parallel server (ADR-015 + the TP serving
work). The one throughput item left open is **batched continuous batching for
V4**: serving many requests through a single forward pass, the way
`ContinuousScheduler` + `PagedKVCache` already do for every non-V4 family.

ADR-014 deferred this on purpose. It rejected "plug compressed entries into
PagedKVCache instead of StateCache" for that stage, noting the follow-up
condition directly: *"A follow-up can move the compressed history into a paged
stream once cross-request prefix-sharing becomes a workload."* It also flagged
the wart this ADR removes: *"The StateCache lives outside PagedKVCache ... a
real V4-serving workload would unify them."* This ADR is that follow-up.

**Why it is hard (the root cause).** The V4 decode path keys off a single
scalar `start_pos` ([state_cache.py:197](../../src/mini_infer/cache/state_cache.py),
plus per-layer `int` `n_compressed_blocks` / `swa_count`). Every request in a
batch must therefore sit at the same position. The batch dimension exists for
prefilling equal-length sequences and for TP replication, not for ragged decode.
The decode write puts the whole batch in one window slot
(`swa_kv[:, start_pos % n_win]`, [swa.py:316](../../src/mini_infer/models/blocks/swa.py)),
and the compression flush is a batch-uniform scalar condition
(`(start_pos + 1) % m == 0`). Continuous batching needs each request at its own
position, which breaks all three assumptions.

**Parity anchor situation (checked 2026-06-23).** The vendored DeepSeek
reference ([third_party/deepseek_v4_reference/model.py](../../third_party/deepseek_v4_reference/model.py))
batches up to `max_batch_size=4` but is **lockstep**: `start_pos` is a scalar
threaded top to bottom (`Transformer.forward` to `Attention.forward` to
`Compressor`/`Indexer`), every batch row writes the same slot, and the window /
compressed index sets are computed once and `.expand(bsz, ...)`. So there is
**no bit-parity anchor for ragged decode**.

There are, however, two **design** anchors. vLLM and SGLang both shipped Day-0
ragged V4 (April 2026):

- **vLLM**: fixes one logical block at "256 native token positions for every
  compressed layer", then sorts cache types into three page-size buckets (c4a
  main KV + sliding-window KV + compressor state; C4 indexer KV; c128a main KV).
  The compressor residual is treated as "sliding-window KV". Leans on fused
  kernels (compressor+RMSNorm+RoPE+insert; fused Q-norm+KV-RoPE+K-insert) plus
  FlashMLA + FlashInfer.
- **SGLang**: "ShadowRadix", a radix tree over *virtual* token slots ("a
  unified coordinate system shared by all layers"), with separate SWA / C4 /
  C128 physical pools and compression-state ring buffers; per-pool lifetimes let
  compressed shadows be shared across requests (their prefix caching). Kernels:
  FlashMLA (fuses SWA + extra attention), TileLang mHC, DeepGEMM MegaMoE.

Both fold V4 state into a **paged pool with per-request virtual indexing**, and
both optimize for throughput with FP8 indexer keys and custom kernels, so
neither is bit-identical to the reference. They tell us the design, not the
exact numbers.

**Readable template (mini-sglang).** Our closest readable reference has **no
DeepSeek-V4 model** (only llama / mistral / qwen2 / qwen3 / qwen3_moe). It is
the template for the *paged-pool + scheduler machinery* (`kvcache/mha_pool.py`,
`kvcache/base.py`, `scheduler/{prefill,decode,table}.py`), which mini-infer
already mirrors, not for V4's math. So the V4-specific design must come from
SGLang/vLLM proper, re-expressed in mini-infer's readable style.

## Proposed approach

Move V4's per-request window + compressed + indexer state out of the
scalar-position `StateCache` and into the existing paged `BlockPool` as
per-request streams, so positions become per-request page-table lookups and V4
plugs into the `ContinuousScheduler` we already run for every other family.
Build a ragged decode path for CSA / HCA. Hold parity by self-validating the
batched ragged path against single-stream `StateCache` runs (which are
themselves bit-parity'd against the reference per ADR-014), since no ragged
reference exists to diff against.

## What it touches (precise)

**Reusable primitives that already exist (these lower the cost):**

- `StreamSpec` + `BlockPool` per-stream paged allocation
  ([block_pool.py:27,83](../../src/mini_infer/cache/block_pool.py)). ADR-014
  already names "MLA + V4's compressed branch" as the two motivating consumers
  of per-stream allocation; MLA exercises it today.
- `PagedKVCache.append_stream_packed` / `materialize_packed_stream`
  ([paged_kv_cache.py:522,597](../../src/mini_infer/cache/paged_kv_cache.py)):
  per-stream packed-varlen read/write, the ragged primitive for non-"k"/"v"
  streams.
- `PrefixCache` radix tree ([prefix_cache.py](../../src/mini_infer/cache/prefix_cache.py)):
  cross-request prefix sharing, the workload ADR-014 said would justify this.
- `ContinuousScheduler` ([continuous_scheduler.py:50](../../src/mini_infer/scheduler/continuous_scheduler.py)):
  the ragged orchestration (join/leave per step). V4 routes here instead of the
  one-at-a-time `StateCacheScheduler`.
- `packed_attention.py`: varlen packing for the standard attention path, the
  pattern to mirror for V4's ragged attention.

Note: `BlockPool.read_compressed_block` / `write_compressed_block`
([block_pool.py:768,833](../../src/mini_infer/cache/block_pool.py)) are
**TurboQuant KV quantization** (ADR-013), a different axis of "compressed" than
V4's token-level compression. Not reusable for this, but they prove the pool
already stores non-standard per-block layouts.

**New work (the actual port):**

1. **V4 paged streams.** Define per-layer `StreamSpec`s on `BlockPool`: a SWA
   window stream, a compressed-history stream (one entry per `m` tokens), a CSA
   indexer compressed stream, and the in-flight compressor accumulator, each
   with a per-request page table. This is `StateCache` re-expressed as paged
   streams. (cache/)
2. **Per-request position state.** Replace scalar `start_pos` /
   `n_compressed_blocks` / `swa_count` with length-`B` vectors threaded through
   decode.
3. **Ragged decode for CSA/HCA** (the hard, parity-critical core): new variants
   of `forward_decode` ([csa.py:358](../../src/mini_infer/models/blocks/csa.py),
   [hca.py:505](../../src/mini_infer/models/blocks/hca.py)) that take per-request
   positions, scatter each row into its own window slot, apply a per-request
   **predicated** compression flush, and run a per-request **masked** indexer
   top-k.
4. **Ragged / chunked prefill admission**, building on chunked prefill (ADR-006)
   so new requests join between decode steps.
5. **Scheduler wiring.** Route V4 (`USES_STATE_CACHE`) to `ContinuousScheduler`
   with the paged V4 layout; keep `StateCacheScheduler` as the single-stream
   oracle / fallback.
6. **The attention-kernel decision** (below).

## Kernel decision (where it lands)

The ragged sparse attention (per-request top-k gather + MQA-with-sink over
window+compressed at per-request seqlens) is the kernel question.

- **(a) Depend on FlashMLA-sparse / DeepGEMM indexer kernels**, as vLLM/SGLang
  do. Fast, but couples us to their Hopper/Blackwell stack and is unreadable.
  Conflicts with the "readable" thesis and the "not a custom-kernel project at
  scale" non-goal.
- **(b) Readable varlen path**: a torch/Triton ragged variant of the existing
  `hca_mqa_with_sink` ([cache/hca_attention.py](../../src/mini_infer/cache/hca_attention.py))
  that accepts per-request seqlens + top-k indices, mirroring how
  `packed_attention.py` does varlen for standard attention. Slower, readable,
  on-thesis.

**Recommendation: (b) as the reference implementation, with (a) as an optional
fast backend behind the same interface** (the existing `flashinfer_backend`
toggle is the precedent for a swappable backend).

## Parity decision (where it lands)

No ragged bit-parity anchor exists (reference is lockstep; vLLM/SGLang are not
bit-exact). So the oracle is **our own single-request `StateCache` path**, which
ADR-014 validated against the reference to cosine > 0.9999:

- Test: for a batch of requests at ragged positions, assert the batched ragged
  decode is token-identical (greedy) to running each request alone through
  `StateCacheGenerator`. This makes the lockstep reference the *transitive*
  oracle without needing a ragged reference.
- Lockstep-cohort path (Phase 0) anchor, as built: the batched attention math is
  already reference-anchored at B=2 by the existing decode-parity tests, so the
  cohort generator + scheduler on top are validated by self-consistency (batched
  output == each request run alone, token-for-token). Wiring the full reference
  Transformer for a B=4 generation run was a disproportionate lift for no extra
  coverage, so it was not done.

## Effort and phasing (proposed)

Correctness work needs no GPU (gloo / CPU synthetic, like the TP serving tests);
GPU is only for throughput numbers.

- **Phase 0 (DONE 2026-06-23): lockstep cohort batching.** `generate_ids_batched`
  + `iter_generate_ids_batched` on `StateCacheGenerator` (B>1 through the existing
  scalar-position path), `StateCacheCohortScheduler` (groups equal-length /
  same-sampling requests and serves each group through one lockstep forward), and
  the `/v1/completions` server defaulting V4 to it. Validated by self-consistency
  on gloo/CPU (batched output == each request run alone, token-for-token); the
  batched attention math itself is already reference-anchored at B=2 by the
  existing decode-parity tests.
- **Phases 1-2 (DONE 2026-06-24): ragged continuous batching, all modes.**
  Built differently from the sketch above, and more simply: it turned out the
  paged `BlockPool` streams were NOT needed. The per-request `StateCache` already
  carries a batch dim, and every counter (`n_compressed_blocks`, `swa_count`,
  write slots) derives from position, so ragged decode needed only per-request
  `positions` plus per-row scatter/gather and a per-row masked indexer top-k, not
  a cache rewrite. Shipped: `forward_decode_ragged` on SWA / HCA / CSA, ragged
  `forward_decode_step` on the compressor + LightningIndexer, threaded through
  `DeepseekV4ForCausalLM.forward_decode_with_cache_ragged`; a new
  `StateCacheContinuousScheduler` (single-process dynamic admit/evict, NOT a reuse
  of the packed-varlen `ContinuousScheduler`) and `TensorParallelStateCacheContinuousServer`
  (TP leader/follower). Parity self-validated token-for-token vs single-stream
  scalar on CPU + gloo (the reference is lockstep-only, so no ragged anchor, as
  predicted). Chunked-prefill admission was not needed (admit prefills a whole
  prompt into a free slot).
- **Phase 3 (not done):** the fast fused-kernel backend and cross-request prefix
  sharing remain future. Prefix sharing is the one place the deferred paged
  compressed-stream design would still pay off; revisit if it becomes a workload.

## Benchmark (2026-06-24, real V4-Flash, 2x B200, TP)

`scripts/modal_v4_flash_cb_bench.py`, 16 requests x 64 tokens, ragged continuous
batching vs one-at-a-time, identical output:

| path | wall | throughput |
|---|---|---|
| one-at-a-time | 298.5 s | 3.4 tok/s |
| ragged continuous batching | 82.9 s | 12.3 tok/s |

**3.60x throughput.** This is a readable (unfused) implementation, so absolute
tok/s is low (a ~158 GB MoE on a PyTorch path); the relative speedup is the
result, consistent with the project not competing on absolute throughput. A fused
attention/MoE path (Phase 3) would shift the decode toward memory-bound and raise
the speedup further.

## Alternatives Considered

- **Status quo (keep `StateCache`, one at a time):** zero risk, no throughput.
  Fine while throughput is a non-goal.
- **Lockstep cohort only (do Phase 0, stop):** cheap and reference-anchored, but
  stalls on the slowest member and cannot admit mid-flight. A good 80/20 if the
  throughput need is mild.
- **Depend wholesale on vLLM/SGLang kernels:** fastest route to their
  performance, but off-thesis (unreadable, kernel-coupled) and still no
  bit-parity vs the DeepSeek reference.
- **Full ragged with a readable kernel (this proposal):** most work, on-thesis,
  self-validated parity.

## Consequences

**Positive:**

- V4 gets real serving throughput on the same paged / continuous path as every
  other family, removing the "StateCache lives outside PagedKVCache" wart
  ADR-014 flagged.
- Enables V4 prefix caching (the compressed stream becomes shareable, the
  workload ADR-014 named as the trigger).
- Produces a readable continuous-batched-sparse-attention reference, which is
  on-mission per the CLAUDE.md technique-inventory goal (a textbook
  implementation a paper reader can follow), distinct from competing on
  throughput.

**Negative:**

- A substantial cache + attention rewrite. The ragged decode is parity-fragile
  and has no direct reference; it is self-validated only.
- A readable kernel will be slower than vLLM/SGLang. Acceptable under the
  non-goals, but it means we do not claim competitive throughput.
- Two code paths during migration (the `StateCache` oracle and the paged path).

**Trade-offs:**

- Readable-but-slower kernel (b) vs fast-but-coupled kernel (a). Resolved by
  shipping (b) and leaving (a) as an optional backend.
- Self-validated parity vs (nonexistent) reference parity. Resolved by making
  single-stream `StateCache` the transitive oracle.

## References

- ADRs this builds on: ADR-005 (continuous batching), ADR-006 (chunked prefill),
  ADR-008 (paged FA varlen), ADR-009 (prefix caching), ADR-013 (TurboQuant KV;
  note its "compressed block" is quantization, not V4 compression), ADR-014
  (V4 attention; deferred this exact follow-up), ADR-015 (tensor parallelism).
- Design anchors (not bit-parity): vLLM DeepSeek-V4 blog
  (`vllm.ai/blog/2026-04-24-deepseek-v4`, 256-token blocks + 3 page buckets);
  SGLang/LMSYS DeepSeek-V4 blog (`lmsys.org/blog/2026-04-25-deepseek-v4/`,
  ShadowRadix + SWA/C4/C128 pools).
- Readable machinery template: mini-sglang `kvcache/mha_pool.py`,
  `kvcache/base.py`, `scheduler/{prefill,decode,table}.py` (no V4 model present).
- Current code: [state_cache.py](../../src/mini_infer/cache/state_cache.py)
  (scalar `start_pos`), [csa.py:358](../../src/mini_infer/models/blocks/csa.py)
  / [hca.py:505](../../src/mini_infer/models/blocks/hca.py) (decode entry points
  to gain ragged variants), [block_pool.py:27,83](../../src/mini_infer/cache/block_pool.py)
  (`StreamSpec` / `BlockPool`), [paged_kv_cache.py:522,597](../../src/mini_infer/cache/paged_kv_cache.py)
  (per-stream packed), [continuous_scheduler.py](../../src/mini_infer/scheduler/continuous_scheduler.py),
  [prefix_cache.py](../../src/mini_infer/cache/prefix_cache.py).
- Lockstep reference: [third_party/deepseek_v4_reference/model.py](../../third_party/deepseek_v4_reference/model.py)
  (`max_batch_size=4`, scalar `start_pos`).
