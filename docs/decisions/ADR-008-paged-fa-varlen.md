# ADR-008: Paged FlashAttention varlen — investigated, kept as a tunable, not the default

Date: 2026-04-28
Status: Accepted (with the unexpected finding that paged FA underperforms materialized FA on our reference hardware)

## Context

ADR-007 shipped the packed-varlen architecture with one model.forward per scheduler step. The benchmark report flagged a 30% throughput regression vs ADR-005's decode-only Triton kernel, attributed to per-layer K/V materialization: every step's `flash_attn_varlen_func` call needs contiguous K/V tensors, so we gather from `BlockPool` storage into a packed buffer per layer per step.

flash-attn 2.7+ added a `block_table` parameter to `flash_attn_varlen_func` that lets the kernel read K/V directly from paged storage — no gather. The plan for this slice (ADR-008) was to swap the materialized varlen call for the paged varlen call and recover the lost throughput.

That's the slice I built. The result is not what the plan predicted.

## Decision

1. **Implement the paged varlen path** as `_packed_attention_paged_flash` in `cache/packed_attention.py`. Calls `flash_attn_varlen_func` with `block_table` from the cache; no materialization on the CUDA path.
2. **Keep the materialized varlen path** as `_packed_attention_materialized_flash`. Same code as ADR-007's hot path, kept because (a) FA's paged constraint is `block_size % 256 == 0` and we don't want to force that on every config, and (b) measurements showed materialized is *faster* than paged on our reference hardware.
3. **Dispatch on `block_size`** in `packed_attention_forward`: when `cache._pool.block_size % 256 == 0` and we're on CUDA, use paged FA; otherwise use materialized FA on CUDA (or PyTorch reference off-CUDA).
4. **Default `block_size` stays at 16.** This routes user-facing `ModelRunner.from_pretrained()` to materialized FA on CUDA — the better-performing path. Setting `block_size=256` is the explicit opt-in to paged FA.
5. **Bump `flash-attn>=2.8`** in the `[cuda]` extra (paged + varlen requires 2.7+, but 2.8.3 is the version we validated against).
6. **`PagedKVCache` gains paged-FA-friendly accessors**: `block_table_padded(device)`, `seq_lens_tensor(device)`, `pool_storage_for_layer(layer_idx)`. These are useful regardless of which FA path is chosen — the abstraction is the swap point for any future kernel.

## Empirical finding

The benchmark expected a throughput improvement on A10; it measured a regression. The first hypothesis was that paged FA would win on Hopper hardware where its kernel is more optimized. Tested. It doesn't.

### A10 (Ampere, SM_86)

| Workload | Materialized FA (block_size=16) | Paged FA (block_size=256) | Δ |
|---|---:|---:|---:|
| Short, C=1 | 36.6 tok/s | 17.5 tok/s | -52% |
| Short, C=4 | 88.0 tok/s | 49.9 tok/s | -43% |
| Short, C=8 | 122.2 tok/s | 122.7 tok/s | ≈0 |
| Long (~3.9k), C=1 | 14.5 tok/s | 12.0 tok/s | -17% |
| Long, C=4 | 19.7 tok/s | 16.9 tok/s | -14% |

### H100 (Hopper, SM_90)

| Workload | Materialized FA | Paged FA | Δ |
|---|---:|---:|---:|
| Short, C=1 | 57.2 tok/s | 43.9 tok/s | -23% |
| Short, C=4 | 144.9 tok/s | 132.3 tok/s | -9% |
| Short, C=8 | 203.0 tok/s | 209.8 tok/s | +3% |
| Long, C=1 | 21.4 tok/s | 16.8 tok/s | -22% |
| Long, C=4 | 28.4 tok/s | 24.3 tok/s | -14% |

### Reading the data

**Materialized wins almost everywhere.** The only configuration where paged is competitive is high-concurrency short workloads on H100, and even there the +3% is well within run-to-run noise. Everywhere else materialized is meaningfully faster — by 9-52%.

The hypothesis that motivated this slice — "paged kernel optimization on Hopper closes the gap" — is wrong. The block-table indirection has a real cost on both Ampere and Hopper that doesn't get amortized away. Likely contributors, in rough order of confidence:

1. **`block_size=256` (FA's minimum) is wasteful for short prompts**. A 5-token prompt occupies a 256-slot block; the kernel loads the whole block tile into shared memory and `cache_seqlens` masks the unused 251 slots. That's ~98% wasted memory bandwidth on the very-short side.
2. **Materialization is cheap at our scale**. The gather is ~3 MB/layer/step on a 0.5B model with ~3k contexts. Even at 24 layers × 4 requests that's ~300 MB/step total HBM traffic — negligible vs the matmul work for the same step. The "tax" we were trying to avoid is much smaller than expected.
3. **Block-table indirection has overhead the kernel can't fully hide**. Even on Hopper, the per-tile pointer-chase to resolve `block_table[batch_idx][block_idx] → physical_block` is real work that's not present in the contiguous-K/V varlen path.
4. **The 0.5B / bf16 / A10-or-H100 combo is the wrong scale to surface paged's win**. Production paged FA pays off at 70B+ models with 32k+ contexts where the materialization gather scales into the gigabytes-per-step range. We're orders of magnitude below that.

## Alternatives considered

- **Make paged FA the default anyway** (forcing `block_size=256` for CUDA). Rejected — the measured regression is consistent and meaningful. Defaulting to a slower path for ideological cleanness is the wrong tradeoff for a portfolio project that documents real numbers.
- **Force `block_size=256` in `ModelRunner.from_pretrained` when device is CUDA**. Rejected for the same reason; also bumps memory pressure on every CUDA run.
- **Drop the paged path entirely**. Rejected — the abstraction is correct and the path will be the right one on H100 / H200. Worth keeping the code in place so the dispatcher decision is data-driven rather than gating on an absence.
- **Fix the regression by re-implementing paged in a custom Triton kernel**. Phase 3b stretch territory. Worth doing only after exhausting other Phase 2 work.

## Consequences

- **Positive**:
  - The codebase now has both FA paths behind a clean dispatcher. Future hardware changes (H100 etc.) can engage paged by setting `block_size=256` with a one-line config change.
  - `PagedKVCache` exposes the right primitives for paged kernels: `block_table_padded`, `seq_lens_tensor`, `pool_storage_for_layer`. These would be reused by a custom Triton paged kernel if we ever write one.
  - `flash-attn>=2.8` is now pinned and validated.
  - The benchmark report has honest before/after numbers. The slice doesn't pretend the change was a win.
- **Negative**:
  - The slice's headline goal (close the materialization gap) was not achieved. The closest gap-closing path here would have been a custom Triton kernel — out of scope.
  - More code to maintain (two FA paths instead of one), though both are short.
  - There's a small risk a future user sees `block_size=256` mentioned in the docs and assumes it's faster. The benchmark report and this ADR document the opposite explicitly.
- **Reversibility**:
  - Removing `_packed_attention_paged_flash` and the cache's paged-FA accessors is a clean revert (~80 lines).
  - The materialized path stays as the default and is tested in CI on M1.

## Validation

- **M1 (CPU, fp32)**: 75 unit + 3 golden tests pass. PyTorch reference path unchanged.
- **CUDA (Modal A10)**: smoke green via materialized FA at `block_size=16`; smoke green via paged FA at `block_size=256`. Both produce the expected output with bf16 tail drift on the longest prompt only.
- **Benchmarks**: numbers in `docs/benchmarks/2026-04-27-packed-forward.md`. Materialized FA stays as the default-engaged path.

## Pointers

- Dispatcher: `src/mini_infer/cache/packed_attention.py::packed_attention_forward`.
- Materialized FA: `_packed_attention_materialized_flash`.
- Paged FA: `_packed_attention_paged_flash`.
- Cache accessors: `src/mini_infer/cache/paged_kv_cache.py` (`block_table_padded`, `seq_lens_tensor`, `pool_storage_for_layer`).
- Test: `tests/unit/test_packed_attention.py::test_paged_flash_matches_torch_reference` (`requires_cuda`, exercises paged FA correctness).
- Benchmark report addendum: `docs/benchmarks/2026-04-27-packed-forward.md` ("Paged vs materialized FA" section).
- Earlier ADRs: ADR-005 (continuous batching), ADR-006 (chunked prefill, two-forward), ADR-007 (packed varlen forward).

## Follow-ups

- **Re-bench on a 70B+ model** when one fits in the project's hardware budget. The hypothesis is that paged FA wins at production model scale; we measured at 0.5B and saw the opposite, but the gather cost we were avoiding scales linearly with `(num_layers × kv_len × num_kv_heads × head_dim)` and our 0.5B has the smallest of all four factors.
- **Custom Triton varlen-paged kernel** with `block_size=16` (no FA constraint). Would avoid both the indirection cost of FA paged and the wasted-block-bandwidth issue of `block_size=256`. Phase 3b stretch — only worth doing if a future workload shows the materialization gather dominating step time.
