# ADR-007: Packed-varlen forward — one model.forward per scheduler step

Date: 2026-04-27
Status: Accepted

## Context

ADR-006 shipped chunked prefill with a "two forwards per step" pattern: one `model.forward` for the prefill chunk, one for the batched decoders. That removed the head-of-line-blocking pathology but left throughput on the table — at any step where both prefill and decode work was pending, we paid two model.forward overheads instead of one. The dominant matmul cost (Q/K/V projections + MLP) ran twice over the same weights.

This slice (B) collapses both into a single packed forward per step using FlashAttention's varlen attention. ADR-006's structural pieces (chunked prefill state machine, `append_kv_packed`, `get_mask_sizes` override) carry over; what changes is the kernel call site and the scheduler's step shape.

## Decision

1. **One forward per step**, end of story. The scheduler builds a packed sequence — `(1, total_q, hidden)` — where `total_q = sum(per-request q_lens)`. A prefilling request contributes `chunk_size` tokens; a decoding request contributes 1. cu_seqlens_q + cu_seqlens_k define per-request boundaries.
2. **`cache/packed_attention.py`** (already shipped in `f140c20`) is the new attention abstraction. `packed_attention_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale)` dispatches `flash_attn_varlen_func` on CUDA and a per-request SDPA loop in PyTorch elsewhere.
3. **`PagedKVCache.materialize_packed_kv(layer_idx)`** is the new gather primitive. Unlike `_materialize` (which pads to a 4D rectangular tensor for HF's stock attention), this returns the K/V history of every request concatenated along dim 0 — exactly the shape FA varlen wants. Per-request offsets come back as `cu_seqlens_k`.
4. **The Qwen2 patch is fully replaced** by a packed-forward path. Q/K/V projections + RoPE on the packed sequence, `cache.append_kv_packed(...)` for the new K/V, `materialize_packed_kv(...)` for the full history, then `packed_attention_forward(...)`. The previous `paged_attention_decode_batched` decode path is no longer wired to the patch (the kernel module remains; it can be removed in a cleanup slice).
5. **`runner.forward_step(cache, packed_input_ids, cu_seqlens_q, position_offsets)`** is the unified entry point. All other runner methods (`prefill`, `decode`, `decode_batch`, `prefill_chunk`) become 3-line wrappers that pack their inputs appropriately. Golden tests, the decode-latency benchmark, and the scheduler all flow through this one path.
6. **The scheduler's `_step` is restructured** into: admit → sample decoders → reap DONE → `_packed_forward(alive)`. New requests get a slot in the shared `_batched_cache` immediately on admission (no per-request temp prefill cache; that field is gone from `RunningRequest`). The reap happens BEFORE the forward so the forward only runs over the alive set.
7. **Patch is applied on every device, not just CUDA.** The patched forward calls `packed_attention_forward`, which dispatches FA varlen on CUDA and a PyTorch per-request SDPA loop on CPU/MPS. Both produce the same outputs (validated by unit tests and the golden tests).

## Alternatives considered

- **Keep the two-forward design as a fallback flag.** Tempting for safety but doubles the test surface. Slice A's two-forward code path is preserved in git history (commit `9ea3ff4`); reverting is a clean revert.
- **Write a custom Triton varlen-paged kernel** instead of `flash_attn_varlen_func` + materialization. vLLM's design. Materialization adds an HBM gather per layer per step; for our 0.5B model on A10 that's tiny (~3 MB/layer/step) but at 7B+ contexts it bites. Earmarked as a follow-up if benchmarks demand it; the abstraction in `packed_attention.py` is the swap point.
- **Use `flash_attn_with_kvcache`** (FA's paged-aware varlen API in 2.7+). Avoids materialization at the cost of taking on more of FA's API. Tried briefly; the API requires `q` shape `(B, q_len, num_heads, head_dim)` with a uniform per-batch `q_len`, which doesn't naturally fit our packed mixed-q-len case. Could be made to work by padding to max q_len; not worth it given materialization is fast at our scale.
- **Sample decoders AFTER the forward** instead of before. Cleaner-looking but requires keeping DONE slots in the cache through the forward (or a more complex reap scheme). Sampling-then-reap-then-forward keeps the forward strictly over the alive set and is correctness-easier to reason about.

## Consequences

- **Positive**:
  - One model.forward per scheduler step. The matmul cost amortizes across all in-flight q-tokens (chunk + decoders) — this is the production "Approach 2" win.
  - `runner.forward_step` is the single primitive; everything else is a thin wrapper. The runner's API surface is smaller and clearer.
  - The patch is now device-agnostic: same code path on CUDA and CPU/MPS, same correctness invariants.
  - `RunningRequest` is leaner — `prefill_cache` is gone; one less ownership concern.
- **Negative**:
  - Materialization cost per layer per step. Negligible at our scale; potentially a bottleneck at very long contexts + high concurrency. Watching for this in the bench.
  - The packed Qwen2 patch is more code than the old decode-only patch. Worth it for the unified shape.
  - flash-attn is now a runtime dependency on CUDA installs (still optional via the `[cuda]` extra; non-CUDA installs work without it). Modal images install it as part of the smoke/bench scripts.
- **Reversibility**:
  - The two-forward design is one revert away (commit `9ea3ff4` has the prefill_chunk + advance_chunked_prefill code).
  - The kernel choice (FA varlen + materialization) is hidden behind `packed_attention_forward`. Swapping to a custom Triton kernel doesn't touch the scheduler or the patch.

## Validation

- **Local (M1, CPU PyTorch path)**:
  - All 75 unit tests pass: cache, packed_attention (parity vs SDPA, GQA, causality, isolation, B=1), scheduler (chunked-prefill parity, batched decode parity, drift), golden tests (token-for-token vs HF).
  - 2 stress tests pass: 8 concurrent mixed-length requests with parity vs serial reference, memory pressure with a 32-block pool.
- **Modal**: deferred to a follow-up benchmark slice. The integration's correctness is validated locally; the benchmark numbers (chunked-prefill on vs off, mixed-length workload, throughput sweep) are the next deliverable.

## Pointers

- Packed attention: `src/mini_infer/cache/packed_attention.py`
- Cache primitive: `src/mini_infer/cache/paged_kv_cache.py` (`materialize_packed_kv`, `append_kv_packed`)
- Patched Qwen2: `src/mini_infer/engine/attention_patches/qwen2.py`
- Runner: `src/mini_infer/engine/model_runner.py` (`forward_step` + thin wrappers)
- Scheduler: `src/mini_infer/scheduler/continuous_scheduler.py` (`_step`, `_sample_decoders`, `_packed_forward`)
- Tests: `tests/unit/test_packed_attention.py`, `tests/unit/test_scheduler.py`, `tests/stress/test_chunked_prefill_load.py`
- Earlier ADRs: ADR-005 (continuous batching), ADR-006 (chunked prefill).
