# Paged attention kernel vs materialization, decode latency

**Date:** 2026-04-25
**Hardware:** Modal A10 24GB, NVIDIA driver via Modal default image
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`, bf16, single batch
**Workload:** 4 prompt lengths × 2 paths (kernel, materialization) × 50 measured iters + 5 warmup iters per config. Prompts generated synthetically to hit each target seq_len.
**Reproducer:** `uv run modal run scripts/modal_bench_paged.py`.

## Numbers

Decode-step latency, μs (median across 50 iters; lower is better):

| seq_len | Materialization | Triton kernel | Kernel speedup (median) |
|---|---|---|---|
| 16 | ~18,440 | **17,851** | 1.03x |
| 64 | ~19,043 | **17,849** | 1.07x |
| 256 | ~19,005 | **17,710** | 1.07x |
| 1024 | 19,269 | **17,856** | 1.08x |

Kernel p99 latencies are tight (within ~1ms of median) for seq_len ≥ 64; the seq_len=16 column has a one-time long-tail at 210 ms — the first iteration's Triton autotune / JIT compile. Median is robust to this and is the right metric here.

## Reading the numbers

- **The kernel is correct**: paired with the smoke (`modal_paged_smoke.py`), which confirmed the patched runner produces the same tokens as the materialization path on the same prompt.
- **The speedup is modest**: 3–8% across the tested seq_lens. For a 0.5B model with GQA (14 Q heads / 2 KV heads / head_dim 64), the per-decode attention compute is a small fraction of total decode time — most cycles go to the linear projections (Q, K, V, O) and the MLP. The kernel optimizes the attention portion only.
- **The speedup grows slowly with seq_len**: materialization cost scales linearly with cached length (gather every step); the paged kernel processes one block at a time, so its per-step cost stays roughly constant. We see this trend (1.03 → 1.08 from seq_len 16 → 1024) but it plateaus because total decode is bottlenecked elsewhere.
- **Where bigger wins should appear**: larger models (more layers + bigger head_dim → attention is a bigger fraction of decode), longer contexts (materialization scaling becomes painful past ~4k), and quantized variants (where attention dominates because matmuls are cheaper in INT8/FP8). Phase 2.5 quantization and Phase 4's vLLM comparison are the natural places to observe these.

## What this validates

- Correctness: the Triton kernel produces the same outputs as the PyTorch reference (Smoke A) and the SDPA-on-materialized-K/V reference (cosine sim > 0.99 in unit tests, deferred to Modal).
- Integration: the per-architecture monkey-patch (`attention_patches/qwen2.py`) wires the kernel in cleanly; `use_paged_kernel=False` cleanly disables it for A/B.
- Memory: the block pool's allocate/free cycle is balanced (`pool_blocks=1024/1024` after the bench finishes).

## What this does NOT validate

- Multi-batch performance — the kernel grid is `(batch * num_q_heads,)` so it should scale, but we haven't measured concurrent requests yet (Phase 2.3 territory).
- Prefill — prefill still goes through HF's stock attention path (with our materialization in `cache.update()`); the kernel is decode-only by design.
- vLLM-class numbers — we're not chasing those yet. Phase 4's public benchmark report does the head-to-head with proper hardware (likely H100) and longer contexts.
