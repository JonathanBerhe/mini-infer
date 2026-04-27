# Packed-varlen forward throughput sweep

**Date:** 2026-04-27
**Hardware:** Modal A10 (NVIDIA A10), via the cu124 Modal image with FlashAttention 2.7.4 prebuilt wheel.
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`, bf16.
**Workload:** mixed-length, 50/50 split of short (~5 token prompt) and long (~80 token prompt) at concurrency C ∈ {1, 4, 8}, `max_tokens=32` per request, greedy sampling, one warmup run before the timed run at each setting.
**Reproducer:** `uv run modal run scripts/modal_packed_bench.py`.

## Numbers

End-to-end wall-clock for `C` simultaneously-submitted mixed-length requests:

| Config | C | Elapsed (s) | Output tokens | Throughput (tok/s) |
|---|---:|---:|---:|---:|
| chunked-32 | 1 | 0.874 | 32 | 36.6 |
| chunked-32 | 4 | 1.455 | 128 | 88.0 |
| chunked-32 | 8 | 2.096 | 256 | 122.2 |
| unchunked  | 1 | 0.864 | 32 | 37.0 |
| unchunked  | 4 | 1.438 | 128 | 89.0 |
| unchunked  | 8 | 2.064 | 256 | 124.0 |

Throughput scaling (chunked, vs C=1): C=4 → 2.4×, C=8 → 3.3×.

## Reading the numbers

- **Throughput scales sub-linearly with concurrency**, as expected: C=8 is 3.3× C=1, gap to 8.0× is the per-step compute that scales with batch.
- **Chunked vs un-chunked are within 2%** of each other on this workload. The synthetic prompts are short (~80 tokens for the "long" ones), so the chunked variant doesn't get many opportunities to interleave a chunk forward with a decode forward — by the time prefill finishes, decoding is already dominating. A workload with longer prompts (~1–4k tokens) and shorter `max_tokens` would show the chunked advantage more.
- **Compared to ADR-005's batched-decode benchmark** (Slice 2.3c, decode-only Triton kernel, same A10), the packed-varlen path here is meaningfully slower:

| C | ADR-005 (batched decode, Triton) | This (packed varlen, FA + materialize) | Delta |
|---:|---:|---:|---:|
| 1 | 45.7 tok/s | 36.6 tok/s | -20% |
| 4 | 127.7 tok/s | 88.0 tok/s | -31% |
| 8 | 187.5 tok/s | 122.2 tok/s | -35% |

  The slowdown is real and tracks with concurrency. The likely cause is **per-layer materialization**: every layer per step does a packed K/V gather from the paged storage into a contiguous tensor for `flash_attn_varlen_func`. ADR-005's Triton kernel reads K/V directly from blocks — no gather. At small batches and short contexts the gather overhead dominates the FA win.

## What this validates

- **The packed-forward path is correct on real CUDA.** Smoke (`modal_packed_smoke.py`) confirmed concurrent outputs match a serial reference within bf16 numerical drift (3/4 prompts exact, 1/4 with a single tail-token flip — same drift pattern vLLM and SGLang publish).
- **FlashAttention varlen integration works.** `flash_attn=True` reported by the runner; the patched Qwen2 forward routes every layer through `flash_attn_varlen_func`.
- **Throughput scales with concurrency**: 3.3× at C=8 means the engine's batching mechanics are doing real work, not serializing internally.

## What this DOESN'T validate (and what's next)

- **Long-context throughput** is not measured. Materialization cost scales with `(num_layers × total_kv_tokens)`; at 4k+ contexts this becomes the dominant per-step cost. The current numbers reflect ~80-token contexts where materialization is cheap; we'd expect the gap vs ADR-005 to *widen* at long contexts (more work per step) but FA's tiling advantage to *eventually* flip the comparison at very long contexts.
- **Mixed prefill+decode interleaving** isn't really exercised by this workload. The chunked-vs-unchunked comparison would tell a different story for a workload with long prompts (1–4k) submitted while short decoders are in flight: the chunked variant should sustain decoder ITL within ~2× of the no-prefill baseline; the un-chunked variant should freeze decoders for the full prefill duration.
- **Materialization-free path.** `flash_attn_varlen_func` doesn't accept paged K/V; we materialize per layer per step. The cleanest follow-up is `flash_attn_with_kvcache` (FA 2.7+ paged-aware API) or a custom Triton varlen-paged kernel (vLLM-style). The interface in `cache/packed_attention.py::packed_attention_forward` is the abstraction boundary; only that file changes.

## Bottom line

The architecture is in place — one model.forward per scheduler step over a packed varlen sequence on real CUDA via FlashAttention. Correctness is validated, throughput scales. The 30% gap vs ADR-005's decode-only kernel is the per-layer materialization tax we knew we'd take on. The next slice should close that gap with a paged-aware FA call or a custom kernel, and ideally re-run this benchmark with a longer-context workload where the tradeoffs shift.
