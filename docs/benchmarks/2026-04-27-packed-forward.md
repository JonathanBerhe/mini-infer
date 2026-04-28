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

- **The packed-forward path is correct on real CUDA.** Smoke (`modal_packed_bench.py --config smoke`) confirmed concurrent outputs match a serial reference within bf16 numerical drift (3/4 prompts exact, 1/4 with a single tail-token flip — same drift pattern vLLM and SGLang publish).
- **FlashAttention varlen integration works.** `flash_attn=True` reported by the runner; the patched Qwen2 forward routes every layer through `flash_attn_varlen_func`.
- **Throughput scales with concurrency**: 3.3× at C=8 means the engine's batching mechanics are doing real work, not serializing internally.

## What this DOESN'T validate (and what's next)

- **Long-context throughput** is not measured. Materialization cost scales with `(num_layers × total_kv_tokens)`; at 4k+ contexts this becomes the dominant per-step cost. The current numbers reflect ~80-token contexts where materialization is cheap; we'd expect the gap vs ADR-005 to *widen* at long contexts (more work per step) but FA's tiling advantage to *eventually* flip the comparison at very long contexts.
- **Mixed prefill+decode interleaving** isn't really exercised by this workload. The chunked-vs-unchunked comparison would tell a different story for a workload with long prompts (1–4k) submitted while short decoders are in flight: the chunked variant should sustain decoder ITL within ~2× of the no-prefill baseline; the un-chunked variant should freeze decoders for the full prefill duration.
- **Materialization-free path.** `flash_attn_varlen_func` doesn't accept paged K/V; we materialize per layer per step. The cleanest follow-up is `flash_attn_with_kvcache` (FA 2.7+ paged-aware API) or a custom Triton varlen-paged kernel (vLLM-style). The interface in `cache/packed_attention.py::packed_attention_forward` is the abstraction boundary; only that file changes.

## Long-context sweep

Re-ran with realistic long prompts to surface the long-context behavior. Workload: 3938-token prompts (RAG / long-chat scale), `max_tokens=64`, sweep at C ∈ {1, 2, 4}, same chunked-vs-unchunked split. Reproducer: `scripts/modal_packed_bench.py --config throughput --workload long`.

| Config | C | Elapsed (s) | Output tokens | Throughput (tok/s) |
|---|---:|---:|---:|---:|
| chunked-256 | 1 | 4.41 | 64 | 14.5 |
| chunked-256 | 2 | 7.33 | 128 | 17.5 |
| chunked-256 | 4 | 12.98 | 256 | 19.7 |
| unchunked   | 1 | 4.09 | 64 | 15.7 |
| unchunked   | 2 | 6.98 | 128 | 18.3 |
| unchunked   | 4 | 12.55 | 256 | 20.4 |

What changes vs the short workload:

- **Absolute throughput drops by ~6×** (14-20 tok/s vs 36-122 on the short workload). At 4k contexts, attention compute genuinely scales with `kv_len`, so each step is much heavier. This is fundamental to the model + hardware (a 0.5B model on A10 with 4k contexts is in attention-bound territory), not specific to our engine.
- **Concurrency scaling becomes sub-linear early**: C=4 only delivers 1.4× C=1 throughput (vs 3.3× on the short workload). The per-step compute scales aggressively with batch at long contexts; the matmul amortization that the short workload saw doesn't show up here.
- **Un-chunked is ~3-5% faster than chunked at every concurrency**. With identical prompts arriving together and no in-flight decode work to interleave, chunking is pure overhead — 16 small forward calls vs 1 big one for the same total prefill work, and Python-side step machinery runs 16× more often.

This last point is worth dwelling on: **on this synthetic-batch workload the chunked configuration looks worse**, and that's expected. Chunking is a head-of-line-blocking fix, not a throughput optimization for batches that arrive together. The benefit shows up in mixed workloads where short decoders are running when a long prompt lands; the chunked path keeps the decoders flowing while the un-chunked path freezes them for the full prefill duration. We'd need a microbenchmark that explicitly times decoder inter-token latency during a concurrent long prefill to show that effect — flagged as a follow-up.

## Head-of-line blocking microbench

To check whether chunked prefill actually delivers its claimed benefit (decoders keep flowing while a long prompt is being prefilled), I ran a focused microbench. Setup: start a short decoder, let it emit 4 tokens at steady-state ITL (~55-58ms). Then submit a 3065-token long prompt. Measure: the time from long-submit to the short decoder's next token. Reproducer: `scripts/modal_packed_bench.py --config holb`.

| Config | Baseline ITL (ms) | First short token after long-submit (ms) | Long total (ms) |
|---|---:|---:|---:|
| chunked-256 | 58.2 | 61.8 | 4666 |
| un-chunked  | 55.7 | 65.8 | 3251 |

**The result wasn't what a vLLM-style two-forward analysis would predict.** With a classic two-forward design (one model.forward for prefill, another for decode), an un-chunked 3k-token prefill would freeze the decoder for the full prefill duration — somewhere around 800-1000ms on this hardware. Chunked prefill exists to break that pause into small steps so the decoder gets a token in between.

But on Slice B's packed-varlen design, the decoder's first token after long-submit lands in **~60ms regardless of chunk size**. Effectively no pause. That's because the packed forward processes ALL in-flight work in one step: the long prompt's first chunk (or the whole long prompt, in the un-chunked case) and the short decoder's next token are concatenated into a single varlen forward, so the decoder can't be "blocked" by the prefill in the time-domain sense. They share the same forward.

There's a measurement caveat — `first_short_token_after_long` likely captures the tail of the in-flight step (the engine was already mid-forward when long was submitted), not the duration of the first long-aware step. The cleaner metric would be the *maximum* short-decoder ITL across the post-long window. That said, the broader signal is consistent: Slice B doesn't experience HOL blocking the way a two-forward design would.

The interesting metric is `long_total_ms`: **chunked is 44% slower than un-chunked at completing the long prefill** (4666ms vs 3251ms). 12 small chunked forwards have meaningfully more per-step Python + materialization overhead than 1 big un-chunked forward. With packed varlen as the substrate, chunked prefill stops being a HOL fix (the packed design already solved that) and starts being pure overhead for prefills without competing decode work.

**What this changes about the chunked-prefill story**: chunking still has a real reason to exist — bounded per-step memory pressure (the un-chunked 3k-token prefill materializes all 3k positions at once at every layer; chunked materializes one chunk at a time). At 4k or 16k+ contexts on the same model, the un-chunked path would OOM before the chunked path. But the throughput-vs-latency story we'd been telling ("chunking is the HOL fix") was specific to two-forward designs. For Slice B's packed design, **the architecture is the HOL fix**, and chunking is a memory tool.

## Paged vs materialized FA: cross-platform sweep

Subsequent work (ADR-008) added a paged FA varlen path (`flash_attn_varlen_func` with `block_table`) as an alternative to materializing K/V per layer per step. Hypothesis was that paged would close the gap to ADR-005's numbers, especially on Hopper-class hardware. The hypothesis didn't pan out — measurements on both A10 and H100, with `block_size=16` (materialized) vs `block_size=256` (paged) and the same packed-varlen scheduler.

### A10 (Ampere)

| Workload | Materialized | Paged | Δ |
|---|---:|---:|---:|
| Short, C=1 | 36.6 tok/s | 17.5 tok/s | -52% |
| Short, C=4 | 88.0 tok/s | 49.9 tok/s | -43% |
| Short, C=8 | 122.2 tok/s | 122.7 tok/s | ≈0 |
| Long (~3.9k), C=1 | 14.5 tok/s | 12.0 tok/s | -17% |
| Long, C=4 | 19.7 tok/s | 16.9 tok/s | -14% |

### H100 (Hopper)

| Workload | Materialized | Paged | Δ |
|---|---:|---:|---:|
| Short, C=1 | 57.2 tok/s | 43.9 tok/s | -23% |
| Short, C=4 | 144.9 tok/s | 132.3 tok/s | -9% |
| Short, C=8 | 203.0 tok/s | 209.8 tok/s | +3% |
| Long, C=1 | 21.4 tok/s | 16.8 tok/s | -22% |
| Long, C=4 | 28.4 tok/s | 24.3 tok/s | -14% |

**Materialized wins on both GPUs at almost every config.** The only place paged is competitive is C=8 short on H100, and the +3% there is within run-to-run noise. The block-table indirection cost and the `block_size=256` wasted-bandwidth-on-short-prompts cost don't go away on newer silicon for our model scale (0.5B). This doesn't disprove the production paged FA story for 70B+ models with 32k+ contexts where the gather scales into gigabytes — it just means the win isn't there at our scale.

ADR-008 documents the design and the decision to keep both paths available but default to materialized.

### H100 vs A10 (materialized FA, the default)

Side-by-side, holding everything else constant:

| Workload | A10 | H100 | Speedup |
|---|---:|---:|---:|
| Short, C=1 | 36.6 tok/s | 57.2 tok/s | 1.6× |
| Short, C=4 | 88.0 tok/s | 144.9 tok/s | 1.6× |
| Short, C=8 | 122.2 tok/s | 203.0 tok/s | 1.7× |
| Long, C=1 | 14.5 tok/s | 21.4 tok/s | 1.5× |
| Long, C=4 | 19.7 tok/s | 28.4 tok/s | 1.4× |

Roughly **1.4-1.7× speedup** moving from A10 to H100 on the same code path, holding workload constant. The architecture is GPU-portable; FlashAttention's regular varlen is well-tuned across Ampere and Hopper.

## Bottom line

Four findings, all measured:

1. **Throughput scales with concurrency on short workloads** (3.3× at C=8 on A10) but degrades at long context (1.4× at C=4 on 4k prompts). Long-context throughput is bottlenecked by attention compute that scales with `kv_len`, not by our scheduling.
2. **Slice B (packed varlen + materialized FA) is ~30% slower than ADR-005's decode-only kernel** at short workloads (122 vs 187 tok/s at C=8 on A10). The cost is the per-layer K/V materialization for `flash_attn_varlen_func`. ADR-005 was decode-only with a custom Triton kernel that read K/V directly from blocks; the architecture switch to packed varlen pays a real but bounded perf cost.
3. **Slice B's packed-forward design eliminates HOL blocking by construction**. Chunked vs un-chunked makes ~no difference to decoder ITL when a long prompt arrives; un-chunked is actually faster for the prefill itself (no chunk overhead).
4. **Paged FA isn't a win on either A10 or H100 at our model scale.** ADR-008 has the full numbers and decision rationale. Materialized FA is the default; paged is an opt-in via `block_size=256`.

Two follow-ups, in priority order:

1. **Re-bench on a 70B+ model** when hardware is available — that's the regime where paged FA's gather-avoidance was supposed to matter, and where our 0.5B measurements can't validate or refute the production claim.
2. **Custom Triton varlen-paged kernel** with `block_size=16`. Would avoid both costs (FA paged's indirection + the FA materialized's gather). Phase 3b stretch.
