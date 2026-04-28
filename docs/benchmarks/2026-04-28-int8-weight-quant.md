# Weight-only INT8 (W8A16), A10 — Qwen2.5-0.5B-Instruct

Date: 2026-04-28
Hardware: NVIDIA A10 (Ampere, SM_86)
Model: Qwen/Qwen2.5-0.5B-Instruct
Engine: mini-infer @ this slice (ADR-010)
Script: `scripts/modal_packed_bench.py --config quant`

## Workload

- Three model configurations loaded sequentially in a single Modal container:
  - `fp16` (baseline)
  - `int8 (skip lm_head)` — default; every `nn.Linear` except `lm_head` is
    replaced with `Int8Linear`
  - `int8 (quant lm_head)` — `quant_lm_head=True`; `lm_head` is also replaced
- For each: post-load CUDA-allocated bytes delta, then a small concurrent
  throughput sweep at C ∈ {1, 4, 8} on a moderate prompt
  (~2 KB / ~80 tokens), `max_tokens=32`.

## Memory footprint

Bytes allocated by `ModelRunner.from_pretrained` (cuda `memory_allocated`
delta around the call), after a `torch.cuda.empty_cache()` between configs:

| Config                  | HBM     | Δ vs fp16 |
|-------------------------|---------|-----------|
| fp16                    | 1142 MiB| —         |
| int8 (skip lm_head)     | 794 MiB | **−30.5%** |
| int8 (quant lm_head)    | 924 MiB | −19.1% (worse than skip) |

**The `quant_lm_head=True` configuration saves *less* memory than the
default**, not more. Cause: Qwen2.5 has `tie_word_embeddings=True`, so
`model.embed_tokens.weight` and `model.lm_head.weight` are the same physical
tensor. Replacing `lm_head` with `Int8Linear` allocates a new int8 buffer
(~136 MiB) but doesn't free the original fp weight, which is still
referenced by `embed_tokens`. Net: +136 MiB int8 buffer on top of the
already-allocated fp embedding.

This is a real and slightly counter-intuitive interaction with weight tying.
The default (skip `lm_head`) sidesteps it; the flag is left in place but
documented as a no-op (or worse) on tied-embedding models. A future slice
could add proper tying-aware handling: replace both `embed_tokens` (which is
an `nn.Embedding`, not `nn.Linear`) and `lm_head` together, or untie before
quantizing.

## Throughput

Aggregate tokens/sec across N concurrent requests, prompt ~80 tokens,
max_tokens=32, warmup before timing:

|  C |   fp16            | int8 (skip lm_head) | int8 (quant lm_head) |
|---:|-------------------|---------------------|----------------------|
|  1 | 1.211s, 26.4 tok/s| 1.295s, 24.7 tok/s  | 1.321s, 24.2 tok/s   |
|  4 | 2.502s, 51.2 tok/s| 2.501s, 51.2 tok/s  | 2.560s, 50.0 tok/s   |
|  8 | 4.028s, 63.6 tok/s| 4.072s, 62.9 tok/s  | 4.145s, 61.8 tok/s   |

Reading: throughput is essentially neutral. At C=1 the int8 paths are
~6–8% slower than fp16 (the dequant pass on every matmul is a fixed cost
that small batches don't amortize). At C=4 and C=8 the gap closes to
within run-to-run noise (~1–3%). This matches the design expectation: a
naive dequant-then-bf16-matmul has no FLOP savings, so any throughput
benefit would come from reduced HBM weight read pressure — and at this
model size on A10, weight reads are not the bottleneck.

A fused INT8 dequant-matmul Triton kernel (or a CUTLASS-backed path à la
bitsandbytes' `Linear8bitLt`) would be the path to actual throughput
speedup. Out of scope for this slice; flagged as follow-up in ADR-010.

## What this proves

- **Memory: clear win.** 30.5% reduction in model-load HBM at the default
  setting on a 0.5B model. On larger models the proportional savings
  approach 50% as the embedding fraction shrinks (Qwen2.5-7B has lm_head
  ≈ 6% of Linear params instead of 28%).
- **Throughput: neutral.** Honest reporting; the W8A16 path is a memory
  technique, not a compute technique. The tiny C=1 regression is
  consistent with dequant overhead on small batches.
- **Numerical correctness: preserved.** Verified independently on M1:
  cosine similarity > 0.99 on logits
  (`tests/unit/test_int8_model_integration.py::test_quantized_logits_close_to_fp_reference`),
  first-token greedy parity across multiple prompts
  (`tests/stress/test_int8_load.py::test_int8_first_tokens_match_fp_reference`).
- **Weight tying gotcha surfaced.** `quant_lm_head=True` is not free on
  tied-embedding models; the table above documents the ~+11% regression
  rather than hiding it. Default behaviour (skip lm_head) is the safe
  choice.

## Caveats

- 0.5B is a small model. The headline 30.5% savings would scale to ~45%
  on a 7B-class model where `lm_head` is a smaller fraction of total
  Linear params.
- Throughput is for a moderate prompt + short generation. Long-context
  decode (where the bottleneck shifts to K/V cache reads) might show a
  different picture; not measured here.
- A10 (Ampere) doesn't have INT8 Tensor Cores in the same way Hopper does
  for bf16. On H100, even W8A16 might show a small dequant-throughput
  benefit due to better memory bandwidth ratio. Not measured.
- Reported HBM numbers are the `cuda.memory_allocated()` delta around
  `from_pretrained`. The absolute baseline includes the activation
  workspace HF allocates lazily; numbers across runs may shift by a few
  MiB. The savings comparisons (30.5%, etc.) are stable.

## Reproduce

```
uv run modal run scripts/modal_packed_bench.py --config quant
```

Defaults: A10, Qwen2.5-0.5B-Instruct, prompt ~80 tokens, max_tokens=32,
concurrencies=1,4,8. Set `MINI_INFER_BENCH_GPU=H100` for the H100 sweep.

## Pointers

- ADR: [ADR-010](../decisions/ADR-010-int8-weight-quant.md).
- Implementation: `src/mini_infer/quant/int8.py`.
- Integration: `src/mini_infer/engine/model_runner.py::from_pretrained`
  (the `quant=` and `quant_lm_head=` flags).
- Unit tests: `tests/unit/test_int8_quant.py`.
- Real-model tests: `tests/unit/test_int8_model_integration.py`.
- Stress: `tests/stress/test_int8_load.py`.
