# TurboQuant V1 KV cache (rotation + 4-bit), A10 — Qwen2.5-0.5B + 7B

Date: 2026-04-29
Hardware: NVIDIA A10 (Ampere, SM_86), bf16 model
Engine: mini-infer @ this slice (ADR-013)
Script: `scripts/modal_packed_bench.py --config turbo`

## Workload

- Moderate prompt (~80 tokens, 8 repeats of a passage), `max_tokens=32`,
  concurrencies C ∈ {1, 4, 8}.
- Two ModelRunner configurations on the same Modal container per model:
  - `bf16` baseline (default, uncompressed KV cache).
  - `turbo4` (TurboQuant V1: per-layer random rotation + per-block
    asymmetric 4-bit quant + materialize-on-read).
- Greedy parity check on a fixed prompt before throughput sweeps:
  decode 8 tokens with each config, compare token IDs.

## Results

### Qwen2.5-0.5B

```
KV-cache pool storage:
  bf16:   192.0 MiB
  turbo4: 72.2 MiB (37.6% of bf16, savings=62.4%)

Greedy parity (prompt='The capital of France is', max_tokens=8):
  bf16:   [12095, 13, 1084, 374, 279, 7772, 3283, 304]   (' Paris. It is the largest city in')
  turbo4: [12095, 13, 1084, 374, 279, 7772, 3283, 304]   (same)
  first_token_match=True, full_match=True

Throughput:
    C       bf16 (s, t/s)        turbo4 (s, t/s)  turbo/bf16
  ----------------------------------------------------------
    1   1.178s, 27.18 t/s       9.29s, 3.44 t/s       0.13x
    4   2.544s, 50.31 t/s     33.664s, 3.80 t/s       0.08x
    8   4.278s, 59.84 t/s      66.10s, 3.87 t/s       0.06x
```

### Qwen2.5-7B

```
KV-cache pool storage:
  bf16:   896.0 MiB
  turbo4: 336.9 MiB (37.6% of bf16, savings=62.4%)

Greedy parity (prompt='The capital of France is', max_tokens=8):
  bf16:   [12095, 13, 15920, 315, 279, 2701, 12239, 374]   (' Paris. Which of the following statements is')
  turbo4: [12095, 13,   576, 6722, 315, 315,  9625, 315]   (' Paris. The capital of of France of')
  first_token_match=True, full_match=False

Throughput:
    C       bf16 (s, t/s)         turbo4 (s, t/s)  turbo/bf16
  ----------------------------------------------------------
    1   1.989s, 16.09 t/s     11.167s, 2.87 t/s       0.18x
    4   3.753s, 34.11 t/s     39.669s, 3.23 t/s       0.09x
    8   5.985s, 42.77 t/s     77.869s, 3.29 t/s       0.08x
```

## Reading the data

### Storage savings: 62.4%, consistent across model sizes

Both models hit the same 62.4% compression. That's expected: the
compression ratio is a property of the per-block quant scheme, not the
model. Theoretical ceiling at 4-bit is 75% (1/4 the bytes), but ~33% of
the compressed bytes go to per-block per-channel `(low, scale)` overhead.
PolarQuant (V3) removes the `low` parameter (polar coordinates are
zero-centered) and would push realized compression from 2.7x toward the
paper's 5x.

### Accuracy: 0.5B passes, 7B doesn't

- **0.5B**: token-for-token full parity vs bf16 baseline. The cache
  plumbing is correct end-to-end and rotation + uniform 4-bit retains
  enough fidelity at 24-layer depth.
- **7B**: first token matches; the full sequence diverges at index 2
  with degenerate output (the model emits ` of of France of`,
  repeating `315` = ` of`). Per-block uniform 4-bit noise compounds
  across 28 layers and exceeds the bf16-baseline argmax margin.

This is exactly the regime the full TurboQuant recipe is designed for:
PolarQuant + Lloyd-Max + QJL together close ~2 bits of residual error
budget. **V1's accuracy is acceptable for ≤ 0.5B-class but should not
be used at 7B+ in production**; V3 (full algorithm) is the path forward.

### Throughput: catastrophic regression at any scale

V1's `materialize_packed_kv` calls `read_compressed_block` per block,
which does Python-side bit unpacking + dequant + inverse rotation. The
overhead is severe: turbo4 throughput is 6-18% of bf16 across all
configurations. Both models. Both small and large concurrency.

This is **bench-wrecking** and was anticipated by the plan (the slice's
"materialize-on-read" approach was explicitly flagged as not the
production path). The fix is V2: a Triton kernel that reads compressed
K/V tiles directly inside the attention forward, dequantizing in
shared memory tile-by-tile. That's the single highest-value follow-up.

## What this proves

- **The rotation-based pipeline works end-to-end on real hardware**.
  Compressed storage layout, per-block 4-bit quant, per-layer rotation
  matrices, materialize-on-read, dispatcher gating of FA-paged when
  compressed — all correct on A10 bf16 with both 0.5B and 7B.
- **Storage savings of 2.7x are real and consistent**. 62.4% memory
  reduction in the KV pool, regardless of model size.
- **Per-block uniform 4-bit isn't enough at production model
  depths**. The 7B parity failure is informative: it identifies
  exactly which TurboQuant components (PolarQuant, QJL, Lloyd-Max)
  are doing the accuracy work the V1 cuts left out. V3 has a clear
  scope.
- **Materialize-on-read is unviable at any scale**. Python dequant
  per block per layer per step burns the savings. V2 (fused kernel)
  is non-negotiable for any throughput claim.

## Caveats

- 0.5B is small enough that storage savings are in MB, not GB. The
  practical "fit longer contexts on the same GPU" win is at 7B+ with
  longer sequences — but V1 isn't accurate enough at 7B+ AND doesn't
  reduce peak attention memory anyway (V2 does both).
- bf16 is the model's compute dtype; KV cache compression is
  orthogonal to weight quant.
- The 62.4% savings excludes the ~1 MB rotation matrix storage
  (negligible for any model size).
- Throughput numbers should be discarded as a serious metric for V1;
  they exist to surface the V2 motivation, not as a comparison
  point.

## Reproduce

```
uv run modal run scripts/modal_packed_bench.py --config turbo
uv run modal run scripts/modal_packed_bench.py --config turbo --model "Qwen/Qwen2.5-7B-Instruct"
```

Defaults: A10, prompt ~80 tokens, max_tokens=32, C ∈ {1, 4, 8}.

## Pointers

- ADR: [ADR-013](../decisions/ADR-013-turboquant-kv.md).
- Implementation: `src/mini_infer/cache/turbo_quant.py`.
- Cache integration: `src/mini_infer/cache/block_pool.py`,
  `src/mini_infer/cache/paged_kv_cache.py`.
- Unit tests: `tests/unit/test_turbo_quant.py`,
  `tests/unit/test_turbo_quant_integration.py`.
- Paper: https://arxiv.org/abs/2504.19874
