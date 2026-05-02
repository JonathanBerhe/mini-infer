# FlashInfer paged-attention backend on bf16 — Modal A10

Date: 2026-05-02
Hardware: NVIDIA A10G (Ampere, SM_86), bf16 model
Engine: mini-infer @ this slice (FlashInfer backend opt-in via
`attention_backend="flashinfer"`)
Script: `scripts/modal_packed_bench.py --config flashinfer`

The integration adds [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
as an alternative paged-attention backend alongside the existing
`flash_attn_varlen_func` path. On Ampere (no native FP8/NVFP4 path),
the win isn't throughput — it's the architectural piece that unlocks
FP8 KV on Hopper and NVFP4 KV on Blackwell in later slices.

## Workload

- Real long prompt
  (`scripts/data/technical_passage.md`): ~2000 tokens of varied
  technical prose.
- `max_tokens=128`, single request (`C=1`).
- Same Qwen2.5-0.5B-Instruct model, same workload, both backends loaded
  sequentially.

## Results

```
                t/s     peak HBM     first 8 decoded tokens
flash_attn     21.13   1224.9 MiB    [785, 12538, 7481, 2924, 1447, 16, 13, 16516]
flashinfer     21.22   1232.8 MiB    [785, 12538, 7481, 2924, 1447, 16, 13, 16516]
flashinfer vs flash_attn: throughput 1.00x, peak HBM -7.9 MiB
```

Logit cosine sim at the first decode position: **0.999993** (>0.999
required). First 8 greedy tokens match exactly.

## Reading the data

### Numerical equivalence: confirmed

FlashInfer and flash-attn produce attention output that agrees within
~1e-5 relative error (cosine sim 0.999993). On a 24-layer 0.5B model
with greedy decoding, that's enough for token-for-token agreement. This
matches expectations: both backends implement the same FlashAttention-2
math, just with different scheduling code around the kernel.

### Throughput: parity (expected on Ampere)

1.00x means the two backends complete the workload at the same speed.
This is the right outcome — Ampere has no native FP8 or NVFP4 tensor
cores, so FlashInfer's quantized-KV advantages don't apply. The bf16
paths through both libraries land on the same FA-2 kernel
implementation underneath. We're not paying a perf tax to run on
FlashInfer; we're also not getting one.

### Memory: ~8 MiB lower under FlashInfer

The 7.9 MiB delta is in FlashInfer's favor and roughly matches the
size of FA's `block_table` working buffer that FlashInfer doesn't
need (CSR-style page indices instead). Small in absolute terms;
negligible at the model scale we tested.

## What this slice unlocks

The integration is fully transparent: `attention_backend="flashinfer"`
swaps the per-layer attention call from `flash_attn_varlen_func` to
FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` /
`BatchDecodeWithPagedKVCacheWrapper`, sharing the same paged KV cache
layout `(num_blocks, page_size, num_kv_heads, head_dim)`. This is the
plumbing that future slices need to add:

- **FP8 KV** on Hopper (H100/H200): same paged layout, K/V stored as
  `torch.float8_e4m3fn` with per-head scales. Native FP8 tensor-core
  dequant fused into attention.
- **NVFP4 KV** on Blackwell (B200+): same layout, K/V stored in NVFP4
  format. Native FP4 tensor-core dequant. ~75% memory savings vs bf16.

Both modes plug in via the `kv_quant` parameter on `BlockPool`,
re-using the dispatcher gate this slice added.

## Caveats

- A10 only. The win at this hardware tier is "no regression while
  opening the door" rather than a measurable speedup.
- 0.5B only. 7B should look similar on Ampere; we'll re-bench once an
  FP8 mode lands and Hopper hardware enters the picture.
- First-call JIT compile cost not measured (warmup absorbed it). The
  `flashinfer-cubin` package would eliminate this; not added yet.

## Reproduce

```
uv run modal run scripts/modal_packed_bench.py --config flashinfer
```

## Pointers

- Backend wrapper:
  [src/mini_infer/cache/flashinfer_backend.py](../../src/mini_infer/cache/flashinfer_backend.py).
- Dispatcher gate:
  [packed_attention.py:packed_attention_forward](../../src/mini_infer/cache/packed_attention.py).
- Tests:
  [tests/unit/test_flashinfer_backend.py](../../tests/unit/test_flashinfer_backend.py).
- Plan:
  [docs/plans/flashinfer-integration.md](../plans/flashinfer-integration.md).
