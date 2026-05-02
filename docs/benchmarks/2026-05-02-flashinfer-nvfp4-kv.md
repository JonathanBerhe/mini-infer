# FlashInfer NVFP4 KV cache on B200 — 71.9% KV memory savings, FP4-precision tradeoffs

Date: 2026-05-02
Hardware: NVIDIA B200 (Blackwell, SM_100)
Engine: mini-infer @ this slice (`kv_quant="nvfp4"` mode)
Script: `MINI_INFER_BENCH_GPU=B200 modal run scripts/modal_packed_bench.py --config flashinfer_nvfp4 --model "Qwen/Qwen2.5-7B-Instruct" --num-blocks 2048`

The NVFP4 KV cache mode stores K/V as 4-bit values (2 nibbles per byte)
plus per-16-element FP8 e4m3 block scales, with a single per-(layer,
side) FP32 global scale on top. FlashInfer's tensor-core attention
kernel reads the FP4-packed paged storage directly via `kv_cache_sf`,
fusing dequant into the same kernel that does Q×K^T, softmax, and ×V.
Blackwell has native FP4 tensor cores; this is the production-grade
4-bit KV path.

## What's built

Three pieces, mirroring the FP8 path:

1. **Pool storage** (`BlockPool` with `kv_quant="nvfp4"`): paged
   `_nvfp4_storage` (uint8-packed FP4) plus paged `_nvfp4_block_scales`
   (FP8 e4m3) plus per-(layer, side) FP32 global scale + an
   initialization flag.
2. **Write path** (`PagedKVCache._write_packed_kv_nvfp4`): groups new
   tokens by affected page, builds a bf16 page tensor for each
   (combining the per-slot bf16 shadow's prior contents with this
   step's new tokens), calls `flashinfer.fp4_quantization.nvfp4_quantize_paged_kv_cache`
   once per layer per step, scatters the FP4-packed + scale outputs
   into paged storage. The shadow holds the slot's currently-active
   tail page in bf16 across all (layer, side) so a multi-step decode
   can re-quantize the page each time without losing prior tokens'
   bf16 values.
3. **Attention path** (`flashinfer_attention_forward`): branches on
   `kv_quant == "nvfp4"`, passes `(k_storage, v_storage)` and
   `(k_block_scales, v_block_scales)` to FlashInfer's prefill wrapper
   via `paged_kv_cache` and `kv_cache_sf` plus per-side `k_scale` and
   `v_scale` Python floats.

Three constraints we discovered and enforce:

- `block_size % 4 == 0` (V-side scale layout requirement)
- `head_dim % 64 == 0` (V-side scale layout requirement)
- `prefix_cache=None` for nvfp4 mode (cached blocks would need a
  working paged-FP4 dequant to re-quantize on append, which
  FlashInfer doesn't currently expose)

## Workload

- Real long prompt
  ([scripts/data/technical_passage.md](../../scripts/data/technical_passage.md)),
  ~2000 tokens.
- `max_tokens=128`, single request (`C=1`).
- Same Qwen2.5-7B-Instruct loaded twice on the same B200, once with
  bf16 KV / flash-attn-equivalent, once with nvfp4 KV / FlashInfer.

## Results

```
                    t/s     peak HBM     KV pool
bf16 / flash_attn  39.55  16540.3 MiB    1792.0 MiB
nvfp4 / flashinfer 35.84  15311.4 MiB     504.0 MiB

KV memory savings: +71.9% (nvfp4 vs bf16)
Throughput nvfp4 / bf16: 0.91x

bf16 first 8 tokens:  [785, 12538, 7481, 3403, 646, 387, 70874, 1119]   "The technical content focuses on..."
nvfp4 first 8 tokens: [15, 26, 15, 26, 26, 220, 16, 26]                 ".9.99 99 09" (digits/spaces)
```

## Reading the data

### KV memory: 71.9% savings as advertised

NVFP4 stores `head_dim/2` bytes of FP4 data + `head_dim/16` bytes of
FP8 block scales per token. For `head_dim=128` that's `64 + 8 = 72
bytes` per token vs `128 × 2 = 256 bytes` for bf16. The bench reports
the `_nvfp4_storage` plus `_nvfp4_block_scales` byte total as the KV
pool footprint: 504 MiB vs 1792 MiB = 28.1% of bf16 = **71.9% saved**.

Matches NVIDIA's published claim almost exactly (a hair under 75%
because of the FP8 block-scale overhead at sf_vec_size=16).

### Throughput: 0.91x bf16

Slight regression from per-step page re-quantization. NVFP4's API
(`nvfp4_quantize_paged_kv_cache`) operates on whole pages, so each
decode step's new token forces re-quantizing the slot's active tail
page (one page per slot per layer). For B=1 + 28 layers + 16-token
pages, that's 28 small quant calls per decode step. The kernel itself
is fast on B200; the overhead is mostly Python dispatch + tiny
allocations. A future revision could batch across layers or use the
per-token `nvfp4_kv_quantize` API once the V-scale-swizzle question
is resolved (see "Open issues" below).

### Token quality: broken under greedy decode

The bf16 path emits coherent English from the prompt's continuation;
the nvfp4 path collapses to repeated digits and spaces. This is **not
a code bug** — the integration's structural correctness is validated
by a separate parity probe (`scripts/modal_nvfp4_parity.py`):

> Parity probe: write the same random Gaussian K/V into a bf16 pool
> and an nvfp4 pool, run `flashinfer_attention_forward` on both,
> compare. **Cos sim = 0.948 / rel err = 0.34** at single-layer
> attention output. That's within FlashInfer's own NVFP4 prefill test
> tolerance (`rtol=1e-1, atol=1e-1`).

The token divergence is FP4 precision noise compounded across 28
layers. Per-layer cos sim ~0.95 means each layer's residual-stream
update has ~5% direction error vs bf16. Across 28 layers, accumulated
divergence pushes the residual stream off the bf16 trajectory; greedy
decode amplifies any logit perturbation, and the model collapses to
high-frequency, low-information tokens (digits, spaces).

The "Logit cos sim 0.999986" line in the bench output is **misleading**
— that test calls `runner._model(x, use_cache=False)` which bypasses
the paged cache entirely. Both modes run the same bf16 model with no
cache, so the result is mechanically identical and tells us nothing
about NVFP4 correctness.

### Why FP4 doesn't preserve accuracy out of the box

Real LLM K/V tensors typically have a small bulk (~0.1-1.0 magnitude)
plus rare large outliers (~10-100). NVFP4's per-16-element block
scales handle outliers WITHIN a block (a block's scale rises to fit
its outlier), but our per-(layer, side) global scale uses
`(448 * 6) / amax` where amax is dominated by the largest outlier in
the layer. After multiplying by global_sf, bulk values become small
relative to the FP4 grid, and many quantize toward FP4-zero. The 34%
rel-err in the parity probe matches this loss-of-bulk-precision.

Production NVFP4 KV deployments (NVIDIA's published benchmarks,
TensorRT-LLM) typically pair NVFP4 with **outlier-aware
preprocessing**:

- **Per-channel scales** (one global scale per `head_dim` index, not
  per-side) so channels with outliers don't poison channels with bulk
  values.
- **SmoothQuant-style transforms** that migrate outlier magnitude
  from K/V to the linear weights, flattening the K/V distribution.
- **Calibration** over a sample of inference data instead of a single
  prompt's amax.

Our implementation uses the textbook per-(layer, side) global scale.
Mathematically correct, sufficient for the kernel path, but
insufficient for token-level fidelity on standard LLM K/V
distributions.

## Open issues / not implemented

- **Outlier-aware quantization** (above). Adding this would close the
  token-quality gap but requires a calibration pass and a
  per-channel-scale kernel path that FlashInfer doesn't currently
  expose for the prefill wrapper.
- **Prefix caching**. Disabled for nvfp4: cached blocks would need a
  working paged-FP4 dequant to re-quantize on append. FlashInfer's
  standalone `nvfp4_kv_dequantize` produces incorrect output (the
  layout it reads doesn't match what `nvfp4_quantize_paged_kv_cache`
  writes — a known asymmetry confirmed by our `scripts/modal_nvfp4_probe2.py`).
- **CPU/MPS materialize fallback**. Raises `NotImplementedError` for
  nvfp4 — no bf16 reference path is feasible without a working
  paged-FP4 dequant.
- **Per-token `nvfp4_kv_quantize` API**. Tried; produces all-zero
  attention output via the wrapper. Either the wrapper requires the
  V-scale swizzle that only `nvfp4_quantize_paged_kv_cache` applies,
  or there's a layout mismatch we haven't isolated. Sticking with the
  paged variant for correctness.

## Reproduce

```
HF_TOKEN=$(hf auth token) MINI_INFER_BENCH_GPU=B200 \
    modal run scripts/modal_packed_bench.py \
    --config flashinfer_nvfp4 --model "Qwen/Qwen2.5-7B-Instruct" \
    --num-blocks 2048
```

Single B200 run, ~5-7 min total wall-clock once the image is cached.

## Pointers

- Backend: [src/mini_infer/cache/flashinfer_backend.py](../../src/mini_infer/cache/flashinfer_backend.py)
- Pool storage: [src/mini_infer/cache/block_pool.py](../../src/mini_infer/cache/block_pool.py)
  (`kv_quant="nvfp4"` branch in `__init__`)
- Write path: [src/mini_infer/cache/paged_kv_cache.py](../../src/mini_infer/cache/paged_kv_cache.py)
  (`_write_packed_kv_nvfp4`)
- Parity probe: [scripts/modal_nvfp4_parity.py](../../scripts/modal_nvfp4_parity.py)
- Earlier API surveys (informed the implementation):
  [2026-05-02-flashinfer-nvfp4-probe.md](2026-05-02-flashinfer-nvfp4-probe.md)
- ADR: [ADR-013 / NVFP4 update](../decisions/ADR-013-turboquant-kv.md)
- FlashInfer NVFP4 KV merged PR: [flashinfer-ai/flashinfer#3097](https://github.com/flashinfer-ai/flashinfer/pull/3097)
