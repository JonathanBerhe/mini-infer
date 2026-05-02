# FlashInfer NVFP4 KV API survey on B200

Date: 2026-05-02
Hardware: NVIDIA B200 (Blackwell, SM_100)
Engine: probe only — does not integrate NVFP4 KV into mini-infer's pool
Script: `scripts/modal_nvfp4_probe.py`

This was a focused API survey rather than a full integration. The
research that informed Stage C of the FlashInfer plan flagged NVFP4 KV
as "experimental, not in vLLM mainline as of May 2026" and lacked a
worked end-to-end example. Rather than burn Blackwell budget chasing
the right call sequence, we ran two short probe scripts on a real B200
to collect concrete API data.

## Confirmed facts

### Hardware + library detection
- **B200 detected**: `torch.cuda.get_device_name() == "NVIDIA B200"`,
  compute capability `(10, 0)` = SM_100.
- `torch>=2.6` with the `cu128` wheel index has Blackwell tensor-core
  kernels; the older `torch==2.5.1+cu124` we use for Hopper would
  fail with `no kernel image is available for execution on the device`
  on the first tensor op.
- FlashInfer 0.6.10rc1 imports cleanly on B200.

### NVFP4 quantization helpers (top-level FlashInfer API)
All three functions exist as `flashinfer.X`:
- `nvfp4_quantize(input, a_global_sf, sf_vec_size=16, sfLayout=...)` —
  generic quantizer.
- `nvfp4_kv_quantize(input: torch.Tensor, global_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
  — KV-specific helper with a simpler signature; returns
  `(fp4_data, block_scales)`.
- `nvfp4_kv_dequantize(fp4_data, block_scales, global_scale, output_dtype=torch.bfloat16) -> torch.Tensor`
  — companion dequant.

### `nvfp4_quantize` output shapes
For an input `(32, 4, 128)` bf16 tensor (16384 values total) with
`sf_vec_size=16`:
- `fp4_data`: `(32, 4, 64)` `torch.uint8` — packed nibbles, last dim
  halved (each byte = 2 fp4 values).
- `block_scales`: `(128, 8)` `torch.uint8` — flat 1024-byte
  scale-factor block (16384 / sf_vec_size = 1024 scale factors,
  reshaped to (128, 8) per the layout).

### `BatchPrefillWithPagedKVCacheWrapper.run()` signature
Confirmed parameters from `inspect.signature(...)`:
```
self, q, paged_kv_cache, *args,
q_scale, k_scale, v_scale,
out, lse, return_lse, enable_pdl, window_left, sinks,
kv_cache_sf,                                # ← NVFP4 scale-factor channel
skip_softmax_threshold_scale_factor
```
The `kv_cache_sf` kwarg is the integration point for NVFP4 — it
exists alongside the FP8 `k_scale` / `v_scale` floats but is meant for
the per-block scale-factor tensor returned by the NVFP4 quantizers.

### Calling-convention gotchas
- `nvfp4_quantize`'s `a_global_sf` argument **must be a CUDA tensor**,
  not a Python float (`AttributeError: 'float' object has no
  attribute 'cuda'`). `nvfp4_kv_quantize` follows the same pattern
  (`global_scale: torch.Tensor`).
- The `kv_cache_sf` argument expects a **3D or 4D** tensor (`ValueError:
  x must be 3D or 4D` from the wrapper). `nvfp4_quantize`'s
  `(128, 8)` 2D scales are the wrong shape; the KV helper
  (`nvfp4_kv_quantize`) likely returns a layout that's already
  paged-compatible. We didn't probe that, but it's the next step.

## Open questions

1. **Scale-factor shape returned by `nvfp4_kv_quantize` for paged input.**
   The probe only ran the generic `nvfp4_quantize`. Running
   `nvfp4_kv_quantize` on `(num_blocks, page_size, num_kv_heads,
   head_dim)` bf16 is the natural next probe, since its return
   `block_scales` should match what `kv_cache_sf` expects.
2. **Exact paged-cache layout** for NVFP4 K/V. Plausible candidates:
   - `(num_blocks, page_size, num_kv_heads, head_dim // 2)` `uint8`
     packed (2 fp4 values per byte, NHD layout).
   - HND alternative if FlashInfer's NVFP4 path requires it.
3. **Whether `q_data_type=torch.bfloat16, kv_data_type=torch.uint8`
   in `plan()` is the right combination** to flag NVFP4 KV. The
   wrapper may have a separate `kv_data_type` value (e.g.,
   `FLOAT4_E2M1X2`) we haven't seen.

## Stage C disposition

The probe data is enough to design a real NVFP4 KV integration in a
future slice:

1. Allocate `_nvfp4_storage`: `(num_layers, 2, num_blocks, page_size,
   num_kv_heads, head_dim // 2)` `torch.uint8`.
2. Allocate `_nvfp4_block_scales` with whatever layout
   `nvfp4_kv_quantize` produces (probe TBD).
3. Allocate `_nvfp4_global_scale`: `(num_layers, 2)` fp32 CUDA tensor
   set on first append (one global scale per layer per side).
4. In `_write_packed_kv_nvfp4`: call `nvfp4_kv_quantize` per layer,
   store outputs.
5. In `flashinfer_attention_forward`: pass quantized K/V via
   `paged_kv_cache=(k_q, v_q)` and the per-block scales via
   `kv_cache_sf=...`.

This is real engineering work, not blocked by speculation anymore.
Deferred to a follow-up slice; this one ships the probe + findings.

## Reproduce

```
MINI_INFER_BENCH_GPU=B200 modal run scripts/modal_nvfp4_probe.py
```

Two B200 runs total to collect the data above (~$5 spend).

## Pointers

- Probe: [scripts/modal_nvfp4_probe.py](../../scripts/modal_nvfp4_probe.py)
- Plan: [docs/plans/flashinfer-integration.md](../plans/flashinfer-integration.md)
- ADR: [ADR-013 / FlashInfer recommendations](../decisions/ADR-013-turboquant-kv.md)
- Stage A bench: [2026-05-02-flashinfer-bf16.md](2026-05-02-flashinfer-bf16.md)
- Stage B bench: [2026-05-02-flashinfer-fp8-kv.md](2026-05-02-flashinfer-fp8-kv.md)
