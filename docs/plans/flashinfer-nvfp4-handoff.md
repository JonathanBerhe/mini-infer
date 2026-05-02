# FlashInfer NVFP4 KV — handoff for next session

## State of `mini-infer`

- Branch `main`, **8 commits ahead of origin** (not pushed):
  - `ddff2cc` NVFP4 KV API survey on B200 (probe only, full integration deferred)
  - `151e6af` FP8 KV cache via FlashInfer: 50% KV memory savings on H100
  - `0064049` FlashInfer fixes: CUDA 12.8 base image, plan-time sm_scale, prefill-only
  - `d62fc1c` FlashInfer paged-attention backend (bf16, opt-in)
  - `4fde1f8` Plan: integrate FlashInfer as the production attention backend
  - `280a317` ADR-013: FlashInfer NVFP4 is the 4-bit production answer on Blackwell
  - `7da28e7` Default to V2a; V2b kept opt-in (V2b 12% slower at 7B)
  - `e208851` Fully-fused TurboQuant decode attention kernel

- Working tree clean. Local tests + lint + mypy all green.

## Goal for the next session

**Ship Stage C of the FlashInfer integration**: a fully working
`kv_quant="nvfp4"` mode end-to-end on B200, validated against a **very
large model** (target: Qwen2.5-32B as the largest single-B200 fit, or
multi-B200 for 70B+). Bench should report KV pool savings (expect
~75% vs bf16), logit cosine sim vs bf16, and throughput.

## What's already in place (don't redo)

- FlashInfer backend wired in via `BlockPool(attention_backend="flashinfer")`.
  Bf16 (Stage A) and FP8 (Stage B) modes work end-to-end.
- Dispatcher in `src/mini_infer/cache/packed_attention.py` already
  routes `kv_quant in (None, "fp8")` to FlashInfer.
- `_FUSED_DISABLED_FOR_BENCH`-style toggle pattern available
  (`_FLASHINFER_DISABLED_FOR_BENCH` in `flashinfer_backend.py`).
- Modal image: `nvidia/cuda:12.8.0-devel-ubuntu22.04` + `torch>=2.6` (cu128 wheels for Blackwell) + `flashinfer-python>=0.6.10rc1`.
  See `scripts/modal_nvfp4_probe.py` for a known-good Blackwell image
  recipe. (The bench uses cu124 torch; that doesn't work on B200.)

## Concrete API knowledge from the B200 probes

From `docs/benchmarks/2026-05-02-flashinfer-nvfp4-probe.md`:

- `nvfp4_kv_quantize(input: Tensor, global_scale: Tensor) -> (fp4_data, block_scales)`
- `nvfp4_kv_dequantize(fp4_data, block_scales, global_scale, output_dtype=bf16) -> Tensor`
- `BatchPrefillWithPagedKVCacheWrapper.run()` has a `kv_cache_sf` kwarg for per-block scale factors (3D or 4D tensor required — 2D fails).
- `a_global_sf` / `global_scale` **must be a CUDA tensor**, not a Python float (same gotcha both helpers).
- `nvfp4_quantize` on `(M, num_kv_heads, head_dim)` returns
  `(M, num_kv_heads, head_dim // 2)` `uint8` packed nibbles + `(128, 8)`
  scale block. Need to confirm `nvfp4_kv_quantize`'s scale layout — that's the obvious next probe.

## Recommended next-session plan

1. **Quick B200 follow-up probe** (~$1): call `nvfp4_kv_quantize` on a
   `(num_blocks, page_size, num_kv_heads, head_dim)` paged input,
   inspect the returned `block_scales` shape. That tells us the
   `kv_cache_sf` layout the wrapper expects.
2. **Implement `kv_quant="nvfp4"`** in `BlockPool` + `PagedKVCache._write_packed_kv_nvfp4` mirroring Stage B (FP8). Storage:
   `_nvfp4_storage` packed-uint8 + `_nvfp4_block_scales` (shape from probe) + `_nvfp4_global_scale` (per-layer-per-side fp32 CUDA tensor).
3. **Wire in `flashinfer_backend.py`**: when `kv_quant == "nvfp4"`,
   pass quantized K/V via `paged_kv_cache=(k_q, v_q)` and per-block
   scales via `kv_cache_sf=...`. May need separate scale tensors per K/V side.
4. **Unit tests** under `@pytest.mark.requires_cuda + requires_model`, similar to `tests/unit/test_fp8_kv.py`.
5. **Modal B200 bench** with **Qwen2.5-32B** (or larger, if multi-B200 is available). Expect ~75% KV savings vs bf16 + cos sim > 0.99 on first decode.

## Standing user preferences (from memory)

- Explain results in **plain non-expert terms**, not jargon.
- **Kill Modal containers** between iterations (`modal app list` + `modal app stop --yes`).
- **Use a bigger model on bigger GPU**: Hopper/Blackwell benches default to Qwen2.5-7B+; smaller models undersell quant wins.
- **No roadmap references** ("Stage N", "ships later") in production
  source / comments — that lives in `docs/plans/` only.
- **No em dashes** in code/docs.
- **Modal cost-conscious but unblocked**: confirm GPU/duration/cost
  per run, but the user OK'd "no worry about costs" for this push.
- **Avoid sed**; use the Edit tool with `replace_all` for in-place changes.
- **No `Co-Authored-By: Claude`** trailer on commits.

## Where to start the next session

> "Continue Stage C of the FlashInfer integration: write a small Modal
> B200 probe to determine the shape of `nvfp4_kv_quantize`'s
> `block_scales` output on a `(num_blocks, page_size, num_kv_heads,
> head_dim)` paged input. Then implement `kv_quant=\"nvfp4\"` in
> `BlockPool` + `PagedKVCache` mirroring the FP8 path, wire it through
> `flashinfer_backend.py` via `kv_cache_sf`, and validate end-to-end on
> Qwen2.5-32B (or larger) on B200. See
> `docs/plans/flashinfer-nvfp4-handoff.md` for the API findings and
> standing preferences."
