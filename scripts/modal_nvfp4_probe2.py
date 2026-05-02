"""Modal probe (follow-up): paged-shape NVFP4 KV API on B200.

The first probe (`scripts/modal_nvfp4_probe.py`) confirmed FlashInfer
exposes `nvfp4_kv_quantize` / `nvfp4_kv_dequantize` and that
`BatchPrefillWithPagedKVCacheWrapper.run()` has a `kv_cache_sf` kwarg.
This follow-up answers the questions left open by that probe so we can
implement `kv_quant="nvfp4"` end-to-end:

A. What is the `block_scales` shape and dtype when `nvfp4_kv_quantize`
   is called on a paged input
   `(num_blocks, page_size, num_kv_heads, head_dim)`? That tensor is
   what the wrapper expects via `kv_cache_sf`.

B. What `kv_data_type` should `plan()` receive for the NVFP4 path?
   The fp4 storage is `uint8`-packed; we try `torch.uint8` first and
   capture whatever exception the wrapper raises if that's wrong.

C. Does the wrapper accept `(k_q, v_q)` as `paged_kv_cache` together
   with `(k_sf, v_sf)` as `kv_cache_sf` and per-side global scales as
   `k_scale` / `v_scale` floats? We try the smallest possible
   single-request prefill attention call and capture the result or
   exception.

Run with:

    MINI_INFER_BENCH_GPU=B200 uv run modal run scripts/modal_nvfp4_probe2.py
"""

import os

import modal

app = modal.App("mini-infer-nvfp4-probe2")

_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "B200")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-python>=0.6.10rc1")
)


@app.function(image=image, gpu=_BENCH_GPU, timeout=600)
def probe() -> str:
    import traceback

    import torch

    lines: list[str] = []
    lines.append(f"GPU: {torch.cuda.get_device_name()}")
    lines.append(f"Compute capability: {torch.cuda.get_device_capability()}")
    lines.append(f"PyTorch: {torch.__version__}")

    try:
        import flashinfer

        lines.append(f"FlashInfer: {flashinfer.__version__}")
    except ImportError as exc:
        lines.append(f"FlashInfer import failed: {exc}")
        return "\n".join(lines)

    # Sizes chosen to mirror Qwen2.5-32B's KV head geometry: head_dim=128,
    # num_kv_heads=8 (group_size=5 for q_heads=40). Page size 16 matches
    # mini-infer's default block_size.
    num_blocks = 4
    page_size = 16
    num_kv_heads = 8
    head_dim = 128

    # First probe (probe1) found the function rejects 4D tensors with
    # `M, K = input.shape`. Flatten to 2D first; we try two flatten
    # strategies (Probe A1 / A2) since the wrapper's `kv_cache_sf`
    # expects a 3D or 4D tensor — whichever flatten round-trips cleanly
    # back into a 3D/4D paged layout is the one we use.
    k_paged = (
        torch.randn(
            num_blocks, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )
    v_paged = k_paged.clone()
    abs_max_k = float(k_paged.float().abs().max().item())
    abs_max_v = float(v_paged.float().abs().max().item())
    # FlashInfer's NVFP4 reference uses (448 * 6) / abs_max as the global
    # scale (FP8 max * FP4 max-amax). Must be a CUDA fp32 tensor (lesson
    # from probe 1).
    global_scale_k = torch.tensor(
        [(448.0 * 6.0) / max(abs_max_k, 1e-6)], dtype=torch.float32, device="cuda"
    )
    global_scale_v = torch.tensor(
        [(448.0 * 6.0) / max(abs_max_v, 1e-6)], dtype=torch.float32, device="cuda"
    )

    # --- Probe A1: flatten to (num_blocks * page_size * num_kv_heads, head_dim) ---
    lines.append("")
    m_a1 = num_blocks * page_size * num_kv_heads
    lines.append(
        f"=== Probe A1: nvfp4_kv_quantize on flat-rows input ({m_a1}, {head_dim}) bf16 ==="
    )
    k_flat_a1 = k_paged.reshape(m_a1, head_dim).contiguous()
    v_flat_a1 = v_paged.reshape(m_a1, head_dim).contiguous()
    k_q_a1: torch.Tensor | None = None
    k_sf_a1: torch.Tensor | None = None
    v_q_a1: torch.Tensor | None = None
    v_sf_a1: torch.Tensor | None = None
    try:
        k_q_a1, k_sf_a1 = flashinfer.nvfp4_kv_quantize(k_flat_a1, global_scale_k)
        v_q_a1, v_sf_a1 = flashinfer.nvfp4_kv_quantize(v_flat_a1, global_scale_v)
        for label, t in (
            ("k_q_a1", k_q_a1),
            ("k_sf_a1", k_sf_a1),
            ("v_q_a1", v_q_a1),
            ("v_sf_a1", v_sf_a1),
        ):
            lines.append(f"  {label}: shape={tuple(t.shape)} dtype={t.dtype} numel={t.numel()}")
        bf16_bytes = k_flat_a1.element_size() * k_flat_a1.numel()
        nvfp4_bytes = (
            k_q_a1.element_size() * k_q_a1.numel() + k_sf_a1.element_size() * k_sf_a1.numel()
        )
        lines.append(
            f"  K bytes: bf16={bf16_bytes} -> nvfp4(data+sf)={nvfp4_bytes} "
            f"ratio={nvfp4_bytes / bf16_bytes:.3f}"
        )
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        lines.append("  " + traceback.format_exc(limit=4).replace("\n", "\n  "))

    # --- Probe A2: flatten to (num_blocks * page_size, num_kv_heads * head_dim) ---
    lines.append("")
    m_a2 = num_blocks * page_size
    k_a2 = num_kv_heads * head_dim
    lines.append(f"=== Probe A2: nvfp4_kv_quantize on input ({m_a2}, {k_a2}) bf16 ===")
    k_flat_a2 = k_paged.reshape(m_a2, k_a2).contiguous()
    try:
        k_q_a2, k_sf_a2 = flashinfer.nvfp4_kv_quantize(k_flat_a2, global_scale_k)
        for label, t in (("k_q_a2", k_q_a2), ("k_sf_a2", k_sf_a2)):
            lines.append(f"  {label}: shape={tuple(t.shape)} dtype={t.dtype} numel={t.numel()}")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")

    # --- Probe A3: introspect nvfp4_kv_quantize source / signature ---
    lines.append("")
    lines.append("=== Probe A3: nvfp4_kv_quantize source location + sf_vec_size default ===")
    try:
        import inspect

        src_file = inspect.getsourcefile(flashinfer.nvfp4_kv_quantize)
        lines.append(f"  source file: {src_file}")
        try:
            src = inspect.getsource(flashinfer.nvfp4_kv_quantize)
            # Print just the first 60 lines so we can see the body shape
            head = "\n".join(src.splitlines()[:60])
            lines.append("  source (first 60 lines):")
            for ln in head.splitlines():
                lines.append(f"    {ln}")
        except Exception as exc:
            lines.append(f"  source unavailable: {exc}")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")

    if k_q_a1 is None or k_sf_a1 is None or v_q_a1 is None or v_sf_a1 is None:
        # Without successful A1 we can't run the rest of the probes.
        return "\n".join(lines)
    k_q, k_sf = k_q_a1, k_sf_a1
    v_q, v_sf = v_q_a1, v_sf_a1

    # --- Probe B: round-trip dequant accuracy ---
    lines.append("")
    lines.append("=== Probe B: nvfp4_kv_dequantize round-trip accuracy ===")
    try:
        k_recovered = flashinfer.nvfp4_kv_dequantize(
            k_q, k_sf, global_scale_k, output_dtype=torch.bfloat16
        )
        ref = k_flat_a1.float().flatten()
        rec = k_recovered.float().flatten()
        cos = torch.nn.functional.cosine_similarity(ref, rec, dim=0).item()
        rel_err = float((ref - rec).abs().mean().item()) / max(float(ref.abs().mean().item()), 1e-9)
        lines.append(
            f"  k_recovered: shape={tuple(k_recovered.shape)} dtype={k_recovered.dtype} "
            f"cos_sim={cos:.6f} rel_err={rel_err:.6f}"
        )
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        lines.append("  " + traceback.format_exc(limit=4).replace("\n", "\n  "))

    # --- Probe C: end-to-end paged attention with NVFP4 K/V ---
    lines.append("")
    lines.append("=== Probe C: BatchPrefillWithPagedKVCacheWrapper with NVFP4 KV ===")
    num_q_heads = 40  # Qwen2.5-32B has 40 q heads, 8 kv heads
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    # Reshape the 2D quantized output back to a 4D paged layout the
    # wrapper can index. fp4_data has last-dim halved (head_dim // 2).
    k_q_paged = k_q.reshape(num_blocks, page_size, num_kv_heads, head_dim // 2).contiguous()
    v_q_paged = v_q.reshape(num_blocks, page_size, num_kv_heads, head_dim // 2).contiguous()
    sf_last = k_sf.shape[-1] if k_sf.dim() >= 1 else 1
    # block_scales come out as (M, sf_last). Try reshaping to 4D paged
    # layout the same way data is reshaped.
    k_sf_paged = k_sf.reshape(num_blocks, page_size, num_kv_heads, sf_last).contiguous()
    v_sf_paged = v_sf.reshape(num_blocks, page_size, num_kv_heads, sf_last).contiguous()
    lines.append(
        f"  reshaped k_q_paged: {tuple(k_q_paged.shape)} dtype={k_q_paged.dtype}; "
        f"k_sf_paged: {tuple(k_sf_paged.shape)} dtype={k_sf_paged.dtype}"
    )

    # Single request, one full page used.
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda")

    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1

    def _try_variant(label: str, plan_kwargs: dict, run_kwargs: dict) -> None:
        try:
            wrapper.plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=True,
                pos_encoding_mode="NONE",
                sm_scale=1.0 / (head_dim**0.5),
                q_data_type=torch.bfloat16,
                **plan_kwargs,
            )
            out = wrapper.run(q, (k_q_paged, v_q_paged), **run_kwargs)
            lines.append(f"    OK: out shape={tuple(out.shape)} dtype={out.dtype}")
        except Exception as exc:
            lines.append(f"    EXCEPTION: {type(exc).__name__}: {exc}")
            lines.append("    " + traceback.format_exc(limit=4).replace("\n", "\n    "))

    lines.append("  -- Variant 1: kv_data_type=k_q.dtype, kv_cache_sf=(k_sf_paged, v_sf_paged) --")
    _try_variant(
        "v1",
        plan_kwargs={"kv_data_type": k_q.dtype},
        run_kwargs={"kv_cache_sf": (k_sf_paged, v_sf_paged)},
    )

    lines.append(
        "  -- Variant 2: kv_data_type=k_q.dtype, kv_cache_sf=stack([k_sf_paged, v_sf_paged]) --"
    )
    kv_sf_stacked = torch.stack([k_sf_paged, v_sf_paged], dim=0).contiguous()
    lines.append(f"    stacked kv_sf shape: {tuple(kv_sf_stacked.shape)}")
    _try_variant(
        "v2",
        plan_kwargs={"kv_data_type": k_q.dtype},
        run_kwargs={"kv_cache_sf": kv_sf_stacked},
    )

    lines.append(
        "  -- Variant 3: + k_scale, v_scale = float(global_scale) (combined global+per-block) --"
    )
    _try_variant(
        "v3",
        plan_kwargs={"kv_data_type": k_q.dtype},
        run_kwargs={
            "kv_cache_sf": (k_sf_paged, v_sf_paged),
            "k_scale": float(global_scale_k.item()),
            "v_scale": float(global_scale_v.item()),
        },
    )

    lines.append("  -- Variant 4: kv_cache_sf=raw 2D k_sf (no reshape) --")
    _try_variant(
        "v4",
        plan_kwargs={"kv_data_type": k_q.dtype},
        run_kwargs={"kv_cache_sf": (k_sf, v_sf)},
    )

    # --- Probe D: attention parity bf16 vs NVFP4 ---
    # The standalone nvfp4_kv_dequantize had cos_sim=0 in Probe B. That
    # might mean the layout expected by the wrapper differs from what the
    # standalone dequant reads, OR the global_scale direction differs
    # between quant/dequant. The only thing that matters for our
    # integration is whether the wrapper produces a correct attention
    # output. We compute a bf16 reference with the same K/V data, then
    # compare the NVFP4 attention output cosine sim.
    lines.append("")
    lines.append("=== Probe D: attention parity bf16 ref vs NVFP4 (Variants 1 and 3) ===")
    try:
        # bf16 reference: use the same wrapper but with bf16 KV.
        bf16_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        bf16_wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            pos_encoding_mode="NONE",
            sm_scale=1.0 / (head_dim**0.5),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        out_bf16 = bf16_wrapper.run(q, (k_paged, v_paged))
        lines.append(f"  bf16 ref out: shape={tuple(out_bf16.shape)} dtype={out_bf16.dtype}")

        def _nvfp4_attn(label: str, run_kwargs: dict) -> None:
            wrapper.plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=True,
                pos_encoding_mode="NONE",
                sm_scale=1.0 / (head_dim**0.5),
                q_data_type=torch.bfloat16,
                kv_data_type=k_q.dtype,
            )
            out_nvfp4 = wrapper.run(q, (k_q_paged, v_q_paged), **run_kwargs)
            ref = out_bf16.float().flatten()
            rec = out_nvfp4.float().flatten()
            cos = torch.nn.functional.cosine_similarity(ref, rec, dim=0).item()
            rel_err = float((ref - rec).abs().mean().item()) / max(
                float(ref.abs().mean().item()), 1e-9
            )
            lines.append(f"  {label}: cos_sim={cos:.6f} rel_err={rel_err:.6f}")

        _nvfp4_attn(
            "Variant 1 (kv_cache_sf only, no global scale)",
            {"kv_cache_sf": (k_sf_paged, v_sf_paged)},
        )
        _nvfp4_attn(
            "Variant 3 (kv_cache_sf + k_scale=global_sf)",
            {
                "kv_cache_sf": (k_sf_paged, v_sf_paged),
                "k_scale": float(global_scale_k.item()),
                "v_scale": float(global_scale_v.item()),
            },
        )
        # Variant 5: try inverse global scale (1 / global_scale_k) in case
        # the dequant pipeline expects multiplication by inverse.
        _nvfp4_attn(
            "Variant 5 (kv_cache_sf + k_scale=1/global_sf)",
            {
                "kv_cache_sf": (k_sf_paged, v_sf_paged),
                "k_scale": 1.0 / float(global_scale_k.item()),
                "v_scale": 1.0 / float(global_scale_v.item()),
            },
        )
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        lines.append("  " + traceback.format_exc(limit=4).replace("\n", "\n  "))

    # --- Probe E: nvfp4_kv_dequantize source ---
    lines.append("")
    lines.append("=== Probe E: nvfp4_kv_dequantize source (to explain round-trip failure) ===")
    try:
        import inspect

        src_file = inspect.getsourcefile(flashinfer.nvfp4_kv_dequantize)
        lines.append(f"  source file: {src_file}")
        try:
            src = inspect.getsource(flashinfer.nvfp4_kv_dequantize)
            head = "\n".join(src.splitlines()[:60])
            lines.append("  source (first 60 lines):")
            for ln in head.splitlines():
                lines.append(f"    {ln}")
        except Exception as exc:
            lines.append(f"  source unavailable: {exc}")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")

    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    print(probe.remote())
