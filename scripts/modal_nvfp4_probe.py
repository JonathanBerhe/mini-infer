"""Modal probe: inspect FlashInfer's NVFP4 KV-cache API surface on B200.

Goal: collect enough concrete data about FlashInfer's NVFP4 quant + paged
attention path that a future slice can wire NVFP4 KV into mini-infer
without speculating. We don't try to make the engine work end-to-end
here — we just call the FlashInfer functions, print shapes/dtypes/error
messages, and exit.

Run with:

    MINI_INFER_BENCH_GPU=B200 uv run modal run scripts/modal_nvfp4_probe.py
"""

import os

import modal

app = modal.App("mini-infer-nvfp4-probe")

_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "B200")

image = (
    # CUDA 12.8 devel image; need a Blackwell-aware torch build (the
    # cu124 wheel only ships kernels through SM_90 / Hopper and would
    # fail on B200 with "no kernel image" at the first tensor op).
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-python>=0.6.10rc1")
)


@app.function(image=image, gpu=_BENCH_GPU, timeout=600)
def probe() -> str:
    """Inspect FlashInfer's NVFP4 KV API on Blackwell.

    Returns a human-readable report. Each section either prints a
    successful sample or captures the exception and reports it, so we
    can survey multiple API surfaces in one run.
    """
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

    # --- Probe 1: nvfp4_quantize with a_global_sf as a CUDA tensor ---
    lines.append("")
    lines.append("=== Probe 1: nvfp4_quantize on (32, 4, 128) bf16 K, a_global_sf=tensor ===")
    k_bf16 = torch.randn(32, 4, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    try:
        from flashinfer.fp4_quantization import nvfp4_quantize

        # Per the prior probe: a_global_sf must be a CUDA tensor, not a float.
        global_sf = torch.tensor(
            [(448.0 * 6.0) / float(k_bf16.float().abs().max().item())],
            dtype=torch.float32,
            device="cuda",
        )
        result = nvfp4_quantize(k_bf16, global_sf, sf_vec_size=16)
        if isinstance(result, tuple):
            for i, t in enumerate(result):
                if isinstance(t, torch.Tensor):
                    lines.append(
                        f"  return[{i}]: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
                    )
                else:
                    lines.append(f"  return[{i}]: {type(t).__name__} = {t!r}")
        else:
            lines.append(f"  return: {type(result).__name__} = {result!r}")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        lines.append("  " + traceback.format_exc(limit=3).replace("\n", "\n  "))

    # --- Probe 1b: signature inspection of nvfp4_kv_quantize / dequantize ---
    lines.append("")
    lines.append("=== Probe 1b: nvfp4_kv_quantize / nvfp4_kv_dequantize signatures ===")
    try:
        import inspect

        for fname in ("nvfp4_kv_quantize", "nvfp4_kv_dequantize"):
            fn = getattr(flashinfer, fname)
            sig = inspect.signature(fn)
            lines.append(f"  {fname}{sig}")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")

    # --- Probe 2: discover NVFP4-specific helper functions ---
    lines.append("")
    lines.append("=== Probe 2: NVFP4 / FP4 helpers exposed by flashinfer ===")
    candidates = [
        "fp4_quantization",
        "nvfp4_quantize",
        "nvfp4_kv_quantize",
        "nvfp4_kv_dequantize",
    ]
    for name in candidates:
        obj = getattr(flashinfer, name, None)
        sub = None
        if obj is None:
            mod = getattr(flashinfer, "fp4_quantization", None)
            if mod is not None:
                sub = getattr(mod, name, None)
        target = obj if obj is not None else sub
        if target is None:
            lines.append(f"  flashinfer.{name}: NOT FOUND")
        else:
            lines.append(f"  flashinfer.{name}: {target!r}")

    # --- Probe 3: BatchPrefillWithPagedKVCacheWrapper signature inspection ---
    lines.append("")
    lines.append("=== Probe 3: BatchPrefillWithPagedKVCacheWrapper.run signature ===")
    try:
        import inspect

        sig = inspect.signature(flashinfer.BatchPrefillWithPagedKVCacheWrapper.run)
        lines.append(f"  parameters: {list(sig.parameters.keys())}")
        for pname, pobj in sig.parameters.items():
            ann = getattr(pobj, "annotation", None)
            default = getattr(pobj, "default", inspect.Parameter.empty)
            default_str = "<no default>" if default is inspect.Parameter.empty else repr(default)
            lines.append(f"    {pname}: annotation={ann!r}, default={default_str}")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")

    # --- Probe 4: attempt a paged NVFP4 attention call to see what breaks ---
    lines.append("")
    lines.append("=== Probe 4: paged NVFP4 attention attempt (expect TypeError) ===")
    try:
        from flashinfer.fp4_quantization import nvfp4_quantize

        # Single-request, 1 page, 16-token block, 4 kv heads, 128 head_dim.
        page_size = 16
        num_kv_heads = 4
        head_dim = 128
        num_q_heads = 28

        # Build the page in bf16, then quantize whole-page to NVFP4.
        k_page = (
            torch.randn(1, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            * 0.1
        )
        v_page = k_page.clone()
        global_sf_k = torch.tensor(
            [(448.0 * 6.0) / float(k_page.float().abs().max().item())],
            dtype=torch.float32,
            device="cuda",
        )
        global_sf_v = torch.tensor(
            [(448.0 * 6.0) / float(v_page.float().abs().max().item())],
            dtype=torch.float32,
            device="cuda",
        )
        k_q, k_sf = nvfp4_quantize(k_page, global_sf_k, sf_vec_size=16)
        v_q, _v_sf = nvfp4_quantize(v_page, global_sf_v, sf_vec_size=16)
        lines.append(
            f"  k_q: shape={tuple(k_q.shape)} dtype={k_q.dtype}; "
            f"k_sf: shape={tuple(k_sf.shape)} dtype={k_sf.dtype}"
        )

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        kv_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
        last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda")

        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
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
        q = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        # Pass scale-factor tensors via the `kv_cache_sf` kwarg discovered
        # in the prior probe of the run() signature.
        out = wrapper.run(q, (k_q, v_q), kv_cache_sf=(k_sf, k_sf))  # placeholder layout
        lines.append(f"  run() succeeded: out shape={tuple(out.shape)} dtype={out.dtype}")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        lines.append("  " + traceback.format_exc(limit=4).replace("\n", "\n  "))

    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    print(probe.remote())
