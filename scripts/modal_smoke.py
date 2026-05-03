"""CUDA smoke test on Modal: load Qwen2.5-0.5B, generate, verify cuda device."""

# Run with:
#   uv run modal run scripts/modal_smoke.py                       # A10 default
#   MINI_INFER_BENCH_GPU=B200 uv run modal run scripts/modal_smoke.py

import os

import modal

_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "A10")
_IS_BLACKWELL = _BENCH_GPU.upper().startswith("B200")

app = modal.App("mini-infer-cuda-smoke")

if _IS_BLACKWELL:
    # Blackwell (SM_100) needs the cu128 torch wheel; the cu124 wheel ships
    # kernels through SM_90 only. flash-attn isn't installed because its
    # prebuilt wheel is pinned to torch 2.5; FlashInfer covers attention.
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
        .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
        .pip_install(
            "transformers>=4.40",
            "fastapi>=0.110",
            "uvicorn[standard]>=0.27",
            "pydantic>=2.5",
        )
        .pip_install("flashinfer-python>=0.6.10rc1")
        .add_local_python_source("mini_infer")
    )
else:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.4",
            "transformers>=4.40",
            "fastapi>=0.110",
            "uvicorn[standard]>=0.27",
            "pydantic>=2.5",
        )
        .add_local_python_source("mini_infer")
    )


@app.function(image=image, gpu=_BENCH_GPU, timeout=900)
def smoke() -> str:
    import torch

    from mini_infer.device import is_blackwell_device
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    gpu_name = torch.cuda.get_device_name()

    # Detect Blackwell at runtime via the CUDA compute-capability check.
    # Module-level env vars (`MINI_INFER_BENCH_GPU`) drive image selection on
    # the local side but don't propagate into the Modal container, so the
    # backend choice has to be made from device info inside the function.
    # On Blackwell the only working attention path is FlashInfer (flash-attn
    # isn't installed in the cu128 image), so we flag it explicitly here.
    attention_backend = "flashinfer" if is_blackwell_device() else "flash_attn"
    runner = ModelRunner.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        attention_backend=attention_backend,
    )
    assert runner.device == "cuda", f"expected cuda, got {runner.device!r}"

    scheduler = ContinuousScheduler(runner)
    scheduler.start()
    try:
        result = scheduler.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=8,
            )
        )
    finally:
        scheduler.stop()
    assert "Paris" in result.text, f"unexpected output: {result.text!r}"

    model_dtype = next(runner._model.parameters()).dtype
    return (
        f"OK | torch={torch.__version__} | gpu={gpu_name} | "
        f"runner.device={runner.device} | dtype={model_dtype} | "
        f"backend={attention_backend} | output={result.text!r}"
    )


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote())
