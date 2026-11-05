"""CUDA smoke test on Modal: load a model, generate, verify cuda device.

A PagedKVCache model runs through `ContinuousScheduler` with the FlashInfer
backend. A model with its own state cache (Kimi Linear, Qwen3.8, Nemotron-H)
runs through `StateCacheGenerator`, as the API server serves it, with the
decode CUDA graph where the model takes one.
"""

# Run with:
#   uv run modal run scripts/smoke/modal_smoke.py                       # A10 default
#   MINI_INFER_BENCH_GPU=B200 uv run modal run scripts/smoke/modal_smoke.py
#   MINI_INFER_BENCH_GPU=B200 MINI_INFER_BENCH_MODEL=google/gemma-4-31B-it \
#       uv run modal run scripts/smoke/modal_smoke.py

import os

import modal

_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "A10")
_BENCH_MODEL = os.environ.get("MINI_INFER_BENCH_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# A cost cap, not an expected duration: a hung container bills the whole cap.
_TIMEOUT_S = int(os.environ.get("MINI_INFER_BENCH_TIMEOUT", "3600"))

# Gated HF repos (Google Gemma 2/3/4, Meta Llama, ...) need an auth token.
# Pulled from the local `HF_TOKEN` env var and inlined into the container
# via `Secret.from_dict`. For ungated repos `HF_TOKEN` can be unset; the
# secrets list collapses to empty and downloads continue anonymously.
_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []

# Persistent volume for HF model cache. First run pays the full download
# cost once and writes safetensors here; subsequent runs of any model
# read from cache and skip the download. Modal auto-commits on clean
# function exit so future invocations see the contents.
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("mini-infer-cuda-smoke")

# One image for every GPU, from the A10 to the B300. The CUDA 13.2 torch wheel
# has Blackwell (sm_100) kernels, and Modal runs a B300 only with CUDA 13.1 or
# newer. flash-attn isn't installed because no wheel exists for torch 2.14 and
# CUDA 13; FlashInfer covers attention and compiles its kernels with the
# image's nvcc.
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.14.1+cu132", extra_index_url="https://download.pytorch.org/whl/cu132")
    .pip_install(
        "triton==3.8.0",
        "transformers==5.14.0",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
    )
    .pip_install("flashinfer-python==0.6.10")
    .add_local_python_source("mini_infer")
)


@app.function(
    image=image,
    gpu=_BENCH_GPU,
    timeout=_TIMEOUT_S,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def smoke(model_name: str) -> str:
    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.fast_paths import fast_path_report
    from mini_infer.models import architecture_uses_state_cache
    from mini_infer.scheduler import ContinuousScheduler, Request

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    gpu_name = torch.cuda.get_device_name()

    if architecture_uses_state_cache(model_name):
        return _state_cache_smoke(model_name, gpu_name)

    # FlashInfer is the attention backend of the image on every GPU. A model
    # that needs another backend (Gemma 4 31B needs "torch") overrides it.
    runner = ModelRunner.from_pretrained(model_name, attention_backend="flashinfer")
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
    # The attention entry is the backend that ran, after the model override.
    paths = fast_path_report(runner.block_pool.attention_backend)
    return (
        f"OK | model={model_name} | gpu={gpu_name} | runner.device={runner.device} | "
        f"dtype={model_dtype} | {paths} | output={result.text!r}"
    )


def _state_cache_smoke(model_name: str, gpu_name: str) -> str:
    """Greedy generation through the state-cache path, with the decode graph if supported."""
    from mini_infer.engine.cuda_graph_decode import supports_decode_cuda_graph
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.engine.state_cache_generator import StateCacheGenerator
    from mini_infer.fast_paths import fast_path_report
    from mini_infer.scheduler import Request, StateCacheContinuousScheduler

    generator = StateCacheGenerator.from_pretrained(model_name)
    cuda_graph = supports_decode_cuda_graph(generator.model, generator.device)
    scheduler = StateCacheContinuousScheduler(generator, cuda_graph=cuda_graph)
    scheduler.start()
    try:
        text = scheduler.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=8,
            )
        ).text
    finally:
        scheduler.stop()
    assert "Paris" in text, f"unexpected output: {text!r}"
    # These models run their own attention, so the attention entry does not apply.
    return (
        f"OK | model={model_name} | gpu={gpu_name} | state_cache=True | "
        f"cuda_graph={cuda_graph} | {fast_path_report('torch')} | output={text!r}"
    )


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote(_BENCH_MODEL))
