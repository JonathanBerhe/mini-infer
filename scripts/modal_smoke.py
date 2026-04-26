"""CUDA smoke test on Modal: load Qwen2.5-0.5B, generate, verify cuda device."""

# Run with: uv run modal run scripts/modal_smoke.py
# Cost: roughly $0.02 per run on an A10 (under a minute including model load).

import modal

app = modal.App("mini-infer-cuda-smoke")

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


@app.function(image=image, gpu="A10", timeout=600)
def smoke() -> str:
    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    gpu_name = torch.cuda.get_device_name()

    runner = ModelRunner.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
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

    return (
        f"OK | torch={torch.__version__} | gpu={gpu_name} | "
        f"runner.device={runner.device} | output={result.text!r}"
    )


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote())
