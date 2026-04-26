"""CUDA smoke for the paged attention kernel + Qwen2 patch.

Run with: uv run modal run scripts/modal_paged_smoke.py
"""

import modal

app = modal.App("mini-infer-paged-smoke")

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

    from mini_infer.cache.paged_attention import supports_paged_kernel
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    gpu_name = torch.cuda.get_device_name()
    kernel_ok = supports_paged_kernel(torch.device("cuda"))

    runner = ModelRunner.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    assert runner.device == "cuda"

    pool_before = runner.block_pool.num_free_blocks
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
    pool_after = runner.block_pool.num_free_blocks

    assert "Paris" in result.text, f"unexpected output: {result.text!r}"
    assert pool_after == pool_before, f"pool leaked blocks: {pool_before} -> {pool_after}"

    return (
        f"OK | torch={torch.__version__} | gpu={gpu_name} | kernel_ok={kernel_ok} | "
        f"output={result.text!r} | pool_blocks={pool_before}/{runner.block_pool.num_blocks}"
    )


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote())
