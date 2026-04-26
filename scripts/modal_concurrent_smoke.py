"""Modal concurrent smoke: 4 in-flight requests through the batched scheduler on CUDA.

Verifies (1) each request produces sensible per-prompt output, and (2) the batched
scheduler's tokens match a serial reference run on the same hardware. The serial
reference proves the batched forward isn't silently corrupting per-request state.
"""

# Run with: uv run modal run scripts/modal_concurrent_smoke.py

import modal

app = modal.App("mini-infer-concurrent-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.40",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "triton>=3.0",
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

    prompts = [
        "The capital of France is",
        "Once upon a time",
        "def fibonacci(n):",
        "The quickest path from A to B is",
    ]
    max_tokens = 8

    # Concurrent run: submit all four, then wait. The scheduler should run them
    # in a shared batched forward (B up to 4) on most steps.
    scheduler = ContinuousScheduler(runner)
    scheduler.start()
    try:
        handles = [
            scheduler.submit(
                Request(prompt=p, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
            for p in prompts
        ]
        concurrent_results = [h.wait() for h in handles]
    finally:
        scheduler.stop()

    # Serial reference: a fresh single-request scheduler. Greedy sampling must
    # produce identical tokens to the concurrent run.
    serial_scheduler = ContinuousScheduler(runner, max_concurrent=1)
    serial_scheduler.start()
    try:
        serial_results = [
            serial_scheduler.run(
                Request(prompt=p, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
            for p in prompts
        ]
    finally:
        serial_scheduler.stop()

    diffs: list[str] = []
    for prompt, c, s in zip(prompts, concurrent_results, serial_results, strict=True):
        if c.tokens != s.tokens:
            diffs.append(f"  diverged on {prompt!r}: concurrent={c.tokens} vs serial={s.tokens}")

    if diffs:
        raise AssertionError("batched != serial on CUDA:\n" + "\n".join(diffs))

    summary = " | ".join(
        f"{p[:24]!r}->{r.text[:24]!r}" for p, r in zip(prompts, concurrent_results, strict=True)
    )
    return f"OK | gpu={gpu_name} | dtype=bf16 | {summary}"


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote())
