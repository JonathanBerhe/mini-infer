"""Modal continuous-batching throughput benchmark on CUDA.

Measures decode throughput (output tokens / second) at concurrency 1, 2, 4, 8.
Same prompt across all requests at each concurrency level so per-step work is
roughly equal. Compares against a single-request baseline run on the same model.
"""

# Run with: uv run modal run scripts/modal_concurrent_bench.py

import statistics
import time

import modal

app = modal.App("mini-infer-continuous-batching-bench")

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


@app.function(image=image, gpu="A10", timeout=900)
def bench() -> str:
    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name()

    runner = ModelRunner.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    prompt = "The quick brown fox jumps over the lazy dog. " * 4  # ~40 tokens
    max_tokens = 32
    concurrencies = [1, 2, 4, 8]

    rows: list[dict[str, float | int | str]] = []
    for concurrency in concurrencies:
        scheduler = ContinuousScheduler(runner, max_concurrent=concurrency)
        scheduler.start()
        try:
            # Warmup: one request to compile kernels + warm up CUDA caches.
            warmup = scheduler.run(
                Request(
                    prompt=prompt,
                    sampling_params=SamplingParams(),
                    max_tokens=max_tokens,
                )
            )
            assert warmup.finish_reason in {"stop", "length"}

            # Timed run: submit `concurrency` requests, wait for all.
            torch.cuda.synchronize()
            start = time.perf_counter()
            handles = [
                scheduler.submit(
                    Request(
                        prompt=prompt,
                        sampling_params=SamplingParams(),
                        max_tokens=max_tokens,
                    )
                )
                for _ in range(concurrency)
            ]
            results = [h.wait() for h in handles]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        finally:
            scheduler.stop()

        total_output_tokens = sum(len(r.tokens) for r in results)
        throughput = total_output_tokens / elapsed
        per_request_latency = statistics.mean([elapsed] * concurrency)
        rows.append(
            {
                "concurrency": concurrency,
                "elapsed_s": round(elapsed, 3),
                "output_tokens": total_output_tokens,
                "throughput_tok_per_s": round(throughput, 2),
                "per_req_latency_s": round(per_request_latency, 3),
            }
        )

    header = f"{'C':>3} {'elapsed_s':>10} {'tokens':>8} {'tok/s':>10} {'lat_s':>8}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['concurrency']:>3} {row['elapsed_s']:>10} {row['output_tokens']:>8} "
            f"{row['throughput_tok_per_s']:>10} {row['per_req_latency_s']:>8}"
        )
    table = "\n".join(lines)

    return f"\nGPU: {gpu_name}\nModel: Qwen/Qwen2.5-0.5B-Instruct (bf16)\n\n{table}\n"


@app.local_entrypoint()
def main() -> None:
    print(bench.remote())
