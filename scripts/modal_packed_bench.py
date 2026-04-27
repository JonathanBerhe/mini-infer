"""Throughput benchmark for the packed-varlen forward on CUDA.

Mixed-length workload (50% short ~5 tokens, 50% long ~80 tokens) at concurrency
C ∈ {1, 4, 8}. Compares two chunk-size settings:
  - chunked (chunk_size=32): long prompts get chunked; chunks interleave with decodes
  - un-chunked (chunk_size=4096): long prompts processed in one shot per step

The chunked configuration is the head-of-line-blocking-friendly one: while a
long prompt's prefill is in flight, decoders make progress. The un-chunked
configuration shows what we lose when that happens.
"""

# Run with: uv run modal run scripts/modal_packed_bench.py

import statistics
import time

import modal

app = modal.App("mini-infer-packed-bench")

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=4.40",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "triton>=3.0",
    )
    .pip_install(FLASH_ATTN_WHEEL)
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

    short_prompt = "The capital of France is"  # ~5 tokens
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 8  # ~80 tokens
    max_tokens = 32
    concurrencies = [1, 4, 8]
    chunk_settings: list[tuple[str, int]] = [("chunked-32", 32), ("unchunked", 4096)]

    rows: list[dict[str, float | int | str]] = []
    for label, chunk_size in chunk_settings:
        for concurrency in concurrencies:
            # Mixed workload: alternate short/long up to the requested concurrency.
            workload = [
                short_prompt if i % 2 == 0 else long_prompt for i in range(concurrency)
            ]

            scheduler = ContinuousScheduler(
                runner, max_concurrent=concurrency, chunk_size=chunk_size
            )
            scheduler.start()
            try:
                # Warmup with one full request so kernel caches are warm.
                scheduler.run(
                    Request(
                        prompt=long_prompt,
                        sampling_params=SamplingParams(),
                        max_tokens=max_tokens,
                    )
                )

                torch.cuda.synchronize()
                start = time.perf_counter()
                handles = [
                    scheduler.submit(
                        Request(
                            prompt=p,
                            sampling_params=SamplingParams(),
                            max_tokens=max_tokens,
                        )
                    )
                    for p in workload
                ]
                results = [h.wait() for h in handles]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                scheduler.stop()

            total_output = sum(len(r.tokens) for r in results)
            throughput = total_output / elapsed
            rows.append(
                {
                    "config": label,
                    "concurrency": concurrency,
                    "elapsed_s": round(elapsed, 3),
                    "output_tokens": total_output,
                    "throughput_tok_per_s": round(throughput, 2),
                    "per_req_latency_s": round(statistics.fmean([elapsed] * concurrency), 3),
                }
            )

    header = (
        f"{'config':<14} {'C':>3} {'elapsed_s':>10} "
        f"{'tokens':>8} {'tok/s':>10} {'lat_s':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['config']:<14} {row['concurrency']:>3} {row['elapsed_s']:>10} "
            f"{row['output_tokens']:>8} {row['throughput_tok_per_s']:>10} "
            f"{row['per_req_latency_s']:>8}"
        )
    table = "\n".join(lines)

    return f"\nGPU: {gpu_name}\nModel: Qwen/Qwen2.5-0.5B-Instruct (bf16, FlashAttention varlen)\n\n{table}\n"


@app.local_entrypoint()
def main() -> None:
    print(bench.remote())
