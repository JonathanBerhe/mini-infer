"""Long-context throughput benchmark for the packed-varlen forward.

Realistic workload: 3000-token prompts (typical RAG / long-chat scale),
max_tokens=64 (typical short response). Sweeps:
  - concurrency C ∈ {1, 2, 4} (the number of simultaneously-in-flight requests)
  - chunk_size: chunked-256 (default) vs un-chunked (chunk_size=8192)

The chunked variant is the head-of-line-blocking-friendly one: a 3k-token
prefill is split into 12 chunks of 256, each step interleaving with any
in-flight decoders. The un-chunked variant processes the full prompt in one
forward, blocking decoders for the duration.

Block pool sized for C=4 x 3000+64 tokens with headroom.
"""

# Run with: uv run modal run scripts/modal_packed_bench_long.py

import statistics
import time

import modal

app = modal.App("mini-infer-packed-bench-long")

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


# Realistic 3000-token prompt: a synthetic "RAG context" + question.
# Built from repeated paragraphs so the tokenizer produces ~3k tokens deterministically.
def _build_long_prompt(target_tokens: int = 3000) -> str:
    paragraph = (
        "The mini-infer engine is an open-source LLM inference server that "
        "demonstrates production-grade serving techniques. It implements "
        "continuous batching, paged attention with a paged KV cache, chunked "
        "prefill, and a packed varlen attention forward via FlashAttention. "
        "These techniques are the foundation of modern inference engines like "
        "vLLM and SGLang. The engine is structured to be readable, modular, "
        "and testable: a single engine thread owns the model and the running "
        "batch, while API threads only enqueue requests and drain output queues. "
    )
    # ~80 tokens per paragraph after BPE; 38 paragraphs ≈ 3040 tokens.
    repeats = max(1, target_tokens // 80)
    body = paragraph * repeats
    return (
        body + "\n\nQuestion: Summarize the techniques described above in one sentence.\n\nAnswer:"
    )


@app.function(image=image, gpu="A10", timeout=1800)
def bench() -> str:
    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name()

    # 4096 blocks * 16 tokens/block = 65k token capacity. C=4 * 3064 tokens = 12256 tokens used.
    # Comfortable headroom for chunk allocation churn.
    runner = ModelRunner.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        num_blocks=4096,
        block_size=16,
    )

    long_prompt = _build_long_prompt(target_tokens=3000)
    tokenizer = runner.tokenizer
    actual_prompt_len = len(tokenizer.encode(long_prompt))
    max_tokens = 64
    concurrencies = [1, 2, 4]
    chunk_settings: list[tuple[str, int]] = [("chunked-256", 256), ("unchunked", 8192)]

    rows: list[dict[str, float | int | str]] = []
    for label, chunk_size in chunk_settings:
        for concurrency in concurrencies:
            scheduler = ContinuousScheduler(
                runner, max_concurrent=concurrency, chunk_size=chunk_size
            )
            scheduler.start()
            try:
                # Warmup: one full request to compile kernels + warm caches.
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
                            prompt=long_prompt,
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

    header = f"{'config':<14} {'C':>3} {'elapsed_s':>10} {'tokens':>8} {'tok/s':>10} {'lat_s':>8}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['config']:<14} {row['concurrency']:>3} {row['elapsed_s']:>10} "
            f"{row['output_tokens']:>8} {row['throughput_tok_per_s']:>10} "
            f"{row['per_req_latency_s']:>8}"
        )
    table = "\n".join(lines)

    workload_note = (
        f"Workload: prompt_len≈{actual_prompt_len} tokens (RAG / long-chat scale), "
        f"max_tokens={max_tokens}\n"
        f"GPU: {gpu_name}\nModel: Qwen/Qwen2.5-0.5B-Instruct (bf16, FlashAttention varlen)"
    )
    return f"\n{workload_note}\n\n{table}\n"


@app.local_entrypoint()
def main() -> None:
    print(bench.remote())
