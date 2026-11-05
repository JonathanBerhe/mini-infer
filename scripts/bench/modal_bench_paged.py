"""Modal benchmark: paged kernel vs materialization decode latency.

Run with: uv run modal run scripts/bench/modal_bench_paged.py
"""

import json

import modal

app = modal.App("mini-infer-paged-bench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1+cu132", extra_index_url="https://download.pytorch.org/whl/cu132")
    .pip_install(
        "triton==3.8.0",
        "transformers==5.14.0",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
    )
    .add_local_python_source("mini_infer", "tests")
)


@app.function(image=image, gpu="A10", timeout=1200)
def bench() -> str:
    from tests.benchmarks.bench_decode_attention import run_benchmark

    results = run_benchmark(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        seq_lens=(16, 64, 256, 1024),
        n_iters=50,
        warmup=5,
    )
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(bench.remote())
