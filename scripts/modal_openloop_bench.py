"""Open-loop, rate-swept HTTP benchmark on Modal GPU.

Boots the mini-infer HTTP server inside the Modal container, then drives
it from the same container with scripts/http_openloop_bench.py over
localhost. One Modal function call gets you a full rate sweep.

GPU selection follows the same `MINI_INFER_BENCH_GPU` env var as
scripts/modal_packed_bench.py (default A10). Modal 1.4 fixes gpu at
decorator time.

Examples:

    # default A10, Qwen-0.5B sweep
    uv run modal run scripts/modal_openloop_bench.py

    # explicit args
    uv run modal run scripts/modal_openloop_bench.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --rates 1,2,4,8 --duration 20 --max-tokens 128

    # H100 with a 7B target
    MINI_INFER_BENCH_GPU=H100 uv run modal run scripts/modal_openloop_bench.py \
        --model Qwen/Qwen2.5-7B-Instruct --rates 1,2,4,8 --duration 30

Cost (per slice budget rule, default per-slice cap ~$2):
  A10 ~$1.10/hr, 0.5B Qwen, ~3 min wall time -> ~$0.06.
  H100 ~$4/hr,  7B Qwen,   ~4 min wall time -> ~$0.30.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("mini-infer-openloop-bench")

_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "A10")

# Same image base as scripts/modal_packed_bench.py (non-Blackwell path).
# Pinned torch 2.5.1 + flash-attn 2.8.3 wheel; FlashInfer 0.6.10 JITs prefill
# kernels on first call, which is why we use the CUDA dev image (nvcc).
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=4.40",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "triton>=3.0",
        # httpx for the open-loop driver. Already a transitive of fastapi
        # in most setups, but pin it explicitly so the script doesn't crash
        # if the image base drops it.
        "httpx>=0.27",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_file(
        str(Path(__file__).parent / "data" / "technical_passage.md"),
        "/root/scripts/data/technical_passage.md",
    )
    .add_local_file(
        str(Path(__file__).parent / "http_openloop_bench.py"),
        "/root/scripts/http_openloop_bench.py",
    )
    .add_local_python_source("mini_infer")
)


def _wait_for_server(url: str, model: str, timeout_s: float = 240.0) -> None:
    """Poll the server with a 1-token completion until it returns 200.

    Model load on A10 for Qwen-0.5B is ~30-60s; 7B on H100 is similar.
    We send a real (tiny) request because TCP-bind isn't enough; the
    engine has to be loaded before the first /v1/completions succeeds.
    """
    import httpx

    deadline = time.time() + timeout_s
    last_err = "(no attempt yet)"
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = httpx.post(
                f"{url}/v1/completions",
                json={
                    "model": model,
                    "prompt": "ready?",
                    "max_tokens": 1,
                    "stream": False,
                    "temperature": 0.0,
                },
                timeout=10.0,
            )
            if r.status_code == 200:
                print(f"server ready after {attempt} attempts", flush=True)
                return
            last_err = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5)
    raise TimeoutError(
        f"server at {url} not ready after {timeout_s}s ({attempt} attempts); last error: {last_err}"
    )


@app.function(image=image, gpu=_BENCH_GPU, timeout=1800)
def run_openloop(
    model: str,
    rates: str,
    duration: float,
    warmup: int,
    max_tokens: int,
    num_blocks: int,
    block_size: int,
) -> str:
    """Boot the server, then drive it with the open-loop bench. All in one container."""
    import torch

    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name()

    env = os.environ.copy()
    env["MINI_INFER_MODEL"] = model
    env["MINI_INFER_HOST"] = "127.0.0.1"
    env["MINI_INFER_PORT"] = "8000"
    # Size the KV block pool to fit the offered load. Default 1024 blocks of
    # 16 tokens each = 16K slots, which OOMs under a rate sweep that puts
    # dozens of long-prompt requests in flight at once. block_size trades
    # per-block granularity (less internal fragmentation at small sizes)
    # against FlashAttention's paged-varlen fast path (needs block_size % 256).
    env["MINI_INFER_NUM_BLOCKS"] = str(num_blocks)
    env["MINI_INFER_BLOCK_SIZE"] = str(block_size)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Server logs are best-effort; if the run hangs they help diagnose,
    # otherwise they just clutter the Modal log stream. Inherit stderr/stdout.
    server = subprocess.Popen(
        [sys.executable, "-m", "mini_infer.api.server"],
        env=env,
    )

    try:
        _wait_for_server("http://127.0.0.1:8000", model)

        bench_cmd = [
            sys.executable,
            "/root/scripts/http_openloop_bench.py",
            "--url",
            "http://127.0.0.1:8000",
            "--model",
            model,
            "--rates",
            rates,
            "--duration",
            str(duration),
            "--warmup",
            str(warmup),
            "--max-tokens",
            str(max_tokens),
        ]
        print("running:", " ".join(bench_cmd), flush=True)
        result = subprocess.run(bench_cmd, capture_output=True, text=True)
        body = result.stdout
        if result.returncode != 0:
            body += f"\n\n!! bench exited {result.returncode}\nSTDERR:\n{result.stderr}\n"
    finally:
        server.terminate()
        try:
            server.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5.0)

    header = f"GPU: {gpu_name} | model: {model} | rates: {rates} | duration/rate: {duration}s\n"
    return header + "\n" + body


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    rates: str = "1,2,4",
    duration: float = 15.0,
    warmup: int = 3,
    max_tokens: int = 128,
    num_blocks: int = 4096,
    block_size: int = 16,
) -> None:
    print(
        run_openloop.remote(
            model=model,
            rates=rates,
            duration=duration,
            warmup=warmup,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
        )
    )
