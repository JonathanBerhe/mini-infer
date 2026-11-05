"""Paged decode speed of Qwen2.5-7B on one GPU, in a CUDA 13.2 image.

The image has torch 2.14.1+cu132, Triton 3.8.0 and FlashInfer 0.6.10 on a
CUDA 13.2 devel base. Modal runs a B300 only with CUDA 13.1 or newer, and the
same image runs on a B200, so two runs give a comparison in one software
stack.

Two configurations run one after the other in one container, with the
FlashInfer backend, the decode CUDA graph and the prefill CUDA graph:

- bf16 weights
- int8 weights (the W8A16 split-K decode kernel)

The workload is the one of `docs/benchmarks/2026-11-02-paged-decode-chain.md`:
the eight bench prompts, greedy decode, 32 new tokens, through
`ContinuousScheduler` on `PagedKVCache`. For each configuration one warm-up
sweep per concurrency level compiles the kernels and captures the graphs.
Then three timed sweeps per level give the median tokens per second. Last,
one profiled sweep at concurrency 8 gives the GPU time per decode step.

Run:
    uv run modal run scripts/bench/modal_paged_graph_bench.py
    MINI_INFER_TEST_GPU=B200 uv run modal run scripts/bench/modal_paged_graph_bench.py
"""

from __future__ import annotations

import os
import statistics
import time

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
# The volume of `scripts/bench/modal_bench_all.py`, which holds Qwen2.5-7B.
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_GPU = os.environ.get("MINI_INFER_TEST_GPU", "B300")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TIMEOUT_S = 1200
_MODEL_NAME = "Qwen/Qwen2.5-7B"
_LEVELS = (8, 1, 32)
_MAX_TOKENS = 32
_REPEATS = 3

app = modal.App("mini-infer-paged-graph-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.14.1+cu132", extra_index_url="https://download.pytorch.org/whl/cu132")
    .pip_install(
        "triton==3.8.0",
        "transformers==5.14.0",
        "safetensors",
        "huggingface_hub[hf_xet]",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "pytest",
    )
    .pip_install("flashinfer-python==0.6.10")
    .add_local_python_source("mini_infer")
)


def _log(line: str = "") -> None:
    print(line, flush=True)


def _measure(name: str, quant: str | None) -> dict[str, object]:
    """Timed sweeps and one profiled sweep of one configuration."""
    import torch

    from mini_infer.bench import DEFAULT_PROMPTS
    from mini_infer.bench.harness import drive_scheduler, profile_scheduler_steps
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler

    began = time.monotonic()
    runner = ModelRunner.from_pretrained(
        _MODEL_NAME,
        device="cuda",
        attention_backend="flashinfer",
        quant=quant,
        cuda_graph=True,
        prefill_graph=True,
    )
    _log(f"{name}: loaded and captured in {time.monotonic() - began:.0f} s")

    def prompts(level: int) -> list[str]:
        return [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(level)]

    rates: dict[int, float] = {}
    signatures: dict[int, tuple[tuple[int, ...], ...]] = {}
    for level in _LEVELS:
        drive_scheduler(ContinuousScheduler(runner), prompts(level), _MAX_TOKENS)
        samples = []
        for _ in range(_REPEATS):
            seconds, tokens, _latencies, signature = drive_scheduler(
                ContinuousScheduler(runner), prompts(level), _MAX_TOKENS
            )
            samples.append(tokens / seconds)
            signatures[level] = signature
        rates[level] = statistics.median(samples)
        spread = (max(samples) - min(samples)) / rates[level]
        _log(
            f"{name}: concurrency {level}: {rates[level]:.0f} tok/s "
            f"(median of {_REPEATS}, spread {100 * spread:.1f}%)"
        )

    summary = profile_scheduler_steps(
        runner, ContinuousScheduler(runner), prompts(8), _MAX_TOKENS, top=8
    )
    decode = [row for row in summary["steps"] if row.get("phase") == "decode"]
    gpu_ms = statistics.median(row["gpu_busy_ms"] for row in decode) if decode else float("nan")
    wall_ms = statistics.median(row["wall_ms"] for row in decode) if decode else float("nan")
    kernels = statistics.median(row["kernels"] for row in decode) if decode else float("nan")
    _log(
        f"{name}: profiled decode step at concurrency 8: GPU busy {gpu_ms:.2f} ms, "
        f"wall {wall_ms:.2f} ms (profiler on), {kernels:.0f} kernels, {len(decode)} steps"
    )
    del runner
    torch.cuda.empty_cache()
    return {"rates": rates, "signatures": signatures, "gpu_ms": gpu_ms}


@app.function(
    image=image,
    gpu=_GPU,
    timeout=_TIMEOUT_S,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def bench() -> str:
    import torch
    import triton

    began = time.monotonic()
    header = (
        f"gpu={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
        f"SMs={torch.cuda.get_device_properties(0).multi_processor_count} "
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}"
    )
    _log(header)
    results = {name: _measure(name, quant) for name, quant in (("bf16", None), ("int8", "int8"))}

    lines = [
        header,
        f"Qwen2.5-7B, FlashInfer, decode and prefill graphs, {_MAX_TOKENS} new tokens, "
        f"median of {_REPEATS} sweeps",
        "",
        "| Weights | " + " | ".join(f"tok/s at {level}" for level in _LEVELS) + " | GPU ms per "
        "decode step at 8 |",
        "|---|" + "---:|" * (len(_LEVELS) + 1),
    ]
    for name, result in results.items():
        rates = result["rates"]
        assert isinstance(rates, dict)
        cells = " | ".join(f"{rates[level]:.0f}" for level in _LEVELS)
        lines.append(f"| {name} | {cells} | {result['gpu_ms']:.2f} |")
    bf16_sig = results["bf16"]["signatures"]
    int8_sig = results["int8"]["signatures"]
    assert isinstance(bf16_sig, dict) and isinstance(int8_sig, dict)
    for level in _LEVELS:
        same = sum(a == b for a, b in zip(bf16_sig[level], int8_sig[level], strict=True))
        lines.append(
            f"int8 against bf16 at concurrency {level}: {same} of {level} requests decode "
            "the same tokens (int8 is lossy)"
        )
    lines.append(f"total in the container: {time.monotonic() - began:.0f} s")
    report = "\n".join(lines)
    _log(report)
    return report


@app.local_entrypoint()
def main() -> None:
    print("\n==== report ====")
    print(bench.remote())
