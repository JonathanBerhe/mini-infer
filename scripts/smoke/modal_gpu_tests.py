"""Run the `gpu` and `requires_cuda` unit tests on a real CUDA GPU.

CI and a Mac skip both markers, so ~38 tests (Triton kernels, FlashInfer, FP8 / NVFP4 KV) never
run there. This script runs that set in a container with nvcc, flash-attn, and FlashInfer, and
fails if any test fails. Some tests skip on a small GPU (NVFP4 needs Blackwell, SM_100+), so a
green run means "nothing failed", not "every test ran".

Run with:
    uv run modal run scripts/smoke/modal_gpu_tests.py
    MINI_INFER_TEST_GPU=H100 uv run modal run scripts/smoke/modal_gpu_tests.py
"""

import os
import subprocess
from pathlib import Path

import modal

app = modal.App("mini-infer-gpu-tests")

# Modal 1.4 fixes `gpu` when the decorator runs, so the GPU comes from the environment.
# An A10 (SM_86) runs the Triton kernels, the flash-attn paths, and FP8 KV.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "A10")

# A cost cap, not an expected duration. The tests take about 2 min, plus a ~1 GB download and
# the FlashInfer kernel build, and a short cap costs a rerun. A hung container bills the whole
# cap, at several times the A10 rate on a large GPU (lower it with `MINI_INFER_TEST_TIMEOUT=900`).
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1800"))

# The torch and flash-attn pins of `modal_packed_bench.py` (`block_table` needs flash-attn 2.8+).
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

_REPO = Path(__file__).parent.parent.parent

image = (
    # A devel image, because FlashInfer compiles its kernels with nvcc at the first call.
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=5.14,<5.15",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "triton>=3.0",
        "pytest>=8.0",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install("flashinfer-python>=0.6.10rc1")
    # `tests/unit/conftest.py` imports other test modules, so the image gets the whole tree.
    # `pyproject.toml` supplies the marker definitions.
    .add_local_dir(str(_REPO / "tests"), "/root/tests")
    .add_local_file(str(_REPO / "pyproject.toml"), "/root/pyproject.toml")
    .add_local_python_source("mini_infer")
)


@app.function(image=image, gpu=_TEST_GPU, timeout=_TEST_TIMEOUT)
def run_gpu_tests() -> str:
    import torch

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    gpu_name = torch.cuda.get_device_name()

    proc = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "-m",
            "gpu or requires_cuda",
            "-v",
            "-rs",  # print the reason for each skip: on a small GPU, skips are a result
            "--no-header",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ],
        cwd="/root",
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.splitlines()
    summary = next(
        (
            line
            for line in reversed(lines)
            if " passed" in line or " failed" in line or " error" in line
        ),
        "no pytest summary line",
    )
    if proc.returncode != 0:
        # Keep all output from the FAILURES banner on: it shows how far off a numerics failure
        # is, and a truncated traceback costs a second GPU run.
        start = proc.stdout.find("= FAILURES =")
        detail = proc.stdout[start:] if start != -1 else proc.stdout
        # The counts and verdicts go first, so that the truncation keeps them. One JIT compile
        # error can print tens of thousands of characters.
        verdicts = "\n".join(
            line for line in lines if line.startswith(("FAILED", "ERROR", "SKIPPED"))
        )
        raise AssertionError(
            f"gpu tests failed on {gpu_name} (exit {proc.returncode})\n"
            f"{summary}\n{verdicts}\n\n{detail[:20000]}"
            f"\n--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
    # Print every skip in full, because a test that skipped did not pass.
    skips = "\n".join(line for line in lines if line.strip().startswith("SKIPPED"))
    return f"OK | gpu={gpu_name} | torch={torch.__version__} | {summary}\n{skips}"


@app.local_entrypoint()
def main() -> None:
    print(run_gpu_tests.remote())
