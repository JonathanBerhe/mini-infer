"""Run the `gpu` / `requires_cuda` unit tests on a real CUDA GPU.

`tests/conftest.py` skips both markers when CUDA is absent, so ~38 tests never
execute on a Mac dev box or in CI: the Triton kernels (paged attention, INT8
W8A16, TurboQuant dequant, MSA decode, hc_sinkhorn), the FlashInfer backend, and
the FP8 / NVFP4 KV paths. This runs exactly that deselected set in a container
that has nvcc, flash-attn and FlashInfer, and fails loudly if any of them fail.

Capability-gated tests (FP8 / NVFP4 need Hopper or newer, the HC kernel needs a
power-of-2 multiplier) skip themselves through their `supports_*` predicates, so
a green run on a small GPU means "nothing broken here", not "everything ran".
The summary line reports passed / skipped so the difference is visible.

Run with:
    uv run modal run scripts/modal_gpu_tests.py
    MINI_INFER_TEST_GPU=H100 uv run modal run scripts/modal_gpu_tests.py
"""

import os
import subprocess
from pathlib import Path

import modal

app = modal.App("mini-infer-gpu-tests")

# Modal 1.4 fixes `gpu` at decorator evaluation time, so it comes from the
# environment rather than a CLI flag. A10 (SM_86) runs the Triton kernels and
# flash-attn paths; FP8 / NVFP4 self-skip below Hopper.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "A10")

# Worst-case spend bound, not an expected duration: a hung container bills for
# the whole timeout, and the per-second rate on the big GPUs is several times
# the A10's. Override downward when running somewhere expensive
# (`MINI_INFER_TEST_TIMEOUT=900`).
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1800"))

# Same pinned torch + flash-attn pair as `modal_packed_bench.py`; flash-attn
# 2.8+ is what has `flash_attn_varlen_func`'s `block_table` parameter.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

_REPO = Path(__file__).parent.parent

image = (
    # CUDA dev image (has nvcc) so FlashInfer's JIT-compiled kernels can build
    # on first call.
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
    # The tests are a package (`tests/unit/conftest.py` imports
    # `tests.unit._kimi_reference_helpers`), so the tree goes in whole, and
    # pyproject.toml comes along for the marker definitions.
    .add_local_dir(str(_REPO / "tests"), "/root/tests")
    .add_local_file(str(_REPO / "pyproject.toml"), "/root/pyproject.toml")
    .add_local_python_source("mini_infer")
)


# Model-loading tests download ~1 GB and FlashInfer JIT-compiles on first call,
# so the budget is generous relative to the ~2 min of actual test time. A
# too-tight timeout costs a whole second run.
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
            "-rs",  # report why each test skipped: on a small GPU that IS the result
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
        # Everything from the FAILURES banner on: assertion messages carry the
        # numbers a numerics failure has to be judged on, and a GPU minute is
        # too expensive to spend twice because the traceback was truncated.
        start = proc.stdout.find("= FAILURES =")
        detail = proc.stdout[start:] if start != -1 else proc.stdout
        # Counts and the per-test verdict list go FIRST. A single JIT compile
        # error can dump tens of thousands of characters of C++ template
        # diagnostics, and when that lands ahead of the summary the truncation
        # eats the one line that says what else passed or failed.
        verdicts = "\n".join(
            line for line in lines if line.startswith(("FAILED", "ERROR", "SKIPPED"))
        )
        raise AssertionError(
            f"gpu tests failed on {gpu_name} (exit {proc.returncode})\n"
            f"{summary}\n{verdicts}\n\n{detail[:20000]}"
            f"\n--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
    # Skips are printed in full: a capability-gated test that skipped is NOT a
    # test that passed, and on a pre-Hopper GPU several of these do skip.
    skips = "\n".join(line for line in lines if line.strip().startswith("SKIPPED"))
    return f"OK | gpu={gpu_name} | torch={torch.__version__} | {summary}\n{skips}"


@app.local_entrypoint()
def main() -> None:
    print(run_gpu_tests.remote())
