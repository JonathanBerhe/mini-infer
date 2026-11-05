"""First check of mini-infer on a new GPU: versions, the GPU unit tests, one bench.

Written for the B300 (Blackwell Ultra, compute capability 10.3). Modal runs
a B300 only with CUDA 13.1 or newer in the image, so this image uses the
CUDA 13.2 build of torch 2.14.1. torch 2.14 pins Triton 3.8, the version
that the Gluon MoE kernels are tested with. FlashInfer compiles its kernels
with the image's `nvcc` at first use, so the base is a CUDA 13.2 devel image.

The function runs three steps and prints each result when it is done, so a
timeout keeps the results that are finished:

1. The versions, the GPU, one bf16 matmul against an fp32 reference, and
   one device-to-device copy.
2. The GPU unit tests, one pytest call per file. The FlashInfer tests run
   last, because their first use compiles kernels.
3. The uniform bench harness on Qwen2.5-7B: bf16 and int8 weights at
   concurrency 1 and 8, with the torch attention backend.

Run:
    uv run modal run scripts/smoke/modal_b300_check.py
    # the same image on another GPU, for a comparison in one software stack:
    MINI_INFER_TEST_GPU=B200 uv run modal run scripts/smoke/modal_b300_check.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
# The volume of `scripts/bench/modal_bench_all.py`, which holds Qwen2.5-7B.
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_GPU = os.environ.get("MINI_INFER_TEST_GPU", "B300")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TIMEOUT_S = 1500
# The tests stop starting new files after this many seconds of the run.
_TESTS_DEADLINE_S = 900
# One test file that runs longer than this is stopped and reported as a timeout.
_TEST_FILE_TIMEOUT_S = 300
# The bench starts only if this many seconds of the cap are left.
_BENCH_MIN_LEFT_S = 420
_PER_TECHNIQUE_TIMEOUT_S = 300.0
_MODEL_NAME = "Qwen/Qwen2.5-7B"

_REPO = Path(__file__).resolve().parent.parent.parent

_KERNEL_TESTS = [
    "tests/unit/models/blocks/moe/test_fp8_grouped_kernel.py",
    "tests/unit/models/blocks/moe/test_fp8_grouped_gluon.py",
    "tests/unit/models/blocks/moe/test_grouped_experts.py",
    "tests/unit/models/blocks/moe/test_fp4_grouped_kernel.py",
    "tests/unit/models/blocks/moe/test_fp4_grouped_gluon.py",
    "tests/unit/models/deepseek_v4/test_fp4_grouped.py",
    "tests/unit/models/blocks/test_partial_rope_kernel.py",
    "tests/unit/models/blocks/test_rms_norm_kernel.py",
    "tests/unit/models/blocks/test_rope_write_kernel.py",
    "tests/unit/models/blocks/test_silu_mul_kernel.py",
    "tests/unit/models/deepseek_v4/test_hc_sinkhorn_kernel.py",
    "tests/unit/models/minimax_m3/test_msa_kernel_gpu.py",
    "tests/unit/models/qwen/test_gdn_decode_kernel.py",
    "tests/unit/quant/test_int8_decode_kernel.py",
    "tests/unit/quant/test_int8_kernel.py",
    "tests/unit/cache/test_paged_attention.py",
    "tests/unit/cache/test_turbo_kernel.py",
]
_ENGINE_TESTS = [
    "tests/unit/engine/test_cuda_graph_decode.py",
    "tests/unit/models/deepseek_v4/test_v4_cuda_graph_decode.py",
    "tests/unit/engine/test_sampler.py",
    "tests/unit/engine/test_decode_step_timer.py",
    "tests/unit/engine/test_paged_decode_graph.py",
    "tests/unit/engine/test_paged_prefill_graph.py",
]
_FLASHINFER_TESTS = [
    "tests/unit/cache/test_flashinfer_backend.py",
    "tests/unit/cache/test_fp8_kv.py",
    "tests/unit/cache/test_nvfp4_kv.py",
]

app = modal.App("mini-infer-b300-check")

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
    .add_local_dir(str(_REPO / "tests"), remote_path="/root/tests")
    .add_local_file(str(_REPO / "pyproject.toml"), remote_path="/root/pyproject.toml")
)


def _log(line: str = "") -> None:
    print(line, flush=True)


def _shell(command: list[str]) -> str:
    import subprocess

    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"failed: {exc}"
    return (done.stdout or done.stderr).strip()


def _environment() -> list[str]:
    """Versions, the GPU, one bf16 matmul and one copy."""
    import torch
    import triton

    from mini_infer.models.blocks.moe import fp8_grouped_gluon

    try:
        import flashinfer

        flashinfer_version = flashinfer.__version__
    except Exception as exc:  # any import failure is a result here
        flashinfer_version = f"import failed: {exc}"

    props = torch.cuda.get_device_properties(0)
    smi = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    lines = [
        f"gpu: {torch.cuda.get_device_name()}",
        f"nvidia-smi: {_shell(smi)}",
        f"capability: {torch.cuda.get_device_capability()}",
        f"SMs: {props.multi_processor_count}, memory: {props.total_memory / 2**30:.1f} GiB",
        f"torch: {torch.__version__}, CUDA {torch.version.cuda}, "
        f"cuDNN {torch.backends.cudnn.version()}",
        f"torch arch list: {torch.cuda.get_arch_list()}",
        f"triton: {triton.__version__}, "
        f"Gluon kernels available: {fp8_grouped_gluon._GLUON_AVAILABLE}",
        f"flashinfer: {flashinfer_version}",
        f"nvcc: {_shell(['nvcc', '--version']).splitlines()[-1]}",
    ]

    torch.manual_seed(0)
    size = 8192
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    reference = a.float() @ b.float()
    out = a @ b
    rel = ((out.float() - reference).norm() / reference.norm()).item()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        a @ b
    start.record()
    for _ in range(20):
        a @ b
    end.record()
    torch.cuda.synchronize()
    tflops = 20 * 2 * size**3 / (start.elapsed_time(end) / 1e3) / 1e12
    lines.append(f"bf16 matmul {size}^3: {tflops:.0f} TFLOPS, relative error vs fp32 {rel:.2e}")

    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    dst.copy_(src)
    start.record()
    for _ in range(10):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    # A copy reads and writes every byte.
    gbps = 10 * 2 * src.numel() / (start.elapsed_time(end) / 1e3) / 1e9
    lines.append(f"device copy 1 GiB: {gbps:.0f} GB/s (read + write)")
    del a, b, reference, out, src, dst
    torch.cuda.empty_cache()
    return lines


def _run_test_file(path: str) -> tuple[str, bool]:
    """One pytest call. Returns its summary lines and whether it passed."""
    import subprocess
    import sys

    began = time.monotonic()
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-q", "-rs", "-p", "no:cacheprovider"],
            cwd="/root",
            capture_output=True,
            text=True,
            timeout=_TEST_FILE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        tail = "\n".join(output.splitlines()[-20:])
        return f"TIMEOUT {path} after {_TEST_FILE_TIMEOUT_S} s\n{tail}", False
    seconds = time.monotonic() - began
    lines = done.stdout.strip().splitlines()
    summary = lines[-1] if lines else "(no output)"
    skips = [line for line in lines if line.startswith("SKIPPED")]
    status = "PASS" if done.returncode == 0 else f"FAIL (exit {done.returncode})"
    text = f"{status} {path} in {seconds:.0f} s: {summary}"
    if skips:
        text += "\n" + "\n".join(f"    {line}" for line in skips[:8])
    if done.returncode != 0:
        failures = [line for line in lines if line.startswith(("FAILED", "ERROR", "E "))]
        text += "\n" + "\n".join(f"    {line}" for line in (failures or lines)[-30:])
    return text, done.returncode == 0


def _bench(left_s: float) -> list[str]:
    from mini_infer.bench import (
        DEFAULT_PROMPTS,
        Workload,
        build_registry,
        format_table,
        parity_violations,
        run_suite,
    )

    if left_s < _BENCH_MIN_LEFT_S:
        return [f"bench skipped: {left_s:.0f} s left of the cap"]
    workload = Workload(
        model=_MODEL_NAME,
        prompts=DEFAULT_PROMPTS,
        concurrency_levels=[1, 8],
        max_tokens=32,
        device="cuda",
        attention_backend="torch",
    )
    registry = [t for t in build_registry() if t.name in {"baseline", "int8_w8a16"}]
    results, skipped = run_suite(workload, registry, per_technique_timeout=_PER_TECHNIQUE_TIMEOUT_S)
    lines = [
        f"Workload: model={workload.model} concurrency={workload.concurrency_levels} "
        f"max_tokens={workload.max_tokens} attention=torch",
        format_table(results, skipped),
    ]
    violations = parity_violations(results)
    if violations:
        lines.append("PARITY VIOLATIONS (lossless techniques decoded different tokens):")
        lines.extend(f"  - {ref} != {other}" for ref, other in violations)
    else:
        lines.append("Parity: all lossless techniques agree (lossy techniques excluded).")
    return lines


@app.function(
    image=image,
    gpu=_GPU,
    timeout=_TIMEOUT_S,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def check(skip_bench: bool = False) -> str:
    import torch

    began = time.monotonic()
    assert torch.cuda.is_available(), "CUDA not available in the container"
    report: list[str] = []

    def emit(text: str) -> None:
        _log(text)
        report.append(text)

    emit(f"== step 1: environment ({_GPU})")
    for line in _environment():
        emit(line)

    emit("")
    emit("== step 2: GPU unit tests")
    failed: list[str] = []
    for path in _KERNEL_TESTS + _ENGINE_TESTS + _FLASHINFER_TESTS:
        if time.monotonic() - began > _TESTS_DEADLINE_S:
            emit(f"NOT RUN {path}: the test deadline of {_TESTS_DEADLINE_S} s passed")
            failed.append(path)
            continue
        text, ok = _run_test_file(path)
        emit(text)
        if not ok:
            failed.append(path)
    emit(f"tests: {len(failed)} file(s) not passed: {failed}" if failed else "tests: all passed")

    emit("")
    emit("== step 3: bench")
    if skip_bench:
        emit("bench skipped by flag")
    else:
        for line in _bench(_TIMEOUT_S - (time.monotonic() - began)):
            emit(line)
    emit(f"total in the container: {time.monotonic() - began:.0f} s")
    return "\n".join(report)


@app.local_entrypoint()
def main(skip_bench: bool = False) -> None:
    report = check.remote(skip_bench)
    print("\n==== report ====")
    print(report)
