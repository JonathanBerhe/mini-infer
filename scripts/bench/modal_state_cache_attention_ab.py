"""FlashInfer against the masked PyTorch attention of the state-cache models, on one GPU.

Qwen3.8 and Nemotron-H keep each request's attention history in a buffer of
`max_seq_len` positions. Their PyTorch path reads the whole buffer at every
decode step and masks the unwritten slots. The FlashInfer path
(`cache/contiguous_attention.py`) reads each row only up to its length. Both
paths run here on one model load: `contiguous._DISABLED_FOR_BENCH` switches
the FlashInfer path off.

Step 0 runs the unit tests of the path, the CUDA-graph decode tests and the
GPU tests of both model families, so that a broken kernel call fails before
a model loads. Then, for each model:

  1. Parity: prefill a 4096-token prompt of real prose with each path, and
     compare the last-position logits (cosine, top-1). Decode 16 greedy
     tokens with each path and count the equal ones.
  2. Speed: milliseconds per decode step from the CUDA graph, for a short
     request in a short buffer, a short request in a long buffer (a server
     sized for long prompts) and a long request. Then the prefill time of
     the long prompt with each path.

Each row prints as soon as it is measured, and a row that would not finish
before the timeout is skipped.

The prose is War and Peace in the `mini-infer-hf-cache` volume, from
`scripts/qwen/modal_qwen38_prefill_ab.py`. The weights come from the same
volume (`modal_qwen38_smoke.py::download`, `modal_nemotron_h_smoke.py::download`).

Run with:
    MINI_INFER_TEST_GPU='H100!' uv run modal run scripts/bench/modal_state_cache_attention_ab.py
Dry run on tiny random models, on this machine (both paths are PyTorch here):
    uv run python scripts/bench/modal_state_cache_attention_ab.py
"""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("mini-infer-state-cache-attention-ab")

# key -> (repo, revision, module, class)
MODELS = {
    "qwen3.8-27b": (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "mini_infer.models.qwen.qwen3_5",
        "Qwen35ForConditionalGeneration",
    ),
    "nemotron-h-30b": (
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
        "mini_infer.models.nemotron_h.nemotron_h",
        "NemotronHForCausalLM",
    ),
}
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja"]
PROSE_PATH = Path("/hf-cache/prose/war-and-peace.txt")

# (prompt tokens, buffer tokens, batch sizes). A 64k buffer at batch 4 does
# not fit next to the 27B on an 80 GB GPU, so that case runs at batch 1.
CASES = ((2048, 2048, (1, 4)), (2048, 65536, (1,)), (32768, 32768, (1, 4)))
PARITY_TOKENS = 4096
PARITY_DECODE = 16
PREFILL_CHUNK = 4096
WARMUP_STEPS = 3
TIMED_STEPS = 20
# Seconds a speed row may take, for the time guard. The 32k prefill of the
# 27B took 8.4 s on an H100 with the PyTorch path.
ROW_BUDGET_S = 60

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "H100!")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1800"))
# Time kept free at the end of the cap for the report to reach the client.
_END_MARGIN_S = 60

hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.14.1+cu132", extra_index_url="https://download.pytorch.org/whl/cu132")
    .pip_install(
        "triton==3.8.0",
        "transformers==5.14.0",
        "safetensors",
        "huggingface_hub[hf_xet]",
        "pytest",
    )
    .pip_install("flashinfer-python==0.6.10")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
    .add_local_dir("tests", remote_path="/root/tests")
)

GPU_TESTS = (
    "tests/unit/cache/test_contiguous_flashinfer.py",
    "tests/unit/engine/test_cuda_graph_decode.py",
    "tests/unit/models/qwen/test_models_qwen3_5.py",
    "tests/unit/models/nemotron_h/test_models_nemotron_h.py",
)


def _log(message: str) -> None:
    """Print and flush, so that the Modal log shows where a slow step is."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _set_flashinfer(on: bool) -> None:
    from mini_infer.cache import contiguous_attention

    contiguous_attention._DISABLED_FOR_BENCH = not on


def _placement(model: Any) -> tuple[Any, Any]:
    weight = model.lm_head.weight
    return weight.device, weight.dtype


def _sync(device: Any) -> None:
    import torch

    from mini_infer.device import is_cuda_device

    if is_cuda_device(device):
        torch.cuda.synchronize(device)


def _prefill(model: Any, ids: list[int], buffer: int) -> tuple[Any, Any, float]:
    """Chunked prefill into a one-row cache of `buffer` positions plus the decode steps.

    Returns the cache, the last position's logits and the seconds it took.
    """
    import torch

    device, dtype = _placement(model)
    cache = model.build_state_cache(
        max_seq_len=buffer + WARMUP_STEPS + TIMED_STEPS + PARITY_DECODE + 1,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    prompt = torch.tensor([ids], device=device)
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(ids), PREFILL_CHUNK):
            chunk = prompt[:, offset : offset + PREFILL_CHUNK]
            logits = model.forward_prefill_with_cache(chunk, state_cache=cache)
            cache.advance_start_pos(chunk.shape[1])
    _sync(device)
    return cache, logits[0, -1].float(), time.perf_counter() - start


def _greedy(model: Any, cache: Any, first: int, start: int, steps: int) -> list[int]:
    """Eager greedy decode of one row, continuing a prefilled cache."""
    import torch

    device, _ = _placement(model)
    tokens = [first]
    with torch.inference_mode():
        for step in range(steps):
            logits = model.forward_decode_with_cache_ragged(
                torch.tensor([[tokens[-1]]], device=device),
                positions=torch.tensor([start + step], device=device),
                state_cache=cache,
            )
            tokens.append(int(logits[0, -1].argmax()))
    return tokens[1:]


def _decode_ms(model: Any, prefilled: Any, first: int, start: int, batch: int) -> float:
    """Milliseconds per greedy decode step from the CUDA graph, every row a copy
    of the prefilled request. Eager on a CPU dry run."""
    import torch

    from mini_infer.device import is_cuda_device
    from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph

    device, dtype = _placement(model)
    cache = model.build_state_cache(
        max_seq_len=prefilled.max_seq_len, batch_size=batch, device=device, dtype=dtype
    )
    graph = is_cuda_device(device)
    # Capture first: warmup and capture write into every row of the cache.
    captured = DecodeCudaGraph(model, cache, batch_size=batch, device=device) if graph else None
    for row in range(batch):
        cache.copy_row_from(prefilled, src_row=0, dst_row=row)
    tokens = torch.full((batch, 1), first, dtype=torch.long)
    positions = torch.full((batch,), start, dtype=torch.long)

    def step() -> None:
        if captured is not None:
            logits = captured.replay(tokens, positions)
        else:
            with torch.inference_mode():
                logits = model.forward_decode_with_cache_ragged(
                    tokens.to(device), positions=positions.to(device), state_cache=cache
                )
        # One host sync per step for sampling, as in serving.
        tokens[:, 0] = torch.tensor(logits[:, -1].argmax(-1).tolist())
        positions.add_(1)

    for _ in range(WARMUP_STEPS):
        step()
    _sync(device)
    begin = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    _sync(device)
    return 1000 * (time.perf_counter() - begin) / TIMED_STEPS


def run_ab(model: Any, ids: list[int], *, cases: Any, deadline: float, emit: Any) -> None:
    """The parity row and the speed rows of one model."""
    from torch.nn import functional

    from mini_infer.device import is_cuda_device

    parity = ids[:PARITY_TOKENS]
    results = {}
    for on in (True, False):
        _set_flashinfer(on)
        cache, last, _ = _prefill(model, parity, len(parity))
        first = int(last.argmax())
        tokens = _greedy(model, cache, first, len(parity), PARITY_DECODE)
        results[on] = (last, tokens)
        del cache
    cosine = float(functional.cosine_similarity(results[True][0], results[False][0], dim=0))
    same_top1 = int(results[True][0].argmax()) == int(results[False][0].argmax())
    equal = sum(a == b for a, b in zip(results[True][1], results[False][1], strict=True))
    emit(
        f"parity, {len(parity)} tokens: logits cos {cosine:.5f}, top-1 same {same_top1}, "
        f"greedy {equal}/{PARITY_DECODE} equal"
    )

    mode = "graph" if is_cuda_device(_placement(model)[0]) else "eager"
    emit(f"prompt | buffer | batch | torch ms/step ({mode}) | flashinfer ms/step | speedup")
    for prompt_len, buffer, batches in cases:
        for batch in batches:
            if time.monotonic() + ROW_BUDGET_S > deadline:
                emit(f"{prompt_len} | {buffer} | {batch} | skipped: not enough time left")
                continue
            times = {}
            for on in (False, True):
                _set_flashinfer(on)
                prefilled, last, _ = _prefill(model, ids[:prompt_len], buffer)
                times[on] = _decode_ms(model, prefilled, int(last.argmax()), prompt_len, batch)
                del prefilled
            emit(
                f"{prompt_len} | {buffer} | {batch} | {times[False]:.2f} | "
                f"{times[True]:.2f} | {times[False] / times[True]:.2f}x"
            )
    longest = max(prompt_len for prompt_len, _, _ in cases)
    if time.monotonic() + 2 * ROW_BUDGET_S > deadline:
        emit(f"prefill {longest}: skipped: not enough time left")
    else:
        seconds = {}
        for on in (False, True):
            _set_flashinfer(on)
            prefilled, _, seconds[on] = _prefill(model, ids[:longest], longest)
            del prefilled
        emit(
            f"prefill {longest} tokens: torch {seconds[False]:.2f} s, "
            f"flashinfer {seconds[True]:.2f} s, {seconds[False] / seconds[True]:.2f}x"
        )
    _set_flashinfer(True)


@app.function(
    image=image,
    gpu=_TEST_GPU,
    memory=65536,
    timeout=_TEST_TIMEOUT,
    volumes={"/hf-cache": hf_cache},
)
def attention_ab() -> str:
    import gc
    import subprocess
    import sys

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from mini_infer.fast_paths import fast_path_report

    deadline = time.monotonic() + _TEST_TIMEOUT - _END_MARGIN_S
    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    lines = [fast_path_report("flashinfer")]
    print(lines[0], flush=True)

    def emit(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    _log("step 0: unit tests on the GPU")
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *GPU_TESTS,
            "-q",
            "-rfEs",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ],
        cwd="/root",
        capture_output=True,
        text=True,
    )
    emit(f"step 0: {tests.stdout.strip().splitlines()[-1]}")
    if tests.returncode != 0:
        # The failure section, not the tail: compiler warnings can fill the tail.
        out = tests.stdout
        start = out.find("= FAILURES =")
        end = out.find("short test summary")
        emit(out[start:end][:8000] if start >= 0 else out[-3000:])
        for line in out.splitlines():
            if line.startswith(("FAILED", "ERROR")):
                emit(line[:400])
        return "\n".join(lines)

    text = PROSE_PATH.read_text()
    text = text[text.find("*** START OF") :]
    for key, (repo, revision, module, cls) in MODELS.items():
        try:
            model_dir = snapshot_download(
                repo, revision=revision, allow_patterns=ALLOW_PATTERNS, local_files_only=True
            )
        except Exception:
            emit(f"{key}: skipped, the weights are not in the volume")
            continue
        ids = AutoTokenizer.from_pretrained(model_dir).encode(text)
        _log(f"load {key}")
        model_cls = getattr(importlib.import_module(module), cls)
        model = model_cls.from_checkpoint(model_dir, device="cuda", dtype=torch.bfloat16)
        emit(f"{key}@{revision[:7]}: {torch.cuda.memory_allocated() / 2**30:.1f} GiB of weights")
        run_ab(model, ids, cases=CASES, deadline=deadline, emit=emit)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    print(attention_ab.remote())


if __name__ == "__main__":
    import torch

    from mini_infer.models.qwen.qwen3_5 import (
        FULL_ATTENTION,
        Qwen35Config,
        Qwen35ForConditionalGeneration,
    )

    torch.manual_seed(0)
    tiny = Qwen35ForConditionalGeneration(
        Qwen35Config(
            vocab_size=1024,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=8,
            num_attention_heads=6,
            num_key_value_heads=2,
            head_dim=32,
            layer_types=(["linear_attention"] * 3 + [FULL_ATTENTION]) * 2,
            linear_num_key_heads=2,
            linear_num_value_heads=6,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
            tie_word_embeddings=False,
            max_position_embeddings=262144,
        )
    ).eval()
    PARITY_TOKENS = 64
    run_ab(
        tiny,
        [i % 1024 for i in range(600)],
        cases=((64, 64, (1, 2)), (64, 512, (1,)), (512, 512, (1,))),
        deadline=time.monotonic() + 600,
        emit=print,
    )
