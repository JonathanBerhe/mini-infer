"""TTFT and tokens per second of each model that fits one node, on an H100 and a B200.

One container runs one model on one GPU, or on two B200s for DeepSeek-V4-Flash.
It loads the model on the fastest path that the code has for it: FlashInfer
attention and the decode and prefill CUDA graphs for a paged model, where the
model takes them, and the decode CUDA graph for a model with its own state
cache. Then it runs one workload through the scheduler that the API server
uses:

  - Prompts: 1,024-token windows of `scripts/data/technical_passage.md`, each
    at another offset, so that no two prompts share a prefix and the prefix
    cache never hits.
  - Output: up to 128 tokens, greedy, bf16 weights. A request can stop
    earlier at an end-of-text token, so the result records the mean output
    length.
  - TTFT: the median over 5 requests, sent one at a time, from the submit to
    the first token.
  - Decode speed at 1 request: the median over the same 5 requests of
    (output tokens - 1) / (time of the last token - time of the first token).
  - Throughput at 16 requests: 16 requests submitted together, all output
    tokens divided by the time from the first submit to the last token.

Warm-up requests run first, so the timed requests include no kernel compile.
Each container prints one `RESULT` line per finished case, so a timeout keeps
the cases that ran.

Run from the repository root, with the token of a Hugging Face account that
can read the gated checkpoints:

    HF_TOKEN=$(cat ~/.cache/huggingface/token) uv run modal run scripts/bench/modal_model_metrics.py
    ... --models qwen2.5-7b,qwen3-8b --gpus h100

`MINI_INFER_SRC` mounts another copy of the `mini_infer` package (for example
an export of `main`) instead of the one in this checkout.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

# Inside the container the script sits in /root, which has no repository above it.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
_SRC = os.environ.get("MINI_INFER_SRC")
_PASSAGE = os.environ.get(
    "MINI_INFER_PASSAGE", str(_REPO / "scripts" / "data" / "technical_passage.md")
)
_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

PROMPT_TOKENS = 1024
OUTPUT_TOKENS = 128
SINGLE_REQUESTS = 5
CONCURRENCY = 16
MAX_SEQ_LEN = 2048


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    # The GPUs whose memory holds the bf16 weights and the KV cache of the workload.
    gpus: tuple[str, ...] = ("h100", "b200")
    # 2: the model runs on two GPUs through the tensor-parallel state-cache server.
    tensor_parallel: int = 1


MODELS: dict[str, ModelSpec] = {
    "qwen2.5-7b": ModelSpec("Qwen/Qwen2.5-7B"),
    "qwen3-8b": ModelSpec("Qwen/Qwen3-8B"),
    "llama3.1-8b": ModelSpec("unsloth/Meta-Llama-3.1-8B-Instruct"),
    "mistral-7b": ModelSpec("mistralai/Mistral-7B-Instruct-v0.3"),
    "gemma3-1b": ModelSpec("unsloth/gemma-3-1b-it"),
    "gemma4-31b": ModelSpec("google/gemma-4-31B-it"),
    "deepseek-v2-lite": ModelSpec("deepseek-ai/DeepSeek-V2-Lite-Chat"),
    "qwen3.8-27b": ModelSpec("Qwen/Qwen3.8-27B"),
    "nemotron-h-30b": ModelSpec("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"),
    # 93 GB and 98 GB of bf16 weights: one 80 GB H100 is too small.
    "mixtral-8x7b": ModelSpec("mistralai/Mixtral-8x7B-Instruct-v0.1", gpus=("b200",)),
    "kimi-linear-48b": ModelSpec("moonshotai/Kimi-Linear-48B-A3B-Instruct", gpus=("b200",)),
    # FP4 experts in about 150 GB: two B200s, sharded.
    "deepseek-v4-flash": ModelSpec(
        "deepseek-ai/DeepSeek-V4-Flash", gpus=("b200",), tensor_parallel=2
    ),
}

app = modal.App("mini-infer-model-metrics")

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.14.1+cu132", extra_index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("triton==3.8.0", "transformers==5.14.0", "safetensors>=0.4")
    .pip_install("flashinfer-python==0.6.10")
    .add_local_file(_PASSAGE, "/root/technical_passage.md")
)
image = (
    image.add_local_dir(_SRC, "/root/mini_infer", ignore=["**/__pycache__/**"])
    if _SRC
    else image.add_local_python_source("mini_infer")
)

Emit = Callable[..., None]


def _emitter(key: str, gpu: str, results: list[dict[str, Any]]) -> Emit:
    """Print one `RESULT` line per case and keep it for the return value."""
    import torch

    device = torch.cuda.get_device_name()

    def emit(kind: str, **fields: object) -> None:
        row = {"kind": kind, "model": key, "gpu": gpu, "device": device, **fields}
        results.append(row)
        print("RESULT " + json.dumps(row), flush=True)

    return emit


def _workload(scheduler: Any, tokenizer: Any, emit: Emit) -> None:
    """Warm up, then time the single requests and the batch of 16 (see the module docstring)."""
    import statistics
    import threading
    import time

    import torch

    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import Request

    passage = tokenizer.encode(Path("/root/technical_passage.md").read_text())
    doubled = passage + passage

    def prompt(index: int) -> str:
        start = (index * 131) % len(passage)
        return tokenizer.decode(doubled[start : start + PROMPT_TOKENS])

    def run(index: int, max_tokens: int, out: list[dict[str, float]]) -> None:
        request = Request(
            prompt=prompt(index), sampling_params=SamplingParams(), max_tokens=max_tokens
        )
        submit = time.perf_counter()
        handle = scheduler.submit(request)
        first = None
        tokens = 0
        while True:
            step = handle.get_step(timeout=600)
            if step.failed:
                raise RuntimeError(f"request {index} failed")
            if step.num_tokens and first is None:
                first = time.perf_counter()
            tokens += step.num_tokens
            if step.finish_reason is not None:
                break
        end = time.perf_counter()
        assert first is not None
        out.append({"submit": submit, "first": first, "end": end, "tokens": tokens})

    def run_together(indices: list[int], max_tokens: int) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        threads = [threading.Thread(target=run, args=(i, max_tokens, out)) for i in indices]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if len(out) != len(indices):
            raise RuntimeError(f"{len(indices) - len(out)} requests did not finish")
        return out

    # Warm-up: single requests and one full batch, so the timed runs compile nothing.
    for index in (100, 101):
        run_together([index], 16)
    run_together(list(range(200, 200 + CONCURRENCY)), 16)

    single = [run_together([index], OUTPUT_TOKENS)[0] for index in range(SINGLE_REQUESTS)]
    emit(
        "single",
        ttft_ms=round(statistics.median((r["first"] - r["submit"]) * 1000 for r in single), 1),
        decode_tok_s=round(
            statistics.median(
                (r["tokens"] - 1) / (r["end"] - r["first"]) for r in single if r["tokens"] > 1
            ),
            1,
        ),
        mean_output_tokens=round(statistics.mean(r["tokens"] for r in single), 1),
    )

    batch = run_together(list(range(10, 10 + CONCURRENCY)), OUTPUT_TOKENS)
    wall = max(r["end"] for r in batch) - min(r["submit"] for r in batch)
    total = sum(r["tokens"] for r in batch)
    emit(
        "batch",
        concurrency=CONCURRENCY,
        throughput_tok_s=round(total / wall, 1),
        ttft_p50_ms=round(statistics.median((r["first"] - r["submit"]) * 1000 for r in batch), 1),
        mean_output_tokens=round(total / len(batch), 1),
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
    )


def _measure(key: str, gpu: str) -> list[dict[str, Any]]:
    """One model on one GPU, through the scheduler that the API server picks for it."""
    import time

    import torch

    from mini_infer.models import architecture_uses_state_cache

    spec = MODELS[key]
    results: list[dict[str, Any]] = []
    emit = _emitter(key, gpu, results)
    started = time.perf_counter()
    if architecture_uses_state_cache(spec.repo):
        from mini_infer.engine.cuda_graph_decode import supports_decode_cuda_graph
        from mini_infer.engine.state_cache_generator import StateCacheGenerator
        from mini_infer.scheduler import StateCacheContinuousScheduler

        generator = StateCacheGenerator.from_pretrained(spec.repo)
        cuda_graph = supports_decode_cuda_graph(generator.model, generator.device)
        scheduler: Any = StateCacheContinuousScheduler(
            generator,
            max_batch_size=CONCURRENCY,
            max_seq_len=MAX_SEQ_LEN,
            cuda_graph=cuda_graph,
        )
        tokenizer = generator.tokenizer
        path = f"state cache, decode graph {'on' if cuda_graph else 'off'}"
    else:
        from mini_infer.engine.model_runner import ModelRunner
        from mini_infer.scheduler import ContinuousScheduler

        # 2048 blocks of 16 tokens hold 16 requests of 1,152 tokens with room to spare.
        runner = ModelRunner.from_pretrained(
            spec.repo, attention_backend="flashinfer", num_blocks=2048
        )
        graphs = []
        for name, enable in (
            ("decode", runner.enable_decode_graph),
            ("prefill", runner.enable_prefill_graph),
        ):
            try:
                enable()
                graphs.append(name)
            except ValueError as error:
                emit("info", note=f"{name} graph off: {error}")
        scheduler = ContinuousScheduler(runner, max_concurrent=CONCURRENCY)
        tokenizer = runner.tokenizer
        backend = runner.block_pool.attention_backend
        path = f"paged, {backend} attention, graphs: {', '.join(graphs) or 'none'}"
    emit(
        "load",
        path=path,
        load_s=round(time.perf_counter() - started, 1),
        allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 1),
    )
    scheduler.start()
    try:
        _workload(scheduler, tokenizer, emit)
    finally:
        scheduler.stop()
    return results


def _tp_rank(rank: int, world_size: int, key: str, gpu: str) -> list[dict[str, Any]] | None:
    """One rank of a two-GPU model. Rank 0 serves the workload, rank 1 follows its steps."""
    import time

    import torch

    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.engine.tp_state_cache_continuous_server import (
        TensorParallelStateCacheContinuousServer,
    )
    from mini_infer.models import REGISTRY, _raw_config_namespace
    from mini_infer.scheduler import TensorParallelStateCacheContinuousScheduler

    spec = MODELS[key]
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    started = time.perf_counter()
    architecture = _raw_config_namespace(spec.repo).architectures[0]
    model = REGISTRY.lookup(architecture).from_checkpoint(
        spec.repo, device=device, dtype=torch.bfloat16
    )
    tokenizer = Tokenizer.from_pretrained(spec.repo) if rank == 0 else None
    server = TensorParallelStateCacheContinuousServer(
        model,
        tokenizer,
        device=device,
        dtype=torch.bfloat16,
        max_batch_size=CONCURRENCY,
        max_seq_len=MAX_SEQ_LEN,
        cuda_graph=True,
    )
    if rank != 0:
        server.run_follower_loop()
        return None
    results: list[dict[str, Any]] = []
    emit = _emitter(key, gpu, results)
    emit(
        "load",
        path=f"tensor parallel x{world_size}, state cache, decode graph on",
        load_s=round(time.perf_counter() - started, 1),
        allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 1),
    )
    scheduler = TensorParallelStateCacheContinuousScheduler(server)
    scheduler.start()
    try:
        _workload(scheduler, tokenizer, emit)
    finally:
        scheduler.stop()  # also releases the follower
    return results


def _tp_child(rank: int, world_size: int, key: str, gpu: str, queue: Any) -> None:
    """`spawn` entry of one rank. It sends its result before it destroys the process group,
    so a teardown that hangs cannot hold the result back."""
    import torch.distributed as dist

    from mini_infer.distributed.group import destroy_distributed, init_distributed

    try:
        init_distributed(
            world_size=world_size,
            rank=rank,
            backend="nccl",
            master_addr="127.0.0.1",
            master_port=29500,
        )
        queue.put(("ok", _tp_rank(rank, world_size, key, gpu)))
    except Exception:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}"))
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _measure_tp(key: str, gpu: str) -> list[dict[str, Any]]:
    import torch.multiprocessing as mp

    world_size = MODELS[key].tensor_parallel
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_tp_child, args=(rank, world_size, key, gpu, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    replies = []
    try:
        while len(replies) < world_size:
            replies.append(queue.get(timeout=1500))
            if replies[-1][0] == "err":
                break  # the other rank can wait in a collective until the timeout
    finally:
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
    errors = [payload for status, payload in replies if status == "err"]
    if errors:
        raise RuntimeError("\n".join(errors))
    return next(payload for status, payload in replies if status == "ok" and payload is not None)


def _guarded(measure: Callable[[str, str], list[dict[str, Any]]], key: str, gpu: str) -> Any:
    try:
        return measure(key, gpu)
    except Exception as error:
        # One model's failure must not hide the results of the others.
        row = {"kind": "error", "model": key, "gpu": gpu, "error": repr(error)[:800]}
        print("RESULT " + json.dumps(row), flush=True)
        return [row]


_COMMON: dict[str, Any] = dict(
    image=image,
    # A cost cap, not an expected duration: a hung container bills the whole cap.
    timeout=1200,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)


# "H100!" forbids Modal's upgrade to an H200, which would change the numbers.
@app.function(gpu="H100!", **_COMMON)
def measure_h100(key: str) -> list[dict[str, Any]]:
    return _guarded(_measure, key, "h100")


@app.function(gpu="B200", **_COMMON)
def measure_b200(key: str) -> list[dict[str, Any]]:
    return _guarded(_measure, key, "b200")


@app.function(gpu="B200:2", **{**_COMMON, "timeout": 1800})
def measure_b200x2(key: str) -> list[dict[str, Any]]:
    return _guarded(_measure_tp, key, "b200")


@app.local_entrypoint()
def main(models: str = "all", gpus: str = "h100,b200", out: str = "") -> None:
    keys = list(MODELS) if models == "all" else [k.strip() for k in models.split(",")]
    calls = []
    for key in keys:
        spec = MODELS[key]
        for gpu in [g.strip() for g in gpus.split(",")]:
            if gpu not in spec.gpus:
                print(f"skip {key} on {gpu}: the bf16 weights do not fit")
                continue
            if spec.tensor_parallel == 2:
                function = measure_b200x2
            else:
                function = {"h100": measure_h100, "b200": measure_b200}[gpu]
            calls.append((key, gpu, function.spawn(key)))
    rows: list[dict[str, Any]] = []
    for key, gpu, call in calls:
        try:
            rows.extend(call.get())
        except Exception as error:
            rows.append({"kind": "error", "model": key, "gpu": gpu, "error": repr(error)[:800]})
    if out:
        Path(out).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"{len(rows)} rows from {len(calls)} containers")
