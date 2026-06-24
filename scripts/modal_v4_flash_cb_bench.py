"""Benchmark ragged continuous batching vs one-at-a-time on V4-Flash, 2x B200.

V4-Flash is sharded across 2 B200s (tensor parallelism), so both paths run under
the leader / follower split. Two paths, same model + prompts:

  - **one-at-a-time:** each request decoded alone via
    `TensorParallelStateCacheServer.generate_ids` (batch of 1).
  - **continuous batching:** all requests decoded together via
    `TensorParallelStateCacheContinuousServer.generate_cohort` (one ragged
    forward per step over the whole batch, each request at its own position).

Both produce identical tokens (the ragged decode is bit-parity self-consistent
with the scalar path, validated on gloo); this measures the decode-throughput
win of batching on the GPU. Reports tokens/sec for each path and the speedup.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_cb_bench.py
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_cb_bench.py \
        --num-requests 16 --max-new-tokens 64
"""

import os

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"
_MAX_SEQ_LEN = 1024  # generous: real prompts here are << this, plus max_new_tokens

app = modal.App("mini-infer-v4-flash-cb-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=4.40", "safetensors>=0.4", "huggingface_hub>=0.20")
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_python_source("mini_infer")
)

# A spread of realistic, varied-length prompts so the batch is genuinely ragged
# (requests at different positions, finishing at different steps).
_PROMPTS = [
    "Summarize the causes of the 1929 stock market crash in three sentences.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "What is the capital of France?",
    "Explain how a transformer attention layer works to a first-year CS student.",
    "Translate 'good morning, how are you?' into French, German, and Japanese.",
    "List five practical tips for reducing household energy consumption this winter.",
    "Describe the plot of Hamlet in a single paragraph without spoilers for the ending.",
    "Give me a regular expression that matches a valid IPv4 address.",
    "Why is the sky blue? Answer in plain language.",
    "Outline a weekly meal plan for a vegetarian trying to increase protein intake.",
    "What are the trade-offs between TCP and UDP for real-time video streaming?",
    "Compose a four-line poem about the changing of the seasons.",
    "Explain the difference between supervised and unsupervised machine learning.",
    "How do I safely defrost a frozen turkey, and how long does it take?",
    "What were the main consequences of the invention of the printing press?",
    "Write a SQL query to find the second-highest salary in an employees table.",
]


def _run_one_rank(
    rank: int, world_size: int, prompts: list[str], max_new_tokens: int, max_seq_len: int
) -> dict | None:
    """Per-rank work: load the shard; rank 0 benchmarks, followers mirror."""
    import time

    import torch
    import torch.distributed as dist

    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.engine.tp_state_cache_continuous_server import (
        TensorParallelStateCacheContinuousServer,
    )
    from mini_infer.engine.tp_state_cache_server import TensorParallelStateCacheServer
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

    print(f"[rank {rank}] init_distributed (nccl)...", flush=True)
    init_distributed(
        world_size=world_size, rank=rank, backend="nccl", master_addr="127.0.0.1", master_port=29500
    )
    try:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        print(f"[rank {rank}] loading V4-Flash shard onto {device}...", flush=True)
        model = DeepseekV4ForCausalLM.from_checkpoint(
            _MODEL_NAME, device=device, dtype=torch.bfloat16
        )
        batch_size = len(prompts)
        # Baseline (one-at-a-time) server + continuous-batching server, both per rank.
        baseline = TensorParallelStateCacheServer(model, device=device, dtype=torch.bfloat16)
        continuous = TensorParallelStateCacheContinuousServer(
            model,
            device=device,
            dtype=torch.bfloat16,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        if rank != 0:
            # Mirror the baseline phase, then the continuous phase.
            baseline.run_follower_loop()
            continuous.run_follower_loop()
            return None

        tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
        prompt_ids = [tokenizer.encode(p) for p in prompts]

        def sync() -> None:
            torch.cuda.synchronize(rank)

        # ---- one-at-a-time baseline ----
        baseline.generate_ids(prompt_ids[0], max_new_tokens=4)  # warmup (untimed)
        sync()
        t0 = time.perf_counter()
        for ids in prompt_ids:
            baseline.generate_ids(ids, max_new_tokens=max_new_tokens, eos_token_id=None)
        sync()
        baseline_seconds = time.perf_counter() - t0
        baseline.shutdown()

        # ---- continuous batching ----
        continuous.generate_cohort(prompt_ids[:2], max_new_tokens=4)  # warmup (untimed)
        sync()
        t0 = time.perf_counter()
        continuous.generate_cohort(prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=None)
        sync()
        continuous_seconds = time.perf_counter() - t0
        continuous.shutdown()

        total_tokens = batch_size * max_new_tokens
        baseline_tps = total_tokens / baseline_seconds
        continuous_tps = total_tokens / continuous_seconds
        return {
            "num_requests": batch_size,
            "max_new_tokens": max_new_tokens,
            "total_tokens": total_tokens,
            "baseline_seconds": baseline_seconds,
            "continuous_seconds": continuous_seconds,
            "baseline_tokens_per_sec": baseline_tps,
            "continuous_tokens_per_sec": continuous_tps,
            "speedup": continuous_tps / baseline_tps,
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank, world_size, prompts, max_new_tokens, max_seq_len, queue) -> None:
    try:
        queue.put(("ok", _run_one_rank(rank, world_size, prompts, max_new_tokens, max_seq_len)))
    except Exception:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}"))


@app.function(
    image=image,
    gpu=_GPU,
    timeout=1800,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def bench(prompts: list[str], max_new_tokens: int) -> dict:
    import torch.multiprocessing as mp

    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_child_entry,
            args=(rank, world_size, prompts, max_new_tokens, _MAX_SEQ_LEN, queue),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    try:
        results = [queue.get(timeout=1680) for _ in range(world_size)]
    finally:
        for p in procs:
            p.join(timeout=10)
    errors = [payload for status, payload in results if status == "err"]
    if errors:
        raise RuntimeError("\n".join(errors))
    return next(payload for status, payload in results if status == "ok" and payload is not None)


@app.local_entrypoint()
def main(num_requests: int = 16, max_new_tokens: int = 64) -> None:
    prompts = [_PROMPTS[i % len(_PROMPTS)] for i in range(num_requests)]
    result = bench.remote(prompts, max_new_tokens)
    print("\n=== V4-Flash continuous batching benchmark (2x B200, TP) ===")
    print(f"requests: {result['num_requests']}, max_new_tokens: {result['max_new_tokens']}")
    print(f"total decoded tokens: {result['total_tokens']}")
    print(
        f"one-at-a-time: {result['baseline_seconds']:.2f}s  "
        f"({result['baseline_tokens_per_sec']:.1f} tok/s)"
    )
    print(
        f"continuous   : {result['continuous_seconds']:.2f}s  "
        f"({result['continuous_tokens_per_sec']:.1f} tok/s)"
    )
    print(f"speedup: {result['speedup']:.2f}x")
