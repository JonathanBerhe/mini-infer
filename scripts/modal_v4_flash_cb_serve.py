"""Serve V4-Flash with ragged continuous batching over HTTP on 2x B200.

The GPU gate for the server wiring: a model too big for one GPU, served through
the single `/v1/completions` endpoint with continuous batching across ranks.
Rank 0 builds a `TensorParallelStateCacheContinuousScheduler` over the TP
continuous server, mounts the real `completions` route, and fires several
CONCURRENT requests (so the engine actually batches them); rank 1 mirrors via
`run_follower_loop()`. Confirms the full path on real NCCL + real V4-Flash:
HTTP -> TP CB scheduler -> ragged batched decode across both shards.

The leader/follower coordination and the ragged decode are already proven on CPU
with gloo + a synthetic V4 (`tests/unit/test_tp_state_cache_continuous_server.py`);
this confirms the same path on the real checkpoint and real NCCL.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_cb_serve.py
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_cb_serve.py --max-tokens 48
"""

import os

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"
_MAX_SEQ_LEN = 1024

app = modal.App("mini-infer-v4-flash-cb-serve")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=4.40",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
        "fastapi>=0.110",
        "httpx>=0.27",
    )
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_python_source("mini_infer")
)

_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "Explain photosynthesis in one sentence.",
    "List three primary colors.",
    "What is 17 times 23?",
    "Name a famous painting by Van Gogh.",
]


def _run_one_rank(
    rank: int, world_size: int, prompts: list[str], max_tokens: int, max_seq_len: int
) -> dict | None:
    """Per-rank work: load the shard; rank 0 serves concurrent requests, rank 1 follows."""
    import concurrent.futures
    import time

    import torch
    import torch.distributed as dist
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mini_infer.api.server import _unhandled_exception_handler, completions
    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.engine.tp_state_cache_continuous_server import (
        TensorParallelStateCacheContinuousServer,
    )
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM
    from mini_infer.scheduler import TensorParallelStateCacheContinuousScheduler

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
        tokenizer = Tokenizer.from_pretrained(_MODEL_NAME) if rank == 0 else None
        server = TensorParallelStateCacheContinuousServer(
            model,
            tokenizer,
            max_batch_size=len(prompts),
            max_seq_len=max_seq_len,
            device=device,
            dtype=torch.bfloat16,
        )
        if rank != 0:
            print(f"[rank {rank}] entering follower loop...", flush=True)
            server.run_follower_loop()
            print(f"[rank {rank}] follower loop done.", flush=True)
            return None

        scheduler = TensorParallelStateCacheContinuousScheduler(server)
        scheduler.start()
        http_app = FastAPI()
        http_app.add_api_route(
            "/v1/completions", completions, methods=["POST"], response_model=None
        )
        http_app.add_exception_handler(Exception, _unhandled_exception_handler)
        http_app.state.scheduler = scheduler
        try:
            print(
                f"[rank 0] serving {len(prompts)} CONCURRENT requests (max_tokens={max_tokens})...",
                flush=True,
            )
            with TestClient(http_app) as client:

                def fire(prompt: str) -> dict:
                    response = client.post(
                        "/v1/completions",
                        json={"model": _MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens},
                    )
                    return {"status": response.status_code, "body": response.json()}

                torch.cuda.synchronize(rank)
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                    responses = list(pool.map(fire, prompts))
                elapsed = time.perf_counter() - started
            completions_out = [
                {"prompt": p, "text": r["body"]["choices"][0]["text"], "status": r["status"]}
                for p, r in zip(prompts, responses, strict=True)
            ]
            print(f"[rank 0] all requests done in {elapsed:.2f}s.", flush=True)
            return {
                "elapsed_seconds": elapsed,
                "num_requests": len(prompts),
                "max_tokens": max_tokens,
                "all_ok": all(r["status"] == 200 for r in responses),
                "completions": completions_out,
            }
        finally:
            scheduler.stop()  # stops the engine and broadcasts shutdown so rank 1 exits
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank, world_size, prompts, max_tokens, max_seq_len, queue) -> None:
    try:
        queue.put(("ok", _run_one_rank(rank, world_size, prompts, max_tokens, max_seq_len)))
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
def serve(prompts: list[str], max_tokens: int) -> dict:
    import torch.multiprocessing as mp

    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_child_entry, args=(rank, world_size, prompts, max_tokens, _MAX_SEQ_LEN, queue)
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
def main(max_tokens: int = 48) -> None:
    result = serve.remote(_PROMPTS, max_tokens)
    print("\n=== V4-Flash continuous batching over HTTP (2x B200, TP) ===")
    print(f"{result['num_requests']} concurrent requests, max_tokens={result['max_tokens']}")
    print(f"all HTTP 200: {result['all_ok']}, wall: {result['elapsed_seconds']:.2f}s")
    for item in result["completions"]:
        print(f"  [{item['status']}] {item['prompt']!r} -> {item['text']!r}")
