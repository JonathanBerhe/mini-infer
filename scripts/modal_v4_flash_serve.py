"""Serve real DeepSeek-V4-Flash over HTTP under tensor parallelism on 2x B200.

The final gate for option (c): a model too big for one GPU, reachable through
the single `/v1/completions` interface. V4-Flash is sharded across 2 B200s, so
serving it needs the front-door / follower split (see
`mini_infer.engine.tp_state_cache_server`):

  - rank 0 loads its shard, builds the `/v1/completions` app backed by a
    `TensorParallelStateCacheScheduler`, sends one request through the real
    endpoint (via `TestClient`), and checks the response.
  - rank 1 loads its shard and mirrors generation in lockstep via
    `run_follower_loop()` until rank 0 broadcasts shutdown.

The leader/follower coordination and the HTTP path are already proven on CPU
with gloo + a synthetic V4 in `tests/unit/test_tp_state_cache_server.py`; this
confirms the same path on the real checkpoint and real NCCL.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_serve.py
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_serve.py \
        --prompt "The capital of France is" --max-tokens 32
"""

import os

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"

app = modal.App("mini-infer-v4-flash-serve")

# Same Blackwell image as the generate smoke, plus fastapi + httpx so the
# in-process TestClient can drive the real /v1/completions endpoint.
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


def _checkpoint(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _run_one_rank(rank: int, world_size: int, prompt: str, max_tokens: int) -> dict | None:
    """Per-rank work: load the shard; rank 0 serves a request, rank 1 follows."""
    import torch
    import torch.distributed as dist
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mini_infer.api.server import _unhandled_exception_handler, completions
    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.engine.tp_state_cache_server import TensorParallelStateCacheServer
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM
    from mini_infer.scheduler import TensorParallelStateCacheScheduler

    _checkpoint(rank, "init_distributed (nccl)...")
    init_distributed(
        world_size=world_size,
        rank=rank,
        backend="nccl",
        master_addr="127.0.0.1",
        master_port=29500,
    )
    try:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        _checkpoint(rank, f"loading V4-Flash shard via from_checkpoint onto {device}...")
        model = DeepseekV4ForCausalLM.from_checkpoint(
            _MODEL_NAME, device=device, dtype=torch.bfloat16
        )
        _checkpoint(rank, "shard loaded.")

        if rank != 0:
            # Follower: mirror the leader's generation until it broadcasts shutdown.
            server = TensorParallelStateCacheServer(model, device=device, dtype=torch.bfloat16)
            _checkpoint(rank, "entering follower loop...")
            server.run_follower_loop()
            _checkpoint(rank, "follower loop done.")
            return None

        # Leader: build the real /v1/completions app over a TP scheduler, then
        # send one request through it.
        tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
        server = TensorParallelStateCacheServer(
            model, tokenizer, device=device, dtype=torch.bfloat16
        )
        scheduler = TensorParallelStateCacheScheduler(server)
        scheduler.start()
        http_app = FastAPI()
        http_app.add_api_route(
            "/v1/completions", completions, methods=["POST"], response_model=None
        )
        http_app.add_exception_handler(Exception, _unhandled_exception_handler)
        http_app.state.scheduler = scheduler
        try:
            _checkpoint(rank, f"serving one /v1/completions request (max_tokens={max_tokens})...")
            with TestClient(http_app) as client:
                response = client.post(
                    "/v1/completions",
                    json={"model": _MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens},
                )
            result = {"status_code": response.status_code, "body": response.json()}
            _checkpoint(rank, f"request done (status {response.status_code}).")
        finally:
            scheduler.stop()
            server.shutdown()
        return result
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank: int, world_size: int, prompt: str, max_tokens: int, queue) -> None:
    try:
        result = _run_one_rank(rank, world_size, prompt, max_tokens)
        queue.put(("ok", result))
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
def serve(prompt: str, max_tokens: int) -> dict:
    """Spawn 2 ranks (one per B200), serve one HTTP request through the TP leader."""
    import torch.multiprocessing as mp

    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(target=_child_entry, args=(rank, world_size, prompt, max_tokens, queue))
        for rank in range(world_size)
    ]
    for p in processes:
        p.start()
    try:
        results = [queue.get(timeout=1680) for _ in range(world_size)]
    finally:
        for p in processes:
            p.join(timeout=10)
    errors = [payload for status, payload in results if status == "err"]
    if errors:
        raise RuntimeError("\n".join(errors))
    leader = next(payload for status, payload in results if status == "ok" and payload is not None)
    return leader


@app.local_entrypoint()
def main(prompt: str = "The capital of France is", max_tokens: int = 32) -> None:
    result = serve.remote(prompt, max_tokens)
    print(f"\nHTTP status: {result['status_code']}")
    body = result["body"]
    print(f"prompt: {prompt!r}")
    print(f"served completion: {body['choices'][0]['text']!r}")
    print(f"finish_reason: {body['choices'][0]['finish_reason']}, usage: {body['usage']}")
