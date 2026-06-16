"""PD disaggregation smoke on Modal: prefill on one GPU, decode on the other.

Spawns a 2x H100 container, launches two processes inside it (rank 0 =
prefill, rank 1 = decode), each loads its own `ModelRunner` on its own
GPU, and runs a single request through the full PD pipeline:

  1. Rank 0 calls `PrefillWorker.prefill(request)` on its GPU.
  2. Rank 0 ships the resulting `KVHandoff` to rank 1 via
     `kv_transfer.send_handoff` (NCCL P2P over NVLink between the two
     H100s).
  3. Rank 1 receives the handoff via `recv_handoff` and hands it to a
     `DecodeWorker`, which materializes the KV into its paged cache and
     runs the decode loop on its GPU.
  4. Rank 1 returns the decoded token list back to the parent process.

The smoke validates:
  - Per-rank `ModelRunner` load (Qwen2.5-7B at bf16, ~14 GB / GPU).
  - Per-stream KV extraction on the prefill rank.
  - NCCL P2P `send_handoff` / `recv_handoff` over a 2-rank PG.
  - Per-stream KV materialization on the decode rank.
  - Greedy decode loop produces sensible tokens.

It does NOT measure throughput. A separate bench script wraps the same
`PrefillWorker.prefill_batch` / `DecodeWorker.decode_batch` calls in a
timing loop to compare PD against single-GPU mixed-mode.

Run with:
    uv run modal run scripts/modal_pd_smoke.py
    HF_TOKEN=<token> uv run modal run scripts/modal_pd_smoke.py --prompt "..."

Cost (Modal H100:2): ~$0.04/min wall, smoke runs ~5-7 min including
model load. Budget ~$2-4 worst case (cold-start + retry).
"""

import os

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Qwen2.5-7B at bf16 fits each H100 comfortably (~14 GB out of 80) with
# room for KV cache + activations. The point is to exercise PD on a real
# 7B model where the throughput gap between PD and mixed-mode shows up.
_MODEL_NAME = "Qwen/Qwen2.5-7B"
_GPU = "H100:2"

app = modal.App("mini-infer-pd-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.40",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
    )
    .add_local_python_source("mini_infer")
)


def _run_one_rank(rank: int, world_size: int, prompt: str, max_tokens: int) -> dict:
    """Per-rank entry. Rank 0 prefills + sends; rank 1 receives + decodes.

    Each rank initialises a NCCL process group, loads its own
    `ModelRunner` on its own GPU, runs its phase, and returns a small
    diagnostic dict. The decode rank's dict carries the emitted token
    list; the prefill rank's dict carries only metadata (prefill_len,
    first sampled token).
    """
    import torch

    from mini_infer.distributed.group import (
        destroy_distributed,
        init_distributed,
        replica_scope,
    )
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler.request_state import Request
    from mini_infer.workers import (
        DECODE_RANK,
        PREFILL_RANK,
        DecodeWorker,
        PrefillWorker,
    )
    from mini_infer.workers.kv_transfer import recv_handoff, send_handoff

    init_distributed(
        world_size=world_size,
        rank=rank,
        backend="nccl",
        master_addr="127.0.0.1",
        master_port=29500,
    )
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        # PD runs a full model per rank; the ranks talk only via the
        # send/recv handoff, not TP collectives. `replica_scope` makes the
        # TP-aware layers see world_size=1 so they run with no all-reduce
        # (which would deadlock against the handoff). The handoff uses
        # dist.send/recv directly and is unaffected.
        with replica_scope():
            runner = ModelRunner.from_pretrained(_MODEL_NAME, device=device, dtype=torch.bfloat16)

            if rank == PREFILL_RANK:
                worker = PrefillWorker(runner)
                request = Request(
                    prompt=prompt,
                    sampling_params=SamplingParams(),  # greedy
                    max_tokens=max_tokens,
                )
                handoff = worker.prefill(request)
                send_handoff(handoff, dst_rank=DECODE_RANK)
                return {
                    "rank": rank,
                    "role": "prefill",
                    "prefill_len": handoff.prefill_len,
                    "first_sampled_token_id": handoff.first_sampled_token_id,
                    "kv_layers": handoff.num_layers,
                }

            if rank == DECODE_RANK:
                worker = DecodeWorker(runner)
                handoff = recv_handoff(src_rank=PREFILL_RANK, pool=runner.block_pool)
                tokens = list(worker.decode(handoff))
                text = runner.tokenizer.decode(tokens)
                return {
                    "rank": rank,
                    "role": "decode",
                    "n_tokens": len(tokens),
                    "token_ids": tokens,
                    "text": text,
                }

            raise ValueError(f"unexpected rank {rank}")
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank: int, world_size: int, prompt: str, max_tokens: int, queue) -> None:
    """Top-level entry for `mp.spawn` workers (must be module-scope to pickle)."""
    try:
        result = _run_one_rank(rank, world_size, prompt, max_tokens)
        queue.put(("ok", result))
    except Exception:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}"))


@app.function(
    image=image,
    gpu=_GPU,
    # Hard wall-clock ceiling. A warm-cache load plus prefill + handoff +
    # decode finishes well inside this; the cap just bounds a hung run
    # rather than holding both GPUs to the longer default.
    timeout=1200,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def smoke(prompt: str, max_tokens: int) -> dict:
    """Container-side: spawn 2 worker processes, return per-rank diagnostics."""
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
        results = [queue.get(timeout=1100) for _ in range(world_size)]
    finally:
        for p in processes:
            p.join(timeout=10)
    errors = [payload for status, payload in results if status == "err"]
    if errors:
        raise RuntimeError("\n".join(errors))
    return {"per_rank": [r[1] for r in results]}


@app.local_entrypoint()
def main(
    prompt: str = "The capital of France is",
    max_tokens: int = 12,
) -> None:
    """Run the PD smoke and pretty-print the per-rank result.

    The interesting payload is the decode rank's `text` field: it should
    contain a coherent continuation of `prompt` ("Paris." or similar
    for the default prompt).
    """
    result = smoke.remote(prompt, max_tokens)
    for entry in result["per_rank"]:
        print(f"\n--- rank {entry['rank']} ({entry['role']}) ---")
        for k, v in entry.items():
            if k in ("rank", "role"):
                continue
            print(f"  {k}: {v}")
