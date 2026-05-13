"""DeepSeek-V4-Flash smoke test on Modal: load real V4 weights on 2x B200.

The exit condition for the mini-infer + V4 portfolio piece. V4-Flash
(158 GB on disk in mixed FP8/FP4) doesn't fit a single B200 (192 GB HBM)
after activations + KV + MoE routing, so this is also the first
load-real-V4 path the project supports.

What this validates:
  1. The TP infrastructure (column / row / vocab-parallel layers,
     expert-parallel MoE) holds together on real Blackwell GPUs.
  2. `DeepseekV4ForCausalLM.load_weights` round-trips through the HF
     safetensors index, FP8 e4m3fn dequant, V2/V3-style MoE renames,
     and per-rank slicing.
  3. The model produces FINITE logits for a real prompt.

What this does NOT validate (deferred follow-ups):
  - Full greedy generation through the scheduler (the scheduler isn't
    TP-aware yet).
  - A fused FP4 GEMM kernel. Today's loader dequantises NVFP4 expert
    weights to BF16 at load time, which doubles per-rank expert memory
    over keeping them packed. On 2x B200 (192 GB HBM each) V4-Flash
    still fits comfortably, but a fused FP4 GEMM is the throughput
    follow-up for higher-density configurations.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_smoke.py
"""

import os

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"

app = modal.App("mini-infer-v4-flash-smoke")

# Blackwell needs the cu128 torch wheel; cu124 ships kernels through SM_90.
# flash-attn isn't installed for the same reason (prebuilt wheel pinned
# to an older torch); FlashInfer covers attention. The V4 attention path
# uses our SDPA reference plus the per-block HCA/CSA dispatchers, so
# even FlashInfer isn't strictly needed for the smoke.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=4.40",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
    )
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_python_source("mini_infer")
)


def _run_one_rank(rank: int, world_size: int, prompt: str) -> dict:
    """Per-rank container work: init PG, build model, load weights, run forward."""
    import torch
    import torch.distributed as dist

    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models import REGISTRY
    from mini_infer.models.loader import load_safetensors_state_dict

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

        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(_MODEL_NAME)
        arch = hf_config.architectures[0]
        model_cls = REGISTRY.lookup(arch)
        cfg = model_cls.Config.from_hf(hf_config)
        model = model_cls(cfg).to(device=device, dtype=torch.bfloat16).eval()

        # The state_dict carries mixed FP8/FP4 weights. Our V4 loader
        # dequantises FP8 -> BF16 and raises for FP4 until the dequant
        # kernel lands.
        state_dict = load_safetensors_state_dict(
            _MODEL_NAME, device=device, dtype=torch.bfloat16
        )
        model_cls.load_weights(model, state_dict)

        tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
        encoded = tokenizer.encode(prompt)
        # V4 expects T to be a multiple of every layer's compression_ratio.
        # Pad up to the LCM if the prompt is too short.
        compression_ratios = list(cfg.compress_ratios)
        from math import lcm

        ratio_lcm = compression_ratios[0]
        for r in compression_ratios[1:]:
            ratio_lcm = lcm(ratio_lcm, r)
        target_len = ((len(encoded) + ratio_lcm - 1) // ratio_lcm) * ratio_lcm
        if target_len > len(encoded):
            pad_token = tokenizer.eos_id or 0
            encoded = encoded + [pad_token] * (target_len - len(encoded))
        input_ids = torch.tensor([encoded], device=device, dtype=torch.long)

        with torch.inference_mode():
            logits = model(input_ids)

        finite = bool(torch.isfinite(logits).all().item())
        argmax_first_steps = logits[0, : min(5, target_len), :].argmax(dim=-1).tolist()
        return {
            "rank": rank,
            "world_size": world_size,
            "n_tokens": target_len,
            "logits_shape": tuple(logits.shape),
            "finite": finite,
            "argmax_first_steps": argmax_first_steps,
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank: int, world_size: int, prompt: str, queue) -> None:
    """Module-level entry-point for `mp.spawn` workers.

    Must be top-level (not nested inside `smoke`) so that
    `multiprocessing.spawn` can pickle it.
    """
    try:
        result = _run_one_rank(rank, world_size, prompt)
        queue.put(("ok", result))
    except Exception as exc:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}\n{exc}"))


@app.function(
    image=image,
    gpu=_GPU,
    timeout=3600,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def smoke(prompt: str) -> dict:
    """Spawn 2 worker processes, each on one B200, run the per-rank load + forward."""
    import torch.multiprocessing as mp

    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()

    processes = [
        ctx.Process(target=_child_entry, args=(rank, world_size, prompt, queue))
        for rank in range(world_size)
    ]
    for p in processes:
        p.start()
    try:
        results = [queue.get(timeout=3000) for _ in range(world_size)]
    finally:
        for p in processes:
            p.join(timeout=10)
    errors = [payload for status, payload in results if status == "err"]
    if errors:
        raise RuntimeError("\n".join(errors))
    return {"per_rank": [r[1] for r in results]}


@app.local_entrypoint()
def main() -> None:
    result = smoke.remote("The capital of France is")
    for entry in result["per_rank"]:
        print(entry)
