"""Greedy generation on real DeepSeek-V4-Flash on 2x B200 (Modal).

Closes the gap the load smoke (`scripts/modal_v4_flash_smoke.py`) explicitly
left open: that smoke runs ONE `model(input_ids)` forward and checks the
logits are finite; it does not generate. This script drives the full
single-request greedy path on the real checkpoint:

    load (from_checkpoint, per rank) -> prefill -> greedy decode -> detokenize

via `StateCacheGenerator`, the same path the CPU unit tests exercise on
synthetic configs.

What this validates:
  1. `DeepseekV4ForCausalLM.from_checkpoint` loads real V4-Flash onto each of
     2 B200s under tensor parallelism (config.json + meta construction +
     `load_weights` + rotary rebuild), reusing the exact path the unit tests
     round-trip on a synthetic checkpoint.
  2. The cache-aware prefill + decode generate COHERENT text from a real
     prompt (real weights + a layer that is bit-parity validated against the
     DeepSeek-V4 reference should continue a prompt sensibly, not emit noise).
  3. The two ranks generate IDENTICAL tokens. Under correct tensor
     parallelism the per-rank head shards all-reduce to the same logits, so
     any divergence between ranks is a TP bug.

Why no side-by-side reference comparison here: the DeepSeek-V4 reference
(`third_party/deepseek_v4_reference/`) is a full runnable model, but two
full V4-Flash models (~158 GB each) do not fit 2x B200 (384 GB HBM)
simultaneously. Bit-parity against the reference is already covered at the
attention-layer level by `tests/unit/test_v4_prefill_cache_aware.py` (cosine
sim > 0.999, prefill + decode). At the whole-model scale the practical bar
is coherent, deterministic, rank-consistent output.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_generate.py
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_generate.py \
        --prompt "The capital of France is" --max-new-tokens 48
"""

import os

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"

app = modal.App("mini-infer-v4-flash-generate")

# Same image as the load smoke: Blackwell needs the cu128 torch wheel.
# FlashInfer covers attention math; the V4 attention path also has our SDPA
# reference plus the per-block HCA/CSA/SWA dispatchers.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=4.40",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
    )
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_python_source("mini_infer")
)


def _checkpoint(rank: int, msg: str) -> None:
    """Print + flush so a hang's location is visible in Modal logs immediately."""
    print(f"[rank {rank}] {msg}", flush=True)


def _run_one_rank(rank: int, world_size: int, prompt: str, max_new_tokens: int) -> dict:
    """Per-rank container work: init PG, load V4-Flash, greedy-generate."""
    import torch
    import torch.distributed as dist

    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.engine.state_cache_generator import StateCacheGenerator
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

    _checkpoint(rank, "entering _run_one_rank; calling init_distributed (nccl)...")
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
        # from_checkpoint reads the raw config.json, builds on meta, runs
        # load_weights (per-rank TP slicing falls out here), and rebuilds the
        # rotary buffer. The download + load of ~158 GB is the slow step; if a
        # run hangs, it hangs inside this call.
        _checkpoint(rank, f"loading real V4-Flash via from_checkpoint onto {device}...")
        model = DeepseekV4ForCausalLM.from_checkpoint(
            _MODEL_NAME, device=device, dtype=torch.bfloat16
        )
        _checkpoint(rank, "load_weights done; building tokenizer + generator...")

        tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
        generator = StateCacheGenerator(model, tokenizer)

        prompt_ids = tokenizer.encode(prompt)
        _checkpoint(
            rank,
            f"prompt encoded ({len(prompt_ids)} tokens); greedy-generating "
            f"{max_new_tokens} tokens...",
        )
        with torch.inference_mode():
            generated_ids = generator.generate_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(generated_ids)
        _checkpoint(rank, f"generation done ({len(generated_ids)} tokens).")
        return {
            "rank": rank,
            "world_size": world_size,
            "n_prompt_tokens": len(prompt_ids),
            "n_generated": len(generated_ids),
            "generated_ids": generated_ids,
            "generated_text": generated_text,
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank: int, world_size: int, prompt: str, max_new_tokens: int, queue) -> None:
    """Module-level entry-point for `mp.spawn` workers (must be picklable)."""
    try:
        result = _run_one_rank(rank, world_size, prompt, max_new_tokens)
        queue.put(("ok", result))
    except Exception:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}"))


@app.function(
    image=image,
    gpu=_GPU,
    # Hard wall-clock ceiling: a warm-cache load plus a short greedy generation
    # completes well inside this; the cap only bounds a hung run.
    timeout=1800,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def generate(prompt: str, max_new_tokens: int) -> dict:
    """Spawn 2 worker processes (one per B200), greedy-generate, check agreement."""
    import torch.multiprocessing as mp

    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()

    processes = [
        ctx.Process(target=_child_entry, args=(rank, world_size, prompt, max_new_tokens, queue))
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

    per_rank = sorted((r[1] for r in results), key=lambda entry: entry["rank"])
    # Tensor-parallel ranks must agree token-for-token; divergence is a TP bug.
    rank0_ids = per_rank[0]["generated_ids"]
    ranks_agree = all(entry["generated_ids"] == rank0_ids for entry in per_rank)
    return {
        "prompt": prompt,
        "ranks_agree": ranks_agree,
        "generated_text": per_rank[0]["generated_text"],
        "per_rank": per_rank,
    }


@app.local_entrypoint()
def main(prompt: str = "The capital of France is", max_new_tokens: int = 32) -> None:
    result = generate.remote(prompt, max_new_tokens)
    print(f"\nprompt: {result['prompt']!r}")
    print(f"ranks agree (TP consistency): {result['ranks_agree']}")
    print(f"generated: {result['generated_text']!r}")
    for entry in result["per_rank"]:
        print(
            f"  rank {entry['rank']}: {entry['n_prompt_tokens']} prompt tokens -> "
            f"{entry['n_generated']} generated"
        )
