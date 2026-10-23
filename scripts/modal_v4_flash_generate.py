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

`--compare-paths` runs the routed experts on both MoE paths of one loaded
model: the grouped path (ADR-036, ADR-040) and the per-expert loop. It
compares the last prefill logits of the prompt and of a long real passage,
times that prefill, and greedy-generates on each path.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_generate.py
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_generate.py \
        --prompt "The capital of France is" --max-new-tokens 48
    uv run modal run scripts/modal_v4_flash_generate.py --compare-paths
"""

import os
from pathlib import Path
from typing import Any

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"
_PASSAGE = Path(__file__).parent / "data" / "technical_passage.md"
# Prefill length of the passage in `--compare-paths`, in tokens.
_PASSAGE_TOKENS = 2048

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


def _compare_one_rank(
    rank: int,
    world_size: int,
    prompt: str,
    max_new_tokens: int,
    passage: str,
    device: str | None = None,
) -> dict[str, Any]:
    """Per-rank work of `--compare-paths`: load once, run both MoE paths.

    `device` defaults to this rank's GPU. A CPU dry run passes `"cpu"`.
    """
    import time

    import torch
    import torch.distributed as dist

    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.engine.state_cache_generator import StateCacheGenerator
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.blocks import grouped_experts
    from mini_infer.models.blocks.fp4_expert import PackedFp4Experts
    from mini_infer.models.blocks.fp4_grouped_kernel import supports_block_fp4_grouped_mm
    from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

    init_distributed(
        world_size=world_size,
        rank=rank,
        backend="nccl",
        master_addr="127.0.0.1",
        master_port=29500,
    )
    try:
        if device is None:
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
        if not (
            grouped_experts.supports_grouped_experts(device)
            and supports_block_fp4_grouped_mm(device)
        ):
            raise RuntimeError("the grouped FP4 path does not run on this device")
        _checkpoint(rank, f"loading real V4-Flash via from_checkpoint onto {device}...")
        model = DeepseekV4ForCausalLM.from_checkpoint(
            _MODEL_NAME, device=device, dtype=torch.bfloat16
        )
        tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
        generator = StateCacheGenerator(model, tokenizer)
        moe_blocks = [m for m in model.modules() if isinstance(m, HashRoutedMoEFFN)]
        prompt_ids = tokenizer.encode(prompt)
        passage_ids = tokenizer.encode(passage)[:_PASSAGE_TOKENS]

        def last_logits(ids: list[int]) -> Any:
            cache = model.build_state_cache(
                max_seq_len=len(ids), batch_size=1, device=device, dtype=torch.bfloat16
            )
            input_ids = torch.tensor([ids], device=device, dtype=torch.long)
            with torch.inference_mode():
                logits = model.forward_prefill_with_cache(input_ids, state_cache=cache)
            return logits[0, -1].float()

        def timed_ms(fn: Any) -> tuple[Any, float]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            return out, (time.perf_counter() - start) * 1e3

        runs: dict[str, dict[str, Any]] = {}
        packed_layers = 0
        for path, disabled in (("grouped", False), ("loop", True)):
            grouped_experts._DISABLED_FOR_BENCH = disabled
            _checkpoint(rank, f"{path} path: prefill and generate...")
            short = last_logits(prompt_ids)
            last_logits(passage_ids)  # warm-up: Triton compiles its kernels here
            long, prefill_ms = timed_ms(lambda: last_logits(passage_ids))
            ids, generate_ms = timed_ms(
                lambda: generator.generate_ids(
                    prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id
                )
            )
            if path == "grouped":
                packed_layers = sum(
                    isinstance(block._expert_pack, PackedFp4Experts) for block in moe_blocks
                )
            runs[path] = {
                "short": short,
                "long": long,
                "prefill_ms": prefill_ms,
                "generate_ms": generate_ms,
                "ids": ids,
                "text": tokenizer.decode(ids),
            }
        grouped_experts._DISABLED_FOR_BENCH = False

        grouped, loop = runs["grouped"], runs["loop"]
        pairs = enumerate(zip(grouped["ids"], loop["ids"], strict=False))
        diverge = next((i for i, (a, b) in pairs if a != b), None)

        def agreement(key: str) -> dict[str, Any]:
            a, b = grouped[key], loop[key]
            return {
                "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                "max_abs_diff": float((a - b).abs().max()),
                "same_argmax": bool(a.argmax() == b.argmax()),
            }

        return {
            "rank": rank,
            "moe_layers": len(moe_blocks),
            "packed_fp4_layers": packed_layers,
            "prompt_tokens": len(prompt_ids),
            "passage_tokens": len(passage_ids),
            "short_logits": agreement("short"),
            "long_logits": agreement("long"),
            "first_divergence": diverge,
            "generated": {path: run["ids"] for path, run in runs.items()},
            "text": {path: run["text"] for path, run in runs.items()},
            "prefill_ms": {path: run["prefill_ms"] for path, run in runs.items()},
            "generate_ms": {path: run["generate_ms"] for path, run in runs.items()},
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(
    rank: int, world_size: int, prompt: str, max_new_tokens: int, queue, passage: str | None = None
) -> None:
    """Module-level entry-point for `mp.spawn` workers (must be picklable)."""
    try:
        if passage is None:
            result = _run_one_rank(rank, world_size, prompt, max_new_tokens)
        else:
            result = _compare_one_rank(rank, world_size, prompt, max_new_tokens, passage)
        queue.put(("ok", result))
    except Exception:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}"))


def _spawn_ranks(prompt: str, max_new_tokens: int, passage: str | None) -> list[dict]:
    """Run `_child_entry` on 2 worker processes (one per B200), results by rank."""
    import torch.multiprocessing as mp

    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()

    processes = [
        ctx.Process(
            target=_child_entry,
            args=(rank, world_size, prompt, max_new_tokens, queue, passage),
        )
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
    return sorted((r[1] for r in results), key=lambda entry: entry["rank"])


@app.function(image=image, timeout=300)
def versions() -> dict:
    """The image's torch and Triton, and whether torch has `grouped_mm` (no GPU)."""
    import torch
    from torch.nn import functional

    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "torch": torch.__version__,
        "triton": triton_version,
        "grouped_mm": hasattr(functional, "grouped_mm"),
    }


@app.function(
    image=image,
    gpu=_GPU,
    timeout=1800,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def compare_moe_paths(prompt: str, max_new_tokens: int, passage: str) -> dict:
    """Both MoE paths on one load of V4-Flash per rank; see `_compare_one_rank`."""
    per_rank = _spawn_ranks(prompt, max_new_tokens, passage=passage)
    grouped_ids = per_rank[0]["generated"]["grouped"]
    return {
        "ranks_agree": all(entry["generated"]["grouped"] == grouped_ids for entry in per_rank),
        "per_rank": per_rank,
    }


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
    per_rank = _spawn_ranks(prompt, max_new_tokens, passage=None)
    # Tensor-parallel ranks must agree token-for-token; divergence is a TP bug.
    rank0_ids = per_rank[0]["generated_ids"]
    ranks_agree = all(entry["generated_ids"] == rank0_ids for entry in per_rank)
    return {
        "prompt": prompt,
        "ranks_agree": ranks_agree,
        "generated_text": per_rank[0]["generated_text"],
        "per_rank": per_rank,
    }


def _print_comparison(result: dict) -> None:
    rank0 = result["per_rank"][0]
    print(f"\nranks agree on the grouped tokens (TP consistency): {result['ranks_agree']}")
    print(f"MoE layers: {rank0['moe_layers']}, on packed FP4 experts: {rank0['packed_fp4_layers']}")
    print(f"prompt {rank0['prompt_tokens']} tokens, passage {rank0['passage_tokens']} tokens")
    for key in ("short_logits", "long_logits"):
        print(f"{key}, grouped against loop: {rank0[key]}")
    print(f"first diverging generated token: {rank0['first_divergence']}")
    for path in ("grouped", "loop"):
        print(
            f"{path}: passage prefill {rank0['prefill_ms'][path]:.1f} ms, "
            f"generate {rank0['generate_ms'][path]:.1f} ms, text {rank0['text'][path]!r}"
        )


@app.local_entrypoint()
def main(
    prompt: str = "The capital of France is", max_new_tokens: int = 32, compare_paths: bool = False
) -> None:
    if compare_paths:
        found = versions.remote()
        print(f"image: {found}")
        if not found["grouped_mm"] or found["triton"] is None:
            raise SystemExit("the image lacks grouped_mm or Triton; the grouped path cannot run")
        _print_comparison(compare_moe_paths.remote(prompt, max_new_tokens, _PASSAGE.read_text()))
        return
    result = generate.remote(prompt, max_new_tokens)
    print(f"\nprompt: {result['prompt']!r}")
    print(f"ranks agree (TP consistency): {result['ranks_agree']}")
    print(f"generated: {result['generated_text']!r}")
    for entry in result["per_rank"]:
        print(
            f"  rank {entry['rank']}: {entry['n_prompt_tokens']} prompt tokens -> "
            f"{entry['n_generated']} generated"
        )
