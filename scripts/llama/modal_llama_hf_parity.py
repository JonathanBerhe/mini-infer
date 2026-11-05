"""Llama 3.1 parity on one GPU: the engine against transformers on the same prompts.

Loads `unsloth/Meta-Llama-3.1-8B-Instruct` (the weights of the gated meta-llama
repo) in bf16 twice: once in transformers and once through `ModelRunner` with
FlashInfer, as `scripts/smoke/modal_smoke.py` does. It then swaps four RoPE
tables into the engine model:

    as_loaded    the table that `load_model` leaves (llama3 rescale, cast to bf16)
    llama3_fp32  the llama3 table in fp32, as transformers holds it
    plain_bf16   the table before the llama3 fix, cast to bf16
    plain_fp32   the table before the llama3 fix, in fp32

For each table it prints the greedy text for "The capital of France is", the
logit gap to transformers on that prompt, and the top-1 agreement and logit
gap on the last positions of a prompt longer than 8192 tokens (the original
context of Llama 3.1). The long prompt joins repo documents, which are real,
varied text. Each case prints as soon as it finishes.
"""

# Run with:
#   uv run modal run scripts/llama/modal_llama_hf_parity.py
#   MINI_INFER_BENCH_GPU=B200 uv run modal run scripts/llama/modal_llama_hf_parity.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "H100!")
_MODEL = os.environ.get("MINI_INFER_BENCH_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")
# A cost cap, not an expected duration: a hung container bills the whole cap.
_TIMEOUT_S = int(os.environ.get("MINI_INFER_BENCH_TIMEOUT", "900"))

SHORT_PROMPT = "The capital of France is"
GREEDY_TOKENS = 8
# 10240 tokens with BOS, so the last TAIL positions all sit past 8192.
LONG_TOKENS = 10240
TAIL = 256

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("mini-infer-llama-hf-parity")

# The image of `scripts/smoke/modal_smoke.py`.
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.14.1+cu132", extra_index_url="https://download.pytorch.org/whl/cu132")
    .pip_install(
        "triton==3.8.0",
        "transformers==5.14.0",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "tiktoken>=0.7",
    )
    .pip_install("flashinfer-python==0.6.10")
    .add_local_python_source("mini_infer")
)


def _long_text(min_chars: int = 80_000) -> str:
    """Repo documents joined until the text is long enough for LONG_TOKENS.

    Runs on the local machine only: the container holds this file at
    `/root/`, without the repo around it.
    """
    root = Path(__file__).resolve().parents[2]
    paths = [root / "scripts" / "data" / "technical_passage.md"]
    paths += sorted((root / "docs" / "decisions").glob("ADR-*.md"))
    parts: list[str] = []
    for path in paths:
        parts.append(path.read_text(encoding="utf-8"))
        if sum(len(part) for part in parts) >= min_chars:
            break
    return "\n\n".join(parts)


def run_parity(
    model_name: str,
    long_text: str,
    *,
    device: str = "cuda",
    attention_backend: str = "flashinfer",
    long_tokens: int = LONG_TOKENS,
    tail: int = TAIL,
) -> None:
    """Load both models, then print one line per case as it finishes."""
    import torch
    from torch.nn import functional
    from transformers import AutoModelForCausalLM

    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.models.blocks import RotaryEmbedding
    from mini_infer.scheduler import ContinuousScheduler, Request

    def say(line: str) -> None:
        print(line, flush=True)

    if device == "cuda":
        say(f"gpu={torch.cuda.get_device_name()} | torch={torch.__version__}")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # `device_map` needs `accelerate`, which the image does not install, so
    # the reference loads on the host and then moves.
    hf = (
        AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
        .to(device)
        .eval()
    )
    runner = ModelRunner.from_pretrained(
        model_name, device=device, dtype=dtype, attention_backend=attention_backend
    )
    ours: Any = runner._model
    tokenizer = runner.tokenizer
    say(
        f"loaded {model_name} | rope_theta={ours.cfg.rope_theta} | "
        f"rope_scaling={ours.cfg.rope_scaling} | "
        f"inv_freq as loaded: {ours.rotary_emb.inv_freq.dtype}"
    )

    # The gate replay: the scheduler path, with the table that load_model leaves.
    scheduler = ContinuousScheduler(runner)
    scheduler.start()
    try:
        replay = scheduler.run(
            Request(prompt=SHORT_PROMPT, sampling_params=SamplingParams(), max_tokens=GREEDY_TOKENS)
        )
    finally:
        scheduler.stop()
    say(f"gate replay (scheduler, as_loaded): {replay.text!r}")

    short_ids = tokenizer.encode(SHORT_PROMPT)
    long_ids = tokenizer.encode(long_text)[:long_tokens]
    if len(long_ids) != long_tokens:
        raise ValueError(f"long text gives {len(long_ids)} tokens, need {long_tokens}")
    say(f"short prompt ids={short_ids} | long prompt tokens={len(long_ids)}")

    def hf_greedy(ids: list[int]) -> tuple[list[int], torch.Tensor]:
        out = hf(input_ids=torch.tensor([ids], device=device), use_cache=True)
        first = out.logits[0, -1].float()
        past = out.past_key_values
        tokens = [int(first.argmax())]
        for _ in range(GREEDY_TOKENS - 1):
            out = hf(
                input_ids=torch.tensor([[tokens[-1]]], device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            tokens.append(int(out.logits[0, -1].argmax()))
        return tokens, first

    def ours_greedy(ids: list[int]) -> tuple[list[int], torch.Tensor]:
        cache, logits = runner.prefill(ids)
        try:
            first = logits.float()
            tokens = [int(first.argmax())]
            for _ in range(GREEDY_TOKENS - 1):
                cache, logits = runner.decode(cache, tokens[-1])
                tokens.append(int(logits.argmax()))
        finally:
            cache.free()
        return tokens, first

    def ours_tail(ids: list[int]) -> torch.Tensor:
        cache = PagedKVCache(runner.block_pool)
        cache.add_request_slot()
        try:
            packed = runner.forward_step_packed(cache, ids, [0, len(ids)], [0])
            return packed[0, -tail:].float()
        finally:
            cache.free()

    def gaps(reference: torch.Tensor, got: torch.Tensor) -> str:
        top1 = (reference.argmax(-1) == got.argmax(-1)).float().mean().item()
        diff = (reference - got).abs()
        kl = functional.kl_div(
            got.log_softmax(-1), reference.log_softmax(-1), log_target=True, reduction="none"
        ).sum(-1)
        return (
            f"top1 agree {100 * top1:.1f}% | mean |gap| {diff.mean().item():.4f} | "
            f"max |gap| {diff.max().item():.3f} | mean KL {kl.mean().item():.5f}"
        )

    with torch.inference_mode():
        hf_tokens, hf_first = hf_greedy(short_ids)
        say(f"transformers short: {tokenizer.decode(hf_tokens)!r}")
        hf_tail = hf(input_ids=torch.tensor([long_ids], device=device), logits_to_keep=tail)
        hf_tail_logits = hf_tail.logits[0].float()
        del hf_tail
        say(f"transformers long: logits for the last {tail} of {len(long_ids)} positions")

        head_dim, base = ours.cfg.head_dim, ours.cfg.rope_theta
        llama3_fp32 = RotaryEmbedding(
            head_dim, base=base, llama3_scaling=ours.cfg.rope_scaling
        ).inv_freq.to(device)
        plain_fp32 = RotaryEmbedding(head_dim, base=base).inv_freq.to(device)
        tables = {
            "as_loaded": ours.rotary_emb.inv_freq.clone(),
            "llama3_fp32": llama3_fp32,
            "plain_bf16": plain_fp32.to(torch.bfloat16),
            "plain_fp32": plain_fp32,
        }
        for name, table in tables.items():
            ours.rotary_emb.inv_freq = table
            tokens, first = ours_greedy(short_ids)
            say(
                f"{name:<12} short: {tokenizer.decode(tokens)!r} | "
                f"same text as transformers: {tokens == hf_tokens} | "
                f"{gaps(hf_first[None], first[None])}"
            )
            say(f"{name:<12} long:  {gaps(hf_tail_logits, ours_tail(long_ids))}")


@app.function(
    image=image,
    gpu=_GPU,
    timeout=_TIMEOUT_S,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def parity(model_name: str, long_text: str) -> None:
    run_parity(model_name, long_text)


@app.local_entrypoint()
def main() -> None:
    parity.remote(_MODEL, _long_text())
