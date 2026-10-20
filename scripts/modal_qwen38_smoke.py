"""Qwen3.8-27B on one CUDA GPU: the real checkpoint against transformers.

The unit tests cover the Qwen hybrid port with synthetic parity, and
`tests/unit/test_qwen3_5_real_weights.py` runs the 4B sibling on a laptop.
This script covers the 27B, the checkpoint with the 3x Gated DeltaNet head
ratio and an untied `lm_head`. On one GPU it runs, in order:

1. transformers' `Qwen3_5ForConditionalGeneration` on the real weights:
   prefill logits and a short greedy continuation for each prompt. Then it
   frees the model, because two copies do not fit one 80 GB GPU.
2. Our port, loaded through `StateCacheGenerator.from_pretrained` (the
   shard-streaming `from_checkpoint`). Prefill logits through the state cache
   must match the reference: cosine above 0.99 and our top-1 token inside the
   reference top-5 at the last position.
3. Serving through `StateCacheContinuousScheduler`: one request at a time,
   twice, which must give identical tokens, then two requests in flight.
   Greedy agreement with the reference and between batch sizes is reported,
   not asserted, because bf16 rounding can flip a near-tie.

The weights live in a Modal Volume. Fill it once with the CPU-only
`download` function, so that the GPU never waits on Hugging Face.

Run with:
    uv run modal run scripts/modal_qwen38_smoke.py::download
    uv run modal run scripts/modal_qwen38_smoke.py
Dry run on a tiny random checkpoint, on this machine (the transformers
`device_map` path needs accelerate):
    uv run --with accelerate python scripts/modal_qwen38_smoke.py
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any

import modal

app = modal.App("mini-infer-qwen38-smoke")

MODEL_NAME = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja"]

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "A100-80GB")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1800"))

hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

# The Qwen hybrid uses the torch attention path, so no flash-attn or FlashInfer.
# accelerate is for transformers' `device_map`, which loads the reference
# straight onto the GPU instead of through 55 GB of host memory.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=5.14,<5.15",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_xet]",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
    )
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
)

MAX_TOKENS = 32
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain in one sentence why the sky looks blue during the day.",
    # Longer than the 64-token chunk of the Gated DeltaNet prefill, so the
    # chunkwise path runs, not only the short-prompt one.
    "A paged key-value cache splits the attention memory of every request into "
    "fixed-size blocks. A block table maps the logical positions of a sequence "
    "to physical blocks, so requests of different lengths share one memory pool "
    "without fragmentation, and a shared prompt prefix can point to the same "
    "blocks. Linear attention layers do not need this structure, because their "
    "state has a fixed size. In a hybrid model, the cache must therefore hold",
]


def _log(message: str) -> None:
    """Print and flush, so that the Modal log shows where a slow step is."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _reference_pass(model_dir: str, device: str, dtype: Any, eos: int | None) -> list[dict]:
    """Prefill logits and greedy tokens from transformers, one prompt at a time."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    start = time.perf_counter()
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, device_map=device
    ).eval()
    _log(f"reference loaded in {time.perf_counter() - start:.0f}s")
    out = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt)["input_ids"]
        with torch.inference_mode():
            step = hf(input_ids=torch.tensor([ids], device=device), use_cache=True)
            logits = step.logits[0].float().cpu()
            past = step.past_key_values
            tokens: list[int] = []
            token = int(step.logits[0, -1].argmax())
            while len(tokens) < MAX_TOKENS and token != eos:
                tokens.append(token)
                step = hf(
                    input_ids=torch.tensor([[token]], device=device),
                    past_key_values=past,
                    use_cache=True,
                )
                past = step.past_key_values
                token = int(step.logits[0, -1].argmax())
        out.append({"ids": ids, "logits": logits, "tokens": tokens})
    del hf, past, step
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def _serve(generator: Any, *, max_batch_size: int) -> tuple[list[Any], float]:
    """Serve PROMPTS through the state-cache scheduler; return results and wall time."""
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import Request, StateCacheContinuousScheduler

    scheduler = StateCacheContinuousScheduler(
        generator, max_batch_size=max_batch_size, max_seq_len=512
    )
    scheduler.start()
    start = time.perf_counter()
    try:
        handles = [
            scheduler.submit(
                Request(prompt=p, sampling_params=SamplingParams(), max_tokens=MAX_TOKENS)
            )
            for p in PROMPTS
        ]
        results = [handle.wait(timeout=600) for handle in handles]
    finally:
        scheduler.stop()
    return results, time.perf_counter() - start


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def run_checks(model_dir: str, device: str, dtype: Any, *, expect_coherent: bool) -> str:
    """Run the three steps; raise AssertionError with the full report on failure."""
    import torch

    from mini_infer.engine.state_cache_generator import StateCacheGenerator
    from mini_infer.models.qwen3_5 import Qwen35ForConditionalGeneration

    failures: list[str] = []
    lines: list[str] = []

    _log("step 1: transformers reference")
    from transformers import AutoTokenizer

    eos = AutoTokenizer.from_pretrained(model_dir).eos_token_id
    reference = _reference_pass(model_dir, device, dtype, eos)

    _log("step 2: our port")
    start = time.perf_counter()
    generator = StateCacheGenerator.from_pretrained(model_dir, device=device, dtype=dtype)
    load_s = time.perf_counter() - start
    model = generator.model
    assert isinstance(model, Qwen35ForConditionalGeneration)
    lines.append(
        f"load: {load_s:.0f}s via from_checkpoint, layers={model.cfg.num_hidden_layers}, "
        f"head_ratio={model.cfg.linear_head_ratio}, tied={model.cfg.tie_word_embeddings}"
    )
    if device.startswith("cuda"):
        lines.append(f"gpu memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    for ref in reference:
        ids = ref["ids"]
        cache = model.build_state_cache(max_seq_len=512, batch_size=1, device=device, dtype=dtype)
        with torch.inference_mode():
            ours = (
                model.forward_prefill_with_cache(
                    torch.tensor([ids], device=device), state_cache=cache
                )[0]
                .float()
                .cpu()
            )
        # float64: a float32 cosine over millions of logits can round above 1.
        cos = float(
            torch.nn.functional.cosine_similarity(
                ours.double().flatten(), ref["logits"].double().flatten(), dim=0
            )
        )
        max_abs = float((ours - ref["logits"]).abs().max())
        ours_top1 = int(ours[-1].argmax())
        ref_top5 = ref["logits"][-1].topk(5).indices.tolist()
        positions_equal = float((ours.argmax(-1) == ref["logits"].argmax(-1)).float().mean())
        lines.append(
            f"prefill {len(ids):3d} tok: cos={cos:.6f} max_abs={max_abs:.3f} "
            f"last_top1_equal={ours_top1 == ref_top5[0]} argmax_agree={positions_equal:.2f}"
        )
        if cos <= 0.99:
            failures.append(f"prefill cosine {cos:.6f} <= 0.99 ({len(ids)} tok)")
        if ours_top1 not in ref_top5:
            failures.append(f"last-token top-1 {ours_top1} not in reference top-5 ({len(ids)} tok)")

    _log("step 3: serving")
    tokenizer = generator.tokenizer
    sequential, seq_s = _serve(generator, max_batch_size=1)
    repeat, _ = _serve(generator, max_batch_size=1)
    concurrent, conc_s = _serve(generator, max_batch_size=2)
    seq_tokens = sum(len(r.tokens) for r in sequential)
    conc_tokens = sum(len(r.tokens) for r in concurrent)
    lines.append(
        f"serving: batch 1 {seq_tokens / seq_s:.1f} tok/s, batch 2 {conc_tokens / conc_s:.1f} "
        "tok/s (end to end, prefill included)"
    )
    for prompt, ref, first, again, both in zip(
        PROMPTS, reference, sequential, repeat, concurrent, strict=True
    ):
        lines.append(
            f"greedy {len(ref['ids']):3d} tok: {_common_prefix(first.tokens, ref['tokens'])}/"
            f"{len(ref['tokens'])} equal to reference, batch 2 equal to batch 1: "
            f"{_common_prefix(both.tokens, first.tokens)}/{len(first.tokens)}, "
            f"finish={first.finish_reason}"
        )
        lines.append(f"  ours: {prompt[-40:]!r} -> {first.text!r}")
        lines.append(f"  ref:  {tokenizer.decode(ref['tokens'])!r}")
        if first.tokens != again.tokens:
            failures.append(f"batch-1 greedy not deterministic ({len(ref['ids'])} tok)")
        for result in (first, both):
            if result.finish_reason not in {"stop", "length"}:
                failures.append(f"request ended with {result.finish_reason!r}")
    if expect_coherent and "Paris" not in sequential[0].text:
        failures.append(f"no 'Paris' in {sequential[0].text!r}")

    report = "\n".join(lines)
    if failures:
        raise AssertionError(report + "\nFAILURES:\n" + "\n".join(failures))
    return report + "\nRESULT: PASS"


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={"/hf-cache": hf_cache})
def download() -> str:
    """Fetch the pinned checkpoint into the volume (CPU only)."""
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    path = snapshot_download(MODEL_NAME, revision=REVISION, allow_patterns=ALLOW_PATTERNS)
    hf_cache.commit()
    return f"{path} ready in {time.perf_counter() - start:.0f}s"


@app.function(
    image=image,
    gpu=_TEST_GPU,
    memory=32768,
    timeout=_TEST_TIMEOUT,
    volumes={"/hf-cache": hf_cache},
)
def qwen38_smoke() -> str:
    import torch
    from huggingface_hub import snapshot_download

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    try:
        model_dir = snapshot_download(
            MODEL_NAME, revision=REVISION, allow_patterns=ALLOW_PATTERNS, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError("weights are not in the volume; run the `download` function") from exc
    header = (
        f"gpu={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
        f"torch={torch.__version__} model={MODEL_NAME}@{REVISION[:7]}"
    )
    return header + "\n" + run_checks(model_dir, "cuda", torch.bfloat16, expect_coherent=True)


@app.local_entrypoint()
def main() -> None:
    print(qwen38_smoke.remote())


def _tiny_checkpoint() -> str:
    """A tiny random model in the released layout, with the real tokenizer."""
    import tempfile

    import torch
    from transformers import (
        AutoTokenizer,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5TextConfig,
    )

    torch.manual_seed(0)
    text = Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
    )
    vision = {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2}
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision)
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    out = tempfile.mkdtemp(prefix="qwen38-tiny-")
    Qwen3_5ForConditionalGeneration(config).eval().save_pretrained(out)
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B").save_pretrained(out)
    return out


if __name__ == "__main__":
    import torch

    print(run_checks(_tiny_checkpoint(), "cpu", torch.float32, expect_coherent=False))
