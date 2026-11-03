"""Qwen3.8-27B with Latent Sparse Attention against dense attention, on one GPU.

The unit tests pin LSA's key selection to the authors' reference and its
attention to a masked dense oracle, on tiny random models. This script runs
the real checkpoint with the authors' distilled maps (release
`d-lsa-weights-v1` of `Prefrontal-AI/latent-sparse-attention`).

For each context length and needle depth it builds a needle-in-a-haystack
prompt from a public-domain novel, then runs it twice through the cached
path: dense attention first, then LSA right after. Each run prefills in
chunks and decodes greedily. A prompt's row is printed as soon as both runs
finish, and a prompt that would not finish before the timeout is skipped, so
a slow run still returns what it measured. Per prompt it reports:

  - whether each run's answer holds the needle,
  - the cosine of the two runs' last-position logits, and whether top-1 agrees,
  - how many greedy tokens agree,
  - prefill and decode wall time.

The numbers are reported, not asserted: LSA reads a few thousand keys out of
tens of thousands, so it is not expected to match dense exactly. The run
fails on a crash or on non-finite logits. A case that runs out of GPU memory
is reported and the run goes on.

Before the 27B loads, the same comparison runs on a tiny random model on the
GPU, so a device bug fails in seconds.

The weights, the maps and the novel live in a Modal Volume. Fill them once
with the CPU-only `download` function, so that the GPU never waits on a
download.

Run with:
    uv run modal run scripts/qwen/modal_qwen38_lsa_gate.py::download
    uv run modal run scripts/qwen/modal_qwen38_lsa_gate.py
Dry run on a tiny random model with random maps, on this machine (needs the
`Qwen/Qwen3.5-4B` tokenizer in the local Hugging Face cache):
    uv run python scripts/qwen/modal_qwen38_lsa_gate.py
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("mini-infer-qwen38-lsa-gate")

MODEL_NAME = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja"]
MAPS_URL = (
    "https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/"
    "d-lsa-weights-v1/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"
)
# From the release's SHA256SUMS.
MAPS_SHA256 = "24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b"
# War and Peace, Project Gutenberg eBook 2600: about 3 MB of real prose.
HAYSTACK_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
ASSET_DIR = "/hf-cache/lsa-gate"

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "H100")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1800"))

hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=5.14,<5.15", "safetensors", "huggingface_hub[hf_xet]")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
)

# Dense attention at 128k does not fit an 80 GB H100 next to the 27B: its
# masked prefill runs out of memory at about 115k keys.
CONTEXTS = (32768, 65536)
DEPTHS = (0.25, 0.75)
# Measured on an H100 for one case, dense and LSA together: about 200 s per
# 32k tokens, nearly all of it prefill. A little over, so the guard is safe.
SECONDS_PER_TOKEN = 0.007
# Time kept free at the end of the cap for the report to reach the client.
_END_MARGIN_S = 60
NEW_TOKENS = 12
PREFILL_CHUNK = 4096
# Each LSA prefill query gathers its own key list. A larger sub-chunk than
# the default means fewer, larger gathers, which the GPU has memory for.
LSA_CHUNK_BYTES = 2 << 30
NEEDLE = "The secret code of the archive in Lisbon is {code}."
QUESTION = (
    "\n\nQuestion: What is the secret code of the archive in Lisbon?\n"
    "Answer: The secret code of the archive in Lisbon is"
)


def _log(message: str) -> None:
    """Print and flush, so that the Modal log shows where a slow step is."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _paragraphs(text: str) -> list[str]:
    """The novel's paragraphs, without the Gutenberg header and licence."""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start >= 0:
        text = text[text.find("\n", start) + 1 : end if end > start else None]
    return [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]


def _needle_prompt(
    tokenizer: Any, paragraphs: list[str], lengths: list[int], target: int, depth: float, code: str
) -> list[int]:
    """About `target` tokens of prose with the needle at `depth`, then the question."""
    budget = target - len(tokenizer.encode(QUESTION)) - 64
    used, chosen = 0, []
    for paragraph, length in zip(paragraphs, lengths, strict=True):
        if used + length > budget:
            break
        chosen.append(paragraph)
        used += length + 2
    if used < budget // 2:
        raise ValueError(f"the haystack holds about {used} tokens, too few for {target}")
    chosen.insert(round(depth * len(chosen)), NEEDLE.format(code=code))
    ids: list[int] = tokenizer.encode("\n\n".join(chosen) + QUESTION)
    return ids


def _generate(model: Any, ids: list[int], device: str, dtype: Any) -> dict[str, Any]:
    """Chunked prefill, then greedy decode. Returns the last prefill logits and the tokens."""
    import torch

    cache = model.build_state_cache(
        max_seq_len=len(ids) + NEW_TOKENS, batch_size=1, device=device, dtype=dtype
    )
    prompt = torch.tensor([ids], device=device)
    with torch.inference_mode():
        start = time.perf_counter()
        for offset in range(0, len(ids), PREFILL_CHUNK):
            logits = model.forward_prefill_with_cache(
                prompt[:, offset : offset + PREFILL_CHUNK], state_cache=cache
            )
            cache.advance_start_pos(logits.shape[1])
        last = logits[0, -1].float()
        if device == "cuda":
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - start
        tokens = [int(last.argmax())]
        start = time.perf_counter()
        for _ in range(NEW_TOKENS - 1):
            step = model.forward_decode_with_cache(
                torch.tensor([[tokens[-1]]], device=device),
                start_pos=cache.start_pos,
                state_cache=cache,
            )
            cache.advance_start_pos(1)
            tokens.append(int(step[0, -1].argmax()))
        decode_s = time.perf_counter() - start
    if not bool(torch.isfinite(last).all()):
        raise RuntimeError("non-finite prefill logits")
    return {"last": last.cpu(), "tokens": tokens, "prefill_s": prefill_s, "decode_s": decode_s}


def run_gate(
    model: Any,
    tokenizer: Any,
    haystack: str,
    maps: dict[int, Any],
    lsa: Any,
    *,
    device: str,
    dtype: Any,
    contexts: tuple[int, ...],
    depths: tuple[float, ...],
    deadline: float | None = None,
    seconds_per_token: float = 0.0,
) -> str:
    """Each needle prompt through dense attention, then at once through LSA.

    A case's row is printed as soon as both of its runs finish, so a timeout
    keeps every finished case. With `deadline` (a `time.monotonic()` value),
    a case that would end past it at `seconds_per_token` is skipped.
    """
    import torch
    from torch.nn import functional

    from mini_infer.models.qwen import qwen3_5

    paragraphs = _paragraphs(haystack)
    lengths = [len(ids) for ids in tokenizer(paragraphs)["input_ids"]]
    rng = random.Random(0)
    cases = [
        (context, depth, f"{rng.randrange(10**5, 10**6)}")
        for context in contexts
        for depth in depths
    ]
    prompts = [_needle_prompt(tokenizer, paragraphs, lengths, c, d, code) for c, d, code in cases]
    qwen3_5._LSA_CHUNK_BYTES = LSA_CHUNK_BYTES

    lines: list[str] = []

    def emit(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    emit(
        "tokens | depth | dense found | lsa found | logits cos | top-1 same | "
        "greedy same | dense prefill s | lsa prefill s | dense ms/tok | lsa ms/tok"
    )
    for (_, depth, code), ids in zip(cases, prompts, strict=True):
        if deadline is not None and time.monotonic() + seconds_per_token * len(ids) > deadline:
            emit(f"{len(ids)} | {depth:.2f} | skipped: not enough time left before the cap")
            continue
        runs: dict[str, dict[str, Any] | str] = {}
        for mode in ("dense", "lsa"):
            if mode == "lsa":
                model.enable_lsa(maps, lsa)
            else:
                model.disable_lsa()
            _log(f"{mode}: {len(ids)} tokens, needle at {depth:.0%}")
            try:
                runs[mode] = _generate(model, ids, device, dtype)
            except torch.cuda.OutOfMemoryError:
                runs[mode] = "out of memory"
                torch.cuda.empty_cache()
        dense, sparse = runs["dense"], runs["lsa"]
        if isinstance(dense, str) or isinstance(sparse, str):
            emit(f"{len(ids)} | {depth:.2f} | dense: {dense} | lsa: {sparse}")
            continue
        cosine = float(functional.cosine_similarity(dense["last"], sparse["last"], dim=0))
        same = next(
            (
                i
                for i, (a, b) in enumerate(zip(dense["tokens"], sparse["tokens"], strict=True))
                if a != b
            ),
            NEW_TOKENS,
        )
        emit(
            f"{len(ids)} | {depth:.2f} | {code in tokenizer.decode(dense['tokens'])} | "
            f"{code in tokenizer.decode(sparse['tokens'])} | {cosine:.5f} | "
            f"{int(dense['last'].argmax()) == int(sparse['last'].argmax())} | "
            f"{same}/{NEW_TOKENS} | {dense['prefill_s']:.1f} | {sparse['prefill_s']:.1f} | "
            f"{1000 * dense['decode_s'] / (NEW_TOKENS - 1):.0f} | "
            f"{1000 * sparse['decode_s'] / (NEW_TOKENS - 1):.0f}"
        )
        emit(f"    dense: {tokenizer.decode(dense['tokens'])!r}")
        emit(f"    lsa:   {tokenizer.decode(sparse['tokens'])!r}")
    model.disable_lsa()
    return "\n".join(lines)


def _tiny_gate(device: str, haystack: str, tokenizer: Any) -> str:
    """The gate on a tiny random model with random maps and small LSA sizes."""
    import torch

    from mini_infer.models.qwen.lsa import LsaConfig
    from mini_infer.models.qwen.qwen3_5 import (
        FULL_ATTENTION,
        Qwen35Config,
        Qwen35ForConditionalGeneration,
    )

    torch.manual_seed(0)
    cfg = Qwen35Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=(["linear_attention"] * 3 + [FULL_ATTENTION]) * 2,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        tie_word_embeddings=False,
        max_position_embeddings=262144,
    )
    with torch.device(device):
        model = Qwen35ForConditionalGeneration(cfg).eval()
    lsa = LsaConfig(top_k=256, top_p=64, window_size=128)
    # 2 KV heads of 32 - 8 rotary dims each.
    maps = {i: (torch.randn(48, 64), torch.randn(48, 64)) for i in (3, 7)}
    return run_gate(
        model,
        tokenizer,
        haystack,
        maps,
        lsa,
        device=device,
        dtype=torch.float32,
        contexts=(2048, 5000),
        depths=(0.5,),
    )


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={"/hf-cache": hf_cache})
def download() -> str:
    """Fetch the pinned checkpoint, the maps and the novel into the volume (CPU only)."""
    import urllib.request

    from huggingface_hub import snapshot_download

    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "mini-infer-lsa-gate"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body: bytes = response.read()
        return body

    start = time.perf_counter()
    snapshot_download(MODEL_NAME, revision=REVISION, allow_patterns=ALLOW_PATTERNS)
    Path(ASSET_DIR).mkdir(parents=True, exist_ok=True)
    maps = fetch(MAPS_URL)
    digest = hashlib.sha256(maps).hexdigest()
    if digest != MAPS_SHA256:
        raise RuntimeError(f"map file sha256 {digest} != published {MAPS_SHA256}")
    (Path(ASSET_DIR) / "qwen38_27b_maps.pt").write_bytes(maps)
    novel = fetch(HAYSTACK_URL)
    (Path(ASSET_DIR) / "haystack.txt").write_bytes(novel)
    hf_cache.commit()
    return (
        f"assets ready in {time.perf_counter() - start:.0f}s: maps {len(maps)} bytes, "
        f"novel {len(novel)} bytes sha256 {hashlib.sha256(novel).hexdigest()[:16]}"
    )


@app.function(
    image=image,
    gpu=_TEST_GPU,
    memory=32768,
    timeout=_TEST_TIMEOUT,
    volumes={"/hf-cache": hf_cache},
)
def lsa_gate() -> str:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from mini_infer.models.qwen.lsa import load_index_maps
    from mini_infer.models.qwen.qwen3_5 import Qwen35ForConditionalGeneration

    deadline = time.monotonic() + _TEST_TIMEOUT - _END_MARGIN_S
    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    try:
        model_dir = snapshot_download(
            MODEL_NAME, revision=REVISION, allow_patterns=ALLOW_PATTERNS, local_files_only=True
        )
        maps = load_index_maps(Path(ASSET_DIR) / "qwen38_27b_maps.pt")
        haystack = (Path(ASSET_DIR) / "haystack.txt").read_text()
    except Exception as exc:
        raise RuntimeError("assets are not in the volume; run the `download` function") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    _log("step 0: the gate on a tiny random model, on the GPU")
    tiny = _tiny_gate("cuda", haystack, tokenizer)
    torch.cuda.empty_cache()

    _log("step 1: load the 27B")
    model = Qwen35ForConditionalGeneration.from_checkpoint(
        model_dir, device="cuda", dtype=torch.bfloat16
    )
    _log(f"loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU")
    report = run_gate(
        model,
        tokenizer,
        haystack,
        maps,
        None,
        device="cuda",
        dtype=torch.bfloat16,
        contexts=CONTEXTS,
        depths=DEPTHS,
        deadline=deadline,
        seconds_per_token=SECONDS_PER_TOKEN,
    )
    peak = torch.cuda.max_memory_allocated() / 2**30
    header = (
        f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"model={MODEL_NAME}@{REVISION[:7]} peak={peak:.1f} GiB"
    )
    return f"{header}\n\ntiny model:\n{tiny}\n\nQwen3.8-27B:\n{report}"


@app.local_entrypoint()
def main() -> None:
    print(lsa_gate.remote())


if __name__ == "__main__":
    from transformers import AutoTokenizer

    passage = Path(__file__).parent.parent / "data" / "technical_passage.md"
    # The passage is short, so the dry run repeats it. The GPU run uses the novel.
    text = "\n\n".join([passage.read_text()] * 6)
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    print(_tiny_gate("cpu", text, qwen_tokenizer))
