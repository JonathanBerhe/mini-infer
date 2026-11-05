"""Qwen3.8-27B prefill on one GPU: the scalar-gate `gdn_chunkwise` against `kda_chunkwise`.

The Gated DeltaNet layers used to run their prefill through the per-channel
`kda_chunkwise`, which loops over the 64 positions of every chunk and
launches about 20k small kernels per layer for 4096 tokens. `gdn_chunkwise`
computes the same chunk algorithm for a per-head scalar gate with a few
kernels per chunk. The unit tests pin the two to fp32 round-off on small
shapes. This script measures the real 27B in bf16:

  1. Warm both forms up on a short prompt.
  2. For each length, prefill a prompt of real prose in 4096-token chunks,
     first with the new form, then with the old one, and print a row as soon
     as both finish: seconds for each, and the logits cosine and top-1
     agreement at 16 positions of the last chunk.

A length that would not finish before the timeout is skipped.

The prose is War and Peace (Project Gutenberg eBook 2600), kept in the
`mini-infer-hf-cache` volume. The function downloads it once if it is
missing.

Run with:
    uv run modal run scripts/qwen/modal_qwen38_prefill_ab.py
Dry run on a tiny random model, on this machine:
    uv run python scripts/qwen/modal_qwen38_prefill_ab.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("mini-infer-qwen38-prefill-ab")

MODEL_NAME = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja"]
PROSE_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
PROSE_PATH = Path("/hf-cache/prose/war-and-peace.txt")

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "H100")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "900"))
# Time kept free at the end of the cap for the report to reach the client.
_END_MARGIN_S = 60

hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0+cu128", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=5.14,<5.15", "safetensors", "huggingface_hub[hf_xet]")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
)

LENGTHS = (4096, 32768)
PREFILL_CHUNK = 4096
COMPARED_POSITIONS = 16
# Measured on an H100 with the old form: about 2.8 ms per prefill token. The
# new form is faster, so this bounds both runs of one length.
OLD_S_PER_TOKEN = 0.0035


def _log(message: str) -> None:
    """Print and flush, so that the Modal log shows where a slow step is."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _old_form(
    q: Any, k: Any, v: Any, g: Any, beta: Any, initial_state: Any = None
) -> tuple[Any, Any]:
    """The previous prefill path: broadcast the scalar gate, run the per-channel form."""
    from mini_infer.models.blocks.kda import kda_chunkwise
    from mini_infer.models.qwen.gdn import broadcast_gate

    gate = broadcast_gate(g, q.shape[-1])
    return kda_chunkwise(q, k, v, gate, beta, initial_state=initial_state)


def _prefill(model: Any, ids: list[int], *, old: bool) -> tuple[float, Any]:
    """Seconds for a chunked prefill, and the logits at the compared positions."""
    import torch

    from mini_infer.device import is_cuda_device
    from mini_infer.models.qwen import qwen3_5

    weight = model.lm_head.weight
    new_form = qwen3_5.gdn_chunkwise
    if old:
        qwen3_5.gdn_chunkwise = _old_form  # type: ignore[assignment]
    try:
        cache = model.build_state_cache(
            max_seq_len=len(ids), batch_size=1, device=weight.device, dtype=weight.dtype
        )
        prompt = torch.tensor([ids], device=weight.device)
        if is_cuda_device(weight.device):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(ids), PREFILL_CHUNK):
                logits = model.forward_prefill_with_cache(
                    prompt[:, offset : offset + PREFILL_CHUNK], state_cache=cache
                )
                cache.advance_start_pos(logits.shape[1])
        if is_cuda_device(weight.device):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    finally:
        qwen3_5.gdn_chunkwise = new_form
    span = logits.shape[1]
    picks = torch.linspace(0, span - 1, min(COMPARED_POSITIONS, span)).round().long()
    return seconds, logits[0, picks.to(logits.device)].float().cpu()


def run_ab(model: Any, ids: list[int], *, lengths: tuple[int, ...], deadline: float) -> str:
    """Rows of new against old prefill, one per length."""
    from torch.nn import functional

    lines: list[str] = []

    def emit(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    for old in (False, True):
        _prefill(model, ids[:256], old=old)  # warm-up: allocator, cuBLAS heuristics
    emit("tokens | new s | old s | speedup | min logits cos | top-1 same")
    for length in lengths:
        if time.monotonic() + 1.3 * OLD_S_PER_TOKEN * length > deadline:
            emit(f"{length} | skipped: not enough time left before the cap")
            continue
        prompt = ids[:length]
        _log(f"new form: {length} tokens")
        new_s, new_logits = _prefill(model, prompt, old=False)
        _log(f"old form: {length} tokens")
        old_s, old_logits = _prefill(model, prompt, old=True)
        cosine = functional.cosine_similarity(new_logits, old_logits, dim=-1)
        same = (new_logits.argmax(-1) == old_logits.argmax(-1)).sum()
        emit(
            f"{length} | {new_s:.1f} | {old_s:.1f} | {old_s / new_s:.1f}x | "
            f"{float(cosine.min()):.5f} | {int(same)}/{len(cosine)}"
        )
    return "\n".join(lines)


def _prose() -> str:
    """The novel from the volume, fetched once if it is missing."""
    import urllib.request

    if not PROSE_PATH.exists():
        request = urllib.request.Request(PROSE_URL, headers={"User-Agent": "mini-infer-bench"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body: bytes = response.read()
        PROSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROSE_PATH.write_bytes(body)
        hf_cache.commit()
    text = PROSE_PATH.read_text()
    return text[text.find("*** START OF") :]


@app.function(
    image=image,
    gpu=_TEST_GPU,
    memory=32768,
    timeout=_TEST_TIMEOUT,
    volumes={"/hf-cache": hf_cache},
)
def prefill_ab() -> str:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from mini_infer.models.qwen.qwen3_5 import Qwen35ForConditionalGeneration

    deadline = time.monotonic() + _TEST_TIMEOUT - _END_MARGIN_S
    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    header = (
        f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"model={MODEL_NAME}@{REVISION[:7]}"
    )
    print(header, flush=True)
    try:
        model_dir = snapshot_download(
            MODEL_NAME, revision=REVISION, allow_patterns=ALLOW_PATTERNS, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            "weights are not in the volume; run modal_qwen38_smoke.py::download"
        ) from exc
    ids = AutoTokenizer.from_pretrained(model_dir).encode(_prose())
    _log("load the 27B")
    model = Qwen35ForConditionalGeneration.from_checkpoint(
        model_dir, device="cuda", dtype=torch.bfloat16
    )
    _log(f"loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU")
    report = run_ab(model, ids, lengths=LENGTHS, deadline=deadline)
    return f"{header}\n{report}"


@app.local_entrypoint()
def main() -> None:
    print(prefill_ab.remote())


if __name__ == "__main__":
    import torch

    from mini_infer.models.qwen.qwen3_5 import (
        FULL_ATTENTION,
        Qwen35Config,
        Qwen35ForConditionalGeneration,
    )

    torch.manual_seed(0)
    tiny = Qwen35ForConditionalGeneration(
        Qwen35Config(
            vocab_size=1024,
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
    ).eval()
    print(
        run_ab(
            tiny,
            [i % 1024 for i in range(9000)],
            lengths=(1000, 8200),
            deadline=time.monotonic() + 600,
        )
    )
