"""Inkling on a CUDA GPU: the serving path vs HF, in fp32 and bf16.

The Inkling parity suite runs on CPU only. This script builds a tiny random
Inkling, writes it in the released checkpoint layout (HF `save_pretrained`),
loads it through `ModelRunner.from_pretrained`, and serves concurrent requests
with chunked prefill through `ContinuousScheduler`. HF loads the same files
through its own conversion and is the reference.

- fp32: prefill logits must match HF to 1e-4 and greedy tokens must be equal,
  also in a run that forces an out-of-memory preemption.
- bf16: prefill logits cosine must exceed 0.99. Greedy token agreement is
  reported, not asserted. HF computes the short convolutions from fp32
  inputs, but our cache stores the conv inputs in the pool dtype, and random
  tiny weights produce near-ties in the argmax.

The checks also run locally: `run_checks("cpu")`.

Run with:
    uv run modal run scripts/inkling/modal_inkling_smoke.py
    MINI_INFER_TEST_GPU=A10 uv run modal run scripts/inkling/modal_inkling_smoke.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import modal

app = modal.App("mini-infer-inkling-smoke")

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "B200")
# A cost cap, not an expected duration: the checks take about a minute. A hung
# container bills the whole cap, at a high rate on a B200.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "300"))

# The cu128 wheel has Blackwell (SM_100) kernels and bundles the CUDA runtime.
# Inkling pins the torch attention backend, so no flash-attn or FlashInfer.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=5.14,<5.15",
        "safetensors",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
    )
    .add_local_python_source("mini_infer")
)

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 24
PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Explain how paged attention stores keys and values in fixed-size blocks.",
    "Hello there, how are you doing today?",
    "In 1905 Einstein published four papers that changed physics, including "
    "special relativity and the photoelectric effect.",
]
# The tiny parity config of `tests/unit/models/inkling/test_inkling_parity.py`, with the
# vocabulary widened to cover the tokenizer. Init scale 0.1, because at HF's
# 0.02 the logits are so small that real bugs stay inside the tolerance.
TEXT_CONFIG: dict[str, Any] = {
    "initializer_range": 0.1,
    "vocab_size": 151936,
    "unpadded_vocab_size": 151928,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 4,
    "swa_head_dim": 16,
    "sliding_window_size": 8,
    "d_rel": 4,
    "rel_extent": 6,
    "log_scaling_n_floor": 4,
    "log_scaling_alpha": 0.1,
    "layer_types": ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse"],
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 4,
    "n_shared_experts": 2,
    "route_scale": 8.0,
    "conv_kernel_size": 4,
    "rms_norm_eps": 1e-6,
    "logits_mup_width_multiplier": 24.0,
    "bos_token_id": None,
    "eos_token_id": 151645,
}
# Patch 2, temporal patch 2, and 3 layers keep the vision tower off its scipy-only branch.
VISION_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "patch_size": 2,
    "temporal_patch_size": 2,
}
AUDIO_CONFIG = {"n_mel_bins": 4, "mel_vocab_size": 8}


def _build_checkpoint(out: Path, dtype: Any) -> None:
    """Write a tiny random Inkling in the released layout, plus a tokenizer."""
    import torch
    from transformers import AutoTokenizer
    from transformers.models.inkling.configuration_inkling import (
        InklingConfig,
        InklingTextConfig,
    )
    from transformers.models.inkling.modeling_inkling import InklingForConditionalGeneration

    torch.manual_seed(0)
    cfg = InklingConfig(
        text_config=InklingTextConfig(**TEXT_CONFIG),
        vision_config=VISION_CONFIG,
        audio_config=AUDIO_CONFIG,
    )
    model = InklingForConditionalGeneration(cfg).eval()
    # HF zeroes the router bias and sets the global scales to one at init,
    # which would hide a loader that drops them.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("e_score_correction_bias"):
                param.normal_(0.0, 0.5)
            elif name.endswith("global_scale"):
                param.uniform_(0.7, 1.3)
    model.to(dtype).save_pretrained(out)
    AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(out)


def _hf_greedy(hf: Any, ids: list[int], eos: int, device: str) -> list[int]:
    import torch

    out: list[int] = []
    past = None
    cur = torch.tensor([ids], device=device)
    with torch.inference_mode():
        for _ in range(MAX_TOKENS):
            step = hf(input_ids=cur, past_key_values=past, use_cache=True)
            past = step.past_key_values
            token = int(step.logits[0, -1].argmax())
            if token == eos:
                break
            out.append(token)
            cur = torch.tensor([[token]], device=device)
    return out


def _serve(runner: Any, *, headroom: int) -> tuple[list[Any], int]:
    """Serve PROMPTS concurrently; return the results and the OOM recovery count."""
    import logging

    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
    from mini_infer.scheduler.request_state import Request

    class _CountOom(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.WARNING)
            self.count = 0

        def emit(self, record: logging.LogRecord) -> None:
            if "engine OOM" in record.getMessage():
                self.count += 1

    counter = _CountOom()
    sched_logger = logging.getLogger("mini_infer.scheduler.continuous_scheduler")
    sched_logger.addHandler(counter)
    sched = ContinuousScheduler(
        runner, max_concurrent=4, chunk_size=5, decode_headroom_blocks=headroom
    )
    sched.start()
    try:
        handles = [
            sched.submit(Request(prompt=p, sampling_params=SamplingParams(), max_tokens=MAX_TOKENS))
            for p in PROMPTS
        ]
        results = [h.wait(timeout=300) for h in handles]
    finally:
        sched.stop()
        sched_logger.removeHandler(counter)
    return results, counter.count


def _check_dtype(device: str, dtype_name: str, failures: list[str]) -> list[str]:
    import torch
    from transformers.models.inkling.modeling_inkling import InklingForConditionalGeneration

    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.engine.model_runner import ModelRunner

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    lines = [f"[{dtype_name}]"]
    ckpt = Path(tempfile.mkdtemp(prefix=f"inkling-{dtype_name}-"))
    _build_checkpoint(ckpt, dtype)

    # HF keeps the SConv modules in fp32 (`_keep_in_fp32_modules_strict`).
    hf = InklingForConditionalGeneration.from_pretrained(
        ckpt, attn_implementation="eager", dtype=dtype
    )
    hf = hf.to(device).eval()
    runner = ModelRunner.from_pretrained(
        str(ckpt), device=device, dtype=dtype, num_blocks=256, block_size=16
    )
    eos = runner.tokenizer.eos_token_id

    # Prefill logits, one request at a time, through a fresh cache.
    for prompt in PROMPTS:
        ids = runner.tokenizer.encode(prompt)
        cache = PagedKVCache(runner.block_pool)
        cache.add_request_slot()
        ours = runner.forward_step_packed(cache, ids, [0, len(ids)], [0])[0].float()
        cache.free()
        with torch.inference_mode():
            ref = hf(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0]
        ref = ref.float()
        # float64: a float32 cosine over ~3.6M logits rounds to values above 1.
        cos = float(
            torch.nn.functional.cosine_similarity(
                ours.double().flatten(), ref.double().flatten(), dim=0
            )
        )
        max_diff = float((ours - ref).abs().max())
        argmax_equal = bool(torch.equal(ours.argmax(-1), ref.argmax(-1)))
        lines.append(
            f"  prefill {len(ids):2d} tok: cos={cos:.6f} max_abs={max_diff:.2e} "
            f"|logits|={float(ref.abs().max()):.2e} argmax_equal={argmax_equal}"
        )
        if dtype_name == "fp32" and (max_diff > 1e-4 or not argmax_equal):
            failures.append(f"fp32 prefill mismatch ({len(ids)} tok): max_abs={max_diff:.2e}")
        if dtype_name == "bf16" and cos <= 0.99:
            failures.append(f"bf16 prefill cosine {cos:.6f} <= 0.99 ({len(ids)} tok)")

    # Concurrent serving with chunked prefill, then a tight pool that forces preemption.
    runs = [("roomy", runner, 8)]
    if dtype_name == "fp32":
        tight = ModelRunner.from_pretrained(
            str(ckpt), device=device, dtype=dtype, num_blocks=9, block_size=16
        )
        runs.append(("tight", tight, 0))
    for label, run_runner, headroom in runs:
        results, oom = _serve(run_runner, headroom=headroom)
        equal = 0
        finished = 0
        for prompt, result in zip(PROMPTS, results, strict=True):
            if result.finish_reason == "cancelled":
                continue
            finished += 1
            want = _hf_greedy(hf, run_runner.tokenizer.encode(prompt), eos, device)
            same = sum(a == b for a, b in zip(result.tokens, want, strict=False))
            if result.tokens == want:
                equal += 1
            elif dtype_name == "fp32":
                failures.append(f"fp32 {label}: tokens differ from HF ({same}/{len(want)} equal)")
            lines.append(
                f"  {label} {len(run_runner.tokenizer.encode(prompt)):2d} tok: "
                f"{same}/{max(len(want), len(result.tokens))} tokens equal to HF"
            )
        lines.append(
            f"  {label}: {equal}/{finished} finished requests token-equal, "
            f"{len(PROMPTS) - finished} preempted, oom_recoveries={oom}"
        )
        if finished == 0:
            failures.append(f"{dtype_name} {label}: no request finished")
    return lines


def run_checks(device: str) -> str:
    """Run the fp32 and bf16 checks; raise AssertionError with the full report on failure."""
    failures: list[str] = []
    lines: list[str] = []
    for dtype_name in ("fp32", "bf16"):
        lines += _check_dtype(device, dtype_name, failures)
    report = "\n".join(lines)
    if failures:
        raise AssertionError(report + "\nFAILURES:\n" + "\n".join(failures))
    return report + "\nRESULT: PASS"


@app.function(image=image, gpu=_TEST_GPU, timeout=_TEST_TIMEOUT)
def inkling_smoke() -> str:
    import torch

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    header = (
        f"gpu={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
        f"torch={torch.__version__}"
    )
    return header + "\n" + run_checks("cuda")


@app.local_entrypoint()
def main() -> None:
    print(inkling_smoke.remote())
