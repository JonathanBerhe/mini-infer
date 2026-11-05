"""Qwen3.8-27B prefill and decode speed with and without Latent Sparse Attention, on one GPU.

`modal_qwen38_lsa_gate.py` showed that LSA keeps the 27B's answers at 32k
and 64k tokens. This script measures what it costs and saves, with dense
attention on its fair path: `flash_attn_with_kvcache`, which reads each row
only up to its length.

Step 0 runs on tiny random models on the GPU, so a broken path fails before
the 27B loads:

  1. The CUDA-graph and fused Gated DeltaNet unit tests, on this torch.
  2. The FlashAttention path against the PyTorch path, in bf16, over a
     chunked prefill and decode steps: cosine and argmax agreement.
  3. LSA decode from a CUDA graph (with `after_decode_step` clustering
     after each replay) against eager LSA decode, across closed blocks.

Then, for each context length, dense first and LSA right after:

  - prefill a prompt of real prose in 4096-token chunks and time it.
  - decode at batch 1 and 4, eager and from a CUDA graph: milliseconds per
    step, with one host sync per step for sampling, as in serving.
  - at the longest context only, prefill 512 more tokens under
    `torch.profiler` and print the top ops by GPU and by CPU time. It runs
    after the timed rows, so the profiler's overhead is not in them.

Each row prints as soon as it is measured, and a context that would not
finish before the timeout is skipped.

The newest prebuilt flash-attn 2 wheels stop at torch 2.8, so the image pins
torch 2.8.0. The weights and the novel come from the volume that
`modal_qwen38_lsa_gate.py::download` filled.

Run with:
    uv run modal run scripts/qwen/modal_qwen38_lsa_speed.py
Dry run on tiny random models, on this machine (no FlashAttention and no
CUDA graph here, so only the eager rows run):
    uv run python scripts/qwen/modal_qwen38_lsa_speed.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("mini-infer-qwen38-lsa-speed")

MODEL_NAME = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja"]
ASSET_DIR = "/hf-cache/lsa-gate"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/"
    "flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "H100")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1500"))
# Time kept free at the end of the cap for the report to reach the client.
_END_MARGIN_S = 60

hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install("transformers>=5.14,<5.15", "safetensors", "huggingface_hub[hf_xet]", "pytest")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
    .add_local_dir("tests", remote_path="/root/tests")
)

CONTEXTS = (32768, 65536)
PREFILL_CHUNK = 4096
# Each LSA prefill query gathers its own key list. A larger sub-chunk than
# the default means fewer, larger gathers, which the GPU has memory for.
LSA_CHUNK_BYTES = 2 << 30
BATCHES = (1, 4)
WARMUP_STEPS, TIMED_STEPS = 4, 32
# Measured on an H100 in the gate run, with PyTorch dense attention: prefill
# took about 2.9 ms (dense) and 3.8 ms (LSA) per token. A little over, plus
# a minute for the decode rows, so the guard is safe.
PREFILL_S_PER_TOKEN = {"dense": 0.0035, "lsa": 0.0045}
DECODE_ROWS_S = 60
# The profiled span. A full 4096-token chunk of the 27B launches about 1M
# kernels, and building its table took about 10 minutes on an H100.
PROFILE_TOKENS = 512
PROFILE_S = 180


def _log(message: str) -> None:
    """Print and flush, so that the Modal log shows where a slow step is."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _tiny_model(device: str, dtype: Any, vocab_size: int) -> Any:
    """A tiny random Qwen hybrid: two full-attention layers among eight."""
    import torch

    from mini_infer.models.qwen.qwen3_5 import (
        FULL_ATTENTION,
        Qwen35Config,
        Qwen35ForConditionalGeneration,
    )

    torch.manual_seed(0)
    cfg = Qwen35Config(
        vocab_size=vocab_size,
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
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Qwen35ForConditionalGeneration(cfg).eval()
    finally:
        torch.set_default_dtype(previous)
    return model


def _tiny_lsa(model: Any) -> None:
    """Random maps and small sizes, so a few hundred tokens close blocks."""
    import torch

    from mini_infer.models.qwen.lsa import LsaConfig

    generator = torch.Generator().manual_seed(1)
    # 2 KV heads of 32 - 8 rotary dims each.
    maps = {
        i: (torch.randn(48, 64, generator=generator), torch.randn(48, 64, generator=generator))
        for i in (3, 7)
    }
    model.enable_lsa(maps, LsaConfig(window_size=64, top_k=128, top_p=16))


def _flash_against_pytorch() -> str:
    """Step 0.2: logits of the FlashAttention path against the PyTorch path."""
    import torch
    from torch.nn import functional

    from mini_infer.cache.contiguous_attention import supports_contiguous_kernel
    from mini_infer.models.qwen import qwen3_5

    model = _tiny_model("cuda", torch.bfloat16, 1024)
    ids = torch.randint(0, 1024, (2, 340), generator=torch.Generator().manual_seed(2)).cuda()

    def run() -> torch.Tensor:
        cache = model.build_state_cache(
            max_seq_len=400, batch_size=2, device="cuda", dtype=torch.bfloat16
        )
        logits = []
        with torch.inference_mode():
            for start in range(0, 300, 128):
                chunk = ids[:, start : min(start + 128, 300)]
                logits.append(model.forward_prefill_with_cache(chunk, state_cache=cache))
                cache.advance_start_pos(chunk.shape[1])
            for pos in range(300, 340):
                logits.append(
                    model.forward_decode_with_cache(
                        ids[:, pos : pos + 1], start_pos=pos, state_cache=cache
                    )
                )
                cache.advance_start_pos(1)
        if not supports_contiguous_kernel(cache.layer(3).k):
            raise RuntimeError("the FlashAttention path is not active; is flash-attn installed?")
        return torch.cat(logits, dim=1).float()

    flash = run()
    real = qwen3_5.supports_contiguous_kernel
    qwen3_5.supports_contiguous_kernel = lambda k_buffer: False  # type: ignore[assignment]
    try:
        reference = run()
    finally:
        qwen3_5.supports_contiguous_kernel = real
    cosine = float(functional.cosine_similarity(flash.flatten(), reference.flatten(), dim=0))
    agree = float((flash.argmax(-1) == reference.argmax(-1)).float().mean())
    if cosine < 0.999 or agree < 0.95:
        raise RuntimeError(f"FlashAttention path diverges: cosine {cosine:.5f}, argmax {agree:.3f}")
    return f"flash vs pytorch: logits cosine {cosine:.6f}, argmax agreement {agree:.3f}"


def _lsa_graph_against_eager() -> str:
    """Step 0.3: LSA decode from a CUDA graph against eager LSA decode."""
    import torch
    from torch.nn import functional

    from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph

    model = _tiny_model("cuda", torch.bfloat16, 1024)
    _tiny_lsa(model)
    generator = torch.Generator().manual_seed(3)
    lengths = (300, 251)
    steps = 160
    caches = [
        model.build_state_cache(max_seq_len=512, batch_size=2, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    # Capture first: warmup and capture write into every row of the cache.
    graph = DecodeCudaGraph(model, caches[1], batch_size=2, device="cuda")
    for row, length in enumerate(lengths):
        single = model.build_state_cache(
            max_seq_len=512, batch_size=1, device="cuda", dtype=torch.bfloat16
        )
        prompt = torch.randint(0, 1024, (1, length), generator=generator).cuda()
        with torch.inference_mode():
            model.forward_prefill_with_cache(prompt, state_cache=single)
        for cache in caches:
            cache.copy_row_from(single, src_row=0, dst_row=row)

    worst, mismatched = 1.0, 0
    tokens = torch.randint(0, 1024, (steps, 2, 1), generator=generator)
    for step in range(steps):
        positions = torch.tensor([length + step for length in lengths])
        with torch.inference_mode():
            eager = model.forward_decode_with_cache_ragged(
                tokens[step].cuda(), positions=positions.cuda(), state_cache=caches[0]
            ).float()
        captured = graph.replay(tokens[step], positions).float()
        worst = min(
            worst, float(functional.cosine_similarity(eager.flatten(), captured.flatten(), dim=0))
        )
        mismatched += int((eager.argmax(-1) != captured.argmax(-1)).sum())
    if worst < 0.9999 or mismatched:
        raise RuntimeError(
            f"LSA graph decode diverges: worst cosine {worst:.6f}, {mismatched} argmax"
        )
    return f"lsa graph vs eager: {steps} steps x 2 rows, worst cosine {worst:.6f}, argmax all equal"


def _placement(model: Any) -> tuple[Any, Any]:
    """The model's device and dtype, which its caches use too."""
    weight = model.lm_head.weight
    return weight.device, weight.dtype


def _sync(device: Any) -> None:
    """Wait for the GPU, so a wall-clock time covers the work. A no-op on CPU."""
    import torch

    from mini_infer.device import is_cuda_device

    if is_cuda_device(device):
        torch.cuda.synchronize(device)


def _prefill(model: Any, ids: list[int]) -> tuple[Any, Any, float]:
    """Chunked prefill into a one-row cache.

    The cache has room for one more chunk after the decode steps, for
    `_profile_chunk`. Returns the cache, the last position's logits and the
    seconds the prefill took.
    """
    import torch

    device, dtype = _placement(model)
    cache = model.build_state_cache(
        max_seq_len=len(ids) + PREFILL_CHUNK + WARMUP_STEPS + TIMED_STEPS + 1,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    prompt = torch.tensor([ids], device=device)
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(ids), PREFILL_CHUNK):
            chunk = prompt[:, offset : offset + PREFILL_CHUNK]
            logits = model.forward_prefill_with_cache(chunk, state_cache=cache)
            cache.advance_start_pos(chunk.shape[1])
    _sync(device)
    return cache, logits[0, -1], time.perf_counter() - start


def _profile_chunk(model: Any, prefilled: Any, ids: list[int]) -> str:
    """The top ops of one more prefill chunk, continuing the prefilled request.

    It runs after the timed rows, so the profiler's overhead stays out of
    them, at the same context length plus one chunk.
    """
    import torch
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    device, _ = _placement(model)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    chunk = torch.tensor([ids], device=device)
    with torch.inference_mode(), torch_profile(activities=activities) as prof:
        model.forward_prefill_with_cache(chunk, state_cache=prefilled)
        _sync(device)
    averages = prof.key_averages()
    return (
        averages.table(sort_by="device_time_total", row_limit=12)
        + "\n"
        + averages.table(sort_by="cpu_time_total", row_limit=8)
    )


def _decode_ms(
    model: Any, prefilled: Any, first_token: int, prompt_len: int, batch: int, *, graph: bool
) -> float:
    """Milliseconds per greedy decode step, every row a copy of the prefilled request."""
    import torch

    from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph

    device, dtype = _placement(model)
    cache = model.build_state_cache(
        max_seq_len=prefilled.max_seq_len, batch_size=batch, device=device, dtype=dtype
    )
    # Capture first: warmup and capture write into every row of the cache.
    captured = DecodeCudaGraph(model, cache, batch_size=batch, device=device) if graph else None
    for row in range(batch):
        cache.copy_row_from(prefilled, src_row=0, dst_row=row)
    tokens = torch.full((batch, 1), first_token, dtype=torch.long)
    positions = torch.full((batch,), prompt_len, dtype=torch.long)

    def step() -> None:
        if captured is not None:
            logits = captured.replay(tokens, positions)
        else:
            with torch.inference_mode():
                logits = model.forward_decode_with_cache_ragged(
                    tokens.to(device), positions=positions.to(device), state_cache=cache
                )
        # One host sync per step for sampling, as in serving.
        tokens[:, 0] = torch.tensor(logits[:, -1].argmax(-1).tolist())
        positions.add_(1)

    for _ in range(WARMUP_STEPS):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    _sync(device)
    return 1000 * (time.perf_counter() - start) / TIMED_STEPS


def run_speed(
    model: Any,
    ids: list[int],
    maps: dict[int, Any],
    *,
    contexts: tuple[int, ...],
    deadline: float,
    use_graph: bool,
) -> str:
    """Prefill and decode rows, dense then LSA, for each context."""
    import torch

    from mini_infer.models.qwen import qwen3_5

    qwen3_5._LSA_CHUNK_BYTES = LSA_CHUNK_BYTES
    lines: list[str] = []

    def emit(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    emit(
        "mode | tokens | prefill s | "
        + " | ".join(
            f"b{batch} {kind} ms/step"
            for batch in BATCHES
            for kind in (("eager", "graph") if use_graph else ("eager",))
        )
    )
    for context in contexts:
        prompt = ids[:context]
        for mode in ("dense", "lsa"):
            needed = PREFILL_S_PER_TOKEN[mode] * len(prompt) + DECODE_ROWS_S
            if context == contexts[-1]:
                needed += PROFILE_S
            if time.monotonic() + needed > deadline:
                emit(f"{mode} | {len(prompt)} | skipped: not enough time left before the cap")
                continue
            if mode == "lsa":
                model.enable_lsa(maps)
            else:
                model.disable_lsa()
            _log(f"{mode}: prefill {len(prompt)} tokens")
            try:
                prefilled, last, seconds = _prefill(model, prompt)
                first = int(last.argmax())
                cells = []
                for batch in BATCHES:
                    for graph in (False, True) if use_graph else (False,):
                        _log(f"{mode}: decode b{batch} {'graph' if graph else 'eager'}")
                        ms = _decode_ms(model, prefilled, first, len(prompt), batch, graph=graph)
                        # The batch cache and its graph are gone; free their memory.
                        torch.cuda.empty_cache()
                        cells.append(f"{ms:.1f}")
                emit(f"{mode} | {len(prompt)} | {seconds:.1f} | " + " | ".join(cells))
                if context == contexts[-1]:
                    _log(f"{mode}: profile one more prefill chunk")
                    following = ids[context : context + PROFILE_TOKENS]
                    table = _profile_chunk(model, prefilled, following)
                    emit(f"{mode} prefill chunk after {len(prompt)} tokens, top ops:\n{table}")
                del prefilled
            except torch.cuda.OutOfMemoryError:
                emit(f"{mode} | {len(prompt)} | out of memory")
            torch.cuda.empty_cache()
    model.disable_lsa()
    return "\n".join(lines)


@app.function(
    image=image,
    gpu=_TEST_GPU,
    memory=32768,
    timeout=_TEST_TIMEOUT,
    volumes={"/hf-cache": hf_cache},
)
def lsa_speed() -> str:
    import subprocess
    import sys

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from mini_infer.models.qwen.lsa import load_index_maps
    from mini_infer.models.qwen.qwen3_5 import Qwen35ForConditionalGeneration

    deadline = time.monotonic() + _TEST_TIMEOUT - _END_MARGIN_S
    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    header = (
        f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI} model={MODEL_NAME}@{REVISION[:7]}"
    )
    print(header, flush=True)
    try:
        model_dir = snapshot_download(
            MODEL_NAME, revision=REVISION, allow_patterns=ALLOW_PATTERNS, local_files_only=True
        )
        maps = load_index_maps(Path(ASSET_DIR) / "qwen38_27b_maps.pt")
        haystack = (Path(ASSET_DIR) / "haystack.txt").read_text()
    except Exception as exc:
        raise RuntimeError(
            "assets are not in the volume; run modal_qwen38_lsa_gate.py::download"
        ) from exc

    _log("step 0.1: CUDA-graph and fused Gated DeltaNet unit tests")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/tests/unit/engine/test_cuda_graph_decode.py",
            "/root/tests/unit/models/qwen/test_gdn_decode_kernel.py",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
        ],
        cwd="/root",
        check=True,
    )
    _log("step 0.2: FlashAttention path against the PyTorch path")
    checks = [_flash_against_pytorch()]
    _log("step 0.3: LSA decode from a CUDA graph against eager")
    checks.append(_lsa_graph_against_eager())
    for line in checks:
        print(line, flush=True)
    torch.cuda.empty_cache()

    _log("step 1: load the 27B")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = tokenizer.encode(haystack[haystack.find("*** START OF") :])
    model = Qwen35ForConditionalGeneration.from_checkpoint(
        model_dir, device="cuda", dtype=torch.bfloat16
    )
    _log(f"loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU")
    report = run_speed(model, ids, maps, contexts=CONTEXTS, deadline=deadline, use_graph=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    return f"{header} peak={peak:.1f} GiB\n" + "\n".join(checks) + f"\n\n{report}"


@app.local_entrypoint()
def main() -> None:
    print(lsa_speed.remote())


if __name__ == "__main__":
    import torch

    # The served LSA sizes on a tiny model: below 4.6k tokens every key is
    # selected, so this checks the plumbing, not the speed.
    print(
        run_speed(
            _tiny_model("cpu", torch.float32, 1024),
            [i % 1024 for i in range(5000)],
            {i: (torch.randn(48, 64), torch.randn(48, 64)) for i in (3, 7)},
            contexts=(1500, 3000),
            deadline=time.monotonic() + 600,
            use_graph=False,
        )
    )
