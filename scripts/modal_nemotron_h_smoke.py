"""Nemotron 3.5 Lightning 30B-A3B on one CUDA GPU: the real checkpoint against transformers.

The unit tests cover the Nemotron-H port with synthetic parity, and
`tests/unit/test_nemotron_h_real_weights.py` runs the 4B sibling (no MoE) on
a laptop. This script covers the 30B-A3B, with its 23 MoE layers. On one GPU
it runs, in order:

0. The Nemotron-H CUDA-graph unit test on the GPU, to fail fast.
1. transformers' `NemotronHForCausalLM` on the real weights: prefill logits
   and a short greedy continuation for each prompt, then the model is freed.
   Its Mamba-2 CPU path is forced (`use_mamba_kernels=False`). In the pinned
   transformers 5.14 that path is wrong for spans longer than one chunk
   (fixed in 5.15.0, see ADR-035), so every prompt is shorter than the
   128-token chunk, and decode runs one token at a time.
2. Our port, loaded through `StateCacheGenerator.from_pretrained` (the
   batch-streaming `from_checkpoint`). Prefill logits must match the
   reference: cosine above 0.99 and our top-1 inside the reference top-5.
3. Serving through `StateCacheContinuousScheduler` with the CUDA graph: one
   request at a time, twice (must be identical, and equal to an eager pass),
   then two in flight. Greedy agreement with the reference is reported.
4. A decode benchmark: milliseconds per step at batch 1 and 4, eager against
   CUDA graph.

Run with:
    uv run modal run scripts/modal_nemotron_h_smoke.py::download
    uv run modal run scripts/modal_nemotron_h_smoke.py
Dry run on a tiny random checkpoint, on this machine (needs the Nemotron 3
Nano 4B tokenizer in the local Hugging Face cache):
    uv run --with accelerate python scripts/modal_nemotron_h_smoke.py
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any

import modal

app = modal.App("mini-infer-nemotron-h-smoke")

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.jinja"]
TOKENIZER_FOR_DRY_RUN = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "A100-80GB")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1200"))

hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

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
        "pytest",
    )
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
    .add_local_dir("tests", remote_path="/root/tests")
)

MAX_TOKENS = 32
# All shorter than the 128-token Mamba chunk (see step 1 above).
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain in one sentence why the sky looks blue during the day.",
    "A paged key-value cache splits the attention memory of every request into "
    "fixed-size blocks. A block table maps the logical positions of a sequence "
    "to physical blocks. State-space layers do not need this structure, because "
    "their state has a fixed size. In a hybrid model, the cache must therefore hold",
]


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _reference_pass(model_dir: str, device: str, dtype: Any, eos: int | None) -> list[dict]:
    """Prefill logits and greedy tokens from transformers, one prompt at a time.

    The prompt ids come from mini-infer's tokenizer, so both sides see the
    same tokens: it prepends BOS, which transformers omits for this
    tokenizer (`add_bos_token` is false).
    """
    import torch
    from transformers import NemotronHForCausalLM

    from mini_infer.engine.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model_dir)
    start = time.perf_counter()
    hf = NemotronHForCausalLM.from_pretrained(
        model_dir, dtype=dtype, device_map=device, use_mamba_kernels=False
    ).eval()
    _log(f"reference loaded in {time.perf_counter() - start:.0f}s")
    out = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        if len(ids) >= hf.config.chunk_size:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds one Mamba chunk")
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


def _serve(
    generator: Any, *, max_batch_size: int, cuda_graph: bool = False
) -> tuple[list[Any], float]:
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import Request, StateCacheContinuousScheduler

    scheduler = StateCacheContinuousScheduler(
        generator, max_batch_size=max_batch_size, max_seq_len=512, cuda_graph=cuda_graph
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


def _decode_bench(
    model: Any, device: str, dtype: Any, *, batch_size: int, cuda_graph: bool
) -> tuple[float, list[list[int]]]:
    """Milliseconds per greedy decode step at a full batch; also the tokens."""
    import torch

    from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph

    prompt_len, warmup, steps = 64, 4, 48
    cache = model.build_state_cache(
        max_seq_len=512, batch_size=batch_size, device=device, dtype=dtype
    )
    # Capture first: warmup and capture write into every row of the cache.
    graph = (
        DecodeCudaGraph(model, cache, batch_size=batch_size, device=device) if cuda_graph else None
    )
    torch.manual_seed(0)
    for row in range(batch_size):
        single = model.build_state_cache(max_seq_len=512, batch_size=1, device=device, dtype=dtype)
        ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
        with torch.inference_mode():
            model.forward_prefill_with_cache(ids, state_cache=single)
        cache.copy_row_from(single, src_row=0, dst_row=row)
    tokens = torch.zeros(batch_size, 1, dtype=torch.long)
    positions = torch.full((batch_size,), prompt_len, dtype=torch.long)
    generated: list[list[int]] = [[] for _ in range(batch_size)]

    def step() -> None:
        if graph is not None:
            logits = graph.replay(tokens, positions)
        else:
            with torch.inference_mode():
                logits = model.forward_decode_with_cache_ragged(
                    tokens.to(device), positions=positions.to(device), state_cache=cache
                )
        chosen = logits[:, -1].argmax(-1).tolist()
        for row, token in enumerate(chosen):
            generated[row].append(token)
            tokens[row, 0] = token
        positions.add_(1)

    for _ in range(warmup):
        step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / steps, generated


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def run_checks(
    model_dir: str,
    device: str,
    dtype: Any,
    *,
    expect_coherent: bool,
    cuda_graph: bool,
) -> str:
    """Run the steps; raise AssertionError with the full report on failure."""
    import torch

    from mini_infer.engine.state_cache_generator import StateCacheGenerator
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.nemotron_h import NemotronHForCausalLM

    failures: list[str] = []
    lines: list[str] = []

    _log("step 1: transformers reference")
    eos = Tokenizer.from_pretrained(model_dir).eos_token_id
    reference = _reference_pass(model_dir, device, dtype, eos)

    _log("step 2: our port")
    start = time.perf_counter()
    generator = StateCacheGenerator.from_pretrained(model_dir, device=device, dtype=dtype)
    load_s = time.perf_counter() - start
    model = generator.model
    assert isinstance(model, NemotronHForCausalLM)
    kinds = model.cfg.layer_types
    lines.append(
        f"load: {load_s:.0f}s via from_checkpoint, layers={len(kinds)} "
        f"(mamba={kinds.count('mamba')}, moe={kinds.count('moe')}, "
        f"attention={kinds.count('attention')}, mlp={kinds.count('mlp')})"
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
        cos = float(
            torch.nn.functional.cosine_similarity(
                ours.double().flatten(), ref["logits"].double().flatten(), dim=0
            )
        )
        max_abs = float((ours - ref["logits"]).abs().max())
        ours_top1 = int(ours[-1].argmax())
        ref_top5 = ref["logits"][-1].topk(5).indices.tolist()
        agree = float((ours.argmax(-1) == ref["logits"].argmax(-1)).float().mean())
        lines.append(
            f"prefill {len(ids):3d} tok: cos={cos:.6f} max_abs={max_abs:.3f} "
            f"last_top1_equal={ours_top1 == ref_top5[0]} argmax_agree={agree:.2f}"
        )
        if cos <= 0.99:
            failures.append(f"prefill cosine {cos:.6f} <= 0.99 ({len(ids)} tok)")
        if ours_top1 not in ref_top5:
            failures.append(f"last-token top-1 {ours_top1} not in reference top-5 ({len(ids)} tok)")

    _log(f"step 3: serving (cuda_graph={cuda_graph})")
    tokenizer = generator.tokenizer
    sequential, seq_s = _serve(generator, max_batch_size=1, cuda_graph=cuda_graph)
    repeat, _ = _serve(generator, max_batch_size=1, cuda_graph=cuda_graph)
    concurrent, conc_s = _serve(generator, max_batch_size=2, cuda_graph=cuda_graph)
    eager, eager_s = _serve(generator, max_batch_size=1)
    count = lambda results: sum(len(r.tokens) for r in results)  # noqa: E731
    lines.append(
        f"serving (cuda_graph={cuda_graph}): batch 1 {count(sequential) / seq_s:.1f} tok/s, "
        f"batch 2 {count(concurrent) / conc_s:.1f} tok/s; eager batch 1 "
        f"{count(eager) / eager_s:.1f} tok/s (end to end, prefill included)"
    )
    for prompt, ref, first, again, both, plain in zip(
        PROMPTS, reference, sequential, repeat, concurrent, eager, strict=True
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
        if first.tokens != plain.tokens:
            failures.append(
                f"graph tokens differ from eager after "
                f"{_common_prefix(first.tokens, plain.tokens)} tokens"
            )
        for result in (first, both):
            if result.finish_reason not in {"stop", "length"}:
                failures.append(f"request ended with {result.finish_reason!r}")
    if expect_coherent and "Paris" not in sequential[0].text:
        failures.append(f"no 'Paris' in {sequential[0].text!r}")

    _log("step 4: decode benchmark")
    for batch_size in (1, 4):
        eager_ms, eager_out = _decode_bench(
            model, device, dtype, batch_size=batch_size, cuda_graph=False
        )
        line = f"decode batch {batch_size}: eager {eager_ms:.1f} ms/step"
        if cuda_graph:
            graph_ms, graph_out = _decode_bench(
                model, device, dtype, batch_size=batch_size, cuda_graph=True
            )
            line += (
                f", cuda graph {graph_ms:.1f} ms/step ({eager_ms / graph_ms:.2f}x), "
                f"tokens equal: {graph_out == eager_out}"
            )
            if graph_out != eager_out:
                failures.append(f"decode bench batch {batch_size}: graph tokens differ")
        lines.append(line)

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
def nemotron_h_smoke() -> str:
    import subprocess
    import sys

    import torch
    from huggingface_hub import snapshot_download

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    _log("step 0: Nemotron-H CUDA-graph unit test")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/tests/unit/test_models_nemotron_h.py",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
        ],
        cwd="/root",
        check=True,
    )
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
    report = run_checks(model_dir, "cuda", torch.bfloat16, expect_coherent=True, cuda_graph=True)
    return header + "\n" + report


@app.local_entrypoint()
def main() -> None:
    print(nemotron_h_smoke.remote())


def _tiny_checkpoint() -> str:
    """A tiny random Nemotron-H in the released layout, with a real tokenizer."""
    import tempfile

    import torch
    import transformers
    from transformers import AutoTokenizer, NemotronHForCausalLM

    torch.manual_seed(0)
    config = transformers.NemotronHConfig(
        vocab_size=131072,
        hidden_size=64,
        layers_block_type=["mamba", "moe", "attention", "mamba", "moe"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=8,
        mamba_head_dim=8,
        ssm_state_size=16,
        n_groups=2,
        chunk_size=128,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=40,
        num_nextn_predict_layers=0,
        use_mamba_kernels=False,
    )
    config.architectures = ["NemotronHForCausalLM"]
    out = tempfile.mkdtemp(prefix="nemotron-h-tiny-")
    NemotronHForCausalLM(config).eval().save_pretrained(out)
    AutoTokenizer.from_pretrained(TOKENIZER_FOR_DRY_RUN).save_pretrained(out)
    return out


if __name__ == "__main__":
    import torch

    print(
        run_checks(
            _tiny_checkpoint(), "cpu", torch.float32, expect_coherent=False, cuda_graph=False
        )
    )
