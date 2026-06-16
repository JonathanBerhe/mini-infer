"""Run the uniform benchmark harness on a GPU via Modal.

The CPU run (`scripts/bench_all.py`) measures the CPU-runnable techniques and
skips the CUDA-only ones (FlashInfer attention backend, FP8 / NVFP4 KV cache)
plus shows INT8 only via its dequant fallback. This wrapper runs the SAME
registry on a real GPU so those techniques actually execute and the INT8
Triton W8A16 path shows its true speedup, all in one comparable table.

The default run uses the "torch" attention backend (materialize K/V + SDPA),
which needs neither flash-attn nor FlashInfer JIT, so it runs on a stock CUDA
image (verified green on H100 with Qwen2.5-7B). The default technique set is
the Hopper-safe one: baseline, int8_w8a16 (the Triton W8A16 kernel on CUDA, vs
the CPU dequant fallback), and prefix_cache. All run single-process on one GPU.

The FlashInfer techniques (attn_flashinfer, kv_fp8, kv_nvfp4) are opt-in via
`--techniques` and currently constrained:
  - FlashInfer JIT-compiles its kernels at first use, which is slow; budget
    generous time or precompile the kernels into the image.
  - The FP8-KV prefill kernel does not compile on Hopper (SM90) for
    head_dim=128 ("no eligible GMMA operator"); run kv_fp8 / kv_nvfp4 on
    Blackwell with a FlashInfer build that supports them.
  - A technique that hangs in JIT compilation (rather than raising) can blow
    the function timeout and lose already-completed results; a per-technique
    timeout in the harness is the follow-up that makes this robust.

Tensor parallelism and spec decode stay pending (separate drivers).

Defaults to Qwen2.5-7B (cached in the hf-cache volume), where the quant /
KV-bandwidth wins are visible; a 0.5B model undersells them.

Run:
    uv run modal run scripts/modal_bench_all.py
    uv run modal run scripts/modal_bench_all.py --concurrency 1,4,8 --max-tokens 32
    # opt into a FlashInfer technique (slow JIT; FP8/NVFP4 KV need Blackwell):
    uv run modal run scripts/modal_bench_all.py --techniques attn_flashinfer
"""

import os

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "Qwen/Qwen2.5-7B"
_GPU = "H100:1"

app = modal.App("mini-infer-bench-all")

# cu128 wheel + FlashInfer so the FlashInfer attention backend and FP8 / NVFP4
# KV techniques have their kernels available; matches the V4 smoke image.
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


@app.function(
    image=image,
    gpu=_GPU,
    # Bounds a hung run; a warm-cache sweep finishes well inside this.
    timeout=1500,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def bench(
    model: str,
    concurrency: list[int],
    max_tokens: int,
    techniques: list[str],
    attention_backend: str,
) -> str:
    """Run the harness registry on the GPU; return the formatted table + parity line."""
    from mini_infer.bench import (
        DEFAULT_PROMPTS,
        Workload,
        build_registry,
        format_table,
        parity_violations,
        run_suite,
    )

    workload = Workload(
        model=model,
        prompts=DEFAULT_PROMPTS,
        concurrency_levels=concurrency,
        max_tokens=max_tokens,
        device="cuda",
        attention_backend=attention_backend,
    )
    registry = build_registry()
    if techniques:
        wanted = set(techniques)
        registry = [technique for technique in registry if technique.name in wanted]

    results, skipped = run_suite(workload, registry)
    lines = [
        f"Workload: model={workload.model} device=cuda "
        f"concurrency={workload.concurrency_levels} max_tokens={workload.max_tokens}",
        "",
        format_table(results, skipped),
        "",
    ]
    violations = parity_violations(results)
    if violations:
        lines.append("PARITY VIOLATIONS (lossless techniques decoded different tokens; a bug):")
        lines.extend(f"  - {ref} != {other}" for ref, other in violations)
    else:
        lines.append("Parity: all lossless techniques agree (lossy quant excluded by design).")
    return "\n".join(lines)


@app.local_entrypoint()
def main(
    model: str = _MODEL_NAME,
    concurrency: str = "1,4,8",
    max_tokens: int = 32,
    # Hopper-safe default set on the torch backend; FlashInfer techniques are
    # opt-in (see the module docstring for why).
    techniques: str = "baseline,int8_w8a16,prefix_cache",
    attention_backend: str = "torch",
) -> None:
    levels = [int(part.strip()) for part in concurrency.split(",") if part.strip()]
    names = [part.strip() for part in techniques.split(",") if part.strip()]
    print(bench.remote(model, levels, max_tokens, names, attention_backend))
