"""Compressor-projection dtype probe on real DeepSeek-V4-Flash (2x B200, Modal).

Question (vllm-project/vllm#53459): vLLM emits the DeepSeek-V4 compressor GEMMs
(main attention compressor + Lightning Indexer compressor) in fp32
(`torch.mm(..., out_dtype=float32)`). Would the native bf16 output be
sufficient? The GEMM accumulates in fp32 either way; a bf16 output rounds
every projected kv / score entry once before the compression softmax and
before the indexer's top-k selection.

What this run measures, on the real checkpoint and a real long prompt, for each
requested pair of policies (one model load serves every pair):
  1. Isolated per-layer drift (main pair only): every compressor and indexer
     re-run on the same input under the other policy (compressed-entry drift,
     top-k set agreement).
  2. End-to-end drift: two prefills + greedy decodes, one per policy, compared
     at the attention-block outputs, the prompt logits (argmax agreement at
     every position), the next-token quality on the prompt itself (NLL, top-1
     accuracy) and the generated tokens.
  3. GEMM microbenchmark at the real shapes: fp32-out vs bf16-out `torch.mm`
     on B200 (latency, TFLOP/s, max error vs an fp32 reference GEMM).

Pairs (`--pairs`, comma-separated):
  fp32_vs_native       the question itself (fp32 output vs native bf16 output)
  fp32_vs_fp32         determinism floor (the same policy twice)
  fp32_vs_fp32_upcast  sensitivity floor (same fp32 math, different rounding order)

Same load path as `scripts/deepseek_v4/modal_v4_flash_generate.py` (per-rank
`from_checkpoint`, tensor parallelism over 2 ranks). Ranks must agree on the
generated tokens; the report is taken from rank 0.

`--simulate-act-quant` loads the model with the reference's FP8 / FP4 round
trips on the KV entries and the indexer (`DeepseekV4Config.simulate_act_quant`);
`--simulate-fp8-gemm-inputs` adds the FP8 rounding of the inputs to the GEMMs
whose weights are FP8 or FP4. Together they are the numerics of DeepSeek's
reference inference code, which vLLM's quantized path follows.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/deepseek_v4/modal_v4_projection_dtype_probe.py \
        --prompt-token-cap 8192 --max-new-tokens 32 \
        --out docs/benchmarks/data/v4-projection-dtype-probe.json
    HF_TOKEN=<token> uv run modal run scripts/deepseek_v4/modal_v4_projection_dtype_probe.py \
        --simulate-act-quant --simulate-fp8-gemm-inputs --skip-gemm-bench \
        --out docs/benchmarks/data/v4-projection-dtype-probe-qat.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"
_WORLD_SIZE = 2

# Real technical prose shipped into the image; concatenated in this order and
# truncated to `prompt_token_cap` tokens.
_PROMPT_FILES = (
    "docs/architectures/deepseek-v4.md",
    "docs/architectures/deepseek-v2-mla.md",
    "docs/architectures/gemma-4.md",
)
_REMOTE_PROMPT_DIR = "/root/data/prompt"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

app = modal.App("mini-infer-v4-projection-dtype-probe")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=4.40", "safetensors>=0.4", "huggingface_hub>=0.20")
    .pip_install("flashinfer-python>=0.6.10rc1")
)
for _rel in _PROMPT_FILES:
    image = image.add_local_file(str(_REPO_ROOT / _rel), f"{_REMOTE_PROMPT_DIR}/{Path(_rel).name}")
image = image.add_local_python_source("mini_infer")


def _checkpoint(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _time_mm(
    x: Any, w: Any, out_dtype: Any, *, warmup: int = 10, iters: int = 50
) -> tuple[float, Any]:
    """Mean microseconds per `torch.mm(x, w.T[, out_dtype])` and one output sample."""
    import torch

    def run() -> Any:
        if out_dtype is None:
            return torch.mm(x, w.t())
        return torch.mm(x, w.t(), out_dtype=out_dtype)

    sample_out = run()
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters, sample_out


def _gemm_microbench(device: str) -> list[dict[str, Any]]:
    """fp32-out vs bf16-out `torch.mm` at the V4-Flash compressor shapes (M=8192, K=4096)."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = (
        ("csa.compressor kv+score (coff=2, c=512)", 8192, 4096, 2048),
        ("hca.compressor kv+score (coff=1, c=512)", 8192, 4096, 1024),
        ("csa.indexer.compressor kv+score (coff=2, c=128)", 8192, 4096, 512),
    )
    rows: list[dict[str, Any]] = []
    for label, m, k, n in shapes:
        torch.manual_seed(0)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)  # (M, K)
        w = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)  # (N, K)
        reference = x.float() @ w.float().t()  # (M, N) fp32, TF32 off
        for variant, out_dtype in (("out_dtype=float32", torch.float32), ("native bf16", None)):
            try:
                micros, out = _time_mm(x, w, out_dtype)
            except (TypeError, RuntimeError, NotImplementedError) as exc:
                rows.append({"shape": label, "variant": variant, "error": repr(exc)})
                continue
            rows.append(
                {
                    "shape": label,
                    "m": m,
                    "k": k,
                    "n": n,
                    "variant": variant,
                    "out_dtype": str(out.dtype),
                    "mean_us": micros,
                    "tflops": 2.0 * m * n * k / (micros * 1e-6) / 1e12,
                    "output_mbytes": out.numel() * out.element_size() / 1e6,
                    "max_abs_err_vs_fp32_reference": float((out.float() - reference).abs().max()),
                }
            )
    return rows


def _build_prompt_ids(tokenizer: Any, prompt_token_cap: int) -> list[int]:
    text = "\n\n".join(
        Path(_REMOTE_PROMPT_DIR, Path(rel).name).read_text(encoding="utf-8")
        for rel in _PROMPT_FILES
    )
    ids: list[int] = tokenizer.encode(text)
    return ids[:prompt_token_cap]


_PAIR_NAMES = ("fp32_vs_native", "fp32_vs_fp32", "fp32_vs_fp32_upcast")


def _pair_specs() -> dict[str, tuple[Any, Any, bool]]:
    """`name -> (baseline, alternate, isolated)` for every supported pair."""
    from mini_infer.bench.v4_projection_dtype_probe import (
        FP32_OUT,
        FP32_OUT_UPCAST_GEMM,
        NATIVE_OUT,
    )

    return {
        "fp32_vs_native": (FP32_OUT, NATIVE_OUT, True),
        "fp32_vs_fp32": (FP32_OUT, FP32_OUT, False),
        "fp32_vs_fp32_upcast": (FP32_OUT, FP32_OUT_UPCAST_GEMM, False),
    }


def _parse_pairs(pairs: str) -> list[str]:
    names = [name.strip() for name in pairs.split(",") if name.strip()]
    unknown = sorted(set(names) - set(_PAIR_NAMES))
    if not names or unknown:
        raise ValueError(f"--pairs must name some of {_PAIR_NAMES}; unknown: {unknown}")
    return names


def _run_one_rank(
    rank: int,
    world_size: int,
    prompt_token_cap: int,
    max_new_tokens: int,
    pairs: list[str],
    run_gemm_bench: bool,
    config_overrides: dict[str, bool],
) -> dict[str, Any]:
    import time

    import torch
    import torch.distributed as dist

    from mini_infer.bench.v4_projection_dtype_probe import format_report, run_paired_prefill
    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.deepseek_v4.compressor import cuda_mm_fp32_out_supported
    from mini_infer.models.deepseek_v4.deepseek_v4 import DeepseekV4ForCausalLM

    _checkpoint(rank, "init_distributed (nccl)...")
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
        _checkpoint(
            rank,
            f"loading V4-Flash via from_checkpoint onto {device} "
            f"(config overrides: {config_overrides or 'none'})...",
        )
        model = DeepseekV4ForCausalLM.from_checkpoint(
            _MODEL_NAME, device=device, dtype=torch.bfloat16, config_overrides=config_overrides
        )
        model.eval()
        tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
        prompt_ids = _build_prompt_ids(tokenizer, prompt_token_cap)
        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)  # (1, T)
        _checkpoint(rank, f"prompt: {len(prompt_ids)} tokens; running pairs {pairs}...")

        specs = _pair_specs()
        reports: dict[str, Any] = {}
        report_texts: dict[str, str] = {}
        pair_seconds: dict[str, float] = {}
        for name in pairs:
            baseline, alternate, isolated = specs[name]
            start = time.perf_counter()
            report = run_paired_prefill(
                model,
                input_ids,
                baseline=baseline,
                alternate=alternate,
                isolated=isolated,
                max_new_tokens=max_new_tokens,
                position_stride=64,
                position_tail=256,
            )
            torch.cuda.synchronize(rank)
            pair_seconds[name] = time.perf_counter() - start
            reports[name] = report.to_dict()
            report_texts[name] = format_report(report)
            _checkpoint(rank, f"pair {name} done in {pair_seconds[name]:.0f}s.")

        gemm_rows: list[dict[str, Any]] = []
        if rank == 0 and run_gemm_bench:
            _checkpoint(rank, "GEMM microbenchmark...")
            gemm_rows = _gemm_microbench(device)
        return {
            "rank": rank,
            "n_prompt_tokens": len(prompt_ids),
            "config_overrides": config_overrides,
            "moe_swiglu_limit": model.cfg.moe_swiglu_limit,
            "reports": reports,
            "report_texts": report_texts,
            "pair_seconds": pair_seconds,
            "gemm_microbench": gemm_rows,
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(rank),
            # True: the fp32-output policy ran torch.mm(..., out_dtype=float32), as vLLM
            # does. False: it ran the equivalent fp32 GEMM over upcast operands.
            "mm_fp32_out_kernel": cuda_mm_fp32_out_supported(rank),
            "peak_memory_gb": torch.cuda.max_memory_allocated(rank) / 1e9,
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(
    rank: int,
    world_size: int,
    prompt_token_cap: int,
    max_new_tokens: int,
    pairs: list[str],
    run_gemm_bench: bool,
    config_overrides: dict[str, bool],
    queue: Any,
) -> None:
    try:
        result = _run_one_rank(
            rank,
            world_size,
            prompt_token_cap,
            max_new_tokens,
            pairs,
            run_gemm_bench,
            config_overrides,
        )
        queue.put(("ok", result))
    except Exception:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}"))


# Wall-clock caps. A warm load is ~3 min and the probe ~2 min; the caps only bound
# a hung run, which bills both GPUs for as long as it lasts.
_FUNCTION_TIMEOUT_S = 1200
_RESULTS_DEADLINE_S = 1080


def _collect_rank_results(processes: list[Any], queue: Any) -> list[dict[str, Any]]:
    """Gather one result per rank; fail fast on the first error or dead worker.

    A rank that raises leaves its peer blocked inside a collective, so on the
    first error (or a worker that exited without reporting, e.g. OOM-killed)
    every worker is terminated at once instead of waiting for the deadline.
    """
    import queue as queue_module
    import time

    deadline = time.monotonic() + _RESULTS_DEADLINE_S
    results: list[dict[str, Any]] = []
    try:
        while len(results) < len(processes):
            if time.monotonic() > deadline:
                raise TimeoutError(f"no result from every rank within {_RESULTS_DEADLINE_S}s")
            try:
                status, payload = queue.get(timeout=5)
            except queue_module.Empty:
                dead = [p.pid for p in processes if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"worker(s) {dead} exited without a result") from None
                continue
            if status == "err":
                raise RuntimeError(payload)
            results.append(payload)
    finally:
        for p in processes:
            if p.is_alive() and len(results) < len(processes):
                p.terminate()
            p.join(timeout=10)
    return results


@app.function(
    image=image,
    gpu=_GPU,
    timeout=_FUNCTION_TIMEOUT_S,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def probe(
    prompt_token_cap: int,
    max_new_tokens: int,
    pairs: list[str],
    run_gemm_bench: bool,
    config_overrides: dict[str, bool],
) -> dict[str, Any]:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    processes = [
        ctx.Process(
            target=_child_entry,
            args=(
                rank,
                _WORLD_SIZE,
                prompt_token_cap,
                max_new_tokens,
                pairs,
                run_gemm_bench,
                config_overrides,
                queue,
            ),
        )
        for rank in range(_WORLD_SIZE)
    ]
    for p in processes:
        p.start()
    results = _collect_rank_results(processes, queue)
    per_rank = sorted(results, key=lambda entry: entry["rank"])
    ranks_agree = all(
        entry["reports"][name]["tokens"] == per_rank[0]["reports"][name]["tokens"]
        and entry["reports"][name]["logits"]["n_argmax_flips"]
        == per_rank[0]["reports"][name]["logits"]["n_argmax_flips"]
        for entry in per_rank
        for name in pairs
    )
    return {"ranks_agree": ranks_agree, "pairs": pairs, "rank0": per_rank[0]}


@app.local_entrypoint()
def main(
    prompt_token_cap: int = 8192,
    max_new_tokens: int = 32,
    pairs: str = ",".join(_PAIR_NAMES),
    skip_gemm_bench: bool = False,
    simulate_act_quant: bool = False,
    simulate_fp8_gemm_inputs: bool = False,
    out: str = "v4_projection_dtype_probe.json",
) -> None:
    pair_names = _parse_pairs(pairs)  # fail before any GPU time is spent
    if simulate_fp8_gemm_inputs and not simulate_act_quant:
        raise ValueError("--simulate-fp8-gemm-inputs requires --simulate-act-quant")
    # The reference's FP8 / FP4 round trips (off by default in the port).
    config_overrides: dict[str, bool] = {}
    if simulate_act_quant:
        config_overrides["simulate_act_quant"] = True
    if simulate_fp8_gemm_inputs:
        config_overrides["simulate_fp8_gemm_inputs"] = True
    result = probe.remote(
        prompt_token_cap, max_new_tokens, pair_names, not skip_gemm_bench, config_overrides
    )
    rank0 = result["rank0"]
    print(f"\nranks agree (TP consistency, every pair): {result['ranks_agree']}")
    print(f"device: {rank0['device_name']}  torch: {rank0['torch_version']}")
    print(
        f"config overrides: {rank0['config_overrides'] or 'none'}  "
        f"moe_swiglu_limit: {rank0['moe_swiglu_limit']}"
    )
    print(
        f"fp32-out policy used torch.mm(out_dtype=float32): {rank0['mm_fp32_out_kernel']}  "
        f"peak memory rank 0: {rank0['peak_memory_gb']:.1f} GB"
    )
    for name in pair_names:
        print(f"\n===== {name} ({rank0['pair_seconds'][name]:.0f}s) =====")
        print(rank0["report_texts"][name])
    if rank0["gemm_microbench"]:
        print("\nGEMM microbench (B200):")
    for row in rank0["gemm_microbench"]:
        if "error" in row:
            print(f"  {row['shape']:<48} {row['variant']:<18} ERROR {row['error']}")
            continue
        print(
            f"  {row['shape']:<48} {row['variant']:<18} {row['mean_us']:>9.1f} us "
            f"{row['tflops']:>7.1f} TFLOP/s  out={row['output_mbytes']:.0f} MB  "
            f"max_err_vs_fp32={row['max_abs_err_vs_fp32_reference']:.3e}"
        )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nfull result written to {out}")
