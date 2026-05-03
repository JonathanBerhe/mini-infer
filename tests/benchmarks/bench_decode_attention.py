"""Decode-step latency benchmark: paged kernel vs materialization.

Hardware-agnostic harness; runs whatever paths the loaded runner supports.
Outputs a structured dict of timings; callers (local script, Modal script) decide
how to render or persist the numbers.

Not a pytest target.
"""

import statistics
import time
from typing import Any

import torch

from mini_infer.engine.model_runner import ModelRunner


def _bench_one_path(
    runner: ModelRunner,
    initial_seq_len: int,
    n_iters: int,
    warmup: int,
) -> dict[str, float]:
    """Time `n_iters` decode steps after a prefill that gets us to ~initial_seq_len."""
    # Build a synthetic prompt of approximately the target length, using simple
    # English so the tokenizer produces a roughly predictable number of tokens.
    repeated = "the quick brown fox jumps over the lazy dog "
    target_chars = initial_seq_len * 5  # rough chars-per-token estimate
    prompt = (repeated * (target_chars // len(repeated) + 1))[:target_chars]

    prompt_ids = runner.tokenizer.encode(prompt)
    cache, logits = runner.prefill(prompt_ids)

    is_cuda = runner.device == "cuda"
    try:
        # Warmup
        for _ in range(warmup):
            next_token = int(torch.argmax(logits).item())
            cache, logits = runner.decode(cache, next_token)

        latencies_us: list[float] = []
        for _ in range(n_iters):
            next_token = int(torch.argmax(logits).item())
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            cache, logits = runner.decode(cache, next_token)
            if is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_us.append((end - start) * 1_000_000.0)
    finally:
        cache.free()

    sorted_lat = sorted(latencies_us)
    return {
        "n_iters": float(n_iters),
        "median_us": statistics.median(sorted_lat),
        "mean_us": statistics.fmean(sorted_lat),
        "p99_us": sorted_lat[int(0.99 * len(sorted_lat))],
        "min_us": sorted_lat[0],
        "max_us": sorted_lat[-1],
    }


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "auto",
    seq_lens: tuple[int, ...] = (16, 64, 256, 1024),
    n_iters: int = 50,
    warmup: int = 5,
) -> dict[str, Any]:
    """Run kernel + materialization paths across seq_lens; return structured numbers."""
    results: dict[str, Any] = {
        "model": model_name,
        "n_iters": n_iters,
        "warmup": warmup,
        "paths": {},
    }

    # Single attention path now: owned model always uses the packed kernel.
    # The historical "materialization vs kernel" A/B disappeared with the
    # owned-model rewrite (Phase 1 of multi-model support); the materialized
    # path is the device-fallback inside `packed_attention_forward` and isn't
    # reachable separately from the model runner.
    runner = ModelRunner.from_pretrained(model_name, device=device)
    results["device"] = runner.device
    results["dtype"] = str(next(runner._model.parameters()).dtype)
    kernel_active = runner.device == "cuda"
    results["paths"]["kernel"] = {
        str(seq_len): _bench_one_path(runner, seq_len, n_iters, warmup) for seq_len in seq_lens
    }
    results["paths"]["kernel"]["_active"] = kernel_active  # type: ignore[assignment]

    return results
