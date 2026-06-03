"""Latency bench for `hc_split_sinkhorn`: PyTorch baseline vs fused Triton kernel.

Runs the same input shapes through both paths, reports per-call latency
(median over N iters, after warmup) and the speedup ratio.

Two run modes:

- **Local** (default): runs whatever the current PyTorch device supports.
  On M1 / CPU the Triton path is skipped (kernel is CUDA-only) and only
  the PyTorch baseline is reported; useful for sanity-checking the bench
  itself but no speedup story.

- **Modal**: spins up a single-GPU container and runs both paths.
  Decorated as a normal Modal app; configure GPU type via the
  `MINI_INFER_BENCH_GPU` env var (default `L40S`).

The shapes swept mirror V4's per-token cost:

    hc_mult = 4              (V4-Flash's value)
    seqlen  ∈ {1, 64, 256, 1024}   (single decode → moderate prefill chunk)
    batch   = 2

For each shape, warmup with 10 iters then measure 100 iters; report the
median (more stable than mean against the occasional outlier).

Run locally:
    uv run python scripts/bench_hc_sinkhorn.py

Run on Modal (L40S, ~$0.30 for the full sweep):
    uv run modal run scripts/bench_hc_sinkhorn.py::modal_run

Run on Modal with H100 (~$1-2 for the full sweep):
    MINI_INFER_BENCH_GPU=H100 uv run modal run scripts/bench_hc_sinkhorn.py::modal_run
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any

import modal
import torch

app = modal.App("mini-infer-hc-sinkhorn-bench")
_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "L40S")

# Lean CUDA image: torch + transformers. Triton arrives pinned by the
# torch wheel itself (torch 2.5.1 requires triton==3.1.0 on linux);
# do NOT install triton separately, that risks a version skew against
# the torch runtime. No flash-attn / FlashInfer needed because this
# bench only exercises the HC Sinkhorn kernel and its PyTorch oracle.
# `transformers` is required even though the bench never loads a model:
# importing any module under `mini_infer.models` fires
# `_register_builtin_models()`, which imports the cache package, which
# imports `transformers.DynamicCache`.
# The cu124 wheel is fine for L40S / H100; B200 would need cu128 +
# torch >= 2.6 (out of scope for this microbench).
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers>=4.40")
    .add_local_python_source("mini_infer")
)


def _make_inputs(
    batch: int, seqlen: int, hc_mult: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the three Sinkhorn inputs at the requested shape + device."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(batch, seqlen, mix_hc, dtype=torch.float32, device=device, generator=g)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device, generator=g)
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device=device, generator=g)
    return mixes, hc_scale, hc_base


def _time_callable(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    """Median per-call latency in microseconds. CUDA events for GPU, perf_counter for CPU."""
    is_cuda = device.type == "cuda"
    # Warmup. Triton compiles the kernel on first call; the warmup makes
    # sure the cached binary is loaded before we start timing.
    for _ in range(warmup):
        fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize()

    samples_us: list[float] = []
    if is_cuda:
        # CUDA events: closer to real device time than CPU clock for short ops.
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    else:
        # CPU: perf_counter is fine; short ops still measurable down to a few us.
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            t1 = time.perf_counter()
            samples_us.append((t1 - t0) * 1_000_000.0)

    return statistics.median(samples_us)


def _run_parity(device: torch.device) -> None:
    """Inline parity contract, mirroring tests/unit/test_hc_sinkhorn_kernel.py.

    Runs the same checks the pytest file enforces (cos-sim > 0.999,
    elementwise tolerances, doubly-stochastic invariant, iters=1 edge)
    so a single Modal container validates correctness before timing.
    Raises AssertionError on any mismatch; the bench doesn't run on a
    kernel that fails parity.
    """
    from mini_infer.models.blocks.hc_sinkhorn_kernel import hc_split_sinkhorn_triton
    from mini_infer.models.blocks.hyper_connections import _hc_split_sinkhorn_torch

    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        a_flat = a.reshape(-1).to(torch.float64)
        b_flat = b.reshape(-1).to(torch.float64)
        return float(
            torch.dot(a_flat, b_flat)
            / (torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat))
        )

    print("parity: Triton vs PyTorch oracle")
    # First Triton call pays the JIT compile; do it on a tiny input with
    # explicit log lines on both sides so a compile-time hang and a
    # launch-time hang are distinguishable in the Modal log.
    warm_mixes, warm_scale, warm_base = _make_inputs(1, 1, 4, device, seed=0)
    print("parity: warming Triton JIT (hc=4, iters=20)...", flush=True)
    hc_split_sinkhorn_triton(
        warm_mixes, warm_scale, warm_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
    )
    torch.cuda.synchronize()
    print("parity: JIT warm, kernel launched and synced", flush=True)
    checked = 0
    # Canonical config (hc=4, iters=20) runs first so a crash on an edge
    # case is distinguishable from a crash on the common path in the log.
    for hc_mult in (4, 2, 8):
        for seqlen in (16, 1, 256):
            for sinkhorn_iters in (20, 1):
                print(f"parity: hc={hc_mult} T={seqlen} iters={sinkhorn_iters}", flush=True)
                mixes, hc_scale, hc_base = _make_inputs(
                    2, seqlen, hc_mult, device, seed=hc_mult * 1000 + seqlen
                )
                kwargs = {"hc_mult": hc_mult, "sinkhorn_iters": sinkhorn_iters, "eps": 1e-6}
                pre_t, post_t, comb_t = hc_split_sinkhorn_triton(mixes, hc_scale, hc_base, **kwargs)
                pre_r, post_r, comb_r = _hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, **kwargs)
                assert pre_t.shape == pre_r.shape
                assert post_t.shape == post_r.shape
                assert comb_t.shape == comb_r.shape
                cs_pre, cs_post, cs_comb = (
                    cos_sim(pre_t, pre_r),
                    cos_sim(post_t, post_r),
                    cos_sim(comb_t, comb_r),
                )
                assert cs_pre > 0.999, f"pre cos-sim {cs_pre} (hc={hc_mult} T={seqlen})"
                assert cs_post > 0.999, f"post cos-sim {cs_post} (hc={hc_mult} T={seqlen})"
                assert cs_comb > 0.999, f"comb cos-sim {cs_comb} (hc={hc_mult} T={seqlen})"
                torch.testing.assert_close(pre_t, pre_r, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(post_t, post_r, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(comb_t, comb_r, rtol=1e-3, atol=1e-4)
                checked += 1

    # Doubly-stochastic sanity on TEMPERED inputs, mirroring the CPU
    # test's convention (test_hyper_connections.py). Raw randn scales
    # produce near-degenerate softmax rows that converge slowly; the
    # invariant tolerance (5e-3) and input scaling (*0.5 / *0.1) match
    # the established CPU test. Kernel-vs-oracle correctness is already
    # covered by the elementwise asserts above on the harsh inputs.
    mixes = torch.randn(2, 16, (2 + 4) * 4, dtype=torch.float32, device=device) * 0.5
    hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn((2 + 4) * 4, dtype=torch.float32, device=device) * 0.1
    _, _, comb_t = hc_split_sinkhorn_triton(
        mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
    )
    row_sums = comb_t.sum(dim=-1)
    col_sums = comb_t.sum(dim=-2)
    ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, ones, atol=5e-3), (
        f"row sums diverge from 1 by max {(row_sums - 1).abs().max().item():.4e}"
    )
    assert torch.allclose(col_sums, ones, atol=5e-3), (
        f"col sums diverge from 1 by max {(col_sums - 1).abs().max().item():.4e}"
    )
    print(f"parity: OK ({checked} configurations + doubly-stochastic sanity)")
    print()


def _run_bench(
    *,
    hc_mult: int,
    seqlens: list[int],
    batch: int,
    sinkhorn_iters: int,
    warmup: int,
    iters: int,
) -> None:
    """Body of the bench. Runs locally on whatever device PyTorch sees."""
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        gpu_name = "(CPU)"

    print(f"device: {device}  ({gpu_name})")
    print(f"hc_mult={hc_mult}  batch={batch}  sinkhorn_iters={sinkhorn_iters}")
    print(f"warmup={warmup} iters  measurement={iters} iters  (median per-call us)")
    print()

    # Import here so the Modal worker doesn't try to load mini_infer
    # before the image is constructed.
    from mini_infer.models.blocks.hc_sinkhorn_kernel import (
        hc_split_sinkhorn_triton,
        supports_hc_kernel,
    )
    from mini_infer.models.blocks.hyper_connections import _hc_split_sinkhorn_torch

    can_triton = supports_hc_kernel(device, hc_mult)

    header = f"{'seqlen':>8} | {'pytorch us':>12} | {'triton us':>12} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for seqlen in seqlens:
        mixes, hc_scale, hc_base = _make_inputs(batch, seqlen, hc_mult, device, seed=seqlen)

        kwargs = {"hc_mult": hc_mult, "sinkhorn_iters": sinkhorn_iters, "eps": 1e-6}
        torch_us = _time_callable(
            _hc_split_sinkhorn_torch,
            args=(mixes, hc_scale, hc_base),
            kwargs=kwargs,
            warmup=warmup,
            iters=iters,
            device=device,
        )

        if can_triton:
            triton_us = _time_callable(
                hc_split_sinkhorn_triton,
                args=(mixes, hc_scale, hc_base),
                kwargs=kwargs,
                warmup=warmup,
                iters=iters,
                device=device,
            )
            speedup = torch_us / triton_us
            print(f"{seqlen:>8} | {torch_us:>12.1f} | {triton_us:>12.1f} | {speedup:>7.2f}x")
        else:
            print(f"{seqlen:>8} | {torch_us:>12.1f} | {'-':>12} | {'-':>8}")

    if not can_triton:
        print()
        print("Triton path skipped (kernel needs CUDA + power-of-2 hc_mult).")
        print("The speedup story requires a GPU run; see --modal mode.")


@app.function(image=image, gpu=_BENCH_GPU, timeout=600)
def _run_on_modal(
    hc_mult: int,
    seqlens: list[int],
    batch: int,
    sinkhorn_iters: int,
    warmup: int,
    iters: int,
) -> None:
    """Modal entrypoint: parity first, then bench, inside one GPU container."""
    device = torch.device("cuda", 0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()
    _run_parity(device)
    _run_bench(
        hc_mult=hc_mult,
        seqlens=seqlens,
        batch=batch,
        sinkhorn_iters=sinkhorn_iters,
        warmup=warmup,
        iters=iters,
    )


@app.local_entrypoint()
def modal_run(
    hc_mult: int = 4,
    seqlens: str = "1,64,256,1024",
    batch: int = 2,
    sinkhorn_iters: int = 20,
    warmup: int = 10,
    iters: int = 100,
) -> None:
    """`modal run scripts/bench_hc_sinkhorn.py::modal_run` entry point."""
    seqlens_parsed = [int(s) for s in seqlens.split(",")]
    _run_on_modal.remote(
        hc_mult=hc_mult,
        seqlens=seqlens_parsed,
        batch=batch,
        sinkhorn_iters=sinkhorn_iters,
        warmup=warmup,
        iters=iters,
    )


def _local_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument(
        "--seqlens",
        type=str,
        default="1,64,256,1024",
        help="Comma-separated seqlens to sweep.",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    _run_bench(
        hc_mult=args.hc_mult,
        seqlens=[int(s) for s in args.seqlens.split(",")],
        batch=args.batch,
        sinkhorn_iters=args.sinkhorn_iters,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    _local_main()
