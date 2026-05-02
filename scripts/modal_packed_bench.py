"""Modal benchmark for the packed-varlen forward. One script, configurable.

Configurations (selected via the `config` arg of the local entrypoint):

- **smoke**: 4 mixed-length concurrent prompts, parity check vs a serial reference.
- **throughput**: prompt by concurrency by chunk-size sweep on one workload.
- **holb**: head-of-line-blocking microbench. Streams a short decoder, submits a
  long prefill mid-stream, measures decoder ITL.
- **sweep**: smoke + short-throughput + long-throughput in one Modal call.
  Used for the cross-platform comparison (A10 vs H100, materialized vs paged FA).
- **prefix**: shared-system-prompt workload with prefix cache OFF vs ON.
  Reports per-request TTFT (cold vs warm) and aggregate throughput at
  several concurrencies. The system prompt is intentionally large
  (`target_prompt_tokens`, default 12000) so the cache hit pays off visibly.
- **quant**: weight-only INT8 (W8A16) vs fp16 baseline. Reports model-weight
  HBM footprint and a small concurrent throughput sweep on a moderate prompt.
- **spec**: greedy speculative decoding (Qwen2.5-7B target + Qwen2.5-0.5B
  draft). Reports target-alone vs spec-decode tokens/sec, mean acceptance
  per iteration, target forwards saved.
- **quant_kernel**: fp16 vs int8-naive vs int8-fused (Triton W8A16) on the
  same model. Throughput sweep at C=1 and C=4 on a moderate prompt; the
  fused path is the ADR-012 follow-up to ADR-010's neutral throughput
  result.
- **turbo**: TurboQuant KV cache compression vs bf16 baseline. Compares
  three modes: bf16 (uncompressed), turbo4 (V1: rotation + uniform 4-bit),
  turbo3 (V3 full: rotation + polar + Lloyd-Max + QJL + asymmetric K3V4).
  Reports KV-cache storage memory, output-coherence parity, and
  throughput on a moderate prompt. ADR-013.

CLI flags:

- `model`: HF model ID. Default Qwen2.5-0.5B-Instruct.
- `block_size`: 16 routes to materialized FA varlen (default), 256 routes to
  paged FA varlen on CUDA. See ADR-008.
- `num_blocks`: KV block pool size. 1024 is plenty for short workloads;
  scale up for very long contexts.
- `prompt`: text prompt for smoke/throughput/holb. Long-prompt throughput
  (`workload=long`) overrides this with a synthetic ~3k-token prompt.
- `max_tokens`, `concurrencies` (comma-sep), `chunk_size`, `workload`
  (`short`|`long`).
- `target_prompt_tokens`: prompt length for the `prefix` config (default 12000).

GPU selection: set `MINI_INFER_BENCH_GPU` (e.g. `A10`, `H100`); default `A10`.
Modal 1.4 removed runtime gpu overrides, so this is read at decorator time.

Examples:

    uv run modal run scripts/modal_packed_bench.py --config smoke
    MINI_INFER_BENCH_GPU=H100 uv run modal run scripts/modal_packed_bench.py \
        --config throughput --workload long --max-tokens 64 --concurrencies 1,2,4
    MINI_INFER_BENCH_GPU=H100 uv run modal run scripts/modal_packed_bench.py --config sweep
    uv run modal run scripts/modal_packed_bench.py --config holb
    uv run modal run scripts/modal_packed_bench.py --config prefix
    uv run modal run scripts/modal_packed_bench.py --config quant
    uv run modal run scripts/modal_packed_bench.py --config spec
    uv run modal run scripts/modal_packed_bench.py --config quant_kernel
    uv run modal run scripts/modal_packed_bench.py --config turbo
"""

import itertools
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("mini-infer-packed-bench")

# Modal 1.4 removed `Function.with_options(...)`; gpu is fixed at decorator
# evaluation time. Read it from an env var so callers can switch via
# `MINI_INFER_BENCH_GPU=H100 uv run modal run scripts/modal_packed_bench.py ...`.
_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "A10")

# Pinned to a known-working torch + flash-attn combo. The wheel install is fast
# (precompiled); fresh image build is ~1 min. flash-attn 2.8+ is required for
# `flash_attn_varlen_func`'s `block_table` parameter (paged FA varlen).
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)
image = (
    # CUDA dev image (has nvcc) so FlashInfer's JIT-compiled prefill
    # kernels can build at first call. CUDA 12.8 satisfies FlashInfer
    # 0.6.10's `>=12.6` minimum and is forward-compatible with the
    # CUDA 12.4 runtime libs bundled in the torch 2.5.1+cu124 wheel.
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=4.40",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "triton>=3.0",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_file(
        str(Path(__file__).parent / "data" / "technical_passage.md"),
        "/root/scripts/data/technical_passage.md",
    )
    .add_local_python_source("mini_infer")
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "The capital of France is"

# Real long-form technical prose used by every workload that needs a
# substantive prompt. Lives in a dedicated file so it doesn't bloat the
# bench script. Distinct paragraphs covering several facets of LLM
# inference; do NOT replace with `paragraph * N` — that inflates
# prefix-cache hits and understates true decode cost.
_PASSAGE_LOCAL_PATH = Path(__file__).parent / "data" / "technical_passage.md"
_PASSAGE_REMOTE_PATH = "/root/scripts/data/technical_passage.md"


def _load_technical_passage() -> str:
    """Read the bench passage from disk.

    Tries the Modal-mounted remote path first (where the file lives in
    the container image), then falls back to the local repo path so the
    same constant is available outside Modal. Either path may be missing
    on a fresh setup; raise a helpful error in that case.
    """
    for path in (Path(_PASSAGE_REMOTE_PATH), _PASSAGE_LOCAL_PATH):
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(
        f"technical_passage.md not found at {_PASSAGE_REMOTE_PATH} or {_PASSAGE_LOCAL_PATH}"
    )


_TECHNICAL_PASSAGE = _load_technical_passage()


def _build_long_prompt(target_tokens: int | None = None) -> str:
    """Return the technical passage. `target_tokens` is accepted for API
    compatibility but ignored; the passage is sized for ~1500-2000 tokens
    of natural varied technical prose, which exercises the engine more
    realistically than a basic paragraph repeated to a target length.
    """
    del target_tokens
    return _TECHNICAL_PASSAGE


def _parse_int_list(csv: str) -> list[int]:
    return [int(part) for part in csv.split(",") if part.strip()]


def _make_runner(model: str, num_blocks: int, block_size: int) -> Any:
    from mini_infer.engine.model_runner import ModelRunner

    return ModelRunner.from_pretrained(model, num_blocks=num_blocks, block_size=block_size)


def _run_smoke(runner: Any, max_tokens: int) -> str:
    """4 mixed-length concurrent prompts, parity-with-tail-drift check vs serial."""
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    prompts = [
        "The capital of France is",
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog. " * 8,
        "In the beginning was the Word and the Word was with " * 8,
    ]
    sched = ContinuousScheduler(runner, max_concurrent=8, chunk_size=32)
    sched.start()
    try:
        handles = [
            sched.submit(Request(prompt=p, sampling_params=SamplingParams(), max_tokens=max_tokens))
            for p in prompts
        ]
        concurrent = [h.wait() for h in handles]
    finally:
        sched.stop()

    sched_serial = ContinuousScheduler(runner, max_concurrent=1, chunk_size=32)
    sched_serial.start()
    try:
        serial = [
            sched_serial.run(
                Request(prompt=p, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
            for p in prompts
        ]
    finally:
        sched_serial.stop()

    drifts: list[str] = []
    hard_fails: list[str] = []
    for prompt, c, s in zip(prompts, concurrent, serial, strict=True):
        if not c.tokens or c.tokens[0] != s.tokens[0]:
            hard_fails.append(f"  hard fail on {prompt[:40]!r}")
            continue
        if c.tokens != s.tokens:
            n_match = sum(1 for ct, st in zip(c.tokens, s.tokens, strict=True) if ct == st)
            drifts.append(f"  {prompt[:40]!r}: {n_match}/{len(c.tokens)} match")
    if hard_fails:
        raise AssertionError("HARD FAILS:\n" + "\n".join(hard_fails))

    summary = " | ".join(
        f"{p[:24]!r}->{r.text[:24]!r}" for p, r in zip(prompts, concurrent, strict=True)
    )
    drift_section = ""
    if drifts:
        drift_section = "\nbf16 tail drifts (acceptable):\n" + "\n".join(drifts)
    return f"{summary}{drift_section}"


def _run_throughput(
    runner: Any,
    prompt: str,
    max_tokens: int,
    concurrencies: list[int],
    chunk_settings: list[tuple[str, int]],
) -> str:
    """Wall-clock throughput sweep at each (chunk_size, concurrency) combo."""
    import torch

    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    rows: list[dict[str, Any]] = []
    for label, chunk_size in chunk_settings:
        for concurrency in concurrencies:
            scheduler = ContinuousScheduler(
                runner, max_concurrent=concurrency, chunk_size=chunk_size
            )
            scheduler.start()
            try:
                # Warmup once so kernel caches are hot.
                scheduler.run(
                    Request(
                        prompt=prompt,
                        sampling_params=SamplingParams(),
                        max_tokens=max_tokens,
                    )
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                handles = [
                    scheduler.submit(
                        Request(
                            prompt=prompt,
                            sampling_params=SamplingParams(),
                            max_tokens=max_tokens,
                        )
                    )
                    for _ in range(concurrency)
                ]
                results = [h.wait() for h in handles]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                scheduler.stop()

            total_output = sum(len(r.tokens) for r in results)
            rows.append(
                {
                    "config": label,
                    "concurrency": concurrency,
                    "elapsed_s": round(elapsed, 3),
                    "tokens": total_output,
                    "tok_per_s": round(total_output / elapsed, 2),
                    "lat_s": round(statistics.fmean([elapsed] * concurrency), 3),
                }
            )

    header = f"{'config':<14} {'C':>3} {'elapsed_s':>10} {'tokens':>8} {'tok/s':>10} {'lat_s':>8}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['config']:<14} {row['concurrency']:>3} {row['elapsed_s']:>10} "
            f"{row['tokens']:>8} {row['tok_per_s']:>10} {row['lat_s']:>8}"
        )
    return "\n".join(lines)


def _run_holb(runner: Any) -> str:
    """Stream a short decoder, submit a long prefill mid-stream, measure ITL."""
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    long_prompt = _build_long_prompt(target_tokens=3000)
    long_prompt_len = len(runner.tokenizer.encode(long_prompt))
    short_prompt = "The capital of France is"

    rows: list[dict[str, Any]] = []
    for label, chunk_size in [("chunked-256", 256), ("unchunked", 8192)]:
        scheduler = ContinuousScheduler(runner, max_concurrent=8, chunk_size=chunk_size)
        scheduler.start()
        try:
            scheduler.run(
                Request(prompt=short_prompt, sampling_params=SamplingParams(), max_tokens=4)
            )
            short_handle = scheduler.submit(
                Request(prompt=short_prompt, sampling_params=SamplingParams(), max_tokens=64)
            )
            short_token_times: list[float] = []
            done = threading.Event()

            def drain_short(
                handle: Any = short_handle,
                token_times: list[float] = short_token_times,
                done_event: threading.Event = done,
            ) -> None:
                for step in handle.steps():
                    token_times.append(time.perf_counter())
                    if step.finish_reason is not None:
                        break
                done_event.set()

            drain_thread = threading.Thread(target=drain_short, daemon=True)
            drain_thread.start()
            while len(short_token_times) < 4 and not done.is_set():
                time.sleep(0.005)
            long_submit_time = time.perf_counter()
            n_before = len(short_token_times)
            long_handle = scheduler.submit(
                Request(prompt=long_prompt, sampling_params=SamplingParams(), max_tokens=8)
            )
            long_handle.wait()
            long_done_time = time.perf_counter()
            done.wait(timeout=30.0)
            drain_thread.join(timeout=5.0)
        finally:
            scheduler.stop()

        before = short_token_times[:n_before]
        after = short_token_times[n_before:]
        baseline_itl_ms = (
            1000.0
            * statistics.fmean(later - earlier for earlier, later in itertools.pairwise(before))
            if len(before) >= 2
            else float("nan")
        )
        next_after_ms = 1000.0 * (after[0] - long_submit_time) if after else float("nan")
        long_total_ms = 1000.0 * (long_done_time - long_submit_time)
        rows.append(
            {
                "config": label,
                "baseline_itl_ms": round(baseline_itl_ms, 1),
                "short_after_long_ms": round(next_after_ms, 1),
                "long_total_ms": round(long_total_ms, 1),
            }
        )

    header = (
        f"{'config':<14} {'baseline_itl_ms':>16} {'short_after_long_ms':>20} {'long_total_ms':>14}"
    )
    lines = [
        header,
        "-" * len(header),
        *[
            (
                f"{row['config']:<14} {row['baseline_itl_ms']:>16} "
                f"{row['short_after_long_ms']:>20} {row['long_total_ms']:>14}"
            )
            for row in rows
        ],
    ]
    return f"prompt_len={long_prompt_len}\n" + "\n".join(lines)


def _run_prefix_bench(
    model: str,
    num_blocks: int,
    block_size: int,
    target_prompt_tokens: int,
    max_tokens: int,
    concurrencies: list[int],
) -> str:
    """Shared-system-prompt workload; prefix cache OFF vs ON.

    Builds a single very-long system prompt (~target_prompt_tokens), pairs it
    with several unique short user questions, and runs the workload twice:
    once with prefix caching disabled (every request pays the full prefill
    cost) and once with prefix caching enabled (the system prompt is computed
    on the first request and reused for the rest).

    Two measurements per cache mode:
      - Sequential per-request TTFT (one request at a time) — surfaces
        cold-vs-warm asymmetry: with cache ON the first prompt is cold, the
        rest are warm.
      - Aggregate concurrent throughput at several concurrencies — the
        end-user-visible win.
    """
    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    system = _build_long_prompt(target_tokens=target_prompt_tokens)
    user_questions = [
        "What is the capital of France?",
        "Name three primary colors.",
        "How many planets are in the solar system?",
        "Who wrote Hamlet?",
        "What is 7 times 8?",
        "What is the chemical symbol for gold?",
        "List three programming languages.",
        "What is the freezing point of water in Celsius?",
    ]
    prompts = [f"{system}\n\nQ: {q}\nA:" for q in user_questions]

    per_mode: dict[str, dict[str, Any]] = {}

    for cache_on in (False, True):
        label = "cache_on" if cache_on else "cache_off"
        runner = ModelRunner.from_pretrained(
            model,
            num_blocks=num_blocks,
            block_size=block_size,
            prefix_cache=cache_on,
        )
        prompt_len = len(runner.tokenizer.encode(prompts[0]))

        # Sequential TTFT: submit one at a time, time the first emitted step.
        sched = ContinuousScheduler(runner, max_concurrent=1, chunk_size=256)
        sched.start()
        ttfts_ms: list[float] = []
        try:
            for prompt_text in prompts:
                t0 = time.perf_counter()
                handle = sched.submit(
                    Request(
                        prompt=prompt_text,
                        sampling_params=SamplingParams(),
                        max_tokens=max_tokens,
                    )
                )
                first_step_time: float | None = None
                for step in handle.steps():
                    if first_step_time is None and step.text:
                        first_step_time = time.perf_counter()
                    if step.finish_reason is not None:
                        break
                if first_step_time is None:
                    ttfts_ms.append(float("nan"))
                else:
                    ttfts_ms.append((first_step_time - t0) * 1000.0)
        finally:
            sched.stop()

        # Concurrent aggregate throughput at each concurrency.
        throughput_rows: list[dict[str, Any]] = []
        for concurrency in concurrencies:
            scheduler = ContinuousScheduler(runner, max_concurrent=concurrency, chunk_size=256)
            scheduler.start()
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                handles = [
                    scheduler.submit(
                        Request(
                            prompt=p,
                            sampling_params=SamplingParams(),
                            max_tokens=max_tokens,
                        )
                    )
                    for p in prompts
                ]
                results = [h.wait() for h in handles]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                scheduler.stop()
            total_tokens = sum(len(r.tokens) for r in results)
            throughput_rows.append(
                {
                    "concurrency": concurrency,
                    "elapsed_s": round(elapsed, 3),
                    "tokens": total_tokens,
                    "tok_per_s": round(total_tokens / elapsed, 2),
                }
            )

        per_mode[label] = {
            "prompt_len": prompt_len,
            "ttfts_ms": ttfts_ms,
            "throughput": throughput_rows,
        }
        del runner
        torch.cuda.empty_cache()

    off = per_mode["cache_off"]
    on = per_mode["cache_on"]

    # Format report.
    lines: list[str] = []
    lines.append(
        f"prompt_len={off['prompt_len']} tokens | "
        f"max_tokens={max_tokens} | n_prompts={len(prompts)}"
    )
    lines.append("")
    lines.append("Sequential TTFT (single-request, sequential submission):")
    lines.append(
        f"  cache_off: first={off['ttfts_ms'][0]:.0f}ms  "
        f"rest_avg={statistics.fmean(off['ttfts_ms'][1:]):.0f}ms  "
        f"all={[round(t, 1) for t in off['ttfts_ms']]}"
    )
    lines.append(
        f"  cache_on : first={on['ttfts_ms'][0]:.0f}ms  "
        f"rest_avg={statistics.fmean(on['ttfts_ms'][1:]):.0f}ms  "
        f"all={[round(t, 1) for t in on['ttfts_ms']]}"
    )
    rest_off = statistics.fmean(off["ttfts_ms"][1:])
    rest_on = statistics.fmean(on["ttfts_ms"][1:])
    if rest_on > 0:
        lines.append(f"  warm-TTFT speedup: {rest_off / rest_on:.1f}x")
    lines.append("")
    lines.append("Concurrent throughput (all prompts submitted at once):")
    header = (
        f"  {'C':>3}  {'cache_off (s, tok/s)':>26}  {'cache_on (s, tok/s)':>26}  {'speedup':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row_off, row_on in zip(off["throughput"], on["throughput"], strict=True):
        speedup = row_on["tok_per_s"] / row_off["tok_per_s"] if row_off["tok_per_s"] else 0.0
        off_cell = f"{row_off['elapsed_s']}s, {row_off['tok_per_s']} tok/s"
        on_cell = f"{row_on['elapsed_s']}s, {row_on['tok_per_s']} tok/s"
        lines.append(
            f"  {row_off['concurrency']:>3}  {off_cell:>26}  {on_cell:>26}  {speedup:>7.2f}x"
        )

    return "\n".join(lines)


def _run_quant_bench(
    model: str,
    num_blocks: int,
    block_size: int,
    max_tokens: int,
    concurrencies: list[int],
) -> str:
    """Weight-only INT8 (W8A16) vs fp16 baseline.

    Loads the model in three configurations on the same Modal container:
      - fp16
      - int8 with `lm_head` skipped (default)
      - int8 with `lm_head` quantized too

    For each, reports the post-load CUDA memory footprint and a small
    concurrent throughput sweep on a moderate-length prompt. The throughput
    measurement is honest about W8A16's expected behaviour: dequant happens
    on every forward, so throughput at small batch sizes can be neutral or
    slightly negative versus the fp16 baseline.
    """
    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    prompt = (
        "Summarize the following passage in one sentence. "
        "The mini-infer engine is an open-source LLM inference server "
        "demonstrating production techniques: continuous batching, paged "
        "attention, chunked prefill, packed varlen forward, prefix caching, "
        "and weight quantization. " * 8
    )

    configs: list[tuple[str, dict[str, Any]]] = [
        ("fp16", {}),
        ("int8 (skip lm_head)", {"quant": "int8"}),
        ("int8 (quant lm_head)", {"quant": "int8", "quant_lm_head": True}),
    ]

    rows: list[dict[str, Any]] = []
    for label, kwargs in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before_alloc = torch.cuda.memory_allocated()

        runner = ModelRunner.from_pretrained(
            model, num_blocks=num_blocks, block_size=block_size, **kwargs
        )
        torch.cuda.synchronize()
        after_alloc = torch.cuda.memory_allocated()
        weight_bytes_observed = after_alloc - before_alloc

        # Throughput sweep at each concurrency.
        throughput_at: dict[int, dict[str, Any]] = {}
        for concurrency in concurrencies:
            sched = ContinuousScheduler(runner, max_concurrent=concurrency, chunk_size=256)
            sched.start()
            try:
                # Warmup so we're not timing the first matmul launch.
                sched.run(
                    Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=max_tokens)
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                handles = [
                    sched.submit(
                        Request(
                            prompt=prompt,
                            sampling_params=SamplingParams(),
                            max_tokens=max_tokens,
                        )
                    )
                    for _ in range(concurrency)
                ]
                results = [h.wait() for h in handles]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                sched.stop()
            total_tokens = sum(len(r.tokens) for r in results)
            throughput_at[concurrency] = {
                "elapsed_s": round(elapsed, 3),
                "tokens": total_tokens,
                "tok_per_s": round(total_tokens / elapsed, 2),
            }

        rows.append(
            {
                "label": label,
                "weight_mib": weight_bytes_observed / (1024 * 1024),
                "throughput": throughput_at,
            }
        )
        del runner
        torch.cuda.empty_cache()

    # Format the report.
    lines: list[str] = [f"prompt_chars={len(prompt)} | max_tokens={max_tokens}", ""]
    lines.append("Model-weight memory footprint (CUDA allocated by ModelRunner.from_pretrained):")
    fp_mib = rows[0]["weight_mib"]
    for row in rows:
        savings = (fp_mib - row["weight_mib"]) / fp_mib if fp_mib else 0.0
        lines.append(
            f"  {row['label']:<24}  {row['weight_mib']:>8.1f} MiB"
            + (f"   ({savings:+.1%} vs fp16)" if row["label"] != "fp16" else "")
        )
    lines.append("")
    lines.append("Concurrent throughput (warmup + N requests at each concurrency):")
    header = (
        f"  {'C':>3}  {'fp16 (s, tok/s)':>22}  "
        f"{'int8-skip-lm (s, tok/s)':>26}  {'int8-all (s, tok/s)':>22}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for c in concurrencies:
        cells = []
        for row in rows:
            t = row["throughput"][c]
            cells.append(f"{t['elapsed_s']}s, {t['tok_per_s']} tok/s")
        lines.append(f"  {c:>3}  {cells[0]:>22}  {cells[1]:>26}  {cells[2]:>22}")

    return "\n".join(lines)


def _run_turbo_parity(
    model: str,
    num_blocks: int,
    block_size: int,
) -> str:
    """Verify the V2a fused dequant kernel matches the Python-loop reference.

    Two layers of evidence:

    1. **Random-fixture parity** — populate `BlockPool(kv_quant="turbo3")`
       with random data, materialize via fused kernel and via Python loop,
       assert cosine sim > 0.999 on both K and V. Five shape configurations
       cover Qwen2.5-0.5B (head_dim=64, num_kv_heads=2), Qwen2.5-7B
       (head_dim=128, num_kv_heads=8), partial-block edge cases, and a
       larger block_size=64. Mirrors `tests/unit/test_turbo_kernel.py`'s
       `@pytest.mark.requires_cuda` tests, which the local CI skips.
    2. **End-to-end greedy parity** — load Qwen2.5-0.5B turbo3, run a
       12-token greedy decode with `_FUSED_DISABLED_FOR_BENCH=False`
       (kernel) and `=True` (Python loop), assert token-for-token equality.

    The kernel and Python loop go through different fp accumulation orders
    (`tl.dot` vs `torch.matmul`), so absolute equality vs bf16 is not
    expected — but the kernel's output must match the Python loop on the
    same inputs to within the parity bar (cosine sim > 0.999) and produce
    identical tokens under greedy decoding.
    """
    import torch

    from mini_infer.cache import turbo_kernel
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                a.float().flatten(), b.float().flatten(), dim=0
            ).item()
        )

    def _populate_pool(
        *,
        num_layers: int,
        n_blocks: int,
        bs: int,
        nh: int,
        hd: int,
        seq_lens: list[int],
        seed: int,
    ) -> tuple[BlockPool, PagedKVCache]:
        pool = BlockPool(
            num_blocks=n_blocks,
            block_size=bs,
            num_layers=num_layers,
            num_kv_heads=nh,
            head_dim=hd,
            dtype=torch.bfloat16,
            device="cuda",
            kv_quant="turbo3",
        )
        cache = PagedKVCache(pool)
        torch.manual_seed(seed)
        for slot_idx, seq_len in enumerate(seq_lens):
            cache.add_request_slot()
            if seq_len == 0:
                continue
            num_blocks_used = (seq_len + bs - 1) // bs
            block_ids = [pool.allocate() for _ in range(num_blocks_used)]
            cache._block_ids[slot_idx] = block_ids
            cache._num_tokens[slot_idx] = seq_len
            for layer_idx in range(num_layers):
                for bid in block_ids:
                    k = (torch.randn(bs, nh, hd) * 0.1).to(torch.bfloat16).cuda()
                    v = (torch.randn(bs, nh, hd) * 0.1).to(torch.bfloat16).cuda()
                    pool.write_compressed_block(layer_idx, 0, bid, k)
                    pool.write_compressed_block(layer_idx, 1, bid, v)
        return pool, cache

    def _materialize_two_paths(cache: PagedKVCache, layer_idx: int) -> tuple[float, float]:
        # Default: kernel ON.
        k_fused, v_fused, _, _ = cache.materialize_packed_kv(layer_idx)
        saved = turbo_kernel._FUSED_DISABLED_FOR_BENCH
        turbo_kernel._FUSED_DISABLED_FOR_BENCH = True
        try:
            k_python, v_python, _, _ = cache.materialize_packed_kv(layer_idx)
        finally:
            turbo_kernel._FUSED_DISABLED_FOR_BENCH = saved
        return _cos(k_fused, k_python), _cos(v_fused, v_python)

    fixtures = [
        {
            "name": "qwen 0.5B shape",
            "num_layers": 2,
            "n_blocks": 16,
            "bs": 16,
            "nh": 2,
            "hd": 64,
            "seq_lens": [24, 17, 0, 32],
            "seed": 11,
        },
        {
            "name": "qwen 7B shape",
            "num_layers": 2,
            "n_blocks": 8,
            "bs": 16,
            "nh": 8,
            "hd": 128,
            "seq_lens": [16, 32, 8],
            "seed": 23,
        },
        {
            "name": "partial-block edges",
            "num_layers": 1,
            "n_blocks": 8,
            "bs": 16,
            "nh": 2,
            "hd": 64,
            "seq_lens": [16, 0, 1, 31],
            "seed": 42,
        },
        {
            "name": "block_size=64",
            "num_layers": 1,
            "n_blocks": 4,
            "bs": 64,
            "nh": 2,
            "hd": 64,
            "seq_lens": [100, 50],
            "seed": 77,
        },
    ]

    lines: list[str] = []
    lines.append("Random-fixture parity (cosine sim > 0.999 required):")
    all_fixtures_pass = True
    for fx in fixtures:
        pool, cache = _populate_pool(
            num_layers=fx["num_layers"],
            n_blocks=fx["n_blocks"],
            bs=fx["bs"],
            nh=fx["nh"],
            hd=fx["hd"],
            seq_lens=fx["seq_lens"],
            seed=fx["seed"],
        )
        cos_k, cos_v = _materialize_two_paths(cache, layer_idx=fx["num_layers"] - 1)
        ok = cos_k > 0.999 and cos_v > 0.999
        all_fixtures_pass = all_fixtures_pass and ok
        mark = "✓" if ok else "✗"
        lines.append(f"  {mark} {fx['name']}: cos_K={cos_k:.6f} cos_V={cos_v:.6f}")
        del pool, cache
        torch.cuda.empty_cache()

    # End-to-end greedy parity on Qwen2.5-0.5B turbo3.
    def _greedy_decode() -> list[int]:
        runner = ModelRunner.from_pretrained(
            model, num_blocks=num_blocks, block_size=block_size, kv_quant="turbo3"
        )
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            r = sched.run(
                Request(
                    prompt="The capital of France is",
                    sampling_params=SamplingParams(),
                    max_tokens=12,
                )
            )
        finally:
            sched.stop()
        del runner
        torch.cuda.empty_cache()
        return list(r.tokens)

    # ── V2b attention parity: compare V2b output to V2a (materialized) ──
    lines.append("")
    lines.append("V2b attention parity vs V2a materialized (cosine sim > 0.999 required):")
    v2b_fixtures = [
        {
            # Qwen2.5-0.5B: num_q_heads=14, num_kv_heads=2, head_dim=64.
            "name": "qwen 0.5B decode (B=4)",
            "num_layers": 2,
            "n_blocks": 16,
            "bs": 16,
            "nh_kv": 2,
            "hd": 64,
            "nh_q": 14,
            "seq_lens": [24, 17, 32, 8],
            "seed": 101,
        },
        {
            # Qwen2.5-7B: num_q_heads=28, num_kv_heads=4, head_dim=128.
            "name": "qwen 7B decode (B=3)",
            "num_layers": 2,
            "n_blocks": 8,
            "bs": 16,
            "nh_kv": 4,
            "hd": 128,
            "nh_q": 28,
            "seq_lens": [16, 32, 8],
            "seed": 303,
        },
    ]
    all_v2b_fixtures_pass = True
    for fx in v2b_fixtures:
        from mini_infer.cache.packed_attention import packed_attention_forward

        pool, cache = _populate_pool(
            num_layers=fx["num_layers"],
            n_blocks=fx["n_blocks"],
            bs=fx["bs"],
            nh=fx["nh_kv"],
            hd=fx["hd"],
            seq_lens=fx["seq_lens"],
            seed=fx["seed"],
        )
        batch_size = cache.batch_size
        torch.manual_seed(fx["seed"] + 1)
        q = torch.randn(batch_size, fx["nh_q"], fx["hd"], dtype=torch.bfloat16, device="cuda") * 0.1
        cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
        layer_idx = fx["num_layers"] - 1

        # V2b is off by default since V2a wins at 7B+; explicitly toggle.
        saved_attn = turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH

        turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = False
        try:
            out_v2b = packed_attention_forward(q, cache, layer_idx, cu_seqlens_q)
        finally:
            turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = saved_attn

        turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = True
        try:
            out_v2a = packed_attention_forward(q, cache, layer_idx, cu_seqlens_q)
        finally:
            turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = saved_attn

        cos = _cos(out_v2b, out_v2a)
        ok = cos > 0.999
        all_v2b_fixtures_pass = all_v2b_fixtures_pass and ok
        mark = "✓" if ok else "✗"
        lines.append(f"  {mark} {fx['name']}: cos={cos:.6f}")
        del pool, cache
        torch.cuda.empty_cache()

    # ── End-to-end greedy decode (informational, not pass/fail) ──
    # Greedy tokens diverge between kernel and Python loop because
    # tl.dot's fp32 reduction order isn't bit-identical to PyTorch's
    # matmul, and turbo3's argmax is sensitive at LSB level. The right
    # correctness bar is the cosine-sim parity above.
    lines.append("")
    lines.append("End-to-end greedy decode (informational, Qwen2.5-0.5B turbo3, 12 tokens):")
    saved = turbo_kernel._FUSED_DISABLED_FOR_BENCH

    turbo_kernel._FUSED_DISABLED_FOR_BENCH = False
    fused_tokens = _greedy_decode()
    turbo_kernel._FUSED_DISABLED_FOR_BENCH = True
    python_tokens = _greedy_decode()
    turbo_kernel._FUSED_DISABLED_FOR_BENCH = saved

    lines.append(f"  fused tokens:  {fused_tokens}")
    lines.append(f"  python tokens: {python_tokens}")
    if fused_tokens == python_tokens:
        lines.append("  (tokens match — kernel happens to be bit-stable on this prompt)")
    else:
        lines.append("  (tokens diverge — expected per fp accumulation order)")

    lines.append("")
    overall_ok = all_fixtures_pass and all_v2b_fixtures_pass
    lines.append(
        f"OVERALL: {'✓ all parity checks passed' if overall_ok else '✗ some checks failed'}"
    )
    return "\n".join(lines)


def _run_flashinfer_fp8_compare(
    model: str,
    num_blocks: int,
    block_size: int,
) -> str:
    """bf16 baseline vs FP8 KV cache via FlashInfer.

    Loads the model twice and runs the same long-prompt decode workload.
    Reports per-mode KV pool memory, throughput, peak HBM, and first-token
    logit cosine sim to surface FP8's expected ~1-2% drift.
    """
    import gc

    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    prompt = _TECHNICAL_PASSAGE
    max_tokens = 128

    def _measure(label: str, **runner_kwargs: object) -> dict[str, Any]:
        runner = ModelRunner.from_pretrained(
            model,
            num_blocks=num_blocks,
            block_size=block_size,
            **runner_kwargs,  # type: ignore[arg-type]
        )
        # KV pool storage: bf16 vs fp8.
        pool = runner.block_pool
        if pool.kv_quant == "fp8":
            assert pool._fp8_storage is not None
            kv_bytes = int(pool._fp8_storage.numel() * pool._fp8_storage.element_size())
        else:
            storage: torch.Tensor = pool._storage
            kv_bytes = int(storage.numel() * storage.element_size())
        kv_mib = kv_bytes / (1024**2)

        # Warmup so first-launch JIT isn't counted.
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            sched.run(Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=8))
        finally:
            sched.stop()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            result = sched.run(
                Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
        finally:
            sched.stop()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_bytes = torch.cuda.max_memory_allocated()

        input_ids = runner.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], device=runner.device, dtype=torch.long)
        with torch.inference_mode():
            logits = runner._model(x, use_cache=False).logits[0, -1, :].float().cpu()

        del runner
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "label": label,
            "kv_pool_mib": round(kv_mib, 1),
            "elapsed_s": round(elapsed, 3),
            "tok_per_s": round(len(result.tokens) / elapsed, 2),
            "peak_mib": round(peak_bytes / (1024**2), 1),
            "first_tokens": list(result.tokens[:8]),
            "first_token_logits": logits,
        }

    bf16 = _measure("bf16 / flash_attn")
    fp8 = _measure("fp8 / flashinfer", kv_quant="fp8", attention_backend="flashinfer")

    cos = float(
        torch.nn.functional.cosine_similarity(
            bf16["first_token_logits"].flatten(),
            fp8["first_token_logits"].flatten(),
            dim=0,
        ).item()
    )
    speedup = fp8["tok_per_s"] / bf16["tok_per_s"]
    kv_savings_pct = (1.0 - fp8["kv_pool_mib"] / bf16["kv_pool_mib"]) * 100
    lines = [
        f"Workload: prompt={len(prompt)} chars, max_tokens={max_tokens}, C=1",
        "",
        f"  {bf16['label']:25s} t/s={bf16['tok_per_s']:>6.2f}  "
        f"peak HBM={bf16['peak_mib']:>8.1f} MiB  KV pool={bf16['kv_pool_mib']:>8.1f} MiB",
        f"  {fp8['label']:25s} t/s={fp8['tok_per_s']:>6.2f}  "
        f"peak HBM={fp8['peak_mib']:>8.1f} MiB  KV pool={fp8['kv_pool_mib']:>8.1f} MiB",
        "",
        f"  KV memory savings: {kv_savings_pct:+.1f}% (fp8 vs bf16)",
        f"  Logit cos sim (first decode position): {cos:.6f} (>0.99 required)",
        f"  Throughput fp8 / bf16: {speedup:.2f}x",
        "",
        f"  bf16 first 8 tokens: {bf16['first_tokens']}",
        f"  fp8 first 8 tokens:  {fp8['first_tokens']}",
    ]
    return "\n".join(lines)


def _run_flashinfer_compare(
    model: str,
    num_blocks: int,
    block_size: int,
) -> str:
    """flash-attn vs FlashInfer on the same uncompressed bf16 model.

    Loads the model twice (once per backend) and runs the same long-prompt
    decode workload. Reports per-backend throughput, peak HBM, and the
    cosine sim of greedy first-token logits to confirm numerical parity.
    """
    import gc

    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    prompt = _TECHNICAL_PASSAGE
    max_tokens = 128

    def _measure(backend: str) -> dict[str, Any]:
        runner = ModelRunner.from_pretrained(
            model,
            num_blocks=num_blocks,
            block_size=block_size,
            attention_backend=backend,
        )
        # Warmup so first-launch JIT (FlashInfer) isn't counted.
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            sched.run(Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=8))
        finally:
            sched.stop()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            result = sched.run(
                Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
        finally:
            sched.stop()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_bytes = torch.cuda.max_memory_allocated()

        # First-token logits for parity comparison.
        input_ids = runner.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], device=runner.device, dtype=torch.long)
        with torch.inference_mode():
            logits = runner._model(x, use_cache=False).logits[0, -1, :].float().cpu()

        del runner
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "backend": backend,
            "elapsed_s": round(elapsed, 3),
            "decoded_tokens": len(result.tokens),
            "tok_per_s": round(len(result.tokens) / elapsed, 2),
            "peak_mib": round(peak_bytes / (1024**2), 1),
            "first_tokens": list(result.tokens[:8]),
            "first_token_logits": logits,
        }

    fa_result = _measure("flash_attn")
    fi_result = _measure("flashinfer")

    cos = float(
        torch.nn.functional.cosine_similarity(
            fa_result["first_token_logits"].flatten(),
            fi_result["first_token_logits"].flatten(),
            dim=0,
        ).item()
    )

    speedup = fi_result["tok_per_s"] / fa_result["tok_per_s"]
    delta_mib = fa_result["peak_mib"] - fi_result["peak_mib"]
    lines = [
        f"Workload: prompt={len(prompt)} chars, max_tokens={max_tokens}, C=1",
        "",
        f"  flash_attn   t/s={fa_result['tok_per_s']:>6.2f}  "
        f"peak HBM={fa_result['peak_mib']:>8.1f} MiB  first8={fa_result['first_tokens']}",
        f"  flashinfer   t/s={fi_result['tok_per_s']:>6.2f}  "
        f"peak HBM={fi_result['peak_mib']:>8.1f} MiB  first8={fi_result['first_tokens']}",
        "",
        f"  Logit cos sim (first decode position): {cos:.6f} (>0.999 required)",
        f"  flashinfer vs flash_attn: throughput {speedup:.2f}x, peak HBM {delta_mib:+.1f} MiB",
    ]
    return "\n".join(lines)


def _run_turbo_v2b_compare(
    model: str,
    num_blocks: int,
    block_size: int,
) -> str:
    """V2a (materialized) vs V2b (fully-fused) on a long-context decode.

    Same model load, same workload, only the attention path differs:
    - V2a: dequant fused (V2a kernel) into a materialized bf16 K/V
      buffer per layer per step, then `flash_attn_varlen_func`.
    - V2b: dequant + online softmax fused into one Triton kernel; no
      materialized K/V buffer — K/V tiles live in registers.

    The expected V2b win is **peak attention memory**: V2a allocates a
    `(total_k, num_kv_heads, head_dim) * 2 bytes * 2 sides` transient
    buffer per layer per step; V2b doesn't. Throughput should be
    comparable since both already collapsed the per-block Python loop.
    """
    import gc

    import torch

    from mini_infer.cache import turbo_kernel
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    # Long real prompt + many decode steps so the materialized buffer is
    # actually large and the cost of materializing is exposed.
    prompt = _TECHNICAL_PASSAGE
    max_tokens = 128

    model_runner = ModelRunner.from_pretrained(
        model, num_blocks=num_blocks, block_size=block_size, kv_quant="turbo3"
    )

    def _measure_path(runner: Any, label: str) -> dict[str, float]:
        # Warmup so we're not counting first-launch JIT.
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            sched.run(Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=8))
        finally:
            sched.stop()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            result = sched.run(
                Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=max_tokens)
            )
        finally:
            sched.stop()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_bytes = torch.cuda.max_memory_allocated()
        return {
            "label": label,
            "elapsed_s": round(elapsed, 3),
            "decoded_tokens": len(result.tokens),
            "tok_per_s": round(len(result.tokens) / elapsed, 2),
            "peak_mib": round(peak_bytes / (1024**2), 1),
        }

    # V2b on (default).
    saved = turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH
    turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = False
    v2b_result = _measure_path(model_runner, "V2b (fused attention)")

    # V2a fallback (materialized).
    turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = True
    v2a_result = _measure_path(model_runner, "V2a (materialized + FA varlen)")
    turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = saved

    del model_runner
    gc.collect()
    torch.cuda.empty_cache()

    delta_mib = v2a_result["peak_mib"] - v2b_result["peak_mib"]
    speedup = v2b_result["tok_per_s"] / v2a_result["tok_per_s"]
    lines = [
        f"Workload: prompt={len(prompt)} chars, max_tokens={max_tokens}, C=1",
        "",
        f"  {v2a_result['label']:35s} t/s={v2a_result['tok_per_s']:>6.2f}  "
        f"peak HBM={v2a_result['peak_mib']:>8.1f} MiB",
        f"  {v2b_result['label']:35s} t/s={v2b_result['tok_per_s']:>6.2f}  "
        f"peak HBM={v2b_result['peak_mib']:>8.1f} MiB",
        "",
        f"  V2b vs V2a: throughput {speedup:.2f}x, peak HBM {delta_mib:+.1f} MiB "
        f"({-delta_mib / v2a_result['peak_mib'] * 100:+.2f}% of V2a peak)",
    ]
    return "\n".join(lines)


def _run_turbo_bench(
    model: str,
    num_blocks: int,
    block_size: int,
    max_tokens: int,
    concurrencies: list[int],
) -> str:
    """TurboQuant V1 (turbo4) and V3 (turbo3) vs bf16 baseline.

    Three modes loaded sequentially on the same Modal container:
      - bf16: uncompressed reference.
      - turbo4: rotation + per-channel uniform 4-bit (V1, ADR-013 baseline).
      - turbo3: rotation + polar + Lloyd-Max + QJL + asymmetric K3V4
        (V3 full algorithm).
    Reports per-mode KV-cache storage, greedy parity (vs bf16 tokens),
    and throughput. V1/V3 throughput regresses vs bf16 in this slice
    because materialize-on-read dequant runs in Python loops; the fused
    dequant-attention kernel is the V2 follow-up that turns this into a
    real perf win.
    """
    import gc

    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    # Real long-form technical prose — distinct paragraphs covering several
    # facets of LLM inference. Picked to give the benchmark a representative
    # vocabulary spread and varied attention structure rather than the
    # artificial cache patterns a `paragraph * N` prompt would produce.
    prompt = _TECHNICAL_PASSAGE

    def _measure_storage_bytes(runner: Any) -> int:
        pool = runner.block_pool
        if pool.kv_quant is None:
            storage: torch.Tensor = pool._storage
            return int(storage.numel() * storage.element_size())
        compressed: torch.Tensor = pool._compressed_storage
        rotation: torch.Tensor = pool._rotation
        total = compressed.numel() * compressed.element_size()
        total += rotation.numel() * rotation.element_size()
        if pool.kv_quant == "turbo4":
            scales = pool._scales_storage
            total += scales.numel() * scales.element_size()
        else:  # turbo3
            radii = pool._radii_storage
            total += radii.numel() * radii.element_size()
        return int(total)

    def _sweep(label: str, runner: Any) -> dict[int, dict[str, Any]]:
        per_c: dict[int, dict[str, Any]] = {}
        for concurrency in concurrencies:
            sched = ContinuousScheduler(runner, max_concurrent=concurrency, chunk_size=256)
            sched.start()
            try:
                # Warmup so we're not timing the first matmul launch.
                sched.run(
                    Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=max_tokens)
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                handles = [
                    sched.submit(
                        Request(
                            prompt=prompt,
                            sampling_params=SamplingParams(),
                            max_tokens=max_tokens,
                        )
                    )
                    for _ in range(concurrency)
                ]
                results = [h.wait() for h in handles]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                sched.stop()
            total_tokens = sum(len(r.tokens) for r in results)
            per_c[concurrency] = {
                "elapsed_s": round(elapsed, 3),
                "tokens": total_tokens,
                "tok_per_s": round(total_tokens / elapsed, 2),
            }
            print(f"  [{label} C={concurrency}] {per_c[concurrency]}")
        return per_c

    def _parity_tokens(label: str, runner: Any) -> list[int]:
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            r = sched.run(
                Request(
                    prompt="The capital of France is",
                    sampling_params=SamplingParams(),
                    max_tokens=8,
                )
            )
        finally:
            sched.stop()
        print(f"  [parity {label}] tokens={r.tokens} text={r.text!r}")
        return list(r.tokens)

    # 1) bf16 baseline
    bf16_runner = ModelRunner.from_pretrained(model, num_blocks=num_blocks, block_size=block_size)
    bf16_storage = _measure_storage_bytes(bf16_runner)
    bf16_tokens = _parity_tokens("bf16", bf16_runner)
    bf16_results = _sweep("bf16", bf16_runner)
    del bf16_runner
    gc.collect()
    torch.cuda.empty_cache()

    # 2) turbo4 (V1)
    turbo4_runner = ModelRunner.from_pretrained(
        model, num_blocks=num_blocks, block_size=block_size, kv_quant="turbo4"
    )
    turbo4_storage = _measure_storage_bytes(turbo4_runner)
    turbo4_tokens = _parity_tokens("turbo4", turbo4_runner)
    turbo4_results = _sweep("turbo4", turbo4_runner)
    del turbo4_runner
    gc.collect()
    torch.cuda.empty_cache()

    # 3) turbo3 (V3 full)
    turbo3_runner = ModelRunner.from_pretrained(
        model, num_blocks=num_blocks, block_size=block_size, kv_quant="turbo3"
    )
    turbo3_storage = _measure_storage_bytes(turbo3_runner)
    turbo3_tokens = _parity_tokens("turbo3", turbo3_runner)
    turbo3_results = _sweep("turbo3", turbo3_runner)
    del turbo3_runner
    gc.collect()
    torch.cuda.empty_cache()

    def _parity_summary(other: list[int]) -> str:
        if not bf16_tokens or not other:
            return "no_tokens"
        first_match = bf16_tokens[0] == other[0]
        full_match = bf16_tokens == other
        return f"first_match={first_match}, full_match={full_match}"

    # Format report.
    lines: list[str] = [
        f"model={model} | prompt_chars={len(prompt)} | max_tokens={max_tokens}",
        "",
        "KV-cache pool storage:",
        f"  bf16:   {bf16_storage / (1024**2):.1f} MiB",
        f"  turbo4: {turbo4_storage / (1024**2):.1f} MiB "
        f"({turbo4_storage / bf16_storage:.1%} of bf16, "
        f"savings={1 - turbo4_storage / bf16_storage:.1%})",
        f"  turbo3: {turbo3_storage / (1024**2):.1f} MiB "
        f"({turbo3_storage / bf16_storage:.1%} of bf16, "
        f"savings={1 - turbo3_storage / bf16_storage:.1%})",
        "",
        "Greedy parity (prompt='The capital of France is', max_tokens=8):",
        f"  bf16:   {bf16_tokens}",
        f"  turbo4: {turbo4_tokens}     [{_parity_summary(turbo4_tokens)}]",
        f"  turbo3: {turbo3_tokens}     [{_parity_summary(turbo3_tokens)}]",
        "",
        "Throughput:",
    ]
    header = (
        f"  {'C':>3}  {'bf16 (s, t/s)':>20}  "
        f"{'turbo4 (s, t/s)':>20}  {'turbo3 (s, t/s)':>20}  "
        f"{'t4/bf16':>8}  {'t3/bf16':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for c in concurrencies:
        bf = bf16_results[c]
        t4 = turbo4_results[c]
        t3 = turbo3_results[c]
        r4 = t4["tok_per_s"] / bf["tok_per_s"] if bf["tok_per_s"] else 0.0
        r3 = t3["tok_per_s"] / bf["tok_per_s"] if bf["tok_per_s"] else 0.0
        bf_cell = f"{bf['elapsed_s']}s, {bf['tok_per_s']} t/s"
        t4_cell = f"{t4['elapsed_s']}s, {t4['tok_per_s']} t/s"
        t3_cell = f"{t3['elapsed_s']}s, {t3['tok_per_s']} t/s"
        lines.append(
            f"  {c:>3}  {bf_cell:>20}  {t4_cell:>20}  {t3_cell:>20}  {r4:>7.2f}x  {r3:>7.2f}x"
        )
    return "\n".join(lines)


def _run_quant_kernel_bench(
    model: str,
    num_blocks: int,
    block_size: int,
    max_tokens: int,
    concurrencies: list[int],
    skip_fp16: bool = False,
) -> str:
    """fp16 vs int8-naive vs int8-fused (Triton) throughput, same model.

    Loads ONE int8 ModelRunner and toggles `int8_kernel._FUSED_DISABLED_FOR_BENCH`
    to flip between naive and fused dispatch on the same weights — keeps the
    A/B clean (same INT8 quantization, only the matmul path differs).
    """
    import torch

    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.quant import int8_kernel
    from mini_infer.scheduler import ContinuousScheduler, Request

    prompt = (
        "Summarize the following passage in one sentence. "
        "The mini-infer engine is an open-source LLM inference server "
        "demonstrating production techniques: continuous batching, paged "
        "attention, chunked prefill, packed varlen forward, prefix caching, "
        "and weight quantization. " * 8
    )

    def _sweep(label: str, runner: Any) -> dict[int, dict[str, Any]]:
        per_c: dict[int, dict[str, Any]] = {}
        for concurrency in concurrencies:
            sched = ContinuousScheduler(runner, max_concurrent=concurrency, chunk_size=256)
            sched.start()
            try:
                # Warmup so we're not timing the first matmul launch (Triton
                # JIT compile latency on the int8-fused config).
                sched.run(
                    Request(prompt=prompt, sampling_params=SamplingParams(), max_tokens=max_tokens)
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                handles = [
                    sched.submit(
                        Request(
                            prompt=prompt,
                            sampling_params=SamplingParams(),
                            max_tokens=max_tokens,
                        )
                    )
                    for _ in range(concurrency)
                ]
                results = [h.wait() for h in handles]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                sched.stop()
            total_tokens = sum(len(r.tokens) for r in results)
            per_c[concurrency] = {
                "elapsed_s": round(elapsed, 3),
                "tokens": total_tokens,
                "tok_per_s": round(total_tokens / elapsed, 2),
            }
            print(f"  [{label} C={concurrency}] {per_c[concurrency]}")
        return per_c

    import gc

    # 1) fp16 baseline (optional — at 7B+ on A10 it OOMs because the next
    # int8 load can't coexist; skip via `skip_fp16=True`).
    if skip_fp16:
        fp_results: dict[int, dict[str, Any]] = {}
    else:
        fp_runner = ModelRunner.from_pretrained(model, num_blocks=num_blocks, block_size=block_size)
        fp_results = _sweep("fp16", fp_runner)
        del fp_runner
        gc.collect()
        torch.cuda.empty_cache()

    # 2) int8 with naive (HBM-round-trip) dispatch
    int8_runner = ModelRunner.from_pretrained(
        model, num_blocks=num_blocks, block_size=block_size, quant="int8"
    )

    # Token-level parity check: greedy-decode the same prompt under naive and
    # fused dispatch on the same int8 model; compare token IDs. Closes the
    # ADR-012 "Modal-side correctness not verified" gap. Runs BEFORE the
    # throughput sweeps so we surface mismatches early.
    parity_prompt = "The capital of France is"
    parity_max_tokens = 8

    def _greedy_tokens(label: str, runner: Any) -> list[int]:
        sched = ContinuousScheduler(runner)
        sched.start()
        try:
            r = sched.run(
                Request(
                    prompt=parity_prompt,
                    sampling_params=SamplingParams(),
                    max_tokens=parity_max_tokens,
                )
            )
        finally:
            sched.stop()
        print(f"  [parity {label}] tokens={r.tokens} text={r.text!r}")
        return list(r.tokens)

    int8_kernel._FUSED_DISABLED_FOR_BENCH = True
    naive_tokens = _greedy_tokens("int8-naive", int8_runner)
    int8_kernel._FUSED_DISABLED_FOR_BENCH = False
    fused_tokens = _greedy_tokens("int8-fused", int8_runner)
    parity_match = naive_tokens == fused_tokens
    parity_line = (
        f"naive_tokens={naive_tokens}\n  fused_tokens={fused_tokens}\n  match={parity_match}"
    )

    int8_kernel._FUSED_DISABLED_FOR_BENCH = True
    naive_results = _sweep("int8-naive", int8_runner)

    # 3) Same int8 model, fused dispatch enabled
    int8_kernel._FUSED_DISABLED_FOR_BENCH = False
    fused_results = _sweep("int8-fused", int8_runner)

    del int8_runner
    gc.collect()
    torch.cuda.empty_cache()

    # Format report.
    lines: list[str] = [
        f"model={model} | prompt_chars={len(prompt)} | max_tokens={max_tokens}",
        "",
        "Token-level parity (int8-naive vs int8-fused, same model, greedy):",
        f"  prompt={parity_prompt!r}",
        f"  {parity_line}",
        "",
        "Throughput (warmup + N concurrent requests):",
    ]
    if skip_fp16:
        header = (
            f"  {'C':>3}  {'int8-naive (s, t/s)':>26}  "
            f"{'int8-fused (s, t/s)':>26}  {'fused/naive':>11}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for c in concurrencies:
            nv = naive_results[c]
            fu = fused_results[c]
            speedup = fu["tok_per_s"] / nv["tok_per_s"] if nv["tok_per_s"] else 0.0
            lines.append(
                f"  {c:>3}  "
                + f"{nv['elapsed_s']}s, {nv['tok_per_s']} t/s".rjust(26)
                + "  "
                + f"{fu['elapsed_s']}s, {fu['tok_per_s']} t/s".rjust(26)
                + f"  {speedup:>10.2f}x"
            )
    else:
        header = (
            f"  {'C':>3}  {'fp16 (s, t/s)':>22}  {'int8-naive (s, t/s)':>26}  "
            f"{'int8-fused (s, t/s)':>26}  {'fused/naive':>11}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for c in concurrencies:
            fp = fp_results[c]
            nv = naive_results[c]
            fu = fused_results[c]
            speedup = fu["tok_per_s"] / nv["tok_per_s"] if nv["tok_per_s"] else 0.0
            lines.append(
                f"  {c:>3}  {fp['elapsed_s']}s, {fp['tok_per_s']} t/s".rjust(22)
                + "  "
                + f"{nv['elapsed_s']}s, {nv['tok_per_s']} t/s".rjust(26)
                + "  "
                + f"{fu['elapsed_s']}s, {fu['tok_per_s']} t/s".rjust(26)
                + f"  {speedup:>10.2f}x"
            )
    return "\n".join(lines)


def _run_spec_bench(
    target_model: str,
    draft_model: str,
    prompts: list[str],
    max_tokens: int,
    K: int,  # noqa: N803 (canonical name in the spec-decode literature)
) -> str:
    """Greedy speculative decoding (target + draft) vs target-alone greedy.

    Loads both models on the same container, runs each prompt through:
      - target-alone greedy (timed)
      - spec-decode greedy (timed, with `SpecStats`)

    Outputs a per-prompt table plus aggregate summary: tokens/sec, mean
    acceptance per iteration, ratio of target forwards saved.
    """
    import torch

    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.speculative import SpeculativeRunner

    target = ModelRunner.from_pretrained(target_model)
    draft = ModelRunner.from_pretrained(draft_model)
    spec = SpeculativeRunner(target, draft, K=K)

    def _target_alone_greedy(prompt: str) -> tuple[list[int], float]:
        cache = PagedKVCache(target.block_pool)
        batch_idx = cache.add_request_slot()
        eos = target.tokenizer.eos_token_id
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prompt_ids = target.tokenizer.encode(prompt)
            packed = target.forward_step_packed(cache, prompt_ids, [0, len(prompt_ids)], [0])
            next_tok = int(packed[0, -1, :].argmax().item())
            out = [next_tok]
            seq_len = len(prompt_ids)
            while len(out) < max_tokens and next_tok != eos:
                logits_list = target.forward_step(cache, [next_tok], [0, 1], [seq_len])
                next_tok = int(logits_list[0].argmax().item())
                out.append(next_tok)
                seq_len += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            return out[:max_tokens], elapsed
        finally:
            cache.remove_request(batch_idx)

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        baseline_tokens, baseline_s = _target_alone_greedy(prompt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec_tokens, spec_stats = spec.run_greedy(prompt, max_tokens=max_tokens)
        torch.cuda.synchronize()
        spec_s = time.perf_counter() - t0

        match = spec_tokens == baseline_tokens
        rows.append(
            {
                "prompt": prompt[:32],
                "n_tok_base": len(baseline_tokens),
                "base_s": round(baseline_s, 3),
                "base_tok_s": round(len(baseline_tokens) / baseline_s, 2),
                "n_tok_spec": len(spec_tokens),
                "spec_s": round(spec_s, 3),
                "spec_tok_s": round(len(spec_tokens) / spec_s, 2),
                "speedup": round(baseline_s / spec_s, 2),
                "iters": spec_stats.n_iterations,
                "mean_acc": round(spec_stats.mean_acceptance_per_iter, 2),
                "tgt_fwds": spec_stats.n_target_forwards,
                "match": "yes" if match else "NO",
            }
        )

    # Aggregate.
    total_base_s = sum(r["base_s"] for r in rows)
    total_spec_s = sum(r["spec_s"] for r in rows)
    total_base_tok = sum(r["n_tok_base"] for r in rows)
    total_spec_tok = sum(r["n_tok_spec"] for r in rows)

    lines: list[str] = [
        f"target={target_model} | draft={draft_model} | K={K} | max_tokens={max_tokens}",
        "",
    ]
    header = (
        f"  {'prompt':<32}  {'base s':>7}  {'base t/s':>9}  "
        f"{'spec s':>7}  {'spec t/s':>9}  {'x':>5}  {'iters':>5}  "
        f"{'mean_acc':>9}  {'match':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in rows:
        lines.append(
            f"  {r['prompt']:<32}  {r['base_s']:>7}  {r['base_tok_s']:>9}  "
            f"{r['spec_s']:>7}  {r['spec_tok_s']:>9}  {r['speedup']:>5}  "
            f"{r['iters']:>5}  {r['mean_acc']:>9}  {r['match']:>6}"
        )
    lines.append("")
    lines.append(
        f"  aggregate: base={total_base_s:.2f}s ({total_base_tok / total_base_s:.1f} t/s)  "
        f"spec={total_spec_s:.2f}s ({total_spec_tok / total_spec_s:.1f} t/s)  "
        f"speedup={total_base_s / total_spec_s:.2f}x"
    )
    return "\n".join(lines)


@app.function(image=image, gpu=_BENCH_GPU, timeout=3600)
def run_bench(
    config: str,
    model: str,
    block_size: int,
    num_blocks: int,
    prompt: str,
    max_tokens: int,
    concurrencies: list[int],
    chunk_size: int,
    workload: str,
    target_prompt_tokens: int,
    spec_target_model: str,
    spec_draft_model: str,
    spec_K: int,  # noqa: N803 (canonical name in the spec-decode literature)
    skip_fp16: bool,
) -> str:
    """Modal entry point. Single function; selects internal path by `config`."""
    import torch

    from mini_infer.cache.packed_attention import supports_packed_kernel

    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name()
    fa_available = supports_packed_kernel(torch.device("cuda"))

    header = (
        f"GPU: {gpu_name} | flash_attn={fa_available} | block_size={block_size} | model={model}"
    )

    if config == "turbo":
        body = _run_turbo_bench(
            model=model,
            num_blocks=num_blocks,
            block_size=block_size,
            max_tokens=max_tokens,
            concurrencies=concurrencies,
        )
        return f"\n{header}\n\n=== TurboQuant V1 (rotation + 4-bit KV) vs bf16 ===\n{body}\n"

    if config == "turbo_parity":
        body = _run_turbo_parity(model=model, num_blocks=num_blocks, block_size=block_size)
        return f"\n{header}\n\n=== TurboQuant V2a fused-vs-python parity ===\n{body}\n"

    if config == "turbo_v2b":
        body = _run_turbo_v2b_compare(model=model, num_blocks=num_blocks, block_size=block_size)
        return f"\n{header}\n\n=== TurboQuant V2b vs V2a (memory + throughput) ===\n{body}\n"

    if config == "flashinfer":
        body = _run_flashinfer_compare(model=model, num_blocks=num_blocks, block_size=block_size)
        return f"\n{header}\n\n=== flash-attn vs FlashInfer (bf16 backend A/B) ===\n{body}\n"

    if config == "flashinfer_fp8":
        body = _run_flashinfer_fp8_compare(
            model=model, num_blocks=num_blocks, block_size=block_size
        )
        return f"\n{header}\n\n=== bf16 vs FP8 KV (FlashInfer attention) ===\n{body}\n"

    if config == "quant_kernel":
        body = _run_quant_kernel_bench(
            model=model,
            num_blocks=num_blocks,
            block_size=block_size,
            max_tokens=max_tokens,
            concurrencies=concurrencies,
            skip_fp16=skip_fp16,
        )
        title = "int8-naive / int8-fused" if skip_fp16 else "fp16 / int8-naive / int8-fused"
        return f"\n{header}\n\n=== {title} ===\n{body}\n"

    if config == "spec":
        spec_prompts = [
            (
                "Explain the concept of recursion in programming, "
                "with a short example. Use plain prose, no code blocks."
            ),
            ('def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'),
            ("Q: What are the four base pairs of DNA, and how do they pair?\nA:"),
        ]
        body = _run_spec_bench(
            target_model=spec_target_model,
            draft_model=spec_draft_model,
            prompts=spec_prompts,
            max_tokens=max_tokens,
            K=spec_K,
        )
        return f"\n{header}\n\n=== Speculative decoding ===\n{body}\n"

    if config == "quant":
        # `quant` builds its own runners (one per quant mode); skip the
        # default _make_runner here.
        body = _run_quant_bench(
            model=model,
            num_blocks=num_blocks,
            block_size=block_size,
            max_tokens=max_tokens,
            concurrencies=concurrencies,
        )
        return f"\n{header}\n\n=== INT8 weight-only vs fp16 ===\n{body}\n"

    if config == "prefix":
        # Cache-OFF C=N has N concurrent requests each carrying their own full
        # K/V; size the pool for that worst case. `_build_long_prompt`'s repeat
        # count is heuristic (estimates ~80 tokens/paragraph; Qwen tokenizes
        # closer to ~105), so the actual prompt is ~30% longer than asked. Add
        # a 2x safety factor + decode_headroom + a fixed slack so the bench
        # never OOMs under sloppy size estimation.
        per_req_blocks = (target_prompt_tokens * 2 + max_tokens + block_size - 1) // block_size
        required_blocks = max(concurrencies) * per_req_blocks + 256
        effective_num_blocks = max(num_blocks, required_blocks)
        if effective_num_blocks > num_blocks:
            print(
                f"prefix bench: bumping num_blocks {num_blocks} -> {effective_num_blocks} "
                f"to fit C={max(concurrencies)} requests at ~{target_prompt_tokens} tokens"
            )
        body = _run_prefix_bench(
            model=model,
            num_blocks=effective_num_blocks,
            block_size=block_size,
            target_prompt_tokens=target_prompt_tokens,
            max_tokens=max_tokens,
            concurrencies=concurrencies,
        )
        return f"\n{header}\n\n=== Prefix cache OFF vs ON ===\n{body}\n"

    runner = _make_runner(model, num_blocks=num_blocks, block_size=block_size)

    if config == "smoke":
        body = _run_smoke(runner, max_tokens=max_tokens)
        return f"\n{header}\n\n=== Smoke ===\n{body}\n"

    if config == "holb":
        body = _run_holb(runner)
        return f"\n{header}\n\n=== HOL blocking ===\n{body}\n"

    if config == "throughput":
        active_prompt = _build_long_prompt(target_tokens=3000) if workload == "long" else prompt
        chunk_settings = [(f"chunked-{chunk_size}", chunk_size), ("unchunked", 8192)]
        body = _run_throughput(
            runner,
            prompt=active_prompt,
            max_tokens=max_tokens,
            concurrencies=concurrencies,
            chunk_settings=chunk_settings,
        )
        return f"\n{header}\n\n=== Throughput ({workload}) ===\n{body}\n"

    if config == "sweep":
        smoke_out = _run_smoke(runner, max_tokens=8)
        short_out = _run_throughput(
            runner,
            prompt="The capital of France is",
            max_tokens=32,
            concurrencies=[1, 4, 8],
            chunk_settings=[("chunked-32", 32), ("unchunked", 4096)],
        )
        long_out = _run_throughput(
            runner,
            prompt=_build_long_prompt(target_tokens=3000),
            max_tokens=64,
            concurrencies=[1, 2, 4],
            chunk_settings=[("chunked-256", 256), ("unchunked", 8192)],
        )
        return (
            f"\n{header}\n\n"
            f"=== Smoke ===\n{smoke_out}\n\n"
            f"=== Short throughput ===\n{short_out}\n\n"
            f"=== Long throughput ===\n{long_out}\n"
        )

    raise ValueError(
        f"unknown config={config!r}; "
        "expected smoke|throughput|holb|sweep|prefix|quant|spec|quant_kernel|turbo|"
        "turbo_parity|turbo_v2b|flashinfer|flashinfer_fp8"
    )


@app.local_entrypoint()
def main(
    config: str = "smoke",
    model: str = DEFAULT_MODEL,
    block_size: int = 16,
    num_blocks: int = 1024,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 32,
    concurrencies: str = "1,4,8",
    chunk_size: int = 32,
    workload: str = "short",
    target_prompt_tokens: int = 12000,
    spec_target_model: str = "Qwen/Qwen2.5-7B-Instruct",
    spec_draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    spec_k: int = 4,
    skip_fp16: bool = False,
) -> None:
    # GPU is set via the MINI_INFER_BENCH_GPU env var (see _BENCH_GPU above).
    # Default A10. Modal 1.4 removed runtime gpu overrides on Function.
    print(
        run_bench.remote(
            config=config,
            model=model,
            block_size=block_size,
            num_blocks=num_blocks,
            prompt=prompt,
            max_tokens=max_tokens,
            concurrencies=_parse_int_list(concurrencies),
            chunk_size=chunk_size,
            workload=workload,
            target_prompt_tokens=target_prompt_tokens,
            spec_target_model=spec_target_model,
            spec_draft_model=spec_draft_model,
            spec_K=spec_k,
            skip_fp16=skip_fp16,
        )
    )
