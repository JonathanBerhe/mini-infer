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
"""

import itertools
import os
import statistics
import threading
import time
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
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=4.40",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "triton>=3.0",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .add_local_python_source("mini_infer")
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "The capital of France is"

# A synthetic ~3k-token "RAG" prompt used when `workload=long`. Built from a
# fixed paragraph repeated to approximately the requested token count.
_LONG_PARAGRAPH = (
    "The mini-infer engine is an open-source LLM inference server that "
    "demonstrates production-grade serving techniques. It implements "
    "continuous batching, paged attention with a paged KV cache, chunked "
    "prefill, and a packed varlen attention forward via FlashAttention. "
    "These techniques are the foundation of modern inference engines like "
    "vLLM and SGLang. The engine is structured to be readable, modular, "
    "and testable: a single engine thread owns the model and the running "
    "batch, while API threads only enqueue requests and drain output queues. "
)


def _build_long_prompt(target_tokens: int = 3000) -> str:
    repeats = max(1, target_tokens // 80)
    return _LONG_PARAGRAPH * repeats + "\n\nQuestion: Summarize.\n\nAnswer:"


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


@app.function(image=image, gpu=_BENCH_GPU, timeout=1800)
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

    raise ValueError(f"unknown config={config!r}; expected smoke|throughput|holb|sweep|prefix")


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
        )
    )
