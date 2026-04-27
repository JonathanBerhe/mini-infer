"""Head-of-line blocking microbench for chunked prefill.

The whole pitch of chunked prefill is: "while a long prompt is being prefilled,
in-flight decoders don't have to wait for it". This script measures that
directly.

For each chunk-size config (chunked vs un-chunked):
1. Submit a short decoder and stream its tokens, recording per-token timestamps.
2. After it has emitted a few tokens (so it's in steady-state decode mode),
   submit a long-prompt request.
3. Continue streaming the short decoder. Measure the inter-token latency (ITL)
   during the window when the long prompt is being prefilled.

Expected: chunked keeps the short decoder's ITL within ~2x of its baseline
(each scheduler step does one prefill chunk + one decode token, so the
decoder gets one token per step). Un-chunked freezes the decoder for the
full duration of the long prefill (one big forward, no decoder progress).
"""

# Run with: uv run modal run scripts/modal_packed_bench_hol.py

import itertools
import statistics
import threading
import time

import modal

app = modal.App("mini-infer-packed-bench-hol")

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
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


def _build_long_prompt(target_tokens: int = 3000) -> str:
    paragraph = (
        "The mini-infer engine is an open-source LLM inference server that "
        "demonstrates production-grade serving techniques. It implements "
        "continuous batching, paged attention with a paged KV cache, chunked "
        "prefill, and a packed varlen attention forward via FlashAttention. "
    )
    repeats = max(1, target_tokens // 50)
    return paragraph * repeats + "\n\nSummarize:"


@app.function(image=image, gpu="A10", timeout=900)
def bench() -> str:
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import ContinuousScheduler, Request

    runner = ModelRunner.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", num_blocks=4096, block_size=16
    )
    long_prompt = _build_long_prompt(target_tokens=3000)
    long_prompt_len = len(runner.tokenizer.encode(long_prompt))
    short_prompt = "The capital of France is"

    rows: list[dict[str, str | float | int]] = []
    for label, chunk_size in [("chunked-256", 256), ("unchunked", 8192)]:
        scheduler = ContinuousScheduler(runner, max_concurrent=8, chunk_size=chunk_size)
        scheduler.start()
        try:
            # Warmup so kernel caches are hot for both the short and long paths.
            scheduler.run(
                Request(
                    prompt=short_prompt,
                    sampling_params=SamplingParams(),
                    max_tokens=4,
                )
            )

            # Submit the short decoder. We'll stream its tokens and timestamp them.
            short_handle = scheduler.submit(
                Request(
                    prompt=short_prompt,
                    sampling_params=SamplingParams(),
                    max_tokens=64,
                )
            )

            short_token_times: list[float] = []
            done = threading.Event()

            def drain_short(
                handle=short_handle,
                token_times=short_token_times,
                done_event=done,
            ) -> None:
                for step in handle.steps():
                    token_times.append(time.perf_counter())
                    if step.finish_reason is not None:
                        break
                done_event.set()

            drain_thread = threading.Thread(target=drain_short, daemon=True)
            drain_thread.start()

            # Wait until the short decoder has emitted a few tokens so the
            # baseline-ITL window is stable.
            while len(short_token_times) < 4 and not done.is_set():
                time.sleep(0.005)

            # Mark the moment we submit the long prefill.
            long_submit_time = time.perf_counter()
            n_short_tokens_before_long = len(short_token_times)
            long_handle = scheduler.submit(
                Request(
                    prompt=long_prompt,
                    sampling_params=SamplingParams(),
                    max_tokens=8,
                )
            )

            long_handle.wait()
            long_done_time = time.perf_counter()
            done.wait(timeout=30.0)
            drain_thread.join(timeout=5.0)
        finally:
            scheduler.stop()

        # Compute the metric of interest: time-to-next-decode-token after the
        # long request lands. Chunked → ~one step's duration; un-chunked → the
        # full prefill duration.
        before_window_times = short_token_times[:n_short_tokens_before_long]
        after_window_times = short_token_times[n_short_tokens_before_long:]
        if len(before_window_times) >= 2:
            baseline_itl_ms = 1000.0 * statistics.fmean(
                later - earlier for earlier, later in itertools.pairwise(before_window_times)
            )
        else:
            baseline_itl_ms = float("nan")

        if after_window_times:
            time_to_next_short_after_long_ms = 1000.0 * (after_window_times[0] - long_submit_time)
        else:
            time_to_next_short_after_long_ms = float("nan")

        long_prefill_duration_ms = 1000.0 * (long_done_time - long_submit_time)

        rows.append(
            {
                "config": label,
                "baseline_itl_ms": round(baseline_itl_ms, 1),
                "next_short_token_after_long_ms": round(time_to_next_short_after_long_ms, 1),
                "long_request_total_ms": round(long_prefill_duration_ms, 1),
                "n_short_tokens_after_long": len(after_window_times),
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
                f"{row['next_short_token_after_long_ms']:>20} "
                f"{row['long_request_total_ms']:>14}"
            )
            for row in rows
        ],
    ]
    return (
        f"\nWorkload: short decoder (steady state) + long prefill "
        f"(prompt_len≈{long_prompt_len})\n"
        f"Metric: time from long-request submit to short decoder's NEXT token (ms)\n\n"
        + "\n".join(lines)
        + "\n"
    )


@app.local_entrypoint()
def main() -> None:
    print(bench.remote())
