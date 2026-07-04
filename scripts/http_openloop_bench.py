"""Open-loop, rate-swept HTTP benchmark.

Fires HTTP requests at a fixed offered rate against the mini-infer
streaming endpoint (`POST /v1/completions` with `stream: true`),
captures per-request TTFT and ITL by reading the SSE stream, and
reports p50/p90/p99 latencies and achieved RPS across a rate sweep.

Open-loop means clients send on a fixed cadence regardless of how
many requests are still in flight on the server. This is the
methodology Modal's LLM Almanac and Neural Magic's guidellm use; it
measures how the engine degrades under load, which closed-loop "N
concurrent workers" benches (scripts/modal_packed_bench.py) cannot.

Usage (two terminals):

  # terminal A: start the server
  uv run python -m mini_infer.api.server

  # terminal B: drive it
  uv run python scripts/http_openloop_bench.py \
      --url http://localhost:8000 \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --rates 1,2,4,8 \
      --duration 20 \
      --max-tokens 128

The default rate sweep and duration are intentionally light for a
local smoke run; bump both for GPU benchmarks. The script prints a
markdown table at the end matching the format used in
docs/benchmarks/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import httpx
except ImportError as exc:
    sys.stderr.write("httpx is required. Install with: uv add --dev httpx\n")
    raise SystemExit(1) from exc


_PASSAGE_PATH = Path(__file__).parent / "data" / "technical_passage.md"


def _load_prompt(prompt_chars: int | None) -> str:
    """Return a realistic prompt clipped to roughly `prompt_chars` characters.

    Uses the same long-form technical passage that the closed-loop
    benches use, so numbers are comparable across harnesses.
    """
    if not _PASSAGE_PATH.exists():
        raise FileNotFoundError(
            f"technical_passage.md not found at {_PASSAGE_PATH}; "
            "this script reuses the same prompt corpus as modal_packed_bench.py"
        )
    text = _PASSAGE_PATH.read_text()
    if prompt_chars is not None and prompt_chars < len(text):
        text = text[:prompt_chars]
    return text


@dataclass
class RequestResult:
    target_rate: float
    submit_t: float
    first_token_t: float | None
    finish_t: float | None
    n_tokens: int
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.first_token_t is not None and self.finish_t is not None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_t is None:
            return None
        return (self.first_token_t - self.submit_t) * 1000.0

    @property
    def total_ms(self) -> float | None:
        if self.finish_t is None:
            return None
        return (self.finish_t - self.submit_t) * 1000.0

    @property
    def itl_ms(self) -> float | None:
        """Inter-token latency: mean ms between successive tokens after the first."""
        if self.first_token_t is None or self.finish_t is None or self.n_tokens <= 1:
            return None
        decode_span = self.finish_t - self.first_token_t
        return (decode_span / (self.n_tokens - 1)) * 1000.0


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    target_rate: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    submit_t = time.perf_counter()
    first_token_t: float | None = None
    finish_t: float | None = None
    n_tokens = 0
    error: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                error = f"HTTP {response.status_code}: {body[:200]!r}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        finish_t = time.perf_counter()
                        break
                    if not body:
                        continue
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or [{}]
                    text = choices[0].get("text", "")
                    if text:
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                        n_tokens += 1
                    if choices[0].get("finish_reason") is not None:
                        finish_t = time.perf_counter()
                if finish_t is None:
                    finish_t = time.perf_counter()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return RequestResult(
        target_rate=target_rate,
        submit_t=submit_t,
        first_token_t=first_token_t,
        finish_t=finish_t,
        n_tokens=n_tokens,
        error=error,
    )


async def _run_at_rate(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    target_rate: float,
    duration_s: float,
    warmup_n: int,
) -> list[RequestResult]:
    """Submit at constant cadence `1/target_rate` for `duration_s` seconds.

    Open-loop: a new request is launched at every scheduled tick regardless
    of how many are still pending. If the server can't keep up, in-flight
    count grows and per-request latency degrades, which is exactly the
    signal we want to measure.
    """
    interval = 1.0 / target_rate
    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=64)
    async with httpx.AsyncClient(limits=limits) as client:
        for _ in range(warmup_n):
            await _one_request(client, url, model, prompt, max_tokens, target_rate)

        tasks: list[asyncio.Task[RequestResult]] = []
        start = time.perf_counter()
        next_tick = start
        while True:
            now = time.perf_counter()
            if now - start >= duration_s:
                break
            sleep_for = next_tick - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            tasks.append(
                asyncio.create_task(
                    _one_request(client, url, model, prompt, max_tokens, target_rate)
                )
            )
            next_tick += interval

        results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


@dataclass
class RateSummary:
    target_rate: float
    n_ok: int
    n_err: int
    achieved_rps: float
    ttft_p50: float | None
    ttft_p90: float | None
    ttft_p99: float | None
    itl_p50: float | None
    itl_p90: float | None
    itl_p99: float | None
    total_p50: float | None
    total_p90: float | None


def _summarize(target_rate: float, results: list[RequestResult]) -> RateSummary:
    ok = [r for r in results if r.ok]
    n_err = len(results) - len(ok)
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    itls = [r.itl_ms for r in ok if r.itl_ms is not None]
    totals = [r.total_ms for r in ok if r.total_ms is not None]

    if ok:
        first_submit = min(r.submit_t for r in ok)
        last_finish = max(r.finish_t for r in ok if r.finish_t is not None)
        achieved = len(ok) / max(last_finish - first_submit, 1e-9)
    else:
        achieved = 0.0

    return RateSummary(
        target_rate=target_rate,
        n_ok=len(ok),
        n_err=n_err,
        achieved_rps=achieved,
        ttft_p50=_percentile(ttfts, 0.50),
        ttft_p90=_percentile(ttfts, 0.90),
        ttft_p99=_percentile(ttfts, 0.99),
        itl_p50=_percentile(itls, 0.50),
        itl_p90=_percentile(itls, 0.90),
        itl_p99=_percentile(itls, 0.99),
        total_p50=_percentile(totals, 0.50),
        total_p90=_percentile(totals, 0.90),
    )


def _format_table(summaries: list[RateSummary]) -> str:
    headers = [
        "target",
        "achieved",
        "n_ok",
        "n_err",
        "TTFT p50",
        "TTFT p90",
        "TTFT p99",
        "ITL p50",
        "ITL p90",
        "ITL p99",
        "total p50",
        "total p90",
    ]

    def cell(v: float | None, fmt: str = "{:.1f}") -> str:
        return fmt.format(v) if v is not None else "-"

    rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for s in summaries:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{s.target_rate:.1f}",
                    f"{s.achieved_rps:.2f}",
                    f"{s.n_ok}",
                    f"{s.n_err}",
                    cell(s.ttft_p50),
                    cell(s.ttft_p90),
                    cell(s.ttft_p99),
                    cell(s.itl_p50),
                    cell(s.itl_p90),
                    cell(s.itl_p99),
                    cell(s.total_p50),
                    cell(s.total_p90),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _parse_rates(spec: str) -> list[float]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"rate must be positive, got {v}")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("at least one rate required")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default="http://localhost:8000", help="server base URL")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="model name in the request payload",
    )
    p.add_argument("--rates", default="1,2,4,8", help="comma-separated target RPS values")
    p.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="seconds of measurement per rate point",
    )
    p.add_argument("--warmup", type=int, default=5, help="discarded warmup requests per rate")
    p.add_argument("--max-tokens", type=int, default=128, help="max output tokens per request")
    p.add_argument(
        "--prompt-chars",
        type=int,
        default=None,
        help="truncate the technical passage to this many chars (default: full passage)",
    )
    p.add_argument("--json-out", type=Path, default=None, help="optional JSON dump path")
    args = p.parse_args()

    rates = _parse_rates(args.rates)
    prompt = _load_prompt(args.prompt_chars)

    print(
        f"# open-loop rate sweep\n"
        f"- url: {args.url}\n"
        f"- model: {args.model}\n"
        f"- rates: {rates}\n"
        f"- duration/rate: {args.duration}s\n"
        f"- warmup/rate: {args.warmup} requests\n"
        f"- max_tokens: {args.max_tokens}\n"
        f"- prompt: {len(prompt)} chars\n",
        flush=True,
    )

    summaries: list[RateSummary] = []
    for rate in rates:
        print(f"## target rate {rate:.1f} RPS", flush=True)
        t0 = time.perf_counter()
        results = asyncio.run(
            _run_at_rate(
                url=args.url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                target_rate=rate,
                duration_s=args.duration,
                warmup_n=args.warmup,
            )
        )
        elapsed = time.perf_counter() - t0
        s = _summarize(rate, results)
        summaries.append(s)
        print(
            f"  took {elapsed:.1f}s, "
            f"achieved {s.achieved_rps:.2f} RPS, "
            f"TTFT p50/p90/p99 = "
            f"{(s.ttft_p50 or 0):.0f}/{(s.ttft_p90 or 0):.0f}/{(s.ttft_p99 or 0):.0f} ms, "
            f"ITL p50 = {(s.itl_p50 or 0):.1f} ms, "
            f"errs = {s.n_err}",
            flush=True,
        )

    print("\n## results (all latencies in ms)\n")
    print(_format_table(summaries))

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "url": args.url,
                    "model": args.model,
                    "duration_s": args.duration,
                    "warmup_n": args.warmup,
                    "max_tokens": args.max_tokens,
                    "prompt_chars": len(prompt),
                    "rates": [s.__dict__ for s in summaries],
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
