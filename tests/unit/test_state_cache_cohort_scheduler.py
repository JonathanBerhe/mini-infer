"""StateCacheCohortScheduler: lockstep cohort serving of V4 (StateCache) requests.

CPU, synthetic V4 config + deterministic fake tokenizers. Covers cohort serving
matching single-stream output, per-request max_tokens within one cohort, EOS ->
stop, the public threaded path, and (length, sampling-params) cohort grouping.

The oracle throughout is self-consistency: a cohort's per-sequence output equals
running each prompt alone through `generate_ids`, since lockstep batching changes
only scheduling, not the math (the per-block batched attention is separately
bit-parity validated against the DeepSeek-V4 reference).
"""

from __future__ import annotations

import queue
import threading

import pytest
import torch

from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from mini_infer.scheduler import Request, RequestHandle, StateCacheCohortScheduler
from mini_infer.scheduler.request_state import RunningRequest


def _make_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=32,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(4, 8, 4, 8),
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


class _FakeTokenizer:
    """Deterministic stand-in: every prompt encodes to 8 ids (so prompts share length)."""

    def __init__(self, vocab_size: int, eos_token_id: int | None = None) -> None:
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        base = sum(ord(c) for c in text)
        return [(base + i) % self._vocab_size for i in range(8)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id


class _VarLenTokenizer(_FakeTokenizer):
    """Encodes to `len(text)` ids, so prompts land in different cohorts (and ""
    encodes to empty, to exercise the empty-prompt path)."""

    def encode(self, text: str) -> list[int]:
        base = sum(ord(c) for c in text)
        return [(base + i) % self._vocab_size for i in range(len(text))]


def _make_generator(
    eos_token_id: int | None = None, *, var_len: bool = False
) -> StateCacheGenerator:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    tokenizer_cls = _VarLenTokenizer if var_len else _FakeTokenizer
    tokenizer = tokenizer_cls(cfg.vocab_size, eos_token_id)
    return StateCacheGenerator(model, tokenizer)  # type: ignore[arg-type]


def _greedy_request(prompt: str, max_tokens: int) -> Request:
    return Request(
        prompt=prompt, sampling_params=SamplingParams(temperature=0.0), max_tokens=max_tokens
    )


def _build_running(gen: StateCacheGenerator, prompt: str, max_tokens: int) -> RunningRequest:
    running = RunningRequest(
        request=_greedy_request(prompt, max_tokens),
        output_queue=queue.Queue(maxsize=256),
    )
    running.prompt_token_ids = gen.tokenizer.encode(prompt)
    return running


# ---------- _serve_cohort: batched serving matches single-stream ----------


def test_serve_cohort_matches_single_stream() -> None:
    """A cohort of 3 equal-length prompts decodes exactly like 3 solo runs."""
    gen = _make_generator()
    scheduler = StateCacheCohortScheduler(gen)
    prompts = ["alpha", "beta!", "gamma"]
    cohort = [_build_running(gen, prompt, 6) for prompt in prompts]
    handles = [RequestHandle(running) for running in cohort]

    scheduler._serve_cohort(cohort)

    for prompt, handle in zip(prompts, handles, strict=True):
        result = handle.wait()
        assert result.tokens == gen.generate_ids(gen.tokenizer.encode(prompt), max_new_tokens=6)
        assert result.finish_reason == "length"
        assert len(result.tokens) == 6


def test_serve_cohort_respects_per_request_max_tokens() -> None:
    """Within one cohort each sequence stops at its own max_tokens (reason length)."""
    gen = _make_generator()
    scheduler = StateCacheCohortScheduler(gen)
    prompts = ["one", "two", "six"]
    per_request_max = [2, 4, 6]
    cohort = [_build_running(gen, p, mt) for p, mt in zip(prompts, per_request_max, strict=True)]
    handles = [RequestHandle(running) for running in cohort]

    scheduler._serve_cohort(cohort)

    for prompt, max_tokens, handle in zip(prompts, per_request_max, handles, strict=True):
        result = handle.wait()
        assert len(result.tokens) == max_tokens
        assert result.finish_reason == "length"
        assert result.tokens == gen.generate_ids(
            gen.tokenizer.encode(prompt), max_new_tokens=max_tokens
        )


def test_serve_cohort_eos_produces_stop() -> None:
    """A sequence that hits EOS finishes with reason stop; the rest are unaffected."""
    baseline = _make_generator()
    eos = baseline.generate_ids(baseline.tokenizer.encode("x"), max_new_tokens=1)[0]

    gen = _make_generator(eos_token_id=eos)
    scheduler = StateCacheCohortScheduler(gen)
    cohort = [_build_running(gen, "x", 8), _build_running(gen, "y", 8)]
    handles = [RequestHandle(running) for running in cohort]

    scheduler._serve_cohort(cohort)

    result_x = handles[0].wait()
    assert result_x.finish_reason == "stop"
    assert result_x.tokens == []  # EOS was the very first token, nothing emitted
    # The sibling matches its own solo run with the same EOS.
    result_y = handles[1].wait()
    assert result_y.tokens == gen.generate_ids(
        gen.tokenizer.encode("y"), max_new_tokens=8, eos_token_id=eos
    )


# ---------- public threaded path ----------


def test_cohort_scheduler_concurrent_matches_single_stream() -> None:
    """Several clients submit at once; each gets its correct generation."""
    gen = _make_generator()
    prompts = ["alpha", "beta!!", "gamma3", "delta", "epsiln"]
    expected = {p: gen.generate_ids(gen.tokenizer.encode(p), max_new_tokens=5) for p in prompts}

    scheduler = StateCacheCohortScheduler(gen)
    scheduler.start()
    results: dict[str, list[int]] = {}
    lock = threading.Lock()

    def worker(prompt: str) -> None:
        out = scheduler.run(_greedy_request(prompt, max_tokens=5))
        with lock:
            results[prompt] = out.tokens

    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        scheduler.stop()

    assert results == expected


# ---------- cohort formation ----------


def test_form_cohorts_groups_by_length_and_params() -> None:
    """Requests bucket by (prompt length, sampling params), split at max_cohort_size."""
    gen = _make_generator(var_len=True)
    scheduler = StateCacheCohortScheduler(gen, max_cohort_size=2)
    greedy = SamplingParams(temperature=0.0)
    hot = SamplingParams(temperature=0.7)

    def running(prompt: str, params: SamplingParams) -> RunningRequest:
        return RunningRequest(
            request=Request(prompt=prompt, sampling_params=params, max_tokens=4),
            output_queue=queue.Queue(maxsize=8),
        )

    batch = [
        running("aa", greedy),  # len 2, greedy
        running("bb", greedy),  # len 2, greedy
        running("cc", greedy),  # len 2, greedy -> overflows the cap-2 cohort
        running("ddd", greedy),  # len 3, greedy -> own group
        running("ee", hot),  # len 2, but different params -> own group
    ]
    cohorts = scheduler._form_cohorts(batch)
    prompts_per_cohort = [[r.request.prompt for r in cohort] for cohort in cohorts]
    assert prompts_per_cohort == [["aa", "bb"], ["cc"], ["ddd"], ["ee"]]


def test_form_cohorts_finishes_cancelled_and_empty_outside_a_cohort() -> None:
    """Cancelled and empty-prompt requests are finished, not placed in a cohort."""
    gen = _make_generator(var_len=True)
    scheduler = StateCacheCohortScheduler(gen)

    good = _build_running(gen, "hello", 4)
    cancelled = _build_running(gen, "world", 4)
    cancelled.cancel_event.set()
    empty = RunningRequest(request=_greedy_request("", 4), output_queue=queue.Queue(maxsize=8))

    cohorts = scheduler._form_cohorts([good, cancelled, empty])

    assert cohorts == [[good]]
    assert cancelled.output_queue.get_nowait().finish_reason == "cancelled"
    assert empty.output_queue.get_nowait().finish_reason == "stop"


# ---------- construction / lifecycle ----------


def test_submit_before_start_raises() -> None:
    scheduler = StateCacheCohortScheduler(_make_generator())
    with pytest.raises(RuntimeError, match="not running"):
        scheduler.submit(_greedy_request("hi", max_tokens=2))


def test_rejects_nonpositive_cohort_size() -> None:
    with pytest.raises(ValueError, match="max_cohort_size"):
        StateCacheCohortScheduler(_make_generator(), max_cohort_size=0)
