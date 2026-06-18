"""StateCacheScheduler: queue-based concurrent serving of V4 (StateCache) requests.

CPU, synthetic V4 config + a deterministic fake tokenizer. Covers single-request
correctness, concurrent requests served in isolation, streaming deltas + a
terminal step, EOS -> stop, max_tokens -> length, cancellation, and the server's
V4-routing detection helper.
"""

from __future__ import annotations

import json
import threading

import pytest
import torch

from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models import architecture_uses_state_cache
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from mini_infer.scheduler import Request, StateCacheScheduler


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
    """Deterministic stand-in: prompt -> ids by char-sum; ids -> space-joined text."""

    def __init__(self, vocab_size: int, eos_token_id: int | None = None) -> None:
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        base = sum(ord(c) for c in text)
        # 8 ids (a multiple of the layers' compression ratios) so different
        # prompts map to different, aligned token sequences.
        return [(base + i) % self._vocab_size for i in range(8)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id


def _make_generator(eos_token_id: int | None = None) -> StateCacheGenerator:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    tokenizer = _FakeTokenizer(cfg.vocab_size, eos_token_id)
    return StateCacheGenerator(model, tokenizer)  # type: ignore[arg-type]


def _greedy_request(prompt: str, max_tokens: int) -> Request:
    return Request(
        prompt=prompt, sampling_params=SamplingParams(temperature=0.0), max_tokens=max_tokens
    )


def test_run_returns_correct_result() -> None:
    gen = _make_generator()
    # Reference: the generator's own greedy decode of the same prompt ids.
    expected = gen.generate_ids(gen.tokenizer.encode("hello"), max_new_tokens=6)

    scheduler = StateCacheScheduler(gen)
    scheduler.start()
    try:
        result = scheduler.run(_greedy_request("hello", max_tokens=6))
    finally:
        scheduler.stop()

    assert result.tokens == expected
    assert len(result.tokens) == 6
    assert result.finish_reason == "length"  # no eos, hit max_tokens
    assert result.prompt_tokens == 8
    assert result.text


def test_concurrent_requests_served_in_isolation() -> None:
    """Several clients submit at once; each gets its own correct generation."""
    gen = _make_generator()
    prompts = ["alpha", "beta!!", "gamma-3", "delta", "epsilon"]
    # Compute references before the engine starts (no overlapping model use).
    expected = {p: gen.generate_ids(gen.tokenizer.encode(p), max_new_tokens=5) for p in prompts}

    scheduler = StateCacheScheduler(gen)
    scheduler.start()
    results: dict[str, list[int]] = {}
    lock = threading.Lock()

    def worker(prompt: str) -> None:
        out = scheduler.run(_greedy_request(prompt, max_tokens=5))
        with lock:
            results[prompt] = out.tokens

    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        scheduler.stop()

    assert results == expected


def test_stream_yields_deltas_then_terminal() -> None:
    gen = _make_generator()
    scheduler = StateCacheScheduler(gen)
    scheduler.start()
    try:
        steps = list(scheduler.stream(_greedy_request("stream me", max_tokens=4)))
    finally:
        scheduler.stop()

    assert steps[-1].finish_reason == "length"
    text_steps = [s for s in steps if s.finish_reason is None]
    assert len(text_steps) == 4  # one delta per generated token
    assert "".join(s.text for s in text_steps)  # non-empty


def test_eos_produces_stop_finish_reason() -> None:
    # Find the first greedy token for this prompt, then make it the EOS token:
    # the scheduler should stop immediately with reason "stop" and no output.
    baseline = _make_generator()
    first_token = baseline.generate_ids(baseline.tokenizer.encode("x"), max_new_tokens=1)[0]

    gen = _make_generator(eos_token_id=first_token)
    scheduler = StateCacheScheduler(gen)
    scheduler.start()
    try:
        result = scheduler.run(_greedy_request("x", max_tokens=8))
    finally:
        scheduler.stop()

    assert result.finish_reason == "stop"
    assert result.tokens == []


def test_cancel_before_serve_produces_cancelled() -> None:
    """A queued request cancelled before the engine reaches it finishes cancelled."""
    gen = _make_generator()
    scheduler = StateCacheScheduler(gen)
    scheduler.start()
    try:
        # First request keeps the engine busy; the second waits behind it.
        occupier = scheduler.submit(_greedy_request("occupy the engine", max_tokens=30))
        victim = scheduler.submit(_greedy_request("cancel me", max_tokens=30))
        victim.cancel()  # set well before the engine finishes the occupier
        result = victim.wait()
        assert result.finish_reason == "cancelled"
        occupier.wait()  # drain so stop() joins cleanly
    finally:
        scheduler.stop()


def test_submit_before_start_raises() -> None:
    scheduler = StateCacheScheduler(_make_generator())
    with pytest.raises(RuntimeError, match="not running"):
        scheduler.submit(_greedy_request("hi", max_tokens=2))


# ---------- server routing detection ----------


def test_architecture_uses_state_cache_detects_v4(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}))
    assert architecture_uses_state_cache(str(tmp_path)) is True


def test_architecture_uses_state_cache_false_for_paged_model(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
    assert architecture_uses_state_cache(str(tmp_path)) is False


def test_architecture_uses_state_cache_false_for_unknown_arch(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["MysteryForCausalLM"]}))
    assert architecture_uses_state_cache(str(tmp_path)) is False
