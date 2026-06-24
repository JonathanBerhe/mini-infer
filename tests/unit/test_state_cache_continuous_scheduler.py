"""StateCacheContinuousScheduler: ragged continuous batching == per-request scalar.

Submits more requests than fit in the batch, with different prompt lengths and
different max_tokens, so the engine admits / evicts dynamically and decodes a
ragged batch (rows at different positions, finishing at different steps). Each
request's output must equal running it alone through StateCacheGenerator (the
scalar oracle, itself bit-parity validated against the DeepSeek-V4 reference),
token-for-token. Full hybrid config (SWA + CSA + HCA) so all attention modes are
exercised under continuous batching.
"""

from __future__ import annotations

import threading

import pytest
import torch

from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from mini_infer.scheduler import Request, StateCacheContinuousScheduler


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
        compress_ratios=(0, 4, 8, 4),  # SWA + CSA + HCA
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


class _VarLenTokenizer:
    """prompt -> `len(text)` ids (so prompts differ in length); ids -> space-joined."""

    def __init__(self, vocab_size: int, eos_token_id: int | None = None) -> None:
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        base = sum(ord(c) for c in text)
        return [(base + i) % self._vocab_size for i in range(len(text))]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id


def _make_generator(eos_token_id: int | None = None) -> StateCacheGenerator:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    return StateCacheGenerator(model, _VarLenTokenizer(cfg.vocab_size, eos_token_id))  # type: ignore[arg-type]


def _greedy(prompt: str, max_tokens: int) -> Request:
    return Request(
        prompt=prompt, sampling_params=SamplingParams(temperature=0.0), max_tokens=max_tokens
    )


def test_continuous_matches_per_request_scalar() -> None:
    """6 requests, varied lengths + max_tokens, batch of 3 (so admit / evict)."""
    gen = _make_generator()
    specs = [("alphaaa", 4), ("bb", 6), ("gammaXYZ", 5), ("de", 3), ("epsilonnn", 7), ("zz", 4)]
    expected = {
        prompt: gen.generate_ids(gen.tokenizer.encode(prompt), max_new_tokens=max_tokens)
        for prompt, max_tokens in specs
    }

    scheduler = StateCacheContinuousScheduler(gen, max_batch_size=3, max_seq_len=64)
    scheduler.start()
    results: dict[str, tuple[list[int], str]] = {}
    lock = threading.Lock()

    def worker(prompt: str, max_tokens: int) -> None:
        out = scheduler.run(_greedy(prompt, max_tokens))
        with lock:
            results[prompt] = (out.tokens, out.finish_reason)

    threads = [threading.Thread(target=worker, args=(p, mt)) for p, mt in specs]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        scheduler.stop()

    for prompt, max_tokens in specs:
        tokens, reason = results[prompt]
        assert tokens == expected[prompt], f"{prompt}: {tokens} != {expected[prompt]}"
        assert reason == "length"  # no EOS, so each hits its own max_tokens
        assert len(tokens) == max_tokens


def test_continuous_respects_eos() -> None:
    """A request stops at its EOS (reason stop), matching its solo run."""
    baseline = _make_generator()
    eos = baseline.generate_ids(baseline.tokenizer.encode("hello"), max_new_tokens=6)[2]

    gen = _make_generator(eos_token_id=eos)
    expected = gen.generate_ids(gen.tokenizer.encode("hello"), max_new_tokens=6, eos_token_id=eos)

    scheduler = StateCacheContinuousScheduler(gen, max_batch_size=2, max_seq_len=64)
    scheduler.start()
    try:
        result = scheduler.run(_greedy("hello", 6))
    finally:
        scheduler.stop()

    assert result.tokens == expected
    assert eos not in result.tokens
    if len(expected) < 6:
        assert result.finish_reason == "stop"


def test_submit_before_start_raises() -> None:
    scheduler = StateCacheContinuousScheduler(_make_generator())
    with pytest.raises(RuntimeError, match="not running"):
        scheduler.submit(_greedy("hi", max_tokens=2))


def test_rejects_invalid_sizes() -> None:
    gen = _make_generator()
    with pytest.raises(ValueError, match="max_batch_size"):
        StateCacheContinuousScheduler(gen, max_batch_size=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        StateCacheContinuousScheduler(gen, max_seq_len=0)
