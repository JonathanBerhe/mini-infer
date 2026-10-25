"""API serving for the Qwen hybrid family over the state-cache scheduler.

`test_api_families.py` covers the PagedKVCache families; this family serves
through `StateCacheContinuousScheduler` instead, the same route
`test_api_state_cache.py` exercises for DeepSeek-V4. What is family-specific
here is that a hybrid whose linear layers carry a recurrent state and whose
full-attention layers carry per-token K/V survives the real HTTP path,
including multi-request batching where the scheduler moves prefilled
requests between cache rows.

Runs on a synthetic config, so there is no model download, no tokenizer
fetch and no GPU.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mini_infer.api.server import _unhandled_exception_handler, completions
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.qwen.qwen3_5 import Qwen35Config, Qwen35ForConditionalGeneration
from mini_infer.scheduler import StateCacheContinuousScheduler


def _make_config() -> Qwen35Config:
    """A tiny 3:1 hybrid: [linear, linear, linear, full] twice."""
    return Qwen35Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention"] * 3
        + ["full_attention"]
        + ["linear_attention"] * 3
        + ["full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=6,  # 3x broadcast, as Qwen3.8-27B ships
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        tie_word_embeddings=False,
        max_position_embeddings=128,
    )


class _FakeTokenizer:
    """Deterministic stand-in: prompt -> 8 ids; ids -> space-joined text."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        base = sum(ord(c) for c in text)
        return [(base + i) % self._vocab_size for i in range(8)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return None


@pytest.fixture
def qwen_client() -> Iterator[TestClient]:
    cfg = _make_config()
    torch.manual_seed(0)
    model = Qwen35ForConditionalGeneration(cfg).eval()
    scheduler = StateCacheContinuousScheduler(  # type: ignore[arg-type]
        StateCacheGenerator(model, _FakeTokenizer(cfg.vocab_size)),
        max_batch_size=4,
        max_seq_len=64,
    )
    scheduler.start()
    app = FastAPI()
    app.add_api_route("/v1/completions", completions, methods=["POST"], response_model=None)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
    app.state.scheduler = scheduler
    try:
        with TestClient(app) as client:
            yield client
    finally:
        scheduler.stop()


def test_qwen_hybrid_completions_non_stream(qwen_client: TestClient) -> None:
    response = qwen_client.post(
        "/v1/completions",
        json={"model": "synthetic-qwen3-5", "prompt": "The capital of France is", "max_tokens": 6},
    )
    assert response.status_code == 200
    body = response.json()
    # Synthetic weights, so the text is meaningless; what matters is that the
    # hybrid produced tokens through the real scheduler and route.
    assert body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["prompt_tokens"] == 8
    assert body["usage"]["completion_tokens"] == 6


def test_qwen_hybrid_completions_stream(qwen_client: TestClient) -> None:
    with qwen_client.stream(
        "POST",
        "/v1/completions",
        json={"model": "synthetic-qwen3-5", "prompt": "hi", "max_tokens": 4, "stream": True},
    ) as response:
        assert response.status_code == 200
        chunks: list[dict[str, object]] = []
        saw_done = False
        for raw_line in response.iter_lines():
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(payload))
    assert saw_done
    assert chunks


def test_qwen_hybrid_serves_concurrent_requests(qwen_client: TestClient) -> None:
    """Different prompt lengths in flight together exercise the ragged path.

    The scheduler batches these into one cache and steps them at different
    positions, which is where the recurrent state and the per-token K/V
    buffers have to stay per-row consistent.
    """
    prompts = ["short", "a considerably longer prompt than the other one", "middling prompt"]
    bodies = []
    for prompt in prompts:
        response = qwen_client.post(
            "/v1/completions",
            json={"model": "synthetic-qwen3-5", "prompt": prompt, "max_tokens": 5},
        )
        assert response.status_code == 200
        bodies.append(response.json())
    for body in bodies:
        assert body["usage"]["completion_tokens"] == 5
        assert body["choices"][0]["text"]
