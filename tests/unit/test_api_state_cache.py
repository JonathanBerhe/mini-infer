"""API integration for the V4 (StateCache) serving path, on CPU, no model download.

The HTTP layer (`/v1/completions`, SSE) is common to all of mini-infer and is
already covered for the PagedKVCache path. What is V4-specific is the
`StateCacheScheduler` behind it. This drives the REAL `completions` endpoint and
the REAL `StateCacheScheduler` end to end (HTTP -> endpoint -> scheduler ->
`StateCacheGenerator` -> response) over a small synthetic V4, so no real model,
no tokenizer download, and no GPU are needed. Serving a 2-GPU model (V4-Flash)
behind HTTP is a separate tensor-parallel deployment concern; real-model
generation itself is proven by `scripts/modal_v4_flash_generate.py`.
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
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from mini_infer.scheduler import StateCacheScheduler


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
def v4_client() -> Iterator[TestClient]:
    """A TestClient over the real completions endpoint backed by a synthetic V4."""
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    scheduler = StateCacheScheduler(StateCacheGenerator(model, _FakeTokenizer(cfg.vocab_size)))  # type: ignore[arg-type]
    scheduler.start()

    # Mount the real route + error handler on a fresh app with the scheduler
    # injected, so we exercise server.py's request/response path without its
    # model-loading lifespan.
    app = FastAPI()
    app.add_api_route("/v1/completions", completions, methods=["POST"], response_model=None)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
    app.state.scheduler = scheduler
    try:
        with TestClient(app) as client:
            yield client
    finally:
        scheduler.stop()


def test_v4_completions_non_stream(v4_client: TestClient) -> None:
    response = v4_client.post(
        "/v1/completions",
        json={"model": "synthetic-v4", "prompt": "The capital of France is", "max_tokens": 6},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"]  # non-empty (synthetic weights, content meaningless)
    assert body["choices"][0]["finish_reason"] == "length"  # no eos, hit max_tokens
    assert body["usage"]["prompt_tokens"] == 8
    assert body["usage"]["completion_tokens"] == 6
    assert body["usage"]["total_tokens"] == 14


def test_v4_completions_stream(v4_client: TestClient) -> None:
    with v4_client.stream(
        "POST",
        "/v1/completions",
        json={"model": "synthetic-v4", "prompt": "hi", "max_tokens": 4, "stream": True},
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
    assert any(c["choices"][0]["text"] for c in chunks)
    finish_reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert finish_reasons[-1] == "length"
