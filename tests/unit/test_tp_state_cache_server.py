"""Tensor-parallel serving parity: TP generation matches single-process.

gloo, 2 CPU processes, a synthetic V4 sharded via `from_checkpoint`'s TP weight
slicing (the same path real V4-Flash takes on 2x B200). Proves the leader /
follower coordination in `TensorParallelStateCacheServer` produces the SAME
tokens as a single-process generation, so serving a sharded model is correct.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from mini_infer.api.server import _unhandled_exception_handler, completions
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.engine.tp_state_cache_server import TensorParallelStateCacheServer
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from mini_infer.scheduler import TensorParallelStateCacheScheduler
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process


class _FakeTokenizer:
    """Deterministic stand-in used on the leader for the HTTP smoke (prompt -> ids)."""

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


def _make_tp_config() -> DeepseekV4Config:
    """Small hybrid (CSA/HCA) V4; every TP-sharded dim is divisible by 2."""
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


def _write_synthetic_checkpoint(checkpoint_dir: Path, cfg: DeepseekV4Config) -> None:
    torch.manual_seed(0)
    source = DeepseekV4ForCausalLM(cfg).eval()
    config_json = {**dataclasses.asdict(cfg), "architectures": ["DeepseekV4ForCausalLM"]}
    (checkpoint_dir / "config.json").write_text(json.dumps(config_json))
    save_file(
        {k: v.detach().clone().contiguous() for k, v in source.state_dict().items()},
        str(checkpoint_dir / "model.safetensors"),
    )


def _tp_generate_target(
    rank: int, world_size: int, checkpoint_dir: str, prompt_ids: list[int], max_new_tokens: int
) -> list[int] | None:
    """Per-rank work: load the sharded model, then generate (leader) or mirror (follower)."""
    model = DeepseekV4ForCausalLM.from_checkpoint(checkpoint_dir, device="cpu", dtype=torch.float32)
    server = TensorParallelStateCacheServer(model, device="cpu", dtype=torch.float32)
    if server.is_leader:
        tokens = server.generate_ids(prompt_ids, max_new_tokens=max_new_tokens)
        server.shutdown()
        return tokens
    server.run_follower_loop()
    return None


@pytest.mark.skipif(not is_multi_process_available(), reason="multi-process / gloo unavailable")
def test_tp_generation_matches_single_process(tmp_path: Path) -> None:
    cfg = _make_tp_config()
    _write_synthetic_checkpoint(tmp_path, cfg)
    prompt_ids = list(range(8))
    max_new_tokens = 6

    # Single-process reference (world_size == 1 in this process).
    reference_model = DeepseekV4ForCausalLM.from_checkpoint(
        str(tmp_path), device="cpu", dtype=torch.float32
    )
    reference = StateCacheGenerator(reference_model).generate_ids(
        prompt_ids, max_new_tokens=max_new_tokens
    )

    # Tensor-parallel over 2 gloo ranks: rank 0 leads, rank 1 follows.
    results = run_multi_process(
        2, _tp_generate_target, str(tmp_path), prompt_ids, max_new_tokens, timeout_sec=120.0
    )
    assert results[0] == reference, (
        f"TP generation diverged from single-process:\n  TP: {results[0]}\n  ref: {reference}"
    )
    assert len(reference) == max_new_tokens


def _tp_http_target(
    rank: int, world_size: int, checkpoint_dir: str, prompt: str, max_tokens: int
) -> dict | None:
    """Leader serves one /v1/completions request over the real endpoint; follower mirrors."""
    model = DeepseekV4ForCausalLM.from_checkpoint(checkpoint_dir, device="cpu", dtype=torch.float32)
    if rank != 0:
        TensorParallelStateCacheServer(model, device="cpu", dtype=torch.float32).run_follower_loop()
        return None

    server = TensorParallelStateCacheServer(
        model, _FakeTokenizer(model.cfg.vocab_size), device="cpu", dtype=torch.float32
    )
    scheduler = TensorParallelStateCacheScheduler(server)
    scheduler.start()
    app = FastAPI()
    app.add_api_route("/v1/completions", completions, methods=["POST"], response_model=None)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
    app.state.scheduler = scheduler
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={"model": "tp-v4", "prompt": prompt, "max_tokens": max_tokens},
            )
        result = {"status_code": response.status_code, "body": response.json()}
    finally:
        # Stop the engine thread (all generate broadcasts done), THEN tell the
        # follower to leave its loop. Order matters: shutdown must follow the
        # last generation broadcast so the rank sequences stay aligned.
        scheduler.stop()
        server.shutdown()
    return result


@pytest.mark.skipif(not is_multi_process_available(), reason="multi-process / gloo unavailable")
def test_tp_serving_over_http(tmp_path: Path) -> None:
    cfg = _make_tp_config()
    _write_synthetic_checkpoint(tmp_path, cfg)
    results = run_multi_process(
        2,
        _tp_http_target,
        str(tmp_path),
        "The capital of France is",
        5,
        timeout_sec=120.0,
    )
    leader = results[0]
    assert leader is not None
    assert leader["status_code"] == 200
    body = leader["body"]
    assert body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 5
