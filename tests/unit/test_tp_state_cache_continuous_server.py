"""TP ragged continuous batching parity: cohort generation == single-process.

gloo, 2 CPU processes, a synthetic full-hybrid V4 (SWA + CSA + HCA) sharded via
`from_checkpoint`'s TP weight slicing (the path real V4-Flash takes on 2x B200).
Proves the leader / follower forward mirroring in
`TensorParallelStateCacheContinuousServer` produces the SAME tokens as
single-process per-request generation, so the TP continuous-batching benchmark
measures a correct path.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.engine.tp_state_cache_continuous_server import (
    TensorParallelStateCacheContinuousServer,
)
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process


def _make_tp_config() -> DeepseekV4Config:
    """Small full-hybrid V4; every TP-sharded dim is divisible by 2."""
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


def _write_synthetic_checkpoint(checkpoint_dir: Path, cfg: DeepseekV4Config) -> None:
    torch.manual_seed(0)
    source = DeepseekV4ForCausalLM(cfg).eval()
    config_json = {**dataclasses.asdict(cfg), "architectures": ["DeepseekV4ForCausalLM"]}
    (checkpoint_dir / "config.json").write_text(json.dumps(config_json))
    save_file(
        {k: v.detach().clone().contiguous() for k, v in source.state_dict().items()},
        str(checkpoint_dir / "model.safetensors"),
    )


def _tp_cohort_target(
    rank: int, world_size: int, checkpoint_dir: str, prompts: list[list[int]], max_new_tokens: int
) -> list[list[int]] | None:
    """Per-rank work: load the sharded model, then cohort-generate (leader) or mirror (follower)."""
    model = DeepseekV4ForCausalLM.from_checkpoint(checkpoint_dir, device="cpu", dtype=torch.float32)
    server = TensorParallelStateCacheContinuousServer(
        model, max_batch_size=len(prompts), max_seq_len=64, device="cpu", dtype=torch.float32
    )
    if server.is_leader:
        tokens = server.generate_cohort(prompts, max_new_tokens=max_new_tokens)
        server.shutdown()
        return tokens
    server.run_follower_loop()
    return None


@pytest.mark.skipif(not is_multi_process_available(), reason="multi-process / gloo unavailable")
def test_tp_continuous_matches_single_process(tmp_path: Path) -> None:
    cfg = _make_tp_config()
    _write_synthetic_checkpoint(tmp_path, cfg)
    prompts = [list(range(8)), list(range(2, 18)), list(range(24))]  # varied lengths -> ragged
    max_new_tokens = 6

    # Single-process per-request reference (world_size == 1 in this process).
    reference_model = DeepseekV4ForCausalLM.from_checkpoint(
        str(tmp_path), device="cpu", dtype=torch.float32
    )
    gen = StateCacheGenerator(reference_model)
    reference = [gen.generate_ids(prompt, max_new_tokens=max_new_tokens) for prompt in prompts]

    # TP ragged continuous batching over 2 gloo ranks: rank 0 leads, rank 1 follows.
    results = run_multi_process(
        2, _tp_cohort_target, str(tmp_path), prompts, max_new_tokens, timeout_sec=180.0
    )
    assert results[0] == reference, (
        f"TP cohort diverged from single-process:\n  TP:  {results[0]}\n  ref: {reference}"
    )
    assert all(len(tokens) == max_new_tokens for tokens in reference)
