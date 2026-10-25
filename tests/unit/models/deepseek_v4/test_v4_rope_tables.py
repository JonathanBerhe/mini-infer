"""Per-layer RoPE tables of the DeepSeek-V4 model against the reference.

The reference builds one table per attention layer (`Attention.__init__`):
compressed layers (CSA and HCA) rotate with `compress_rope_theta` plus YaRN,
and pure sliding-window layers rotate with `rope_theta` and no YaRN. The block
parity tests hand one table to both sides, so they never check this choice.
These tests compare the table the model gives each layer with the reference's
own `precompute_freqs_cis`, at V4-Flash's real RoPE settings.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from mini_infer.models.deepseek_v4.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_rotary_tables,
)

# `reference_module` is the shared fixture from `tests/unit/conftest.py`.

# V4-Flash `config.json`: rope_theta 10000, compress_rope_theta 160000, and
# rope_scaling {type yarn, factor 16, original_max_position_embeddings 65536,
# beta_fast 32, beta_slow 1}, qk_rope_head_dim 64.
_ROPE_THETA = 10000.0
_COMPRESS_ROPE_THETA = 160000.0
_YARN_ORIGINAL = 65536
_YARN_FACTOR = 16.0
_ROPE_HEAD_DIM = 64


def _flash_rope_config(compress_ratios: tuple[int, ...] = (0, 4, 128, 4)) -> DeepseekV4Config:
    """Small model dims with V4-Flash's RoPE settings; ratio 0 marks a sliding-window layer."""
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(compress_ratios),
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=128,
        rope_head_dim=_ROPE_HEAD_DIM,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=compress_ratios,
        index_num_heads=2,
        index_head_dim=128,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=_ROPE_THETA,
        tie_word_embeddings=False,
        yarn_original_seq_len=_YARN_ORIGINAL,
        yarn_scaling_factor=_YARN_FACTOR,
        yarn_beta_fast=32,
        yarn_beta_slow=1,
        compress_rope_theta=_COMPRESS_ROPE_THETA,
    )


def _reference_table(ref: Any, *, compressed: bool, seqlen: int) -> torch.Tensor:
    """The reference's `(seqlen, rope_head_dim // 2)` complex table for one layer type."""
    original, base = (_YARN_ORIGINAL, _COMPRESS_ROPE_THETA) if compressed else (0, _ROPE_THETA)
    table: torch.Tensor = ref.precompute_freqs_cis.__wrapped__(
        _ROPE_HEAD_DIM, seqlen, original, base, _YARN_FACTOR, 32, 1
    )
    return table


def test_from_hf_reads_compress_rope_theta() -> None:
    """The real V4-Flash config fields reach the model config."""
    hf_config = {
        "vocab_size": 129280,
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "num_attention_heads": 64,
        "q_lora_rank": 1024,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "sliding_window": 128,
        "compress_ratios": [0, 0, 4, 128, 0],
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
    }
    cfg = DeepseekV4Config.from_hf(hf_config)
    assert cfg.rope_theta == _ROPE_THETA
    assert cfg.compress_rope_theta == _COMPRESS_ROPE_THETA
    assert cfg.yarn_original_seq_len == _YARN_ORIGINAL
    assert cfg.yarn_scaling_factor == _YARN_FACTOR
    without = DeepseekV4Config.from_hf(
        {k: v for k, v in hf_config.items() if k != "compress_rope_theta"}
    )
    assert without.compress_rope_theta is None


@pytest.mark.parametrize("seqlen", [256, 8192])
def test_each_layer_gets_the_reference_rope_table(reference_module: Any, seqlen: int) -> None:
    cfg = _flash_rope_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    positions = torch.arange(seqlen).unsqueeze(0)  # (1, T)
    half = _ROPE_HEAD_DIM // 2
    for layer_idx, ratio in enumerate(cfg.compress_ratios):
        expected = _reference_table(
            reference_module, compressed=ratio > 0, seqlen=seqlen
        )  # (T, rd/2)
        cos, sin = model.rotary_for_layer(layer_idx)(torch.zeros(1, seqlen, 1), positions)
        # Our tables are `(1, T, rope_head_dim)` with the angle repeated over the
        # two halves; the first half lines up with the reference's complex entries.
        torch.testing.assert_close(cos[0, :, :half], expected.real, rtol=0.0, atol=1e-5)
        torch.testing.assert_close(sin[0, :, :half], expected.imag, rtol=0.0, atol=1e-5)


def test_default_config_keeps_a_single_table() -> None:
    """Without `compress_rope_theta` and YaRN, both tables equal plain RoPE at `rope_theta`."""
    cfg = _flash_rope_config()
    plain = DeepseekV4Config(
        **{
            **{f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
            "compress_rope_theta": None,
            "yarn_original_seq_len": 0,
        }
    )
    window_table, compressed_table = build_rotary_tables(plain)
    expected = 1.0 / (_ROPE_THETA ** (torch.arange(0, _ROPE_HEAD_DIM, 2).float() / _ROPE_HEAD_DIM))
    torch.testing.assert_close(window_table.inv_freq, expected)
    assert torch.equal(window_table.inv_freq, compressed_table.inv_freq)


def test_compressed_layers_no_longer_use_the_window_base() -> None:
    """At position 8191 the two tables disagree by much more than rounding."""
    cfg = _flash_rope_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    position = torch.tensor([[8191]])  # (1, 1)
    window_cos, window_sin = model.window_rotary_emb(torch.zeros(1, 1, 1), position)
    comp_cos, comp_sin = model.compress_rotary_emb(torch.zeros(1, 1, 1), position)
    window_angle = torch.atan2(window_sin, window_cos)  # (1, 1, rope_head_dim)
    comp_angle = torch.atan2(comp_sin, comp_cos)
    gap = torch.remainder(window_angle - comp_angle + math.pi, 2 * math.pi) - math.pi
    assert float(gap.abs().mean()) > 0.5
