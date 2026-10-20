"""Config parsing, cache shapes and serving guards for the Qwen hybrid family.

Reference-facing numerics live in `test_qwen3_5_parity.py`. What is pinned
here is the surface around them: that a multimodal checkpoint's nested
config parses, that the assumptions the math hardcodes are rejected loudly
rather than mis-served, and that the state cache's shapes and lifecycle
match the layer layout.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mini_infer.cache.qwen3_5_state_cache import (
    Qwen35StateCache,
    QwenAttentionLayerState,
    QwenAttentionStateSpec,
    QwenGdnLayerState,
    QwenGdnStateSpec,
)
from mini_infer.models import REGISTRY
from mini_infer.models.qwen3_5 import (
    Qwen35Config,
    Qwen35ForConditionalGeneration,
    build_qwen35_state_cache_specs,
)

BASE_TEXT = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attn_output_gate": True,
    "output_gate_type": "swish",
    "max_position_embeddings": 128,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
    },
}


def _nested(**overrides):
    """A multimodal-style config: text fields under `text_config`."""
    text = {**BASE_TEXT, **overrides}
    return SimpleNamespace(text_config=SimpleNamespace(**text), vision_config=SimpleNamespace())


def _flat(**overrides):
    return SimpleNamespace(**{**BASE_TEXT, **overrides})


def test_parses_nested_and_flat_configs_identically():
    nested = Qwen35Config.from_hf(_nested(layer_types=None))
    flat = Qwen35Config.from_hf(_flat(layer_types=None))
    assert nested == flat
    assert nested.hidden_size == 32


def test_parses_text_config_given_as_a_dict():
    """The loader's trust_remote_code-free fallback hands back raw dicts."""
    raw = SimpleNamespace(text_config=dict(BASE_TEXT))
    assert Qwen35Config.from_hf(raw).num_hidden_layers == 4


def test_layer_types_synthesized_from_full_attention_interval():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None, full_attention_interval=4))
    assert cfg.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


def test_explicit_layer_types_are_honoured():
    explicit = ["full_attention", "linear_attention", "linear_attention", "linear_attention"]
    assert Qwen35Config.from_hf(_nested(layer_types=explicit)).layer_types == explicit


def test_layer_types_length_mismatch_raises():
    with pytest.raises(ValueError, match="layer_types has 2 entries"):
        Qwen35Config.from_hf(_nested(layer_types=["linear_attention", "full_attention"]))


def test_unknown_layer_type_raises():
    bad = ["linear_attention"] * 3 + ["sliding_attention"]
    with pytest.raises(ValueError, match="unsupported layer_types"):
        Qwen35Config.from_hf(_nested(layer_types=bad))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"attention_bias": True}, "attention_bias"),
        ({"attn_output_gate": False}, "attn_output_gate"),
        ({"output_gate_type": "sigmoid"}, "output_gate_type"),
        (
            {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e7}},
            "rope_type",
        ),
    ],
)
def test_hardcoded_assumptions_are_rejected_loudly(override, match):
    """These four are hardcoded in the math, so a differing checkpoint must fail."""
    with pytest.raises(ValueError, match=match):
        Qwen35Config.from_hf(_nested(layer_types=None, **override))


def test_value_heads_must_be_a_multiple_of_key_heads():
    with pytest.raises(ValueError, match="must be a positive multiple"):
        Qwen35Config.from_hf(_nested(layer_types=None, linear_num_value_heads=7))


def test_derived_dimensions():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    assert cfg.rotary_dim == 4  # head_dim 16 * partial_rotary_factor 0.25
    assert cfg.linear_head_ratio == 3
    assert cfg.linear_key_dim == 16
    assert cfg.linear_value_dim == 48
    assert cfg.linear_conv_channels == 2 * 16 + 48


def test_state_cache_layer_kinds_and_shapes_track_layer_types():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    specs = build_qwen35_state_cache_specs(cfg, max_seq_len=32)
    assert [type(s) for s in specs] == [QwenGdnStateSpec] * 3 + [QwenAttentionStateSpec]
    cache = Qwen35StateCache(specs, batch_size=2)
    gdn = cache.layer(0)
    attn = cache.layer(3)
    assert isinstance(gdn, QwenGdnLayerState)
    assert isinstance(attn, QwenAttentionLayerState)
    # One FUSED conv tail, not the three a Kimi KDA layer carries.
    assert gdn.conv.shape == (2, cfg.linear_conv_channels, 4)
    assert gdn.recurrent_state.shape == (2, 6, 8, 8)
    assert gdn.recurrent_state.dtype == torch.float32
    assert attn.k.shape == (2, 32, 2, 16)
    assert attn.v.shape == attn.k.shape


def test_capacity_guard_raises_before_overrunning_the_buffers():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    model = Qwen35ForConditionalGeneration(cfg)
    cache = model.build_state_cache(max_seq_len=4, batch_size=1)
    with pytest.raises(ValueError, match="exceeds the cache's max_seq_len"):
        model.forward_prefill_with_cache(torch.zeros(1, 9, dtype=torch.long), state_cache=cache)


def test_copy_row_from_rejects_a_layer_count_mismatch():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    full = Qwen35StateCache(build_qwen35_state_cache_specs(cfg, max_seq_len=8))
    short = Qwen35StateCache(build_qwen35_state_cache_specs(cfg, max_seq_len=8)[:2])
    with pytest.raises(ValueError, match="layer count mismatch"):
        short.copy_row_from(full, src_row=0, dst_row=0)


def test_advance_start_pos_rejects_negative():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    cache = Qwen35StateCache(build_qwen35_state_cache_specs(cfg, max_seq_len=8))
    with pytest.raises(ValueError, match="non-negative"):
        cache.advance_start_pos(-1)


def test_decode_entry_points_reject_multi_token_input():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    model = Qwen35ForConditionalGeneration(cfg)
    cache = model.build_state_cache(max_seq_len=16, batch_size=1)
    two = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\(B, 1\)"):
        model.forward_decode_with_cache(two, start_pos=0, state_cache=cache)
    with pytest.raises(ValueError, match=r"\(B, 1\)"):
        model.forward_decode_with_cache_ragged(
            two, positions=torch.zeros(1, dtype=torch.long), state_cache=cache
        )


def test_registry_resolves_the_checkpoint_architecture_string():
    """Qwen3.8-27B declares this name, not a Qwen3.8-specific one."""
    assert REGISTRY.lookup("Qwen3_5ForConditionalGeneration") is Qwen35ForConditionalGeneration
    assert Qwen35ForConditionalGeneration.USES_STATE_CACHE is True


def test_load_weights_is_single_rank_only(monkeypatch):
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    model = Qwen35ForConditionalGeneration(cfg)
    monkeypatch.setattr("mini_infer.distributed.group.get_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="single-rank only"):
        Qwen35ForConditionalGeneration.load_weights(model, {})


def test_stateless_forward_produces_finite_logits():
    cfg = Qwen35Config.from_hf(_nested(layer_types=None))
    model = Qwen35ForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        logits = model(torch.randint(0, cfg.vocab_size, (2, 5)))
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert bool(torch.isfinite(logits).all())
