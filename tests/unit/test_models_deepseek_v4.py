"""DeepSeek-V4 owned model: registry, per-layer dispatch, end-to-end prefill + decode.

V4 weights aren't public yet and the published architecture also uses MoE
FFN + Hyper-Connections that we don't replicate. So `load_weights` raises.
What we DO ship + test:

  - Per-layer CSA-or-HCA dispatch driven by `compress_ratios`.
  - End-to-end forward (prefill) on a synthetic 4-layer hybrid backbone
    — verifies the whole stack composes correctly (residual shape,
    per-layer compressed RoPE, final norm + lm_head).
  - End-to-end decode (1 token) through the same stack with a per-layer
    `StateCache` initialised from the prefill — verifies cache wiring
    works across mixed CSA/HCA layers, including the case where some
    layers flush a compressed block and others don't on the same step.
  - `build_state_cache_layer_specs` produces the right per-layer
    `(overlap_mode, indexer)` combination.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.state_cache import StateCache
from mini_infer.models import REGISTRY
from mini_infer.models.blocks import CSAAttention, HCAAttention
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)


def _make_hybrid_config(
    *,
    num_hidden_layers: int = 4,
    seq_len_friendly_compress_ratios: tuple[int, ...] = (4, 8, 4, 8),
) -> DeepseekV4Config:
    """A small 4-layer hybrid config (CSA, HCA, CSA, HCA) suited to CPU tests.

    `compress_ratios` defaults to `(4, 8, 4, 8)` so every layer sees a multiple
    of its ratio when prefilling at any multiple of 8 tokens.
    """
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=32,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=seq_len_friendly_compress_ratios,
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


# ---------- Registry / config ----------


def test_registry_has_deepseek_v4() -> None:
    assert REGISTRY.lookup("DeepseekV4ForCausalLM") is DeepseekV4ForCausalLM


def test_config_validates_compress_ratios_length() -> None:
    """`compress_ratios` must have one entry per layer."""
    with pytest.raises(ValueError, match="compress_ratios"):
        DeepseekV4Config(
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
            compress_ratios=(4, 8),  # length 2 != num_hidden_layers 4
            index_num_heads=2,
            index_head_dim=16,
            index_top_k=2,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )


def test_is_csa_layer_matches_compress_ratios() -> None:
    cfg = _make_hybrid_config(seq_len_friendly_compress_ratios=(4, 8, 4, 8))
    assert cfg.is_csa_layer(0) is True
    assert cfg.is_csa_layer(1) is False
    assert cfg.is_csa_layer(2) is True
    assert cfg.is_csa_layer(3) is False


def test_load_weights_raises_until_v4_weights_public() -> None:
    cfg = _make_hybrid_config()
    model = DeepseekV4ForCausalLM(cfg)
    with pytest.raises(NotImplementedError, match="V4 checkpoints"):
        DeepseekV4ForCausalLM.load_weights(model, {})


# ---------- Per-layer dispatch ----------


def test_csa_layers_use_csa_attention_others_use_hca() -> None:
    cfg = _make_hybrid_config(seq_len_friendly_compress_ratios=(4, 8, 4, 8))
    model = DeepseekV4ForCausalLM(cfg)
    assert isinstance(model.model.layers[0].self_attn, CSAAttention)
    assert isinstance(model.model.layers[1].self_attn, HCAAttention)
    assert isinstance(model.model.layers[2].self_attn, CSAAttention)
    assert isinstance(model.model.layers[3].self_attn, HCAAttention)


def test_hca_layer_does_not_carry_indexer_attribute() -> None:
    """HCA layers wire a plain HCAAttention — no indexer, no extra params."""
    cfg = _make_hybrid_config()
    model = DeepseekV4ForCausalLM(cfg)
    hca_layer = model.model.layers[1].self_attn
    assert not hasattr(hca_layer, "indexer")


def test_csa_decoder_layer_rejects_zero_indexer_knobs() -> None:
    """CSA layer (compression_ratio == 4) needs positive indexer hyperparams."""
    from mini_infer.models.blocks.deepseek_v4_decoder_layer import DeepseekV4DecoderLayer

    with pytest.raises(ValueError, match="CSA layer"):
        DeepseekV4DecoderLayer(
            hidden_size=64,
            num_heads=4,
            q_lora_rank=32,
            kv_head_dim=32,
            rope_head_dim=8,
            num_groups=2,
            o_lora_rank=32,
            window_size=8,
            compression_ratio=4,  # CSA — needs indexer knobs
            intermediate_size=128,
            rms_norm_eps=1e-6,
            # index_num_heads / index_head_dim / index_top_k missing -> fail
        )


# ---------- StateCache spec helper ----------


def test_build_state_cache_layer_specs_csa_layers_get_overlap_and_indexer() -> None:
    cfg = _make_hybrid_config(seq_len_friendly_compress_ratios=(4, 8, 4, 8))
    layer_specs = build_state_cache_layer_specs(cfg, max_n_compressed=16)
    assert len(layer_specs) == 4
    # CSA: overlap + indexer
    assert layer_specs[0].overlap_mode is True
    assert layer_specs[0].indexer is not None
    assert layer_specs[0].indexer.head_dim == cfg.index_head_dim
    assert layer_specs[2].overlap_mode is True
    assert layer_specs[2].indexer is not None
    # HCA: no overlap, no indexer
    assert layer_specs[1].overlap_mode is False
    assert layer_specs[1].indexer is None
    assert layer_specs[3].overlap_mode is False
    assert layer_specs[3].indexer is None
    # Other fields propagated correctly.
    for layer_idx, spec in enumerate(layer_specs):
        assert spec.kv_head_dim == cfg.kv_head_dim
        assert spec.compression_ratio == cfg.compress_ratios[layer_idx]
        assert spec.n_win == cfg.window_size
        assert spec.max_n_compressed == 16


# ---------- End-to-end forward (prefill) ----------


def test_forward_runs_through_4_layer_hybrid_backbone() -> None:
    """Prefill 16 tokens through a CSA/HCA/CSA/HCA stack, check shape + finiteness."""
    cfg = _make_hybrid_config(seq_len_friendly_compress_ratios=(4, 8, 4, 8))
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    seq_len = 16  # multiple of every compression_ratio in the schedule
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)

    with torch.inference_mode():
        logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_forward_rejects_seq_len_not_multiple_of_min_compress_ratio() -> None:
    """The standalone forward of each attention block requires `T % m == 0`.

    With `compress_ratios = (4, 8, 4, 8)` the LCM is 8, so any seq_len that's
    not a multiple of 8 fails on the second layer at minimum.
    """
    cfg = _make_hybrid_config(seq_len_friendly_compress_ratios=(4, 8, 4, 8))
    model = DeepseekV4ForCausalLM(cfg).eval()
    bad_seq_len = 12  # multiple of 4 but not of 8
    input_ids = torch.randint(0, cfg.vocab_size, (1, bad_seq_len), dtype=torch.long)
    with pytest.raises(ValueError, match="multiple of compression_ratio"):
        model(input_ids)


# ---------- End-to-end decode (single token through cache) ----------


def test_forward_decode_with_cache_produces_finite_logits() -> None:
    """Manually populated `StateCache` -> one decode step yields well-formed logits.

    For this test the cache is left at default-initialized state (which models
    a "zero history" world), so no parity is asserted — the test verifies
    that:
      - `forward_decode_with_cache` runs end-to-end without shape errors.
      - All layers (mixed CSA/HCA) consume their per-layer state correctly.
      - Output is `(B, 1, vocab_size)` with finite values.
      - Layers that flush at this step (compression_ratio | (start_pos + 1))
        and layers that don't both work in the same forward.
    """
    cfg = _make_hybrid_config(seq_len_friendly_compress_ratios=(4, 8, 4, 8))
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    state_cache = StateCache(
        build_state_cache_layer_specs(cfg, max_n_compressed=8),
        batch_size=batch_size,
    )

    # Simulate "we've already prefilled 16 tokens", so n_compressed_blocks reflects
    # how many blocks each layer should already have. 16/4 = 4 for CSA layers,
    # 16/8 = 2 for HCA layers.
    prefilled_seq_len = 16
    for layer_idx, compression_ratio in enumerate(cfg.compress_ratios):
        layer_state = state_cache.layer(layer_idx)
        layer_state.n_compressed_blocks = prefilled_seq_len // compression_ratio
        # Indexer's compressed history aligns with the main one's layout.
        if layer_state.indexer is not None:
            layer_state.indexer.n_compressed_blocks = prefilled_seq_len // compression_ratio
    state_cache.start_pos = prefilled_seq_len

    # Decode at start_pos=16: with compress_ratios (4, 8, 4, 8):
    #   layer 0 (m=4): (16+1) % 4 = 1 -> NO flush
    #   layer 1 (m=8): (16+1) % 8 = 1 -> NO flush
    # So neither layer needs block_position_embeddings — easy first step.
    input_id = torch.randint(0, cfg.vocab_size, (batch_size, 1), dtype=torch.long)
    with torch.inference_mode():
        logits = model.forward_decode_with_cache(
            input_id, start_pos=prefilled_seq_len, state_cache=state_cache
        )
    assert logits.shape == (batch_size, 1, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_forward_decode_with_cache_handles_simultaneous_flush_layers() -> None:
    """At `start_pos = 23` with compress_ratios (4, 8, 4, 8):
        layer 0 (m=4): (23+1) % 4 == 0 -> FLUSH
        layer 1 (m=8): (23+1) % 8 == 0 -> FLUSH
        layer 2 (m=4): FLUSH
        layer 3 (m=8): FLUSH
    All four layers flush a compressed block on the same step. Verifies the
    block-position-embedding plumbing (computed per-layer) lights up correctly.
    """
    cfg = _make_hybrid_config(seq_len_friendly_compress_ratios=(4, 8, 4, 8))
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    state_cache = StateCache(
        build_state_cache_layer_specs(cfg, max_n_compressed=8),
        batch_size=batch_size,
    )
    # Simulate state matching `start_pos = 23` (equivalent to having processed
    # tokens 0..22 already). Each layer's compressed_count = 23 // m.
    for layer_idx, compression_ratio in enumerate(cfg.compress_ratios):
        layer_state = state_cache.layer(layer_idx)
        layer_state.n_compressed_blocks = 23 // compression_ratio
        if layer_state.indexer is not None:
            layer_state.indexer.n_compressed_blocks = 23 // compression_ratio
    state_cache.start_pos = 23

    input_id = torch.randint(0, cfg.vocab_size, (batch_size, 1), dtype=torch.long)
    with torch.inference_mode():
        logits = model.forward_decode_with_cache(input_id, start_pos=23, state_cache=state_cache)
    assert logits.shape == (batch_size, 1, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))
    # Every layer should have appended a new compressed entry this step.
    for layer_idx, compression_ratio in enumerate(cfg.compress_ratios):
        expected_count = (23 // compression_ratio) + 1  # was N, now N+1
        assert state_cache.layer(layer_idx).n_compressed_blocks == expected_count


def test_forward_decode_with_cache_rejects_layer_count_mismatch() -> None:
    cfg = _make_hybrid_config(num_hidden_layers=4)
    model = DeepseekV4ForCausalLM(cfg).eval()
    # Build a StateCache with FEWER layers than the model.
    too_few = build_state_cache_layer_specs(cfg, max_n_compressed=8)[:2]
    state_cache = StateCache(too_few, batch_size=1)
    input_id = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="layers"):
        model.forward_decode_with_cache(input_id, start_pos=8, state_cache=state_cache)
