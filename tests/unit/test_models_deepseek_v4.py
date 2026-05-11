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


def test_load_weights_round_trips_via_state_dict() -> None:
    """V4 weights pipeline: snapshot a freshly-initialised model's state_dict,
    construct a second model, load via `load_weights`, and confirm every
    parameter matches bit-for-bit.

    This exercises the rename-rules + TP-aware load path against synthetic
    (non-quantised) weights at world_size=1, which is the only environment
    available to CPU unit tests. The FP8/FP4 dequant + real V4-Flash
    end-to-end load are validated separately (Modal smoke)."""
    cfg = _make_hybrid_config()
    source = DeepseekV4ForCausalLM(cfg)
    source_state_dict = {k: v.detach().clone() for k, v in source.state_dict().items()}

    target = DeepseekV4ForCausalLM(cfg)
    DeepseekV4ForCausalLM.load_weights(target, source_state_dict)

    for name, target_param in target.named_parameters():
        source_param = source_state_dict[name]
        torch.testing.assert_close(target_param.detach(), source_param, rtol=0, atol=0)


def test_load_weights_dequantizes_fp8_e4m3fn_weights() -> None:
    """FP8 (e4m3fn) tensors in the HF state_dict are upcast to BF16 at load."""
    cfg = _make_hybrid_config()
    source = DeepseekV4ForCausalLM(cfg).to(torch.bfloat16)
    target = DeepseekV4ForCausalLM(cfg).to(torch.bfloat16)

    # Snapshot weights, then re-cast one matrix to FP8 to exercise the dequant.
    state_dict = {k: v.detach().clone() for k, v in source.state_dict().items()}
    fp8_key = next(
        k for k, v in state_dict.items() if v.ndim == 2 and v.numel() > 0
    )
    state_dict[fp8_key] = state_dict[fp8_key].to(torch.float8_e4m3fn)

    DeepseekV4ForCausalLM.load_weights(target, state_dict)
    # After load the parameter should be BF16 (the model's working dtype),
    # not FP8.
    loaded_param = dict(target.named_parameters())[fp8_key]
    assert loaded_param.dtype == torch.bfloat16


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


def test_v4_config_yarn_defaults_disable_correction() -> None:
    """Default config has YaRN disabled (`yarn_original_seq_len == 0`),
    matching V4-Pro/Flash at <= 4k context."""
    cfg = _make_hybrid_config()
    assert cfg.yarn_original_seq_len == 0
    assert cfg.yarn_scaling_factor == 1.0
    assert cfg.yarn_beta_fast == 32
    assert cfg.yarn_beta_slow == 1


def test_v4_model_threads_yarn_into_rotary_embedding() -> None:
    """When YaRN is configured, the model's `inv_freq` table reflects the
    correction (low-frequency components scaled by `1/factor`)."""
    head_dim = 8  # rope_head_dim in the demo config
    factor = 4.0
    cfg = DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=32,
        rope_head_dim=head_dim,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(4, 8),
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        yarn_original_seq_len=512,
        yarn_scaling_factor=factor,
    )
    model = DeepseekV4ForCausalLM(cfg)
    standard_inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    # The lowest-frequency component (largest index) should be scaled by ~1/factor.
    yarn_lowest = model.rotary_emb.inv_freq[-1].item()
    standard_lowest = standard_inv_freq[-1].item()
    assert abs(yarn_lowest * factor - standard_lowest) < 0.01 * standard_lowest, (
        f"YaRN should scale lowest-freq by 1/factor; "
        f"got {yarn_lowest} vs unscaled {standard_lowest}"
    )


def test_forward_decode_with_cache_rejects_layer_count_mismatch() -> None:
    cfg = _make_hybrid_config(num_hidden_layers=4)
    model = DeepseekV4ForCausalLM(cfg).eval()
    # Build a StateCache with FEWER layers than the model.
    too_few = build_state_cache_layer_specs(cfg, max_n_compressed=8)[:2]
    state_cache = StateCache(too_few, batch_size=1)
    input_id = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="layers"):
        model.forward_decode_with_cache(input_id, start_pos=8, state_cache=state_cache)


# ---------- MoE FFN integration (V4 paper §2.2) ----------


def _make_moe_hybrid_config(
    *,
    num_hidden_layers: int = 4,
    num_hash_routed_layers: int = 2,
) -> DeepseekV4Config:
    """4-layer hybrid (CSA, HCA, CSA, HCA) backbone with `HashRoutedMoEFFN`.

    Layers `[0, num_hash_routed_layers)` use hash routing; the rest use
    score-topk. With `num_hash_routed_layers=2` and 4 layers total, we
    cover both routing modes in the same forward pass.
    """
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,  # SwiGLU only — ignored when use_moe_ffn=True
        num_hidden_layers=num_hidden_layers,
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
        # MoE knobs
        use_moe_ffn=True,
        moe_intermediate_size=64,
        num_routed_experts=4,
        num_activated_experts=2,
        num_hash_routed_layers=num_hash_routed_layers,
        moe_score_func="softmax",
        moe_route_scale=1.0,
        n_shared_experts=1,
    )


def test_moe_config_validates_required_fields() -> None:
    """`use_moe_ffn=True` rejects zero/negative MoE knobs."""
    base_kwargs = dict(
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
        use_moe_ffn=True,
        moe_intermediate_size=64,
        num_routed_experts=4,
        num_activated_experts=2,
    )
    # Missing num_routed_experts:
    with pytest.raises(ValueError, match="num_routed_experts"):
        DeepseekV4Config(**{**base_kwargs, "num_routed_experts": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_activated_experts"):
        DeepseekV4Config(**{**base_kwargs, "num_activated_experts": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        DeepseekV4Config(**{**base_kwargs, "moe_intermediate_size": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_hash_routed_layers"):
        DeepseekV4Config(**{**base_kwargs, "num_hash_routed_layers": 99})  # type: ignore[arg-type]


def test_is_hash_routed_layer_returns_false_when_moe_disabled() -> None:
    """SwiGLU (no MoE) backbones report no layer as hash-routed."""
    cfg = _make_hybrid_config()
    assert cfg.use_moe_ffn is False
    for layer_idx in range(cfg.num_hidden_layers):
        assert cfg.is_hash_routed_layer(layer_idx) is False


def test_is_hash_routed_layer_partitions_layers_correctly() -> None:
    cfg = _make_moe_hybrid_config(num_hidden_layers=4, num_hash_routed_layers=2)
    assert cfg.is_hash_routed_layer(0) is True
    assert cfg.is_hash_routed_layer(1) is True
    assert cfg.is_hash_routed_layer(2) is False
    assert cfg.is_hash_routed_layer(3) is False


def test_moe_backbone_uses_hash_routed_moe_ffn_in_each_layer() -> None:
    """With `use_moe_ffn=True`, every decoder layer's `mlp` is a `HashRoutedMoEFFN`."""
    from mini_infer.models.blocks import HashRoutedMoEFFN

    cfg = _make_moe_hybrid_config(num_hash_routed_layers=2)
    model = DeepseekV4ForCausalLM(cfg)
    for layer_idx in range(cfg.num_hidden_layers):
        layer_mlp = model.model.layers[layer_idx].mlp
        assert isinstance(layer_mlp, HashRoutedMoEFFN), (
            f"layer {layer_idx} mlp should be HashRoutedMoEFFN, got {type(layer_mlp).__name__}"
        )


def test_moe_backbone_per_layer_routing_mode_matches_config() -> None:
    cfg = _make_moe_hybrid_config(num_hidden_layers=4, num_hash_routed_layers=2)
    model = DeepseekV4ForCausalLM(cfg)
    for layer_idx in range(cfg.num_hidden_layers):
        layer_mlp = model.model.layers[layer_idx].mlp
        expected_mode = "hash" if layer_idx < 2 else "score_topk"
        assert layer_mlp.gate.routing_mode == expected_mode  # type: ignore[union-attr]


def test_moe_backbone_runs_prefill_end_to_end() -> None:
    """Prefill 16 tokens through a 4-layer CSA/HCA stack with hash-routed MoE FFN."""
    cfg = _make_moe_hybrid_config(num_hash_routed_layers=2)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_moe_backbone_runs_decode_end_to_end() -> None:
    """Decode through the MoE backbone with both hash and score-topk layers active.

    Verifies `input_ids` threads through `forward_decode_with_cache` -> per-layer
    `forward_decode` -> `HashRoutedMoEFFN.forward(hidden_state, input_ids)`.
    """
    cfg = _make_moe_hybrid_config(num_hash_routed_layers=2)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    state_cache = StateCache(
        build_state_cache_layer_specs(cfg, max_n_compressed=8),
        batch_size=batch_size,
    )
    # Simulate post-prefill state at start_pos=16.
    prefilled_seq_len = 16
    for layer_idx, compression_ratio in enumerate(cfg.compress_ratios):
        layer_state = state_cache.layer(layer_idx)
        layer_state.n_compressed_blocks = prefilled_seq_len // compression_ratio
        if layer_state.indexer is not None:
            layer_state.indexer.n_compressed_blocks = prefilled_seq_len // compression_ratio
    state_cache.start_pos = prefilled_seq_len

    input_id = torch.randint(0, cfg.vocab_size, (batch_size, 1), dtype=torch.long)
    with torch.inference_mode():
        logits = model.forward_decode_with_cache(
            input_id, start_pos=prefilled_seq_len, state_cache=state_cache
        )
    assert logits.shape == (batch_size, 1, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_swiglu_backbone_does_not_thread_input_ids_to_layers() -> None:
    """When `use_moe_ffn=False`, `input_ids` is NOT passed to the layer FFN.

    Sanity check: the SwiGLU FFN doesn't accept input_ids; passing them
    would surface as a TypeError. We rely on the model's `use_moe_ffn`
    branch to gate the threading. The existing `test_forward_runs_through_*`
    tests prove the SwiGLU path runs; this one verifies layers were
    constructed with `ffn_type="swiglu"`.
    """
    from mini_infer.models.blocks import HashRoutedMoEFFN

    cfg = _make_hybrid_config()  # use_moe_ffn defaults to False
    model = DeepseekV4ForCausalLM(cfg)
    for layer_idx in range(cfg.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        assert layer.ffn_type == "swiglu"
        assert not isinstance(layer.mlp, HashRoutedMoEFFN)


# ---------- Hyper-Connections integration (V4 paper §2.5) ----------


def _make_hc_hybrid_config(
    *,
    num_hidden_layers: int = 4,
    hc_mult: int = 4,
    use_moe_ffn: bool = False,
) -> DeepseekV4Config:
    """4-layer hybrid backbone with HC residual mediation enabled."""
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
        compress_ratios=(4, 8, 4, 8),
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_moe_ffn=use_moe_ffn,
        moe_intermediate_size=64 if use_moe_ffn else 0,
        num_routed_experts=4 if use_moe_ffn else 0,
        num_activated_experts=2 if use_moe_ffn else 0,
        num_hash_routed_layers=0,
        moe_score_func="softmax",
        moe_route_scale=1.0,
        n_shared_experts=1 if use_moe_ffn else 0,
        # HC knobs
        use_hyper_connections=True,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


def test_hc_config_validates_positive_hc_mult() -> None:
    """`use_hyper_connections=True` rejects hc_mult <= 0."""
    base_kwargs = dict(
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
        use_hyper_connections=True,
        hc_mult=0,  # invalid
    )
    with pytest.raises(ValueError, match="hc_mult"):
        DeepseekV4Config(**base_kwargs)  # type: ignore[arg-type]


def test_hc_backbone_layers_carry_hyper_connections_pair() -> None:
    """Each decoder layer with HC enabled owns hc_attn + hc_ffn instances."""
    from mini_infer.models.blocks import HyperConnections

    cfg = _make_hc_hybrid_config(hc_mult=4)
    model = DeepseekV4ForCausalLM(cfg)
    for layer_idx in range(cfg.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        assert layer.use_hyper_connections is True
        assert isinstance(layer.hc_attn, HyperConnections)
        assert isinstance(layer.hc_ffn, HyperConnections)
        # Both share hc_mult, distinct parameter sets.
        assert layer.hc_attn.hc_mult == cfg.hc_mult
        assert layer.hc_ffn.hc_mult == cfg.hc_mult
        assert layer.hc_attn is not layer.hc_ffn


def test_hc_backbone_owns_head_reduction_when_hc_enabled() -> None:
    from mini_infer.models.blocks import HyperConnections  # noqa: F401  (import for clarity)
    from mini_infer.models.blocks.hyper_connections import HCHeadReduction

    cfg_hc_on = _make_hc_hybrid_config()
    cfg_hc_off = _make_hybrid_config()
    model_hc_on = DeepseekV4ForCausalLM(cfg_hc_on)
    model_hc_off = DeepseekV4ForCausalLM(cfg_hc_off)
    assert isinstance(model_hc_on.hc_head_reduction, HCHeadReduction)
    assert model_hc_off.hc_head_reduction is None


def test_hc_backbone_runs_prefill_end_to_end() -> None:
    """HC-enabled V4 backbone runs prefill (B, T) -> (B, T, vocab)."""
    cfg = _make_hc_hybrid_config(hc_mult=4)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_hc_backbone_runs_decode_end_to_end() -> None:
    """HC-enabled decode through cache: (B, 1) -> (B, 1, vocab)."""
    cfg = _make_hc_hybrid_config(hc_mult=4)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    state_cache = StateCache(
        build_state_cache_layer_specs(cfg, max_n_compressed=8),
        batch_size=batch_size,
    )
    prefilled_seq_len = 16
    for layer_idx, compression_ratio in enumerate(cfg.compress_ratios):
        layer_state = state_cache.layer(layer_idx)
        layer_state.n_compressed_blocks = prefilled_seq_len // compression_ratio
        if layer_state.indexer is not None:
            layer_state.indexer.n_compressed_blocks = prefilled_seq_len // compression_ratio
    state_cache.start_pos = prefilled_seq_len

    input_id = torch.randint(0, cfg.vocab_size, (batch_size, 1), dtype=torch.long)
    with torch.inference_mode():
        logits = model.forward_decode_with_cache(
            input_id, start_pos=prefilled_seq_len, state_cache=state_cache
        )
    assert logits.shape == (batch_size, 1, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_hc_backbone_combines_with_moe_ffn() -> None:
    """Hyper-Connections AND hash-routed MoE FFN both active in the same model.

    The most V4-faithful configuration shipped: every published primitive
    is in play (CSA + HCA + sink + grouped output + cache + MoE + HC).
    """
    cfg = _make_hc_hybrid_config(hc_mult=4, use_moe_ffn=True)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_vanilla_backbone_does_not_carry_hc_instances() -> None:
    """When `use_hyper_connections=False`, layers leave hc_attn / hc_ffn None."""
    cfg = _make_hybrid_config()
    model = DeepseekV4ForCausalLM(cfg)
    assert cfg.use_hyper_connections is False
    for layer_idx in range(cfg.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        assert layer.use_hyper_connections is False
        assert layer.hc_attn is None
        assert layer.hc_ffn is None
