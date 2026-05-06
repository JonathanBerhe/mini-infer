"""Gemma 4 owned-model construction, forward, and weight loading.

The 31B target is too large to load on CI / M1 (62 GB at bf16), so all
tests here use a small synthetic config that exercises the per-layer-type
heterogeneous-attention code paths (sliding head_dim=8 / num_kv_heads=4
vs full head_dim=16 / num_kv_heads=2 / k_eq_v=True).

Real-checkpoint integration runs on Modal B200 via `scripts/modal_smoke.py`
with `MINI_INFER_BENCH_MODEL=google/gemma-4-31B-it`. That run produces
`Paris` for `"The capital of France is"`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models import REGISTRY
from mini_infer.models.gemma4 import Gemma4Config, Gemma4ForCausalLM


def _make_cfg(
    *,
    layer_types: list[str] | None = None,
    final_logit_softcapping: float | None = 30.0,
    attention_k_eq_v: bool = True,
    tie_word_embeddings: bool = True,
) -> Gemma4Config:
    """Synthetic mini-config that mirrors the 31B's heterogeneous shape pattern."""
    if layer_types is None:
        # 8 layers: same 5-sliding-then-1-full cadence as the real 31B
        # so any wrap-around bug surfaces. Two full layers (idx 5 and 7).
        layer_types = [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
    return Gemma4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,  # sliding KV
        head_dim=8,  # sliding head_dim
        num_global_key_value_heads=2,  # full KV
        global_head_dim=16,  # full head_dim
        attention_k_eq_v=attention_k_eq_v,
        layer_types=layer_types,
        sliding_window=4,
        rms_norm_eps=1e-6,
        rope_theta_local=10000.0,
        rope_theta_global=1000000.0,
        rope_partial_rotary_factor_global=0.25,
        final_logit_softcapping=final_logit_softcapping,
        tie_word_embeddings=tie_word_embeddings,
    )


def _make_paged_cache(model: Gemma4ForCausalLM) -> PagedKVCache:
    """Heterogeneous-KV pool sized for the synthetic Gemma 4 mini-config.

    Backend nominally `flash_attn`; on CPU/MPS the dispatcher
    auto-falls-back to the PyTorch reference path. That's exactly what
    we want here — no GPU needed for shape/forward sanity.
    """
    pool = BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
        layer_attention=model.per_layer_attention(),
        layer_kv_shape=model.per_layer_kv_shape(),
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def test_registry_has_gemma4_under_conditional_arch() -> None:
    """`Gemma4ForCausalLM` is reachable via the multimodal HF arch string."""
    cls = REGISTRY.lookup("Gemma4ForConditionalGeneration")
    assert cls is Gemma4ForCausalLM


def test_per_layer_kv_shape_alternates_correctly() -> None:
    cfg = _make_cfg()
    model = Gemma4ForCausalLM(cfg)
    shapes = model.per_layer_kv_shape()
    assert len(shapes) == 8
    # Sliding indices get (4, 8); full indices (5, 7) get (2, 16).
    assert shapes[0] == (4, 8)
    assert shapes[4] == (4, 8)
    assert shapes[5] == (2, 16)
    assert shapes[6] == (4, 8)
    assert shapes[7] == (2, 16)


def test_per_layer_attention_translates_layer_types() -> None:
    cfg = _make_cfg()
    model = Gemma4ForCausalLM(cfg)
    pattern = model.per_layer_attention()
    assert pattern[0] == ("sliding", 4)
    assert pattern[5] == "full"
    assert pattern[7] == "full"


def test_unknown_layer_type_raises() -> None:
    cfg = _make_cfg(layer_types=["sliding_attention", "moe_attention"])
    model = Gemma4ForCausalLM(cfg)
    with pytest.raises(ValueError, match="unknown Gemma 4 layer_type"):
        model.per_layer_attention()


def test_full_layer_v_proj_is_none_under_k_eq_v() -> None:
    """`attention_k_eq_v=True` ⇒ full layers have no v_proj parameter."""
    cfg = _make_cfg()
    model = Gemma4ForCausalLM(cfg)
    sliding_layer = model.model.layers[0]
    full_layer = model.model.layers[5]
    assert sliding_layer.self_attn.v_proj is not None  # sliding always builds v_proj
    assert full_layer.self_attn.v_proj is None


def test_attention_k_eq_v_off_keeps_v_proj_everywhere() -> None:
    """If the config disables k_eq_v, full layers still build v_proj."""
    cfg = _make_cfg(attention_k_eq_v=False)
    model = Gemma4ForCausalLM(cfg)
    full_layer = model.model.layers[5]
    assert full_layer.self_attn.v_proj is not None


def test_expected_missing_state_keys_handles_tied_lm_head() -> None:
    cfg_tied = _make_cfg(tie_word_embeddings=True)
    cfg_untied = _make_cfg(tie_word_embeddings=False)
    assert Gemma4ForCausalLM(cfg_tied).expected_missing_state_keys() == {"lm_head.weight"}
    assert Gemma4ForCausalLM(cfg_untied).expected_missing_state_keys() == set()


def test_load_weights_strips_language_model_prefix_and_filters() -> None:
    """Synthetic state dict drives every load-side filter we implement."""
    cfg = _make_cfg()
    model = Gemma4ForCausalLM(cfg)
    target_state = model.state_dict()

    hf_state: dict[str, torch.Tensor] = {}
    # Re-emit our own params under the HF `model.language_model.` prefix
    # so the prefix-strip path is exercised. Skip the lm_head alias since
    # tied embeddings means HF's checkpoint won't ship it either.
    for our_key, tensor in target_state.items():
        if our_key == "lm_head.weight":
            continue
        if our_key.startswith("model."):
            hf_key = "model.language_model." + our_key[len("model.") :]
        else:
            hf_key = our_key
        hf_state[hf_key] = torch.zeros_like(tensor)

    # 1. Multimodal keys: should be filtered out.
    hf_state["model.vision_tower.encoder.layers.0.self_attn.q_proj.weight"] = torch.zeros(8)
    hf_state["model.embed_vision.embedding_projection.weight"] = torch.zeros(8)
    hf_state["model.audio_tower.layers.0.norm.weight"] = torch.zeros(8)
    # 2. v_norm.weight on a sliding layer: HF's `with_scale=False` means our
    # module has no `weight`, so this key should be dropped silently.
    hf_state["model.language_model.layers.0.self_attn.v_norm.weight"] = torch.zeros(8)
    # 3. v_proj on a full layer (idx 5): full layers have no v_proj, so this
    # checkpoint key should be filtered. Sliding-layer v_proj keys are kept.
    hf_state["model.language_model.layers.5.self_attn.v_proj.weight"] = torch.zeros(
        cfg.num_global_key_value_heads * cfg.global_head_dim, cfg.hidden_size
    )

    Gemma4ForCausalLM.load_weights(model, hf_state)


def test_load_weights_raises_on_real_unexpected_key() -> None:
    """A non-multimodal, non-filtered surplus key still surfaces as an error."""
    cfg = _make_cfg()
    model = Gemma4ForCausalLM(cfg)
    hf_state: dict[str, torch.Tensor] = {}
    for our_key, tensor in model.state_dict().items():
        if our_key == "lm_head.weight":
            continue
        if our_key.startswith("model."):
            hf_key = "model.language_model." + our_key[len("model.") :]
        else:
            hf_key = our_key
        hf_state[hf_key] = torch.zeros_like(tensor)
    hf_state["model.language_model.something_unexpected"] = torch.zeros(8)
    with pytest.raises(ValueError, match="weight load mismatch"):
        Gemma4ForCausalLM.load_weights(model, hf_state)


def test_from_hf_navigates_text_config() -> None:
    """`from_hf` must read `text_config` rather than the outer multimodal config."""
    text_cfg = SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_global_key_value_heads=2,
        global_head_dim=16,
        attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        rms_norm_eps=1e-6,
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        },
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
    )
    multimodal_cfg = SimpleNamespace(text_config=text_cfg)
    cfg = Gemma4Config.from_hf(multimodal_cfg)
    assert cfg.vocab_size == 128
    assert cfg.head_dim == 8
    assert cfg.global_head_dim == 16
    assert cfg.rope_partial_rotary_factor_global == 0.25
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.layer_types == ["sliding_attention", "full_attention"]


def test_forward_runs_with_heterogeneous_kv_and_softcap() -> None:
    """End-to-end forward with random init: shape correctness + softcap saturation."""
    cfg = _make_cfg()
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).to(torch.float32).eval()
    cache = _make_paged_cache(model)

    prompt_len = 6
    input_ids = torch.randint(0, cfg.vocab_size, (1, prompt_len), dtype=torch.long)
    position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, prompt_len], dtype=torch.int32)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )

    assert logits.shape == (1, prompt_len, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))
    # Final logit softcap caps |logits| at 30.0; with random init they
    # should comfortably fit, but still — assert the bound for safety.
    cap = cfg.final_logit_softcapping
    assert cap is not None
    assert torch.all(logits.abs() <= cap + 1e-4)


def test_block_pool_accepts_torch_attention_backend() -> None:
    """`attention_backend="torch"` is a valid value (the head_dim-agnostic fallback)."""
    cfg = _make_cfg()
    model = Gemma4ForCausalLM(cfg)
    pool = BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
        layer_attention=model.per_layer_attention(),
        layer_kv_shape=model.per_layer_kv_shape(),
    )
    assert pool.attention_backend == "torch"


def test_required_attention_backend_torch_when_head_dim_above_256() -> None:
    """Gemma 4 31B's head_dim=512 forces our materialized SDPA path.

    FlashInfer's prefill kernel and flash-attn 2 both reject head_dim=512;
    vLLM and SGLang both override the attention backend at the engine
    level for Gemma 4. Mirror that behavior so a Modal B200 run on the
    real 31B doesn't crash mid-prefill.
    """
    cfg_31b = _make_cfg()  # head_dim=8 sliding, global_head_dim=16 full → no override
    assert Gemma4ForCausalLM(cfg_31b).required_attention_backend() is None

    # Bump global_head_dim past 256 to mirror the real 31B's shape pattern.
    cfg_real = _make_cfg()
    cfg_real.global_head_dim = 512
    cfg_real.head_dim = 256
    assert Gemma4ForCausalLM(cfg_real).required_attention_backend() == "torch"


def test_forward_softcap_disabled_when_none() -> None:
    """`final_logit_softcapping=None` ⇒ raw lm_head output (no tanh)."""
    cfg = _make_cfg(final_logit_softcapping=None)
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).to(torch.float32).eval()
    cache = _make_paged_cache(model)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, 4], dtype=torch.int32)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )

    assert torch.all(torch.isfinite(logits))
