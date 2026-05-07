"""DeepSeek-V2 owned model: registry, MLA cache wiring, weight-load filters.

The 16B V2-Lite checkpoint is too large for M1 (~30 GB at bf16); these
tests use a synthetic mini-config that exercises the structurally
distinctive bits — heterogeneous FFN per layer (dense + MoE), MLA
streams, top-k routing with shared experts, weight-name remapping for
HF's per-expert layout — without any model load. The full-model
validation is one Modal B200 smoke run on the real V2-Lite checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models import REGISTRY
from mini_infer.models.deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM


def _make_cfg(
    *,
    num_hidden_layers: int = 4,
    first_k_dense_replace: int = 1,
    n_routed_experts: int = 4,
    n_shared_experts: int = 1,
    num_experts_per_tok: int = 2,
    tie_word_embeddings: bool = False,
) -> DeepseekV2Config:
    return DeepseekV2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        kv_lora_rank=32,
        q_lora_rank=None,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        first_k_dense_replace=first_k_dense_replace,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=tie_word_embeddings,
    )


def _make_paged_cache(model: DeepseekV2ForCausalLM) -> PagedKVCache:
    pool = BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=1,
        head_dim=model.cfg.kv_lora_rank,
        dtype=torch.float32,
        device="cpu",
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def test_registry_has_deepseek_v2() -> None:
    """`DeepseekV2ForCausalLM` registers under its HF arch string."""
    cls = REGISTRY.lookup("DeepseekV2ForCausalLM")
    assert cls is DeepseekV2ForCausalLM


def test_per_layer_streams_returns_kv_latent_and_k_rope() -> None:
    """Stage C3's `per_layer_streams` reports the MLA shape per layer."""
    cfg = _make_cfg()
    model = DeepseekV2ForCausalLM(cfg)
    streams = model.per_layer_streams()
    assert streams is not None
    assert len(streams) == cfg.num_hidden_layers
    for layer_streams in streams:
        names = [s.name for s in layer_streams]
        assert names == ["kv_latent", "k_rope"]
        kv_latent = layer_streams[0]
        k_rope = layer_streams[1]
        assert kv_latent.num_kv_heads == 1
        assert kv_latent.head_dim == cfg.kv_lora_rank
        assert k_rope.num_kv_heads == 1
        assert k_rope.head_dim == cfg.qk_rope_head_dim


def test_required_attention_backend_is_torch() -> None:
    """MLA's asymmetric V head_dim forces materialized SDPA path."""
    cfg = _make_cfg()
    model = DeepseekV2ForCausalLM(cfg)
    assert model.required_attention_backend() == "torch"


def test_dense_layer_uses_swiglu_moe_layer_uses_moeffn() -> None:
    """`first_k_dense_replace` controls the FFN type per layer."""
    from mini_infer.models.blocks import MoEFFN, SwiGLU

    cfg = _make_cfg(num_hidden_layers=4, first_k_dense_replace=1)
    model = DeepseekV2ForCausalLM(cfg)
    assert isinstance(model.model.layers[0].mlp, SwiGLU)
    assert isinstance(model.model.layers[1].mlp, MoEFFN)
    assert isinstance(model.model.layers[3].mlp, MoEFFN)


def test_moe_layer_has_shared_experts_and_routed_experts() -> None:
    """V2-Lite-style MoE: routed `n_routed_experts` + shared MLP."""
    cfg = _make_cfg(n_routed_experts=4, n_shared_experts=2)
    model = DeepseekV2ForCausalLM(cfg)
    moe = model.model.layers[1].mlp
    assert len(moe.experts) == 4
    assert moe.shared_experts is not None
    # Shared experts collapse N MLPs into a single one with N-times the
    # intermediate size; verify the shape.
    assert moe.shared_experts.w1.weight.shape == (
        cfg.moe_intermediate_size * cfg.n_shared_experts,
        cfg.hidden_size,
    )


def test_expected_missing_state_keys_handles_tied_lm_head() -> None:
    cfg_untied = _make_cfg(tie_word_embeddings=False)
    cfg_tied = _make_cfg(tie_word_embeddings=True)
    assert DeepseekV2ForCausalLM(cfg_untied).expected_missing_state_keys() == set()
    assert DeepseekV2ForCausalLM(cfg_tied).expected_missing_state_keys() == {"lm_head.weight"}


def test_load_weights_remaps_hf_per_expert_keys() -> None:
    """HF's `mlp.experts.{j}.{gate,up,down}_proj.weight` -> our `w{1,3,2}.weight`."""
    cfg = _make_cfg(num_hidden_layers=2, n_routed_experts=2, n_shared_experts=1)
    model = DeepseekV2ForCausalLM(cfg)
    target_state = model.state_dict()

    # Build a synthetic state dict in HF naming convention.
    hf_state: dict[str, torch.Tensor] = {}
    rename = {
        ".mlp.experts.0.w1.weight": ".mlp.experts.0.gate_proj.weight",
        ".mlp.experts.0.w2.weight": ".mlp.experts.0.down_proj.weight",
        ".mlp.experts.0.w3.weight": ".mlp.experts.0.up_proj.weight",
        ".mlp.experts.1.w1.weight": ".mlp.experts.1.gate_proj.weight",
        ".mlp.experts.1.w2.weight": ".mlp.experts.1.down_proj.weight",
        ".mlp.experts.1.w3.weight": ".mlp.experts.1.up_proj.weight",
        ".mlp.shared_experts.w1.weight": ".mlp.shared_experts.gate_proj.weight",
        ".mlp.shared_experts.w2.weight": ".mlp.shared_experts.down_proj.weight",
        ".mlp.shared_experts.w3.weight": ".mlp.shared_experts.up_proj.weight",
    }
    for key, tensor in target_state.items():
        new_key = key
        for ours, hf in rename.items():
            if ours in new_key:
                new_key = new_key.replace(ours, hf)
                break
        hf_state[new_key] = torch.zeros_like(tensor)

    DeepseekV2ForCausalLM.load_weights(model, hf_state)


def test_from_hf_parses_lite_config_shape() -> None:
    """Validates `from_hf` against a config matching the V2-Lite checkpoint."""
    hf_cfg = SimpleNamespace(
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        num_hidden_layers=27,
        num_attention_heads=16,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=64,
        n_shared_experts=2,
        num_experts_per_tok=6,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    cfg = DeepseekV2Config.from_hf(hf_cfg)
    assert cfg.qk_head_dim == 192
    assert cfg.is_moe_layer(0) is False
    assert cfg.is_moe_layer(1) is True
    assert cfg.is_moe_layer(26) is True


def test_forward_runs_through_paged_cache() -> None:
    """End-to-end forward: dense layer 0 + MoE layers 1-3 + MLA + lm_head."""
    cfg = _make_cfg(num_hidden_layers=2, first_k_dense_replace=1)
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(cfg).to(torch.float32).eval()
    cache = _make_paged_cache(model)

    total_q = 4
    input_ids = torch.randint(0, cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )
    assert logits.shape == (1, total_q, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def test_load_weights_raises_on_unexpected_key() -> None:
    cfg = _make_cfg(num_hidden_layers=1)
    model = DeepseekV2ForCausalLM(cfg)
    target_state = model.state_dict()
    hf_state: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in target_state.items()}
    hf_state["model.something_unexpected"] = torch.zeros(8)
    with pytest.raises(ValueError, match="weight load mismatch"):
        DeepseekV2ForCausalLM.load_weights(model, hf_state)
