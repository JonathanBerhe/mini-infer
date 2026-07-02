"""Yarn mscale correction on the GLM-MoE-DSA attention softmax scale.

HF `GlmMoeDsaAttention` (transformers >= 5.12) multiplies its softmax scaling
by `yarn_get_mscale(factor, mscale_all_dim)**2` whenever `rope_parameters`
carry a non-"default" rope_type with `mscale_all_dim` set. GLM-5.2's published
config is rope_type="default" (plain 1/sqrt(qk_head_dim)), so the correction
is invisible to the default-rope parity tests; these pin the yarn path
explicitly: the formula on our config, the `from_hf` parsing, and full
attention parity against HF under a yarn rope config.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool, StreamSpec
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models.blocks.glm_dsa_indexer import GlmDsaIndexer
from mini_infer.models.blocks.mla import MLAAttention
from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig

HIDDEN = 64
NUM_HEADS = 4
KV_LORA_RANK = 32
Q_LORA_RANK = 24
QK_NOPE = 16
QK_ROPE = 8
V_HEAD_DIM = 16
INDEX_N_HEADS = 2
INDEX_HEAD_DIM = 16
INDEX_TOPK = 4
SEQ_LEN = 6
RMS_EPS = 1e-5
ROPE_THETA = 10000.0
YARN_FACTOR = 40.0
YARN_MSCALE_ALL_DIM = 0.707


def _make_cfg() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        q_lora_rank=Q_LORA_RANK,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD_DIM,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        attention_bias=False,
        tie_word_embeddings=False,
        index_topk=INDEX_TOPK,
        index_head_dim=INDEX_HEAD_DIM,
        index_n_heads=INDEX_N_HEADS,
        mlp_layer_types=("dense",),
        indexer_types=("full",),
    )


def test_attention_softmax_scale_formula() -> None:
    """Default rope keeps 1/sqrt(qk_head_dim); yarn with mscale_all_dim scales
    it by yarn_get_mscale(factor, mscale_all_dim)**2; yarn WITHOUT
    mscale_all_dim leaves the scale alone (all matching HF)."""
    plain = (QK_NOPE + QK_ROPE) ** -0.5
    cfg = _make_cfg()
    assert cfg.attention_softmax_scale() == pytest.approx(plain)

    yarn = replace(
        cfg,
        rope_type="yarn",
        rope_factor=YARN_FACTOR,
        rope_mscale_all_dim=YARN_MSCALE_ALL_DIM,
    )
    mscale = 0.1 * YARN_MSCALE_ALL_DIM * math.log(YARN_FACTOR) + 1.0
    assert yarn.attention_softmax_scale() == pytest.approx(plain * mscale * mscale)

    yarn_no_mscale = replace(cfg, rope_type="yarn", rope_factor=YARN_FACTOR)
    assert yarn_no_mscale.attention_softmax_scale() == pytest.approx(plain)


def test_from_hf_parses_rope_scaling_parameters() -> None:
    """`from_hf` lifts rope_type / factor / mscale_all_dim out of the HF
    config's `rope_parameters` (and defaults them when absent)."""
    base_fields = dict(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        q_lora_rank=Q_LORA_RANK,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD_DIM,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        rms_norm_eps=RMS_EPS,
        index_topk=INDEX_TOPK,
        index_head_dim=INDEX_HEAD_DIM,
        index_n_heads=INDEX_N_HEADS,
        mlp_layer_types=["dense"],
        indexer_types=["full"],
        quantization_config=None,
    )
    yarn_hf = SimpleNamespace(
        **base_fields,
        rope_parameters={
            "rope_theta": ROPE_THETA,
            "rope_type": "yarn",
            "factor": YARN_FACTOR,
            "mscale_all_dim": YARN_MSCALE_ALL_DIM,
        },
    )
    cfg = GlmMoeDsaConfig.from_hf(yarn_hf)
    assert cfg.rope_type == "yarn"
    assert cfg.rope_factor == YARN_FACTOR
    assert cfg.rope_mscale_all_dim == YARN_MSCALE_ALL_DIM

    default_hf = SimpleNamespace(**base_fields, rope_parameters={"rope_theta": ROPE_THETA})
    cfg_default = GlmMoeDsaConfig.from_hf(default_hf)
    assert cfg_default.rope_type == "default"
    assert cfg_default.rope_factor == 1.0
    assert cfg_default.rope_mscale_all_dim == 0.0


def _make_paged_cache() -> PagedKVCache:
    streams = [[StreamSpec("kv_latent", 1, KV_LORA_RANK), StreamSpec("k_rope", 1, QK_ROPE)]]
    pool = BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK,
        dtype=torch.float32,
        device="cpu",
        layer_streams=streams,
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def test_glm_mla_yarn_mscale_matches_hf() -> None:
    """Full attention parity vs HF under a yarn rope config. Both sides consume
    HF's yarn cos/sin tables, so the mscale-corrected softmax scale is the only
    delta this test can hide; without the correction the outputs diverge."""
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf_mod
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig as HFGlmMoeDsaConfig,
    )

    if not hasattr(hf_mod, "yarn_get_mscale"):
        pytest.skip("installed transformers predates the GLM yarn mscale correction (5.12)")

    torch.manual_seed(0)
    hf_cfg = HFGlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        q_lora_rank=Q_LORA_RANK,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD_DIM,
        attention_bias=False,
        rms_norm_eps=RMS_EPS,
        max_position_embeddings=64,
        index_topk=INDEX_TOPK,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        indexer_types=["full"],
        rope_parameters={
            "rope_theta": ROPE_THETA,
            "rope_type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": 32,
            "mscale": YARN_MSCALE_ALL_DIM,
            "mscale_all_dim": YARN_MSCALE_ALL_DIM,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )
    hf_cfg._attn_implementation = "eager"
    hf_attn = hf_mod.GlmMoeDsaAttention(hf_cfg, layer_idx=0).eval()

    # The yarn correction must actually have engaged on the HF side, and our
    # config-driven scale must reproduce it exactly.
    plain = (QK_NOPE + QK_ROPE) ** -0.5
    assert hf_attn.scaling != pytest.approx(plain)
    ours_cfg = GlmMoeDsaConfig.from_hf(hf_cfg)
    assert ours_cfg.attention_softmax_scale() == pytest.approx(hf_attn.scaling)

    indexer = GlmDsaIndexer(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA_RANK,
        num_heads=INDEX_N_HEADS,
        head_dim=INDEX_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE,
        index_topk=INDEX_TOPK,
    )
    ours = MLAAttention(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD_DIM,
        q_lora_rank=Q_LORA_RANK,
        rms_norm_eps=RMS_EPS,
        attention_bias=False,
        layer_idx=0,
        softmax_scale=ours_cfg.attention_softmax_scale(),
        indexer=indexer,
    ).eval()

    with torch.no_grad():
        ours.q_a_proj.weight.copy_(hf_attn.q_a_proj.weight)
        ours.q_a_layernorm.weight.copy_(hf_attn.q_a_layernorm.weight)
        ours.q_b_proj.weight.copy_(hf_attn.q_b_proj.weight)
        ours.kv_a_proj_with_mqa.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
        ours.kv_a_layernorm.weight.copy_(hf_attn.kv_a_layernorm.weight)
        ours.kv_b_proj.weight.copy_(hf_attn.kv_b_proj.weight)
        ours.o_proj.weight.copy_(hf_attn.o_proj.weight)
        indexer.wq_b.weight.copy_(hf_attn.indexer.wq_b.weight)
        indexer.wk.weight.copy_(hf_attn.indexer.wk.weight)
        indexer.k_norm.weight.copy_(hf_attn.indexer.k_norm.weight)
        indexer.k_norm.bias.copy_(hf_attn.indexer.k_norm.bias)
        indexer.weights_proj.weight.copy_(hf_attn.indexer.weights_proj.weight)

    hidden_states = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)

    # HF's yarn rotary embedding feeds BOTH sides: the rope tables are shared,
    # isolating the softmax-scale correction as the only numeric difference.
    hf_rope = hf_mod.GlmMoeDsaRotaryEmbedding(hf_cfg).eval()
    with torch.no_grad():
        cos, sin = hf_rope(hidden_states, position_ids)
    causal_mask = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), float("-inf"), dtype=torch.float32), diagonal=1
    )[None, None, :, :]
    with torch.no_grad():
        hf_out, _, _ = hf_attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
            past_key_values=None,
            prev_topk_indices=None,
        )

    cache = _make_paged_cache()
    cu_seqlens_q = torch.tensor([0, SEQ_LEN], dtype=torch.int32)
    with torch.no_grad():
        our_out = ours(hidden_states, (cos, sin), cache, cu_seqlens_q)

    assert hf_out.shape == our_out.shape
    assert torch.allclose(hf_out, our_out, atol=1e-4), (
        f"GLM MLA yarn mscale parity failed: "
        f"max_abs_diff={(hf_out - our_out).abs().max().item():.6f}"
    )
