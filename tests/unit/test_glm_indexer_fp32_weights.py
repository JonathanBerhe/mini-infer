"""fp32 residency of the GLM DSA indexer's `weights_proj`.

HF keeps `indexer.weights_proj` fp32 under bf16 models
(`_keep_in_fp32_modules`) and, since transformers 5.12, computes
`weights_proj(hidden_states.to(weights_proj.weight.dtype))`: an fp32 matmul
over upcast hidden states (5.6.x ran the matmul in the native dtype and
upcast the result). The two are identical in fp32 but round differently at
bf16, where they can flip borderline top-k picks. These tests pin the
residency through whole-model dtype casts and the load path, and check bf16
parity against HF's indexer.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from mini_infer.models.blocks.glm_dsa_indexer import GlmDsaIndexer
from mini_infer.models.blocks.rope import RotaryEmbedding
from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

HIDDEN = 64
Q_LORA_RANK = 24
INDEX_N_HEADS = 2
INDEX_HEAD_DIM = 16
INDEX_TOPK = 3
QK_ROPE = 8
SEQ_LEN = 6
ROPE_THETA = 10000.0


def _make_indexer() -> GlmDsaIndexer:
    return GlmDsaIndexer(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA_RANK,
        num_heads=INDEX_N_HEADS,
        head_dim=INDEX_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE,
        index_topk=INDEX_TOPK,
    ).eval()


def _make_model_cfg() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        kv_lora_rank=32,
        q_lora_rank=Q_LORA_RANK,
        qk_nope_head_dim=16,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        rms_norm_eps=1e-5,
        rope_theta=ROPE_THETA,
        attention_bias=False,
        tie_word_embeddings=False,
        index_topk=INDEX_TOPK,
        index_head_dim=INDEX_HEAD_DIM,
        index_n_heads=INDEX_N_HEADS,
        mlp_layer_types=("dense",),
        indexer_types=("full",),
    )


def test_weights_proj_stays_fp32_through_dtype_cast() -> None:
    """A whole-module bf16 cast leaves `weights_proj` fp32 with its ORIGINAL
    bits (no bf16 round-trip), while every other parameter converts."""
    torch.manual_seed(0)
    indexer = _make_indexer()
    before = indexer.weights_proj.weight.detach().clone()

    indexer = indexer.to(torch.bfloat16)

    assert indexer.weights_proj.weight.dtype == torch.float32
    assert torch.equal(indexer.weights_proj.weight, before)
    assert indexer.wq_b.weight.dtype == torch.bfloat16
    assert indexer.wk.weight.dtype == torch.bfloat16
    assert indexer.k_norm.weight.dtype == torch.bfloat16


def test_load_keeps_weights_proj_fp32_under_bf16_model() -> None:
    """The `load_model` shape (construct, cast bf16, load a bf16 checkpoint):
    `weights_proj` ends fp32-resident holding the upcast checkpoint value,
    exactly like HF loading a bf16 checkpoint with `_keep_in_fp32_modules`."""
    torch.manual_seed(0)
    source = GlmMoeDsaForCausalLM(_make_model_cfg()).eval()
    ckpt = {name: tensor.to(torch.bfloat16) for name, tensor in source.state_dict().items()}

    model = GlmMoeDsaForCausalLM(_make_model_cfg()).to(torch.bfloat16).eval()
    GlmMoeDsaForCausalLM.load_weights(model, ckpt)

    indexer = model.model.layers[0].self_attn.indexer
    assert indexer is not None
    key = "model.layers.0.self_attn.indexer.weights_proj.weight"
    assert indexer.weights_proj.weight.dtype == torch.float32
    assert torch.equal(indexer.weights_proj.weight, ckpt[key].float())
    assert indexer.wk.weight.dtype == torch.bfloat16


def test_indexer_topk_and_weights_match_hf_at_bf16() -> None:
    """bf16 parity vs HF's indexer with `weights_proj` fp32-resident on both
    sides: the head-weight vectors match bitwise (same upcast matmul) and the
    selected top-k key sets agree per query."""
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig as HFGlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer

    if "position_ids" not in inspect.signature(GlmMoeDsaIndexer.forward).parameters:
        pytest.skip("installed transformers predates the 5.12 GLM indexer numerics")

    torch.manual_seed(0)
    hf_cfg = HFGlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=Q_LORA_RANK,
        qk_rope_head_dim=QK_ROPE,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        indexer_types=["full"],
        rope_parameters={"rope_theta": ROPE_THETA, "rope_type": "default"},
    )
    hf_indexer = GlmMoeDsaIndexer(hf_cfg, layer_idx=0).to(torch.bfloat16).eval()
    # Emulate HF's from_pretrained residency (`_keep_in_fp32_modules`): the
    # direct constructor + cast used here leaves the weight bf16, so pin it.
    hf_indexer.weights_proj.weight.data = hf_indexer.weights_proj.weight.data.float()

    ours = _make_indexer().to(torch.bfloat16)  # our pin keeps weights_proj fp32
    assert ours.weights_proj.weight.dtype == torch.float32
    with torch.no_grad():
        ours.wq_b.weight.copy_(hf_indexer.wq_b.weight)
        ours.wk.weight.copy_(hf_indexer.wk.weight)
        ours.k_norm.weight.copy_(hf_indexer.k_norm.weight)
        ours.k_norm.bias.copy_(hf_indexer.k_norm.bias)
        ours.weights_proj.weight.copy_(hf_indexer.weights_proj.weight)

    hidden_states = torch.randn(1, SEQ_LEN, HIDDEN).to(torch.bfloat16)
    q_resid = torch.randn(1, SEQ_LEN, Q_LORA_RANK).to(torch.bfloat16)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    cos, sin = RotaryEmbedding(head_dim=QK_ROPE, base=ROPE_THETA)(hidden_states, position_ids)

    # Head-weight vector: bitwise identical to HF's fp32-upcast matmul.
    with torch.no_grad():
        _, _, our_weights = ours._project(hidden_states, q_resid, (cos, sin))
        hf_weights = hf_indexer.weights_proj(
            hidden_states.to(hf_indexer.weights_proj.weight.dtype)
        ).float() * (INDEX_N_HEADS**-0.5)
    assert our_weights.dtype == torch.float32
    assert torch.equal(our_weights, hf_weights)

    # End-to-end selection parity on the 5.12 call convention.
    causal3d = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), float("-inf"), dtype=torch.float32), diagonal=1
    ).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, SEQ_LEN], dtype=torch.int32)
    with torch.no_grad():
        hf_topk = hf_indexer(hidden_states, q_resid, (cos, sin), causal3d, position_ids)
        our_topk = ours(hidden_states, q_resid, (cos, sin), cu_seqlens_q)

    assert hf_topk.shape == (1, SEQ_LEN, min(INDEX_TOPK, SEQ_LEN))
    assert len(our_topk) == 1
    for i in range(SEQ_LEN):
        valid = set(range(i + 1))
        hf_sel = set(hf_topk[0, i].tolist()) & valid
        our_sel = set(our_topk[0][i].tolist()) & valid
        assert hf_sel == our_sel, f"query {i}: HF selected {sorted(hf_sel)}, ours {sorted(our_sel)}"
