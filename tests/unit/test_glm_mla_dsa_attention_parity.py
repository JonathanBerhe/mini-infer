"""GLM-MoE-DSA attention parity vs HF `GlmMoeDsaAttention` WITH sparse selection.

Slice 1 validated MLA + non-interleaved RoPE under an all-pass mask; this test
turns DSA on (`index_topk < seq_len`) so the Lightning Indexer actually drops
keys, and checks the full attention output still matches HF to cosine-sim >
0.999. It exercises the whole DSA path end to end: indexer top-k -> `-inf`
sparse mask in `mla_packed_attention_forward` -> asymmetric SDPA -> o_proj.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool, StreamSpec
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models.blocks.glm_dsa_indexer import GlmDsaIndexer
from mini_infer.models.blocks.mla import MLAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding

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


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().to(torch.float32), b.flatten().to(torch.float32), dim=0
        ).item()
    )


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


def test_glm_mla_dsa_attention_matches_hf() -> None:
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaAttention,
        GlmMoeDsaRotaryEmbedding,
    )

    torch.manual_seed(0)
    cfg = GlmMoeDsaConfig(
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
        index_topk=INDEX_TOPK,  # < SEQ_LEN: selection actually drops keys
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        indexer_types=["full"],
        rope_parameters={"rope_theta": ROPE_THETA, "rope_type": "default"},
    )
    cfg._attn_implementation = "eager"
    hf_attn = GlmMoeDsaAttention(cfg, layer_idx=0).eval()

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
        use_interleaved_rope=False,
        indexer=indexer,
    ).eval()

    with torch.no_grad():
        # MLA projections.
        ours.q_a_proj.weight.copy_(hf_attn.q_a_proj.weight)
        ours.q_a_layernorm.weight.copy_(hf_attn.q_a_layernorm.weight)
        ours.q_b_proj.weight.copy_(hf_attn.q_b_proj.weight)
        ours.kv_a_proj_with_mqa.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
        ours.kv_a_layernorm.weight.copy_(hf_attn.kv_a_layernorm.weight)
        ours.kv_b_proj.weight.copy_(hf_attn.kv_b_proj.weight)
        ours.o_proj.weight.copy_(hf_attn.o_proj.weight)
        # Indexer projections.
        indexer.wq_b.weight.copy_(hf_attn.indexer.wq_b.weight)
        indexer.wk.weight.copy_(hf_attn.indexer.wk.weight)
        indexer.k_norm.weight.copy_(hf_attn.indexer.k_norm.weight)
        indexer.k_norm.bias.copy_(hf_attn.indexer.k_norm.bias)
        indexer.weights_proj.weight.copy_(hf_attn.indexer.weights_proj.weight)

    hidden_states = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)

    hf_rope = GlmMoeDsaRotaryEmbedding(cfg).eval()
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

    our_rope = RotaryEmbedding(head_dim=QK_ROPE, base=ROPE_THETA)
    our_cos, our_sin = our_rope(hidden_states, position_ids)
    cache = _make_paged_cache()
    cu_seqlens_q = torch.tensor([0, SEQ_LEN], dtype=torch.int32)
    with torch.no_grad():
        our_out = ours(hidden_states, (our_cos, our_sin), cache, cu_seqlens_q)

    assert hf_out.shape == our_out.shape
    cs = _cos_sim(hf_out, our_out)
    assert cs > 0.999, f"GLM MLA+DSA parity failed: cos_sim={cs:.6f}"
    assert torch.allclose(hf_out, our_out, atol=1e-4), (
        f"GLM MLA+DSA element-wise parity failed: "
        f"max_abs_diff={(hf_out - our_out).abs().max().item():.6f}"
    )
