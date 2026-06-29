"""GLM-MoE-DSA main-attention RoPE parity vs HF `GlmMoeDsaAttention`.

GLM-5.2's MLA uses non-interleaved (NeoX/Llama) RoPE, unlike DeepSeek-V2/V3
which use interleaved. This test pins that down: with synced MLA weights and
an all-pass DSA mask (`index_topk >= seq_len`, so the indexer selects every
token and the sparse mask degenerates to plain causal), our
`MLAAttention(use_interleaved_rope=False)` must match `GlmMoeDsaAttention`
to cosine-sim > 0.999. The indexer weights stay random because the all-pass
mask makes them irrelevant to the output, which isolates the RoPE convention
and the shared MLA math (low-rank Q, kv_b decompression, asymmetric SDPA).

A regression guard also confirms the default `use_interleaved_rope=True`
still matches the DeepSeek-V2 reference, so V2/V3/Kimi are untouched.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool, StreamSpec
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models.blocks.mla import MLAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding

# Tiny shared shape for the parity tests.
HIDDEN = 64
NUM_HEADS = 4
KV_LORA_RANK = 32
Q_LORA_RANK = 24
QK_NOPE = 16
QK_ROPE = 8
V_HEAD_DIM = 16
TOTAL_Q = 4
RMS_EPS = 1e-5
ROPE_THETA = 10000.0


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().to(torch.float32), b.flatten().to(torch.float32), dim=0
        ).item()
    )


def _make_paged_cache() -> PagedKVCache:
    streams = [
        [
            StreamSpec("kv_latent", 1, KV_LORA_RANK),
            StreamSpec("k_rope", 1, QK_ROPE),
        ]
    ]
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


def _make_ours(use_interleaved_rope: bool) -> MLAAttention:
    return MLAAttention(
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
        use_interleaved_rope=use_interleaved_rope,
    ).eval()


def _sync_mla_weights(ours: MLAAttention, hf_attn: torch.nn.Module) -> None:
    """Copy the shared MLA projections HF -> ours (identical names; world_size=1
    makes our Column/RowParallelLinear plain `.weight` linears)."""
    with torch.no_grad():
        ours.q_a_proj.weight.copy_(hf_attn.q_a_proj.weight)
        ours.q_a_layernorm.weight.copy_(hf_attn.q_a_layernorm.weight)
        ours.q_b_proj.weight.copy_(hf_attn.q_b_proj.weight)
        ours.kv_a_proj_with_mqa.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
        ours.kv_a_layernorm.weight.copy_(hf_attn.kv_a_layernorm.weight)
        ours.kv_b_proj.weight.copy_(hf_attn.kv_b_proj.weight)
        ours.o_proj.weight.copy_(hf_attn.o_proj.weight)


def test_glm_mla_matches_hf_with_allpass_dsa() -> None:
    """Non-interleaved RoPE: our MLA matches `GlmMoeDsaAttention` (all-pass DSA)."""
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
        # All-pass DSA: index_topk >= seq_len so every token is selected and
        # the sparse mask collapses to plain causal.
        index_topk=TOTAL_Q + 4,
        index_n_heads=2,
        index_head_dim=16,
        indexer_types=["full"],
        rope_parameters={"rope_theta": ROPE_THETA, "rope_type": "default"},
    )
    cfg._attn_implementation = "eager"

    hf_attn = GlmMoeDsaAttention(cfg, layer_idx=0).eval()
    ours = _make_ours(use_interleaved_rope=False)
    _sync_mla_weights(ours, hf_attn)

    hidden_states = torch.randn(1, TOTAL_Q, HIDDEN, dtype=torch.float32)
    position_ids = torch.arange(TOTAL_Q, dtype=torch.long).unsqueeze(0)

    hf_rope = GlmMoeDsaRotaryEmbedding(cfg).eval()
    with torch.no_grad():
        cos, sin = hf_rope(hidden_states, position_ids)
    causal_mask = torch.triu(
        torch.full((TOTAL_Q, TOTAL_Q), float("-inf"), dtype=torch.float32),
        diagonal=1,
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
    cu_seqlens_q = torch.tensor([0, TOTAL_Q], dtype=torch.int32)
    with torch.no_grad():
        our_out = ours(hidden_states, (our_cos, our_sin), cache, cu_seqlens_q)

    assert hf_out.shape == our_out.shape
    cs = _cos_sim(hf_out, our_out)
    assert cs > 0.999, f"GLM MLA parity failed: cos_sim={cs:.6f}"
    assert torch.allclose(hf_out, our_out, atol=1e-4), (
        f"GLM MLA element-wise parity failed: "
        f"max_abs_diff={(hf_out - our_out).abs().max().item():.6f}"
    )


def test_interleaved_rope_default_unchanged_vs_deepseek_v2() -> None:
    """Regression: default `use_interleaved_rope=True` still matches DeepSeek-V2.

    Guards that adding the GLM non-interleaved branch did not perturb the
    interleaved path that V2/V3/Kimi rely on.
    """
    pytest.importorskip("transformers.models.deepseek_v2.modeling_deepseek_v2")
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
        DeepseekV2Config,
    )
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2Attention,
        DeepseekV2RotaryEmbedding,
    )

    torch.manual_seed(0)
    cfg = DeepseekV2Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        q_lora_rank=None,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD_DIM,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    cfg.head_dim = QK_ROPE
    cfg.rope_parameters = {"rope_theta": ROPE_THETA, "rope_type": "default"}
    hf_attn = DeepseekV2Attention(cfg, layer_idx=0).eval()

    ours = MLAAttention(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD_DIM,
        q_lora_rank=None,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_idx=0,
    ).eval()  # use_interleaved_rope defaults True

    with torch.no_grad():
        ours.q_proj.weight.copy_(hf_attn.q_proj.weight)
        ours.kv_a_proj_with_mqa.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
        ours.kv_a_layernorm.weight.copy_(hf_attn.kv_a_layernorm.weight)
        ours.kv_b_proj.weight.copy_(hf_attn.kv_b_proj.weight)
        ours.o_proj.weight.copy_(hf_attn.o_proj.weight)

    hidden_states = torch.randn(1, TOTAL_Q, HIDDEN, dtype=torch.float32)
    position_ids = torch.arange(TOTAL_Q, dtype=torch.long).unsqueeze(0)
    hf_rope = DeepseekV2RotaryEmbedding(cfg).eval()
    with torch.no_grad():
        freqs_cis = hf_rope(hidden_states, position_ids)
    causal_mask = torch.triu(
        torch.full((TOTAL_Q, TOTAL_Q), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )[None, None, :, :]
    with torch.no_grad():
        hf_out, _ = hf_attn(
            hidden_states,
            attention_mask=causal_mask,
            past_key_values=None,
            position_embeddings=freqs_cis,
        )

    our_rope = RotaryEmbedding(head_dim=QK_ROPE, base=ROPE_THETA)
    cos, sin = our_rope(hidden_states, position_ids)
    cache = _make_paged_cache()
    cu_seqlens_q = torch.tensor([0, TOTAL_Q], dtype=torch.int32)
    with torch.no_grad():
        our_out = ours(hidden_states, (cos, sin), cache, cu_seqlens_q)

    assert _cos_sim(hf_out, our_out) > 0.999
    assert torch.allclose(hf_out, our_out, atol=1e-4)
