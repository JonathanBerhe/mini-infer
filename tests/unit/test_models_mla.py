"""MLA attention block — shape sanity + bit-parity vs HF DeepSeek-V2.

The HF parity test is the strong correctness gate for Stage C3: with
synced weights, our `MLAAttention` should produce attention output with
cosine similarity > 0.999 against `DeepseekV2Attention` on the same
input. This nails down the unusual bits — interleaved RoPE on q_pe and
the shared k_rope, decompression via `kv_a_layernorm + kv_b_proj`,
asymmetric Q/K vs V head_dim — without needing to download a real
checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool, StreamSpec
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models.blocks.mla import MLAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().to(torch.float32), b.flatten().to(torch.float32), dim=0
        ).item()
    )


def _make_mini_mla(
    *,
    hidden_size: int = 64,
    num_heads: int = 4,
    kv_lora_rank: int = 32,
    qk_nope_head_dim: int = 16,
    qk_rope_head_dim: int = 8,
    v_head_dim: int = 16,
    q_lora_rank: int | None = None,
) -> MLAAttention:
    return MLAAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_idx=0,
    )


def _make_paged_cache(num_layers: int, kv_lora_rank: int, qk_rope_head_dim: int) -> PagedKVCache:
    streams = [
        [
            StreamSpec("kv_latent", 1, kv_lora_rank),
            StreamSpec("k_rope", 1, qk_rope_head_dim),
        ]
    ] * num_layers
    pool = BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=kv_lora_rank,
        dtype=torch.float32,
        device="cpu",
        layer_streams=streams,
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def test_mla_attention_q_lora_path_constructs_correctly() -> None:
    """`q_lora_rank` not None → low-rank Q path (V2 / V3); None → direct (V2-Lite)."""
    direct = _make_mini_mla(q_lora_rank=None)
    assert direct.q_proj is not None
    assert direct.q_a_proj is None and direct.q_a_layernorm is None and direct.q_b_proj is None

    low_rank = _make_mini_mla(q_lora_rank=24)
    assert low_rank.q_proj is None
    assert low_rank.q_a_proj is not None
    assert low_rank.q_a_layernorm is not None
    assert low_rank.q_b_proj is not None


def test_mla_attention_forward_shapes() -> None:
    """End-to-end forward through PagedKVCache + asymmetric SDPA."""
    torch.manual_seed(0)
    block = _make_mini_mla()
    cache = _make_paged_cache(num_layers=1, kv_lora_rank=32, qk_rope_head_dim=8)

    total_q = 4
    hidden_states = torch.randn(1, total_q, 64, dtype=torch.float32)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)
    rope = RotaryEmbedding(head_dim=8, base=10000.0)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cos, sin = rope(hidden_states, position_ids)

    out = block(hidden_states, (cos, sin), cache, cu_seqlens_q)
    assert out.shape == (1, total_q, 64)
    assert torch.all(torch.isfinite(out))


def test_mla_block_matches_hf_reference() -> None:
    """Bit-parity vs HF `DeepseekV2Attention` on synced weights and inputs.

    Verifies the four components most likely to drift:
      1. Q path (direct projection on V2-Lite-shape config)
      2. KV-down + decompression (`kv_a_proj_with_mqa` -> split ->
         `kv_a_layernorm` -> `kv_b_proj` -> split -> per-head k_nope, v)
      3. Interleaved RoPE on q_pe and the shared k_rope
      4. Asymmetric SDPA (Q/K head_dim=qk_nope+qk_rope, V head_dim=v_head_dim)
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
    hidden_size = 64
    num_heads = 4
    kv_lora_rank = 32
    qk_nope_head_dim = 16
    qk_rope_head_dim = 8
    v_head_dim = 16

    cfg = DeepseekV2Config(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=None,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    # HF's RoPE init reads `cfg.head_dim` for inv_freq sizing — for MLA
    # this should equal `qk_rope_head_dim` (only the rope slice is rotated).
    cfg.head_dim = qk_rope_head_dim
    # Pin RoPE to default (no scaling) so synthetic test stays simple.
    cfg.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
    hf_attn = DeepseekV2Attention(cfg, layer_idx=0).eval()

    ours = _make_mini_mla(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=None,
    ).eval()

    # Sync weights (HF -> ours; both have identical names for the layers we
    # use). Note: HF stores (out, in) per nn.Linear convention; same as ours.
    with torch.no_grad():
        ours.q_proj.weight.copy_(hf_attn.q_proj.weight)
        ours.kv_a_proj_with_mqa.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
        ours.kv_a_layernorm.weight.copy_(hf_attn.kv_a_layernorm.weight)
        ours.kv_b_proj.weight.copy_(hf_attn.kv_b_proj.weight)
        ours.o_proj.weight.copy_(hf_attn.o_proj.weight)

    total_q = 4
    hidden_states = torch.randn(1, total_q, hidden_size, dtype=torch.float32)

    # HF path: build freqs_cis the same way DeepseekV2RotaryEmbedding does.
    hf_rope = DeepseekV2RotaryEmbedding(cfg).eval()
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        freqs_cis = hf_rope(hidden_states, position_ids)
    # Build a 4D causal mask for HF (B, 1, T_q, T_k); 0 for keep, -inf for mask.
    causal_mask = torch.triu(
        torch.full((total_q, total_q), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )[None, None, :, :]
    with torch.no_grad():
        hf_out, _ = hf_attn(
            hidden_states,
            attention_mask=causal_mask,
            past_key_values=None,
            position_embeddings=freqs_cis,
        )

    # Our path: same inv_freq → cos/sin emitted by our RotaryEmbedding.
    our_rope = RotaryEmbedding(head_dim=qk_rope_head_dim, base=10000.0)
    cos, sin = our_rope(hidden_states, position_ids)
    cache = _make_paged_cache(
        num_layers=1, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim
    )
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)
    with torch.no_grad():
        our_out = ours(hidden_states, (cos, sin), cache, cu_seqlens_q)

    assert hf_out.shape == our_out.shape
    cs = _cos_sim(hf_out, our_out)
    assert cs > 0.999, f"MLA block parity failed: cos_sim={cs:.6f}"
    assert torch.allclose(hf_out, our_out, atol=1e-4), (
        f"MLA block element-wise parity failed: "
        f"max_abs_diff={(hf_out - our_out).abs().max().item():.6f}"
    )
