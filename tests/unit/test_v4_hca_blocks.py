"""Shape and math tests for V4 attention primitives in isolation.

The HF parity test (`test_v4_hca_parity.py`) is the strong correctness
gate; these tests pin down the things that can drift silently while
working on the parity test:

  - `TokenLevelCompressor` output shape, fp32 internal math, and that
    its softmax is taken across the right axis (per-block, not global).
  - `AttentionSink` parameter shape (one scalar per query head).
  - `GroupedOutputProjection` shape, and equivalence to a plain
    `o_proj` when `num_groups == 1` (i.e. no grouping).
  - `apply_partial_rope_last_n_dims`: only the last `N` dims rotate;
    the rest pass through unchanged. Inverse rotation cancels forward.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from mini_infer.cache.hca_attention import hca_mqa_with_sink
from mini_infer.models.blocks import (
    AttentionSink,
    GroupedOutputProjection,
    HCAAttention,
    TokenLevelCompressor,
)
from mini_infer.models.blocks.rope import RotaryEmbedding, apply_partial_rope_last_n_dims

# ---------- TokenLevelCompressor ----------


def test_compressor_emits_one_entry_per_block() -> None:
    """`(B=2, T=128, d=32)` with `m=16` -> compressed shape `(2, 8, c)`."""
    torch.manual_seed(0)
    comp = TokenLevelCompressor(
        hidden_size=32, kv_head_dim=64, rope_head_dim=16, compression_ratio=16
    )
    rope = RotaryEmbedding(head_dim=16, base=10000.0)
    x = torch.randn(2, 128, 32)
    n_blocks = 128 // 16
    block_positions = torch.arange(n_blocks).unsqueeze(0) * 16  # (1, n_blocks)
    cos, sin = rope(x[:, :n_blocks], block_positions)  # (1, n_blocks, 16)
    cos = cos.expand(2, -1, -1)
    sin = sin.expand(2, -1, -1)
    out = comp(x, (cos, sin))
    assert out.shape == (2, 8, 64)
    assert torch.all(torch.isfinite(out))


def test_compressor_rejects_uneven_seqlen() -> None:
    """Standalone forward expects `seqlen % m == 0`."""
    comp = TokenLevelCompressor(hidden_size=8, kv_head_dim=8, rope_head_dim=0, compression_ratio=4)
    x = torch.randn(1, 7, 8)
    cos = torch.zeros(1, 1, 0)
    sin = torch.zeros(1, 1, 0)
    with pytest.raises(ValueError, match="multiple of compression_ratio"):
        comp(x, (cos, sin))


def test_compressor_softmax_across_block_axis() -> None:
    """Compression weights are softmax-normalized per `m`-token block, not globally.

    Verified by stripping the position bias to zero and feeding deterministic
    inputs that make the per-block softmax a uniform distribution. The output
    should equal the per-block mean of the KV projection.
    """
    torch.manual_seed(1)
    m = 4
    comp = TokenLevelCompressor(hidden_size=8, kv_head_dim=8, rope_head_dim=0, compression_ratio=m)
    # Force uniform softmax: zero out weight_proj weights so logits are zero
    # everywhere (plus zero position_bias by default), then softmax = 1/m.
    with torch.no_grad():
        comp.weight_proj.weight.zero_()
        # Disable RMSNorm scaling so we can compare to plain mean(kv_proj(x)).
        comp.norm.weight.fill_(1.0)

    x = torch.randn(1, 8, 8)
    cos = torch.zeros(1, 2, 0)  # rope_head_dim=0
    sin = torch.zeros(1, 2, 0)
    out = comp(x, (cos, sin))

    # Expected: per-m-block mean of kv_proj(x), then RMSNorm.
    kv = comp.kv_proj(x.float())  # (1, 8, 8)
    expected = kv.unflatten(1, (-1, m)).mean(dim=2)  # (1, 2, 8) -- uniform softmax => mean
    expected = comp.norm(expected.to(x.dtype))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


# ---------- AttentionSink ----------


def test_attention_sink_has_one_logit_per_head() -> None:
    sink = AttentionSink(num_heads=8)
    assert sink.sink_logits.shape == (8,)
    assert sink.sink_logits.dtype == torch.float32


def test_attention_sink_rejects_zero_heads() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        AttentionSink(num_heads=0)


def test_sink_steals_softmax_mass() -> None:
    """Large positive sink logit pulls weight away from real keys.

    With `sink_logits = +10` and unit-scale Q/K/V, the sink absorbs near-all
    softmax mass; the attention output magnitude should be much smaller than
    the equivalent run with `sink_logits = -10` (which makes the sink negligible).
    """
    torch.manual_seed(0)
    bsz, seqlen, n_h, c = 1, 4, 2, 8
    n_kv = 4
    q = torch.randn(bsz, seqlen, n_h, c)
    kv = torch.randn(bsz, n_kv, c)
    topk_idxs = torch.arange(n_kv).unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1)
    scale = 1.0 / math.sqrt(c)

    big_sink = torch.full((n_h,), 10.0)
    out_big = hca_mqa_with_sink(q, kv, big_sink, topk_idxs, scale)

    small_sink = torch.full((n_h,), -10.0)
    out_small = hca_mqa_with_sink(q, kv, small_sink, topk_idxs, scale)

    # Big sink => most mass in the dropped sink column => |out| is small.
    assert out_big.abs().mean() < out_small.abs().mean() * 0.1


# ---------- GroupedOutputProjection ----------


def test_grouped_output_projects_to_hidden_size() -> None:
    proj = GroupedOutputProjection(
        num_heads=8, kv_head_dim=64, num_groups=2, o_lora_rank=32, hidden_size=128
    )
    attn = torch.randn(2, 4, 8, 64)
    out = proj(attn)
    assert out.shape == (2, 4, 128)


def test_grouped_output_with_g1_matches_plain_low_rank() -> None:
    """When `num_groups == 1` the per-group einsum reduces to a single matmul.

    Verifies that the grouped layout doesn't introduce any spurious
    reshape arithmetic — a plain low-rank `o_proj` (one wo_a + one wo_b)
    yields the same output bit-for-bit.
    """
    torch.manual_seed(0)
    n_h, c, r, d = 8, 64, 32, 128
    grouped = GroupedOutputProjection(
        num_heads=n_h, kv_head_dim=c, num_groups=1, o_lora_rank=r, hidden_size=d
    )
    # `wo_a` is `nn.Parameter(torch.empty(...))` — uninitialized memory may contain NaN.
    # Re-init deterministically before the comparison so we're testing the math,
    # not zero-cost garbage propagation.
    with torch.no_grad():
        grouped.wo_a.normal_(mean=0.0, std=0.02)
        grouped.wo_b.weight.normal_(mean=0.0, std=0.02)
    # Plain low-rank: (n_h * c -> r -> d).
    wo_a_plain = nn.Linear(n_h * c, r, bias=False)
    wo_b_plain = nn.Linear(r, d, bias=False)
    with torch.no_grad():
        wo_a_plain.weight.copy_(grouped.wo_a)  # (r, n_h * c)
        wo_b_plain.weight.copy_(grouped.wo_b.weight)  # (d, r)

    attn = torch.randn(1, 4, n_h, c)
    out_grouped = grouped(attn)
    out_plain = wo_b_plain(wo_a_plain(attn.flatten(2)))
    torch.testing.assert_close(out_grouped, out_plain, rtol=1e-5, atol=1e-6)


def test_grouped_output_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        GroupedOutputProjection(
            num_heads=7, kv_head_dim=8, num_groups=2, o_lora_rank=4, hidden_size=8
        )


# ---------- partial RoPE ----------


def test_partial_rope_only_rotates_last_n_dims() -> None:
    """Dims `[0, kv_head_dim - rope_head_dim)` pass through; tail rotates."""
    torch.manual_seed(0)
    bsz, seqlen, n_h, c = 1, 4, 2, 16
    rope_dim = 8
    rope = RotaryEmbedding(head_dim=rope_dim, base=10000.0)
    positions = torch.arange(seqlen).unsqueeze(0)
    cos, sin = rope(torch.zeros(bsz, seqlen), positions)

    x = torch.randn(bsz, seqlen, n_h, c)
    out = apply_partial_rope_last_n_dims(x, cos, sin, rope_dim)

    # Leading nope dims unchanged.
    torch.testing.assert_close(out[..., : c - rope_dim], x[..., : c - rope_dim])
    # Tail dims should differ from input (rotation, not identity) for
    # at least one position. Position 0 is identity (cos=1, sin=0); skip it.
    assert not torch.allclose(out[:, 1:, :, -rope_dim:], x[:, 1:, :, -rope_dim:])


def test_partial_rope_inverse_cancels_forward() -> None:
    """Forward rotate + inverse rotate at the same positions = identity (within fp32 eps)."""
    torch.manual_seed(0)
    bsz, seqlen, n_h, c = 1, 8, 2, 32
    rope_dim = 16
    rope = RotaryEmbedding(head_dim=rope_dim, base=10000.0)
    positions = torch.arange(seqlen).unsqueeze(0)
    cos, sin = rope(torch.zeros(bsz, seqlen), positions)

    x = torch.randn(bsz, seqlen, n_h, c)
    rotated = apply_partial_rope_last_n_dims(x, cos, sin, rope_dim, inverse=False)
    restored = apply_partial_rope_last_n_dims(rotated, cos, sin, rope_dim, inverse=True)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-5)


# ---------- HCAAttention smoke ----------


def test_hca_attention_forward_runs_and_returns_correct_shape() -> None:
    """End-to-end shape check for the composite block (no parity oracle)."""
    torch.manual_seed(0)
    bsz, seqlen, hidden_size = 1, 32, 64
    block = HCAAttention(
        hidden_size=hidden_size,
        num_heads=4,
        q_lora_rank=32,
        kv_head_dim=16,
        rope_head_dim=8,
        num_groups=2,
        o_lora_rank=16,
        window_size=8,
        compression_ratio=8,
    )
    x = torch.randn(bsz, seqlen, hidden_size)

    rope = RotaryEmbedding(head_dim=8, base=10000.0)
    positions = torch.arange(seqlen).unsqueeze(0)
    cos_t, sin_t = rope(x, positions)
    n_compressed = seqlen // 8
    cmp_positions = (torch.arange(n_compressed) * 8).unsqueeze(0)
    cos_c, sin_c = rope(x[:, :n_compressed], cmp_positions)

    out = block(x, (cos_t, sin_t), (cos_c, sin_c))
    assert out.shape == (bsz, seqlen, hidden_size)
    assert torch.all(torch.isfinite(out))
