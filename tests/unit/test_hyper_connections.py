"""Hyper-Connections: kernel transcription + parity vs V4 reference's `Block`.

Three layers of validation:

  1. **Shape contract** — `hc_pre` and `hc_post` produce the right tensor
     shapes for any `(batch, seqlen, hc_mult, hidden_size)` input.

  2. **Doubly-stochastic invariant** — after `sinkhorn_iters` iterations
     of alternating row/column normalization, the comb matrix's row sums
     and column sums are both approximately 1 (modulo additive eps). The
     V4 reference relies on this property for the residual mixing to
     preserve information content across layers.

  3. **Bit-parity vs reference `Block.hc_pre` / `Block.hc_post`** —
     swap our PyTorch `hc_split_sinkhorn` into the kernel stub, build
     the reference's `Block` (which uses our pure-Python `hc_pre`/
     `hc_post` on top of our kernel transcription), and compare against
     our owned `HyperConnections.hc_pre` / `HyperConnections.hc_post`
     when both are seeded with identical weights. The math should be
     element-wise identical.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mini_infer.models.blocks.hyper_connections import (
    HyperConnections,
    hc_split_sinkhorn,
)

# ---------- Shape contracts ----------


def test_hc_split_sinkhorn_returns_correct_shapes() -> None:
    torch.manual_seed(0)
    batch, seqlen, hc_mult = 2, 4, 3
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(batch, seqlen, mix_hc)
    hc_scale = torch.randn(3)
    hc_base = torch.randn(mix_hc)

    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=20, eps=1e-6
    )
    assert pre.shape == (batch, seqlen, hc_mult)
    assert post.shape == (batch, seqlen, hc_mult)
    assert comb.shape == (batch, seqlen, hc_mult, hc_mult)


def test_hyper_connections_hc_pre_returns_reduced_state_plus_post_comb() -> None:
    torch.manual_seed(0)
    hidden_size = 8
    hc_mult = 3
    hc = HyperConnections(hidden_size=hidden_size, hc_mult=hc_mult)
    hc_state = torch.randn(2, 4, hc_mult, hidden_size)
    sublayer_input, post, comb = hc.hc_pre(hc_state)
    assert sublayer_input.shape == (2, 4, hidden_size)
    assert post.shape == (2, 4, hc_mult)
    assert comb.shape == (2, 4, hc_mult, hc_mult)


def test_hyper_connections_hc_post_returns_hc_mult_copies() -> None:
    torch.manual_seed(0)
    hidden_size = 8
    hc_mult = 3
    hc = HyperConnections(hidden_size=hidden_size, hc_mult=hc_mult)
    hc_state = torch.randn(2, 4, hc_mult, hidden_size)
    sublayer_input, post, comb = hc.hc_pre(hc_state)
    sublayer_output = torch.randn_like(sublayer_input)
    next_hc_state = hc.hc_post(sublayer_output, hc_state, post, comb)
    assert next_hc_state.shape == (2, 4, hc_mult, hidden_size)
    assert torch.all(torch.isfinite(next_hc_state))


def test_hyper_connections_rejects_invalid_hc_mult() -> None:
    with pytest.raises(ValueError, match="hc_mult"):
        HyperConnections(hidden_size=8, hc_mult=0)
    with pytest.raises(ValueError, match="sinkhorn_iters"):
        HyperConnections(hidden_size=8, hc_mult=2, sinkhorn_iters=0)


def test_hc_split_sinkhorn_rejects_wrong_hc_scale_shape() -> None:
    with pytest.raises(ValueError, match="hc_scale"):
        hc_split_sinkhorn(
            torch.zeros(1, 1, 12),
            torch.zeros(2),  # should be (3,)
            torch.zeros(12),
            hc_mult=3,
            sinkhorn_iters=10,
            eps=1e-6,
        )


# ---------- Sinkhorn-normalization invariant ----------


def test_comb_matrix_is_approximately_doubly_stochastic_after_sinkhorn() -> None:
    """After 20 Sinkhorn iterations, comb's row sums and col sums ≈ 1.

    Sinkhorn-Knopp iteration converges to a doubly-stochastic matrix when
    the input is positive. Our `eps` regularization prevents singularities;
    the cost is row/col sums slightly off from 1 (by ~hc_mult * eps).
    """
    torch.manual_seed(0)
    hc_mult = 4
    sinkhorn_iters = 20
    eps = 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(2, 4, mix_hc) * 0.5
    hc_scale = torch.randn(3) * 0.1
    hc_base = torch.randn(mix_hc) * 0.1

    _, _, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps
    )
    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)
    # Slack budget: a few times hc_mult * eps to absorb rounding +
    # the eps term added in every normalization step.
    tolerance = 5e-3
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=tolerance), (
        f"row sums diverge from 1 by max {(row_sums - 1).abs().max().item():.4e}"
    )
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=tolerance), (
        f"col sums diverge from 1 by max {(col_sums - 1).abs().max().item():.4e}"
    )


def test_pre_weights_in_unit_interval() -> None:
    """`pre = sigmoid(...) + eps` lives in `(eps, 1 + eps)` — bounded."""
    torch.manual_seed(0)
    hc_mult = 3
    mix_hc = (2 + hc_mult) * hc_mult
    eps = 1e-6
    pre, _, _ = hc_split_sinkhorn(
        torch.randn(1, 1, mix_hc) * 5.0,  # large magnitude to push sigmoid to extremes
        torch.randn(3) * 2.0,
        torch.randn(mix_hc) * 2.0,
        hc_mult=hc_mult,
        sinkhorn_iters=10,
        eps=eps,
    )
    assert torch.all(pre > eps - 1e-9)
    assert torch.all(pre < 1.0 + eps + 1e-9)


def test_post_weights_in_zero_two_interval() -> None:
    """`post = 2 * sigmoid(...)` lives in `(0, 2)` — twice sigmoid, no eps."""
    torch.manual_seed(0)
    hc_mult = 3
    mix_hc = (2 + hc_mult) * hc_mult
    _, post, _ = hc_split_sinkhorn(
        torch.randn(1, 1, mix_hc) * 5.0,
        torch.randn(3) * 2.0,
        torch.randn(mix_hc) * 2.0,
        hc_mult=hc_mult,
        sinkhorn_iters=10,
        eps=1e-6,
    )
    assert torch.all(post > 0.0)
    assert torch.all(post < 2.0)


def test_hc_mult_one_collapses_comb_to_one() -> None:
    """`hc_mult=1` makes comb a 1x1 matrix that Sinkhorn forces to ~1.

    This is the "no multi-residual" degenerate case — the design should
    collapse to a standard residual structure.
    """
    torch.manual_seed(0)
    hc_mult = 1
    mix_hc = (2 + hc_mult) * hc_mult
    _, _, comb = hc_split_sinkhorn(
        torch.randn(1, 4, mix_hc),
        torch.randn(3),
        torch.randn(mix_hc),
        hc_mult=hc_mult,
        sinkhorn_iters=20,
        eps=1e-6,
    )
    # 1x1 doubly-stochastic = scalar 1.0 (with eps slack).
    assert comb.shape == (1, 4, 1, 1)
    assert torch.allclose(comb, torch.ones_like(comb), atol=1e-3)


# ---------- Bit-parity vs the V4 reference's `Block.hc_pre` / `Block.hc_post` ----------


def _build_reference_args(reference_module: Any, *, hc_mult: int) -> Any:
    return reference_module.ModelArgs(
        max_batch_size=2,
        max_seq_len=64,
        dtype="bf16",
        dim=64,
        n_layers=4,
        n_heads=4,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(4,) * 4,
        original_seq_len=0,
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        n_routed_experts=4,
        n_activated_experts=2,
        score_func="softmax",
        route_scale=1.0,
        n_hash_layers=0,
        vocab_size=32,
        moe_inter_dim=32,
        n_shared_experts=1,
        swiglu_limit=0,
        expert_dtype="bf16",
        hc_mult=hc_mult,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


@pytest.mark.parametrize("hc_mult", [2, 3, 4])
def test_hc_pre_matches_reference_block(reference_module: Any, hc_mult: int) -> None:
    """Our `HyperConnections.hc_pre` produces the same `(reduced, post, comb)`
    as the reference's `Block.hc_pre` when both are seeded with identical weights."""
    torch.manual_seed(0)
    args = _build_reference_args(reference_module, hc_mult=hc_mult)
    ref_block = reference_module.Block(0, args)

    # Seed-fill the HC parameters on the reference's block.
    rng = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        ref_block.hc_attn_fn.data = (
            torch.randn(ref_block.hc_attn_fn.shape, generator=rng, dtype=torch.float32) * 0.05
        )
        ref_block.hc_attn_base.data = (
            torch.randn(ref_block.hc_attn_base.shape, generator=rng, dtype=torch.float32) * 0.05
        )
        ref_block.hc_attn_scale.data = (
            torch.randn(ref_block.hc_attn_scale.shape, generator=rng, dtype=torch.float32) * 0.5
        )

    our_hc = HyperConnections(
        hidden_size=args.dim,
        hc_mult=hc_mult,
        sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_eps=args.hc_eps,
        rms_norm_eps=args.norm_eps,
    )
    with torch.no_grad():
        our_hc.fn.copy_(ref_block.hc_attn_fn)
        our_hc.base.copy_(ref_block.hc_attn_base)
        our_hc.scale.copy_(ref_block.hc_attn_scale)

    batch_size, seqlen = 2, 6
    hc_state = torch.randn(batch_size, seqlen, hc_mult, args.dim) * 0.5

    with torch.no_grad():
        ours_reduced, ours_post, ours_comb = our_hc.hc_pre(hc_state)
        theirs_reduced, theirs_post, theirs_comb = ref_block.hc_pre(
            hc_state, ref_block.hc_attn_fn, ref_block.hc_attn_scale, ref_block.hc_attn_base
        )

    torch.testing.assert_close(ours_reduced, theirs_reduced, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ours_post, theirs_post, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ours_comb, theirs_comb, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hc_mult", [2, 3, 4])
def test_hc_post_matches_reference_block(reference_module: Any, hc_mult: int) -> None:
    """Our `HyperConnections.hc_post` produces the same hc_state output as the
    reference's `Block.hc_post` for matching `(post, comb, residual, sublayer_out)`."""
    torch.manual_seed(0)
    args = _build_reference_args(reference_module, hc_mult=hc_mult)
    ref_block = reference_module.Block(0, args)
    our_hc = HyperConnections(
        hidden_size=args.dim,
        hc_mult=hc_mult,
        sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_eps=args.hc_eps,
        rms_norm_eps=args.norm_eps,
    )

    batch_size, seqlen = 2, 5
    sublayer_out = torch.randn(batch_size, seqlen, args.dim) * 0.5
    residual = torch.randn(batch_size, seqlen, hc_mult, args.dim) * 0.5
    post = torch.rand(batch_size, seqlen, hc_mult)
    comb = torch.rand(batch_size, seqlen, hc_mult, hc_mult)
    # Make `comb` doubly stochastic by hand for a non-trivial test input.
    for _ in range(20):
        comb = comb / comb.sum(dim=-1, keepdim=True)
        comb = comb / comb.sum(dim=-2, keepdim=True)

    with torch.no_grad():
        ours = our_hc.hc_post(sublayer_out, residual, post, comb)
        theirs = ref_block.hc_post(sublayer_out, residual, post, comb)

    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)
