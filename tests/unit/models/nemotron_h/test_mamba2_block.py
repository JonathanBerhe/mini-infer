"""Mamba-2 blocks: the chunked SSD scan against the token-by-token recurrence.

`ssd_recurrent` is the definition; `ssd_chunked` must compute the same
outputs and final state for any length, including lengths that leave a
partial last chunk, and when it continues from a carried state.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.nemotron_h.mamba2 import (
    gated_group_rmsnorm,
    mamba_conv_prefill,
    mamba_conv_step,
    ssd_chunked,
    ssd_recurrent,
)


def _inputs(seq_len: int, *, heads: int = 4, head_dim: int = 6, state: int = 5, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(2, seq_len, heads, head_dim, generator=gen)
    dt = torch.rand(2, seq_len, heads, generator=gen) * 0.5 + 0.05
    a = -torch.rand(heads, generator=gen) * 2.0 - 0.1
    b = torch.randn(2, seq_len, heads, state, generator=gen)
    c = torch.randn(2, seq_len, heads, state, generator=gen)
    return x, dt, a, b, c


@pytest.mark.parametrize("seq_len", [1, 7, 8, 9, 23, 32])
def test_chunked_matches_recurrent(seq_len: int) -> None:
    x, dt, a, b, c = _inputs(seq_len)
    y_ref, state_ref = ssd_recurrent(x, dt, a, b, c)
    y, state = ssd_chunked(x, dt, a, b, c, chunk_size=8)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state, state_ref, rtol=1e-5, atol=1e-5)


def test_chunked_continues_from_a_carried_state() -> None:
    """Splitting a sequence and carrying the state equals one pass."""
    x, dt, a, b, c = _inputs(29, seed=1)
    y_full, state_full = ssd_chunked(x, dt, a, b, c, chunk_size=8)
    y_head, state_head = ssd_chunked(x[:, :11], dt[:, :11], a, b[:, :11], c[:, :11], chunk_size=8)
    y_tail, state_tail = ssd_chunked(
        x[:, 11:], dt[:, 11:], a, b[:, 11:], c[:, 11:], chunk_size=8, initial_state=state_head
    )
    torch.testing.assert_close(torch.cat([y_head, y_tail], dim=1), y_full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_tail, state_full, rtol=1e-5, atol=1e-5)


def test_recurrent_does_not_modify_the_initial_state() -> None:
    x, dt, a, b, c = _inputs(3)
    initial = torch.randn(2, 4, 6, 5)
    before = initial.clone()
    ssd_recurrent(x, dt, a, b, c, initial_state=initial)
    ssd_chunked(x, dt, a, b, c, chunk_size=2, initial_state=initial)
    assert torch.equal(initial, before)


def test_conv_steps_match_one_prefill() -> None:
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(2, 9, 12, generator=gen)
    weight = torch.randn(12, 4, generator=gen)
    bias = torch.randn(12, generator=gen)
    y_ref, state_ref = mamba_conv_prefill(x, weight, bias)
    y_head, state = mamba_conv_prefill(x[:, :3], weight, bias)
    outs = [y_head]
    for t in range(3, 9):
        y_t, state = mamba_conv_step(x[:, t], weight, bias, state)
        outs.append(y_t.unsqueeze(1))
    torch.testing.assert_close(torch.cat(outs, dim=1), y_ref, rtol=1e-6, atol=1e-6)
    assert torch.equal(state, state_ref)


def test_gated_group_rmsnorm_normalizes_each_group() -> None:
    y = torch.randn(3, 16) * 5.0
    gate = torch.full((3, 16), 30.0)  # silu(30) == 30 to fp32 precision
    out = gated_group_rmsnorm(y, gate, torch.ones(16), group_size=4, eps=0.0)
    groups = out.view(3, 4, 4)
    torch.testing.assert_close(groups.pow(2).mean(-1), torch.ones(3, 4), rtol=1e-5, atol=1e-5)
