"""KDA core math: chunkwise == recurrent, state carry, conv prefill/step parity.

These pin the internal consistency of `blocks/kda.py` BEFORE any reference
comparison: the one-token-at-a-time recurrence is the oracle, and the
chunkwise prefill form must reproduce it (same math, different evaluation
order), including across chunk boundaries, non-multiple lengths, and a
carried initial state. The reference-facing bit-parity lives in
`test_kimi_linear_parity.py`.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.blocks.kda import (
    causal_conv1d_prefill,
    causal_conv1d_step,
    kda_chunkwise,
    kda_gate,
    kda_recurrent,
    l2norm,
)


def _random_kda_inputs(
    batch: int, seq_len: int, num_heads: int, dim_k: int, dim_v: int, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = l2norm(torch.randn(batch, seq_len, num_heads, dim_k)) * dim_k**-0.5
    k = l2norm(torch.randn(batch, seq_len, num_heads, dim_k))
    v = torch.randn(batch, seq_len, num_heads, dim_v)
    # Realistic log-decays: strictly negative, moderate magnitude.
    g = -torch.rand(batch, seq_len, num_heads, dim_k) * 2.0
    beta = torch.rand(batch, seq_len, num_heads)
    return q, k, v, g, beta


@pytest.mark.parametrize("seq_len", [1, 7, 16, 37])
def test_chunkwise_matches_recurrent(seq_len: int) -> None:
    """Chunked evaluation equals the token-by-token oracle, including when
    the length is not a chunk multiple (zero-pad path) and shorter than one
    chunk."""
    q, k, v, g, beta = _random_kda_inputs(2, seq_len, 3, 8, 16)
    o_step, s_step = kda_recurrent(q, k, v, g, beta)
    o_chunk, s_chunk = kda_chunkwise(q, k, v, g, beta, chunk_size=16)
    assert torch.allclose(o_step, o_chunk, atol=1e-5), (
        f"max_abs_diff={(o_step - o_chunk).abs().max().item():.2e}"
    )
    assert torch.allclose(s_step, s_chunk, atol=1e-5)


def test_chunkwise_matches_recurrent_with_initial_state() -> None:
    q, k, v, g, beta = _random_kda_inputs(1, 20, 2, 8, 8, seed=1)
    initial = torch.randn(1, 2, 8, 8)
    o_step, s_step = kda_recurrent(q, k, v, g, beta, initial_state=initial)
    o_chunk, s_chunk = kda_chunkwise(q, k, v, g, beta, initial_state=initial, chunk_size=8)
    assert torch.allclose(o_step, o_chunk, atol=1e-5)
    assert torch.allclose(s_step, s_chunk, atol=1e-5)


def test_state_carry_across_split_is_exact_for_recurrent() -> None:
    """Running [0:t) then [t:T) with the carried state equals one full pass.

    The split path performs the identical fp ops in the identical order, so
    this is exact equality, the property chunked prefill relies on."""
    q, k, v, g, beta = _random_kda_inputs(1, 12, 2, 4, 4, seed=2)
    o_full, s_full = kda_recurrent(q, k, v, g, beta)
    split = 5
    o_a, s_a = kda_recurrent(
        q[:, :split], k[:, :split], v[:, :split], g[:, :split], beta[:, :split]
    )
    o_b, s_b = kda_recurrent(
        q[:, split:],
        k[:, split:],
        v[:, split:],
        g[:, split:],
        beta[:, split:],
        initial_state=s_a,
    )
    assert torch.equal(torch.cat([o_a, o_b], dim=1), o_full)
    assert torch.equal(s_b, s_full)


def test_chunkwise_state_feeds_recurrent_decode() -> None:
    """Prefill via chunkwise, then decode one token via the recurrence: the
    serving path's exact hand-off."""
    q, k, v, g, beta = _random_kda_inputs(1, 24, 2, 8, 8, seed=3)
    _, s_prefill = kda_chunkwise(q[:, :23], k[:, :23], v[:, :23], g[:, :23], beta[:, :23])
    o_decode, _ = kda_recurrent(
        q[:, 23:], k[:, 23:], v[:, 23:], g[:, 23:], beta[:, 23:], initial_state=s_prefill
    )
    o_full, _ = kda_recurrent(q, k, v, g, beta)
    assert torch.allclose(o_decode, o_full[:, 23:], atol=1e-5)


def test_gate_matches_formula() -> None:
    torch.manual_seed(0)
    num_heads, head_dim = 3, 4
    raw = torch.randn(2, 5, num_heads * head_dim)
    a_log = torch.randn(1, 1, num_heads, 1)  # checkpoint layout
    dt_bias = torch.randn(num_heads * head_dim)
    out = kda_gate(raw, a_log, dt_bias, head_dim)
    expected = -a_log.view(num_heads, 1).exp() * torch.nn.functional.softplus(
        raw.view(2, 5, num_heads, head_dim) + dt_bias.view(num_heads, head_dim)
    )
    assert out.dtype == torch.float32
    assert torch.allclose(out, expected, atol=1e-7)
    assert (out <= 0).all(), "log-decays must be non-positive"


def test_conv_prefill_matches_steps() -> None:
    """Full-sequence conv equals stepping token by token from the rolling state,
    both for a fresh sequence and when continuing from a carried state."""
    torch.manual_seed(0)
    batch, seq_len, channels, kernel = 2, 9, 6, 4
    x = torch.randn(batch, seq_len, channels)
    weight = torch.randn(channels, kernel)

    y_full, state_full = causal_conv1d_prefill(x, weight)
    state = torch.zeros(batch, channels, kernel)
    stepped = torch.empty_like(y_full)
    for t in range(seq_len):
        stepped[:, t], state = causal_conv1d_step(x[:, t], weight, state)
    assert torch.allclose(y_full, stepped, atol=1e-6)
    assert torch.allclose(state_full, state, atol=1e-6)

    # Continuation: prefill [0:4), then prefill [4:9) from the carried state.
    y_a, state_a = causal_conv1d_prefill(x[:, :4], weight)
    y_b, state_b = causal_conv1d_prefill(x[:, 4:], weight, conv_state=state_a)
    assert torch.allclose(torch.cat([y_a, y_b], dim=1), y_full, atol=1e-6)
    assert torch.allclose(state_b, state_full, atol=1e-6)


def test_conv_prefill_shorter_than_kernel() -> None:
    """A 2-token prompt with kernel 4: the state keeps zero left-padding."""
    torch.manual_seed(1)
    x = torch.randn(1, 2, 3)
    weight = torch.randn(3, 4)
    _y, state = causal_conv1d_prefill(x, weight)
    assert state.shape == (1, 3, 4)
    assert torch.equal(state[:, :, :2], torch.zeros(1, 3, 2))
    assert torch.allclose(state[:, :, 2:], x.transpose(1, 2), atol=1e-7)
    # Third token continues correctly.
    x_next = torch.randn(1, 3)
    y_next, _ = causal_conv1d_step(x_next, weight, state)
    y_ref, _ = causal_conv1d_prefill(torch.cat([x, x_next.unsqueeze(1)], dim=1), weight)
    assert torch.allclose(y_next, y_ref[:, -1], atol=1e-6)
