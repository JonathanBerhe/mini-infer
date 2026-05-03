"""Sliding-window attention mask correctness for `packed_attention_torch`.

Locks in the PyTorch reference path's window mask: for window=W and a
query at absolute position q, only keys at [q-W+1, q] should
contribute. Tests are CPU-only, no model load.
"""

import math

import torch

from mini_infer.cache.packed_attention import packed_attention_torch


def _single_request_inputs(seq_len: int, num_heads: int = 2, head_dim: int = 8):
    torch.manual_seed(0)
    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale


def test_window_excludes_old_positions() -> None:
    """With window=4, the last query attends only to the most recent 4 keys.

    Build V so positions 0..(seq_len-window-1) carry value +1 and the last
    `window` positions carry value -1. With window=4 the last query sees
    only -1 values; so the output equals -1 within softmax noise. Without
    a window the output mixes both halves and lands somewhere in (-1, +1).
    """
    seq_len = 16
    window = 4
    num_heads, head_dim = 2, 8
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Constant Q + K → every QK score is identical so the softmax is uniform
    # over whichever positions are valid. Then the output is the simple
    # average of the V values at those positions.
    q = torch.ones(seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.ones(seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.ones(seq_len, num_heads, head_dim, dtype=torch.float32)
    v[seq_len - window :] = -1.0  # last `window` positions carry -1.

    out_full = packed_attention_torch(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    out_window = packed_attention_torch(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, window=window
    )

    last_full = out_full[-1].mean().item()
    last_window = out_window[-1].mean().item()
    # Under full attention, last query averages 12 +1s and 4 -1s → 0.5.
    expected_full = (seq_len - 2 * window) / seq_len
    assert abs(last_full - expected_full) < 1e-4, (
        f"full last-position should equal {expected_full}, got {last_full}"
    )
    # Under window=4, last query sees only the four -1 V positions → -1.
    assert abs(last_window - (-1.0)) < 1e-4, (
        f"windowed last-position should equal -1, got {last_window}"
    )


def test_window_at_or_beyond_seqlen_matches_full() -> None:
    """`window >= seq_len` is a no-op — output should equal the full-attention path."""
    q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale = _single_request_inputs(seq_len=8)

    out_full = packed_attention_torch(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    out_window_eq = packed_attention_torch(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, window=8
    )
    out_window_big = packed_attention_torch(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, window=64
    )

    assert torch.allclose(out_full, out_window_eq, atol=1e-6)
    assert torch.allclose(out_full, out_window_big, atol=1e-6)


def test_window_one_collapses_to_self_only() -> None:
    """`window=1` means each query attends only to its own position."""
    q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale = _single_request_inputs(seq_len=6)

    out_window = packed_attention_torch(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, window=1
    )

    # With window=1 and causal masking, query i attends only to key i.
    # Therefore softmax has weight 1.0 on position i and the output is just v[i].
    assert torch.allclose(out_window, v, atol=1e-5)
