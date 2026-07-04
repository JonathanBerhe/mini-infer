"""The optional block-sparse additive mask on packed_attention_torch (M3 MSA).

The mask is a per-request `(q_len, 1, k_len)` additive bias that folds block
selection AND causality together, so it REPLACES the built-in causal fill.
Checks, on the pure function (no cache plumbing): None matches manual causal
SDPA; an all-causal bias reproduces the None path exactly (replace, not augment);
and a bias that drops a causally-valid key matches manual masked SDPA. Varlen
batch mixes a prefill request (q_len==k_len) and a decode request (q_len=1).
"""

from __future__ import annotations

import torch

from mini_infer.cache.packed_attention import packed_attention_torch

_MIN = torch.finfo(torch.float32).min


def _causal_ref(q_b, k_b, v_b, scale, group_size, extra_mask=None):
    """Per-request causal SDPA (fp32), optional additional boolean drop mask."""
    if group_size > 1:
        k_b = k_b.repeat_interleave(group_size, dim=1)
        v_b = v_b.repeat_interleave(group_size, dim=1)
    q_len, k_len = q_b.shape[0], k_b.shape[0]
    scores = torch.einsum("qhd,khd->qhk", q_b.float(), k_b.float()) * scale
    q_abs = torch.arange(q_len) + (k_len - q_len)
    k_pos = torch.arange(k_len)
    invalid = k_pos[None, :] > q_abs[:, None]
    if extra_mask is not None:
        invalid = invalid | extra_mask
    scores = scores.masked_fill(invalid[:, None, :], -float("inf"))
    return torch.einsum("qhk,khd->qhd", torch.softmax(scores, dim=-1), v_b.float()).to(q_b.dtype)


def _setup():
    torch.manual_seed(0)
    nqh, nkvh, d = 4, 2, 8  # GQA group_size 2
    cu_q = torch.tensor([0, 3, 4])  # req0 prefill q_len=3, req1 decode q_len=1
    cu_k = torch.tensor([0, 3, 8])  # req0 k_len=3, req1 k_len=5
    q = torch.randn(4, nqh, d)
    k = torch.randn(8, nkvh, d)
    v = torch.randn(8, nkvh, d)
    return q, k, v, cu_q, cu_k, nqh // nkvh, d


def test_block_mask_none_matches_manual_causal() -> None:
    q, k, v, cu_q, cu_k, gs, d = _setup()
    scale = d**-0.5
    out = packed_attention_torch(q, k, v, cu_q, cu_k, scale)
    ref0 = _causal_ref(q[0:3], k[0:3], v[0:3], scale, gs)
    ref1 = _causal_ref(q[3:4], k[3:8], v[3:8], scale, gs)
    assert torch.allclose(out[0:3], ref0, atol=1e-6)
    assert torch.allclose(out[3:4], ref1, atol=1e-6)


def _causal_bias(q_len: int, k_len: int) -> torch.Tensor:
    """(q_len, 1, k_len) additive bias equal to the causal mask (0 / finfo.min)."""
    q_abs = torch.arange(q_len) + (k_len - q_len)
    k_pos = torch.arange(k_len)
    invalid = k_pos[None, :] > q_abs[:, None]  # (q_len, k_len)
    bias = torch.zeros(q_len, k_len)
    bias = bias.masked_fill(invalid, _MIN)
    return bias.unsqueeze(1)


def test_all_causal_block_mask_replaces_causal_identically() -> None:
    """A bias that encodes exactly the causal mask reproduces the None path."""
    q, k, v, cu_q, cu_k, _, d = _setup()
    scale = d**-0.5
    baseline = packed_attention_torch(q, k, v, cu_q, cu_k, scale)
    block_mask = [_causal_bias(3, 3), _causal_bias(1, 5)]
    masked = packed_attention_torch(q, k, v, cu_q, cu_k, scale, block_mask=block_mask)
    assert torch.allclose(masked, baseline, atol=1e-6)


def test_block_sparse_mask_drops_a_key() -> None:
    """Masking one causally-valid key matches manual SDPA with that key dropped."""
    q, k, v, cu_q, cu_k, gs, d = _setup()
    scale = d**-0.5
    # req1 (decode, k_len=5): drop key 0 for its single query (position 4).
    bias1 = _causal_bias(1, 5)
    bias1[0, 0, 0] = _MIN
    block_mask = [_causal_bias(3, 3), bias1]
    out = packed_attention_torch(q, k, v, cu_q, cu_k, scale, block_mask=block_mask)

    extra = torch.zeros(1, 5, dtype=torch.bool)
    extra[0, 0] = True  # drop key 0
    ref1 = _causal_ref(q[3:4], k[3:8], v[3:8], scale, gs, extra_mask=extra)
    assert torch.allclose(out[3:4], ref1, atol=1e-6)


def test_block_mask_wrong_length_raises() -> None:
    q, k, v, cu_q, cu_k, _, d = _setup()
    import pytest

    with pytest.raises(ValueError, match="block_mask has"):
        packed_attention_torch(q, k, v, cu_q, cu_k, d**-0.5, block_mask=[_causal_bias(3, 3)])


def test_block_mask_with_window_raises() -> None:
    """The bias replaces the causal fill, so a sliding window would be silently
    dropped; the combination must refuse instead of mis-masking."""
    q, k, v, cu_q, cu_k, _, d = _setup()
    import pytest

    block_mask = [_causal_bias(3, 3), _causal_bias(1, 5)]
    with pytest.raises(ValueError, match="mutually exclusive"):
        packed_attention_torch(q, k, v, cu_q, cu_k, d**-0.5, window=4, block_mask=block_mask)
