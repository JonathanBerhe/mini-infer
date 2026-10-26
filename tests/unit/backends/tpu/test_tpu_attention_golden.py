"""Cross-backend golden test: TPU Pallas kernels vs a PyTorch reference.

Attention is a deterministic forward, so "temperature 0" here means we compare
exact outputs (allclose) against a PyTorch reference, the same ground truth the
project's golden tests rest on. The reference is written in PyTorch (the engine's
framework) rather than NumPy, so this is a genuine torch-vs-jax cross-framework
check that the TPU path matches the CUDA/PyTorch path's math.

The actual CUDA Triton paged kernel needs a GPU, so it cannot run on this CPU
host; comparing to it directly is deferred to a GPU/TPU CI step (ADR-023). Here
the PyTorch reference stands in as the numerical ground truth, and the Pallas
kernels run in interpret mode on CPU via the `tpu` extra's plain `jax`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (after importorskip, by design)

from mini_infer.backends.tpu.dispatch import (  # noqa: E402
    dispatch_attention,
    tpu_backend_available,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _torch_dense_reference(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, causal: bool
) -> np.ndarray:
    """Dense attention in PyTorch: (heads, seq, head_dim) -> same."""
    qt = torch.from_numpy(q).to(torch.float32)
    kt = torch.from_numpy(k).to(torch.float32)
    vt = torch.from_numpy(v).to(torch.float32)
    scores = torch.matmul(qt, kt.transpose(-1, -2)) * scale  # (H, Sq, Sk)
    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        i = torch.arange(seq_q).unsqueeze(-1)
        j = torch.arange(seq_k).unsqueeze(0)
        scores = scores.masked_fill(j > i, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, vt).numpy()


def _torch_paged_reference(
    q: np.ndarray,
    k_pages: np.ndarray,
    v_pages: np.ndarray,
    block_tables: np.ndarray,
    lengths: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Paged, ragged, grouped-query decode attention in PyTorch."""
    num_seqs, num_heads, _ = q.shape
    num_kv_heads = k_pages.shape[0]
    q_per_kv = num_heads // num_kv_heads
    max_pages = block_tables.shape[1]
    out = np.zeros_like(q)
    for s in range(num_seqs):
        length = int(lengths[s])
        k_full = np.concatenate(
            [k_pages[:, block_tables[s, pi]] for pi in range(max_pages)], axis=1
        )
        v_full = np.concatenate(
            [v_pages[:, block_tables[s, pi]] for pi in range(max_pages)], axis=1
        )
        kt = torch.from_numpy(k_full).to(torch.float32)  # (KVH, T, D)
        vt = torch.from_numpy(v_full).to(torch.float32)
        total = kt.shape[1]
        valid = torch.arange(total) < length
        for h in range(num_heads):
            kv = h // q_per_kv
            qh = torch.from_numpy(q[s, h]).to(torch.float32)  # (D,)
            scores = (kt[kv] @ qh) * scale  # (T,)
            scores = scores.masked_fill(~valid, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out[s, h] = (weights @ vt[kv]).numpy()
    return out


def _torch_mixed_reference(
    q: np.ndarray,
    k_pages: np.ndarray,
    v_pages: np.ndarray,
    block_tables: np.ndarray,
    lengths: np.ndarray,
    q_lens: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Mixed prefill/decode paged attention in PyTorch; padding rows stay zero."""
    num_seqs, _max_q_len, num_heads, _ = q.shape
    num_kv_heads = k_pages.shape[0]
    q_per_kv = num_heads // num_kv_heads
    max_pages = block_tables.shape[1]
    out = np.zeros_like(q)
    for s in range(num_seqs):
        length = int(lengths[s])
        q_len_s = int(q_lens[s])
        k_full = np.concatenate(
            [k_pages[:, block_tables[s, pi]] for pi in range(max_pages)], axis=1
        )
        v_full = np.concatenate(
            [v_pages[:, block_tables[s, pi]] for pi in range(max_pages)], axis=1
        )
        kt = torch.from_numpy(k_full).to(torch.float32)  # (KVH, T, D)
        vt = torch.from_numpy(v_full).to(torch.float32)
        k_pos = torch.arange(kt.shape[1])
        for h in range(num_heads):
            kv = h // q_per_kv
            for t in range(q_len_s):
                q_pos = length - q_len_s + t
                qh = torch.from_numpy(q[s, t, h]).to(torch.float32)  # (D,)
                scores = (kt[kv] @ qh) * scale  # (T,)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                out[s, t, h] = (weights @ vt[kv]).numpy()
    return out


def _dense_inputs(num_heads=4, seq=32, head_dim=16, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((num_heads, seq, head_dim)).astype(np.float32)
    k = rng.standard_normal((num_heads, seq, head_dim)).astype(np.float32)
    v = rng.standard_normal((num_heads, seq, head_dim)).astype(np.float32)
    return q, k, v


def _paged_inputs(
    num_seqs=3, num_heads=4, num_kv_heads=4, head_dim=16, page_size=8, max_pages=4, seed=0
):
    rng = np.random.default_rng(seed)
    num_pages = num_seqs * max_pages + 2
    q = rng.standard_normal((num_seqs, num_heads, head_dim)).astype(np.float32)
    k_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    v_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    block_tables = np.zeros((num_seqs, max_pages), dtype=np.int32)
    lengths = np.zeros((num_seqs,), dtype=np.int32)
    cursor = 0
    for s in range(num_seqs):
        length = int(rng.integers(1, max_pages * page_size + 1))
        lengths[s] = length
        n_used = (length + page_size - 1) // page_size
        for pi in range(max_pages):
            block_tables[s, pi] = cursor if pi < n_used else 0
            if pi < n_used:
                cursor += 1
    return q, k_pages, v_pages, block_tables, lengths


def test_backend_available_with_jax():
    assert tpu_backend_available() is True


@pytest.mark.parametrize("causal", [False, True])
def test_dense_matches_torch_reference(causal):
    q, k, v = _dense_inputs(seed=1)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    got = np.asarray(
        dispatch_attention(
            jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), causal=causal, interpret=True
        )
    )
    ref = _torch_dense_reference(q, k, v, scale, causal)
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_paged_matches_torch_reference():
    q, k_pages, v_pages, block_tables, lengths = _paged_inputs(seed=2)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    got = np.asarray(
        dispatch_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            block_tables=jnp.asarray(block_tables),
            lengths=jnp.asarray(lengths),
            interpret=True,
        )
    )
    ref = _torch_paged_reference(q, k_pages, v_pages, block_tables, lengths, scale)
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_paged_grouped_query_matches_torch_reference():
    q, k_pages, v_pages, block_tables, lengths = _paged_inputs(num_heads=8, num_kv_heads=2, seed=4)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    got = np.asarray(
        dispatch_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            block_tables=jnp.asarray(block_tables),
            lengths=jnp.asarray(lengths),
            interpret=True,
        )
    )
    ref = _torch_paged_reference(q, k_pages, v_pages, block_tables, lengths, scale)
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_paged_mixed_matches_torch_reference():
    # Mixed prefill/decode batch (decode rows have q_lens[s] == 1) with grouped
    # queries, routed through the dispatcher against the PyTorch reference.
    rng = np.random.default_rng(10)
    q_lens = np.array([1, 5, 1, 3], dtype=np.int32)
    num_seqs, max_q_len = q_lens.shape[0], int(q_lens.max())
    num_heads, num_kv_heads, head_dim, page_size, max_pages = 8, 2, 16, 8, 4
    num_pages = num_seqs * max_pages + 2
    q = rng.standard_normal((num_seqs, max_q_len, num_heads, head_dim)).astype(np.float32)
    k_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    v_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    block_tables = np.zeros((num_seqs, max_pages), dtype=np.int32)
    lengths = np.zeros((num_seqs,), dtype=np.int32)
    cursor = 0
    for s in range(num_seqs):
        length = int(rng.integers(int(q_lens[s]), max_pages * page_size + 1))
        lengths[s] = length
        n_used = (length + page_size - 1) // page_size
        for pi in range(max_pages):
            block_tables[s, pi] = cursor if pi < n_used else 0
            if pi < n_used:
                cursor += 1
    scale = 1.0 / (head_dim**0.5)

    got = np.asarray(
        dispatch_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            block_tables=jnp.asarray(block_tables),
            lengths=jnp.asarray(lengths),
            q_lens=jnp.asarray(q_lens),
            interpret=True,
        )
    )
    ref = _torch_mixed_reference(q, k_pages, v_pages, block_tables, lengths, q_lens, scale)
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_dispatch_requires_lengths_for_paged():
    q, k_pages, v_pages, block_tables, _ = _paged_inputs(seed=6)
    with pytest.raises(ValueError, match="block_tables and lengths"):
        dispatch_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            block_tables=jnp.asarray(block_tables),
            interpret=True,
        )


def test_dispatch_rejects_lengths_without_block_tables():
    # The mirror misuse: lengths with no block table would silently route to the
    # dense kernel and attend over padded positions (no ragged masking there).
    q, k, v = _dense_inputs(seed=8)
    with pytest.raises(ValueError, match="lengths requires block_tables"):
        dispatch_attention(
            jnp.asarray(q),
            jnp.asarray(k),
            jnp.asarray(v),
            lengths=jnp.array([4, 8, 2, 1], dtype=jnp.int32),
            interpret=True,
        )
