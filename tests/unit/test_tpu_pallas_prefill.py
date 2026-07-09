"""Parity tests for the Pallas paged prefill / multi-query attention kernel.

Runs in Pallas interpret mode on CPU (no TPU). Beyond matching a NumPy reference,
two tests pin the design: `test_qlen1_matches_decode_kernel` checks the prefill
kernel reduces to the decode kernel when q_len == 1, and
`test_causal_masks_future` checks an earlier query cannot see a later query's key.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (after importorskip, by design)

from mini_infer.backends.tpu.dispatch import dispatch_attention  # noqa: E402
from mini_infer.backends.tpu.pallas_paged_attention import (  # noqa: E402
    pallas_paged_attention,
    pallas_paged_prefill_attention,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _prefill_reference(q, k_pages, v_pages, block_tables, lengths, scale):
    """Paged, ragged, grouped-query, causal multi-query attention in NumPy."""
    num_seqs, q_len, num_heads, _ = q.shape
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
        k_ids = np.arange(k_full.shape[1])
        for h in range(num_heads):
            kv = h // q_per_kv
            for t in range(q_len):
                q_pos = length - q_len + t
                scores = (k_full[kv] @ q[s, t, h]) * scale
                scores = np.where(k_ids <= q_pos, scores, -1e30)
                scores = scores - scores.max()
                weights = np.exp(scores)
                weights /= weights.sum()
                out[s, t, h] = weights @ v_full[kv]
    return out


def _make_prefill(
    num_seqs=3, q_len=4, num_heads=4, num_kv_heads=4, head_dim=16, page_size=8, max_pages=4, seed=0
):
    rng = np.random.default_rng(seed)
    num_pages = num_seqs * max_pages + 2
    q = rng.standard_normal((num_seqs, q_len, num_heads, head_dim)).astype(np.float32)
    k_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    v_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    block_tables = np.zeros((num_seqs, max_pages), dtype=np.int32)
    lengths = np.zeros((num_seqs,), dtype=np.int32)
    cursor = 0
    for s in range(num_seqs):
        # length >= q_len so the query tokens fit in the context.
        length = int(rng.integers(q_len, max_pages * page_size + 1))
        lengths[s] = length
        n_used = (length + page_size - 1) // page_size
        for pi in range(max_pages):
            block_tables[s, pi] = cursor if pi < n_used else 0
            if pi < n_used:
                cursor += 1
    return q, k_pages, v_pages, block_tables, lengths


def _run_prefill(case, scale=None):
    q, k_pages, v_pages, block_tables, lengths = case
    eff = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    got = pallas_paged_prefill_attention(
        jnp.asarray(q),
        jnp.asarray(k_pages),
        jnp.asarray(v_pages),
        jnp.asarray(block_tables),
        jnp.asarray(lengths),
        scale=scale,
        interpret=True,
    )
    ref = _prefill_reference(q, k_pages, v_pages, block_tables, lengths, eff)
    return np.asarray(got), ref


def test_prefill_matches_reference():
    got, ref = _run_prefill(_make_prefill(q_len=4, seed=1))
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_prefill_grouped_query_matches_reference():
    got, ref = _run_prefill(_make_prefill(q_len=5, num_heads=8, num_kv_heads=2, seed=2))
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_qlen1_matches_decode_kernel():
    # With q_len == 1 the prefill kernel must equal the decode kernel exactly.
    q, k_pages, v_pages, block_tables, lengths = _make_prefill(q_len=1, seed=3)
    prefill = np.asarray(
        pallas_paged_prefill_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            interpret=True,
        )
    )  # (S, 1, H, D)
    decode = np.asarray(
        pallas_paged_attention(
            jnp.asarray(q[:, 0]),  # (S, H, D)
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            interpret=True,
        )
    )  # (S, H, D)
    np.testing.assert_allclose(prefill[:, 0], decode, rtol=1e-5, atol=1e-5)


def test_causal_masks_future():
    # q_len=2: query 0 sits at position length-2, query 1 at length-1. Perturbing
    # the value at position length-1 must change query 1's output but not query 0's.
    q, k_pages, v_pages, block_tables, lengths = _make_prefill(
        num_seqs=1, q_len=2, num_heads=2, num_kv_heads=2, page_size=8, max_pages=4, seed=8
    )
    base, _ = _run_prefill((q, k_pages, v_pages, block_tables, lengths))

    length = int(lengths[0])
    pos = length - 1  # query 1's own position (the future position for query 0)
    page_size = v_pages.shape[2]
    phys = int(block_tables[0, pos // page_size])
    offset = pos % page_size
    v_perturbed = v_pages.copy()
    v_perturbed[:, phys, offset, :] += 1000.0  # loud change at position length-1
    perturbed, _ = _run_prefill((q, k_pages, v_perturbed, block_tables, lengths))

    # Query 0 (position length-2) must be blind to position length-1.
    np.testing.assert_allclose(perturbed[0, 0], base[0, 0], rtol=1e-5, atol=1e-5)
    # Query 1 (position length-1) attends to it, so it must change.
    assert not np.allclose(perturbed[0, 1], base[0, 1])


def test_dispatch_routes_rank4_to_prefill():
    # dispatch_attention with a rank-4 q and a block table must select the
    # prefill kernel and match a direct call to it.
    q, k_pages, v_pages, block_tables, lengths = _make_prefill(q_len=3, seed=5)
    direct = np.asarray(
        pallas_paged_prefill_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            interpret=True,
        )
    )
    routed = np.asarray(
        dispatch_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            block_tables=jnp.asarray(block_tables),
            lengths=jnp.asarray(lengths),
            interpret=True,
        )
    )
    np.testing.assert_allclose(routed, direct, rtol=1e-6, atol=1e-6)
