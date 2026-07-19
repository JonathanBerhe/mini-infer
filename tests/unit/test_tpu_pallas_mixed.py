"""Parity tests for the Pallas mixed prefill/decode paged attention kernel.

Runs in Pallas interpret mode on CPU (no TPU). The kernel's promise is that ONE
launch serves a batch where some sequences are decoding (q_lens[s] == 1) and
some are prefilling (q_lens[s] > 1). Two tests pin that promise directly:
`test_mixed_rows_match_specialized_kernels` checks every sequence's real rows
against the already-validated decode/prefill kernels, and
`test_mixed_padding_rows_are_zero` checks the padding contract (rows at or past
q_lens[s] are exact zeros, not leftover accumulator noise; see the p-zeroing
comment in the kernel for why that is a real failure mode, not a given).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (after importorskip, by design)

from mini_infer.backends.tpu.dispatch import dispatch_attention  # noqa: E402
from mini_infer.backends.tpu.pallas_paged_attention import (  # noqa: E402
    pallas_paged_attention,
    pallas_paged_mixed_attention,
    pallas_paged_prefill_attention,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _mixed_reference(q, k_pages, v_pages, block_tables, lengths, q_lens, scale):
    """Mixed-batch paged causal attention in NumPy; padding rows stay zero."""
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
        k_ids = np.arange(k_full.shape[1])
        for h in range(num_heads):
            kv = h // q_per_kv
            for t in range(q_len_s):
                q_pos = length - q_len_s + t
                scores = (k_full[kv] @ q[s, t, h]) * scale
                scores = np.where(k_ids <= q_pos, scores, -1e30)
                scores = scores - scores.max()
                weights = np.exp(scores)
                weights /= weights.sum()
                out[s, t, h] = weights @ v_full[kv]
    return out


def _make_mixed(
    q_lens,
    num_heads=4,
    num_kv_heads=4,
    head_dim=16,
    page_size=8,
    max_pages=4,
    max_q_len=None,
    seed=0,
):
    """Build a padded mixed batch.

    Physical pages are assigned through a random permutation, so a kernel that
    ignored the block table and read pages sequentially would fail parity (the
    same trap the decode kernel's shuffled-pages test sets).
    """
    rng = np.random.default_rng(seed)
    q_lens = np.asarray(q_lens, dtype=np.int32)
    num_seqs = q_lens.shape[0]
    if max_q_len is None:
        max_q_len = max(int(q_lens.max()), 1)
    num_pages = num_seqs * max_pages + 2
    perm = rng.permutation(num_pages)
    q = rng.standard_normal((num_seqs, max_q_len, num_heads, head_dim)).astype(np.float32)
    k_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    v_pages = rng.standard_normal((num_kv_heads, num_pages, page_size, head_dim)).astype(np.float32)
    block_tables = np.zeros((num_seqs, max_pages), dtype=np.int32)
    lengths = np.zeros((num_seqs,), dtype=np.int32)
    cursor = 0
    for s in range(num_seqs):
        # length >= q_lens[s] so the query tokens fit in the context (with a
        # floor of 1 so inactive q_lens == 0 slots still carry a valid table).
        length = int(rng.integers(max(int(q_lens[s]), 1), max_pages * page_size + 1))
        lengths[s] = length
        n_used = (length + page_size - 1) // page_size
        for pi in range(max_pages):
            block_tables[s, pi] = perm[cursor] if pi < n_used else int(perm[0])
            if pi < n_used:
                cursor += 1
    return q, k_pages, v_pages, block_tables, lengths, q_lens


def _run_mixed(case, scale=None):
    q, k_pages, v_pages, block_tables, lengths, q_lens = case
    eff = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    got = pallas_paged_mixed_attention(
        jnp.asarray(q),
        jnp.asarray(k_pages),
        jnp.asarray(v_pages),
        jnp.asarray(block_tables),
        jnp.asarray(lengths),
        jnp.asarray(q_lens),
        scale=scale,
        interpret=True,
    )
    ref = _mixed_reference(q, k_pages, v_pages, block_tables, lengths, q_lens, eff)
    return np.asarray(got), ref


def test_mixed_batch_matches_reference():
    # Two decode rows and two prefill chunks in one launch.
    got, ref = _run_mixed(_make_mixed([1, 5, 1, 3], seed=1))
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_mixed_grouped_query_matches_reference():
    got, ref = _run_mixed(_make_mixed([2, 1, 6, 1], num_heads=8, num_kv_heads=2, seed=2))
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_mixed_rows_match_specialized_kernels():
    # The mixed kernel must agree with the specialized kernels on their own
    # turf: each decode row against the decode kernel, each prefill chunk
    # against the prefill kernel, per sequence.
    q, k_pages, v_pages, block_tables, lengths, q_lens = _make_mixed([1, 5, 1, 3], seed=3)
    mixed = np.asarray(
        pallas_paged_mixed_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            jnp.asarray(q_lens),
            interpret=True,
        )
    )
    for s in range(q_lens.shape[0]):
        q_len_s = int(q_lens[s])
        if q_len_s == 1:
            expected = np.asarray(
                pallas_paged_attention(
                    jnp.asarray(q[s : s + 1, 0]),  # (1, H, D)
                    jnp.asarray(k_pages),
                    jnp.asarray(v_pages),
                    jnp.asarray(block_tables[s : s + 1]),
                    jnp.asarray(lengths[s : s + 1]),
                    interpret=True,
                )
            )
            np.testing.assert_allclose(mixed[s, 0], expected[0], rtol=1e-5, atol=1e-5)
        else:
            expected = np.asarray(
                pallas_paged_prefill_attention(
                    jnp.asarray(q[s : s + 1, :q_len_s]),  # (1, q_len, H, D)
                    jnp.asarray(k_pages),
                    jnp.asarray(v_pages),
                    jnp.asarray(block_tables[s : s + 1]),
                    jnp.asarray(lengths[s : s + 1]),
                    interpret=True,
                )
            )
            np.testing.assert_allclose(mixed[s, :q_len_s], expected[0], rtol=1e-5, atol=1e-5)


def test_mixed_padding_rows_are_zero():
    # Padding rows must be EXACT zeros. This is the sentinel trap: with every
    # score masked, exp(score - running_max) is exp(0) == 1, so a kernel that
    # skipped the explicit p-zeroing would return a finite, plausible-looking
    # average of unrelated values here rather than zeros.
    q, k_pages, v_pages, block_tables, lengths, q_lens = _make_mixed([1, 5, 1, 3], seed=4)
    got = np.asarray(
        pallas_paged_mixed_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            jnp.asarray(q_lens),
            interpret=True,
        )
    )
    for s in range(q_lens.shape[0]):
        assert np.all(got[s, int(q_lens[s]) :] == 0.0), f"padding rows of seq {s} are not zero"
    # Sanity: the real rows are not zeros (the kernel did compute something).
    assert np.any(got[1, : int(q_lens[1])] != 0.0)


def test_mixed_zero_qlen_slot_is_inert():
    # q_lens[s] == 0 marks an inactive batch slot (continuous batching leaves
    # holes): all its rows are zeros and its neighbours still match reference.
    got, ref = _run_mixed(_make_mixed([3, 0, 2], seed=5))
    assert np.all(got[1] == 0.0)
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_mixed_all_decode_matches_decode_kernel():
    # A pure-decode batch through the mixed kernel equals the decode kernel.
    q, k_pages, v_pages, block_tables, lengths, q_lens = _make_mixed([1, 1, 1, 1], seed=6)
    mixed = np.asarray(
        pallas_paged_mixed_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            jnp.asarray(q_lens),
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
    )
    np.testing.assert_allclose(mixed[:, 0], decode, rtol=1e-5, atol=1e-5)


def test_dispatch_routes_mixed_batch():
    # dispatch_attention with rank-4 q, a block table, and q_lens must select
    # the mixed kernel and match a direct call to it.
    q, k_pages, v_pages, block_tables, lengths, q_lens = _make_mixed([1, 4, 2, 1], seed=7)
    direct = np.asarray(
        pallas_paged_mixed_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            jnp.asarray(q_lens),
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
            q_lens=jnp.asarray(q_lens),
            interpret=True,
        )
    )
    np.testing.assert_allclose(routed, direct, rtol=1e-6, atol=1e-6)


def test_dispatch_rejects_q_lens_without_block_tables():
    # q_lens with no block table would silently route to the dense kernel and
    # attend over padded query rows as if they were real tokens.
    q = jnp.ones((2, 8, 16), dtype=jnp.float32)
    k = jnp.ones((2, 8, 16), dtype=jnp.float32)
    v = jnp.ones((2, 8, 16), dtype=jnp.float32)
    with pytest.raises(ValueError, match="q_lens requires block_tables"):
        dispatch_attention(q, k, v, q_lens=jnp.array([1, 4], dtype=jnp.int32), interpret=True)


def test_dispatch_rejects_q_lens_with_decode_rank_q():
    # Rank-3 q is the single-token decode shape; a mixed batch must come in as
    # rank-4 padded q with decode rows expressed as q_lens[s] == 1.
    q, k_pages, v_pages, block_tables, lengths, q_lens = _make_mixed([1, 1], seed=8)
    with pytest.raises(ValueError, match="rank-4"):
        dispatch_attention(
            jnp.asarray(q[:, 0]),  # (S, H, D): decode shape
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            block_tables=jnp.asarray(block_tables),
            lengths=jnp.asarray(lengths),
            q_lens=jnp.asarray(q_lens),
            interpret=True,
        )


def test_mixed_rejects_bad_q_lens_shape():
    q, k_pages, v_pages, block_tables, lengths, _q_lens = _make_mixed([1, 3], seed=9)
    with pytest.raises(ValueError, match="q_lens must be"):
        pallas_paged_mixed_attention(
            jnp.asarray(q),
            jnp.asarray(k_pages),
            jnp.asarray(v_pages),
            jnp.asarray(block_tables),
            jnp.asarray(lengths),
            jnp.ones((3,), dtype=jnp.int32),  # wrong num_seqs
            interpret=True,
        )
