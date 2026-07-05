"""Parity tests for the Pallas paged, ragged decode attention kernel (M2).

Runs the kernel in Pallas interpret mode on the host CPU (no TPU needed) and
checks it against a plain NumPy reference. The discriminating test is
`test_paging_indirection`: the physical pages are shuffled so a kernel that
ignored the block table and read the pool contiguously would fail it.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (after importorskip, by design)

from mini_infer.backends.tpu.pallas_paged_attention import (  # noqa: E402
    pallas_paged_attention,
    supports_pallas_paged_attention,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _reference(
    q: np.ndarray,
    k_pages: np.ndarray,
    v_pages: np.ndarray,
    block_tables: np.ndarray,
    lengths: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Materialise each sequence's KV from its pages and do masked attention."""
    num_seqs, num_heads = q.shape[:2]
    page_size = k_pages.shape[1]
    num_kv_heads = k_pages.shape[2]
    q_per_kv = num_heads // num_kv_heads
    max_pages = block_tables.shape[1]
    out = np.zeros_like(q)
    for s in range(num_seqs):
        length = int(lengths[s])
        k_full = np.concatenate(
            [k_pages[block_tables[s, pi]] for pi in range(max_pages)], axis=0
        )  # (max_pages * page_size, num_heads, head_dim)
        v_full = np.concatenate([v_pages[block_tables[s, pi]] for pi in range(max_pages)], axis=0)
        valid = np.arange(max_pages * page_size) < length
        for h in range(num_heads):
            kv = h // q_per_kv  # grouped-query: query head h reads kv head kv
            scores = (k_full[:, kv, :] @ q[s, h]) * scale
            scores = np.where(valid, scores, -1e30)
            scores = scores - scores.max()
            weights = np.exp(scores)
            weights = weights / weights.sum()
            out[s, h] = weights @ v_full[:, kv, :]
    return out


def _make_case(
    num_seqs: int = 3,
    num_heads: int = 2,
    head_dim: int = 16,
    page_size: int = 8,
    max_pages: int = 4,
    seed: int = 0,
    shuffle_pages: bool = False,
    num_kv_heads: int | None = None,
):
    """Build a ragged paged-attention case with distinct pages per sequence."""
    if num_kv_heads is None:
        num_kv_heads = num_heads
    rng = np.random.default_rng(seed)
    # Enough pages for every sequence to use up to max_pages distinct ones.
    num_pages = num_seqs * max_pages + 2
    q = rng.standard_normal((num_seqs, num_heads, head_dim)).astype(np.float32)
    k_pages = rng.standard_normal((num_pages, page_size, num_kv_heads, head_dim)).astype(np.float32)
    v_pages = rng.standard_normal((num_pages, page_size, num_kv_heads, head_dim)).astype(np.float32)

    physical = list(range(num_pages))
    if shuffle_pages:
        rng.shuffle(physical)  # break any contiguous / identity assumption

    block_tables = np.zeros((num_seqs, max_pages), dtype=np.int32)
    lengths = np.zeros((num_seqs,), dtype=np.int32)
    cursor = 0
    for s in range(num_seqs):
        length = int(rng.integers(1, max_pages * page_size + 1))
        lengths[s] = length
        n_used = (length + page_size - 1) // page_size
        for pi in range(max_pages):
            if pi < n_used:
                block_tables[s, pi] = physical[cursor]
                cursor += 1
            else:
                # Slot past the sequence's real pages: any valid, fully-masked page.
                block_tables[s, pi] = physical[-1]
    return q, k_pages, v_pages, block_tables, lengths


def _run(case, scale=None):
    q, k_pages, v_pages, block_tables, lengths = case
    eff_scale = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    got = pallas_paged_attention(
        jnp.asarray(q),
        jnp.asarray(k_pages),
        jnp.asarray(v_pages),
        jnp.asarray(block_tables),
        jnp.asarray(lengths),
        scale=scale,
        interpret=True,
    )
    ref = _reference(q, k_pages, v_pages, block_tables, lengths, eff_scale)
    return np.asarray(got), ref


def test_supports_predicate_true_when_jax_present():
    assert supports_pallas_paged_attention() is True


def test_matches_reference():
    got, ref = _run(_make_case(seed=1))
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_ragged_lengths_per_sequence():
    # Distinct, deliberately uneven lengths across sequences in one batch.
    q, k_pages, v_pages, block_tables, lengths = _make_case(num_seqs=4, seed=7)
    lengths = np.array([1, 8, 17, 32], dtype=np.int32)  # partial, full, spill, max
    for s in range(4):
        n_used = (int(lengths[s]) + 7) // 8
        # keep only n_used real pages; rest point at a valid masked page
        for pi in range(block_tables.shape[1]):
            if pi >= n_used:
                block_tables[s, pi] = block_tables[s, 0]
    case = (q, k_pages, v_pages, block_tables, lengths)
    got, ref = _run(case)
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_paging_indirection():
    # Physical pages are shuffled: only a kernel that honors block_tables (a real
    # gather) can match the reference. A contiguous-read kernel would fail here.
    got, ref = _run(_make_case(seed=3, shuffle_pages=True))
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_trailing_pages_are_masked_out():
    # A single-token context: only position 0 is valid. Every later position,
    # including whole trailing pages pointing at arbitrary valid pages, must be
    # ignored, so the output equals attending to key row 0 alone.
    num_heads, head_dim, page_size = 2, 16, 8
    rng = np.random.default_rng(11)
    num_pages = 6
    q = rng.standard_normal((1, num_heads, head_dim)).astype(np.float32)
    k_pages = rng.standard_normal((num_pages, page_size, num_heads, head_dim)).astype(np.float32)
    v_pages = rng.standard_normal((num_pages, page_size, num_heads, head_dim)).astype(np.float32)
    block_tables = np.array([[0, 3, 5, 2]], dtype=np.int32)  # arbitrary trailing pages
    lengths = np.array([1], dtype=np.int32)
    got, _ = _run((q, k_pages, v_pages, block_tables, lengths))
    # length 1 -> output is exactly value row 0 of the first page, per head.
    expected = v_pages[0, 0, :, :]  # (num_heads, head_dim)
    np.testing.assert_allclose(got[0], expected, rtol=1e-4, atol=1e-4)


def test_stable_on_large_logits():
    # Large scale blows up raw scores; the online softmax must stay finite.
    got, ref = _run(_make_case(seed=5), scale=1000.0)
    assert np.all(np.isfinite(got))
    assert _cosine(got, ref) > 0.99


def test_grouped_query_attention():
    # 8 query heads share 2 kv heads (4:1 grouping).
    got, ref = _run(_make_case(num_heads=8, num_kv_heads=2, seed=9))
    assert _cosine(got, ref) > 0.99
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_rejects_indivisible_head_grouping():
    # num_heads must be a multiple of num_kv_heads; 3 is not a multiple of 2.
    q = jnp.zeros((2, 3, 16), dtype=jnp.float32)  # 3 query heads
    k_pages = jnp.zeros((4, 8, 2, 16), dtype=jnp.float32)  # 2 kv heads
    v_pages = jnp.zeros((4, 8, 2, 16), dtype=jnp.float32)
    block_tables = jnp.zeros((2, 3), dtype=jnp.int32)
    lengths = jnp.ones((2,), dtype=jnp.int32)
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        pallas_paged_attention(q, k_pages, v_pages, block_tables, lengths, interpret=True)
