"""Parity tests for the Pallas TPU dense attention forward (M1 kernel).

The kernel runs in Pallas *interpret* mode (`interpret=True`), which executes
on the host CPU with no TPU hardware, so these tests run anywhere JAX is
installed. `pytest.importorskip("jax")` skips the whole module cleanly when the
optional `tpu` extra is absent (plain M1 / CI), matching pallas_softmax's tests
and how the CUDA-kernel tests gate on their optional deps.

Parity is checked against a plain-JAX softmax-attention reference with two
independent assertions, the same bar ADR-023 sets for every hand-written
kernel:

- cosine similarity > 0.99 (the project-wide kernel parity threshold), and
- elementwise `allclose` (the flash recurrence is algebraically exact in fp32,
  so we can demand more than cosine alone).

Coverage: non-causal parity, causal parity plus a direct check that the mask
zeroes future positions, numerical stability on large logits, and the
shape/divisibility guards.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (must follow the importorskip guard)

from mini_infer.backends.tpu.pallas_attention import (  # noqa: E402
    pallas_attention,
    supports_pallas_attention,
)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Flattened cosine similarity, the project-wide kernel parity metric."""
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf)))


def _reference_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    scale: float,
    causal: bool,
) -> jax.Array:
    """Plain-JAX softmax attention over a 3D `(heads, seq, dim)` layout.

    Materialises the full score matrix on purpose: this is the simple oracle
    the streaming kernel must match, not a second flash implementation.
    """
    scores = jnp.einsum("hqd,hkd->hqk", q, k) * scale
    if causal:
        seq_q, seq_k = scores.shape[1], scores.shape[2]
        q_pos = jnp.arange(seq_q)[:, None]
        k_pos = jnp.arange(seq_k)[None, :]
        scores = jnp.where(k_pos <= q_pos, scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hqk,hkd->hqd", probs, v)


def test_supports_predicate_true_when_jax_present() -> None:
    """With JAX importable, the dispatch predicate must report supported.

    (This module was import-skipped if JAX were absent, so here it is present.)
    """
    assert supports_pallas_attention() is True


@pytest.mark.parametrize(
    ("num_heads", "seq_q", "seq_k", "head_dim", "block_q", "block_k"),
    [
        (2, 128, 128, 64, 128, 128),  # single query/kv block per head
        (1, 256, 256, 64, 128, 128),  # two query blocks, two kv blocks
        (4, 128, 256, 32, 64, 64),  # seq_q != seq_k, smaller blocks
    ],
)
def test_pallas_attention_matches_reference_non_causal(
    num_heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    block_q: int,
    block_k: int,
) -> None:
    """Non-causal kernel output matches the plain-JAX reference within parity."""
    key = jax.random.PRNGKey(seq_q * 100 + seq_k + head_dim)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (num_heads, seq_q, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (num_heads, seq_k, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (num_heads, seq_k, head_dim), dtype=jnp.float32)
    scale = 1.0 / (head_dim**0.5)

    got = np.asarray(
        pallas_attention(q, k, v, causal=False, block_q=block_q, block_k=block_k, interpret=True)
    )
    ref = np.asarray(_reference_attention(q, k, v, scale=scale, causal=False))

    cos = _cosine_sim(got, ref)
    assert cos > 0.99, f"cosine {cos} below 0.99 parity bar for {(num_heads, seq_q, seq_k)}"
    assert np.allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("num_heads", "seq", "head_dim", "block_q", "block_k"),
    [
        (2, 128, 64, 128, 128),  # single block
        (1, 256, 64, 128, 128),  # multiple blocks, exercises the diagonal block
    ],
)
def test_pallas_attention_matches_reference_causal(
    num_heads: int, seq: int, head_dim: int, block_q: int, block_k: int
) -> None:
    """Causal kernel output matches the causal reference within parity."""
    key = jax.random.PRNGKey(seq * 7 + head_dim)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (num_heads, seq, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (num_heads, seq, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (num_heads, seq, head_dim), dtype=jnp.float32)
    scale = 1.0 / (head_dim**0.5)

    got = np.asarray(
        pallas_attention(q, k, v, causal=True, block_q=block_q, block_k=block_k, interpret=True)
    )
    ref = np.asarray(_reference_attention(q, k, v, scale=scale, causal=True))

    cos = _cosine_sim(got, ref)
    assert cos > 0.99, f"cosine {cos} below 0.99 parity bar for causal seq={seq}"
    assert np.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_pallas_attention_causal_mask_zeroes_future_positions() -> None:
    """Directly verify the causal mask hides future keys.

    Construction: the first query row attends only to key 0. We make value row 0
    a known unit vector and all other value rows huge and distinct. If the mask
    works, row 0 of the output equals value row 0 exactly; if future positions
    leaked in, the huge later values would dominate and the output would be far
    from value row 0. This isolates the mask from the softmax numerics.
    """
    seq, head_dim = 4, 8
    q = jnp.ones((1, seq, head_dim), dtype=jnp.float32)
    k = jnp.ones((1, seq, head_dim), dtype=jnp.float32)
    v = jnp.zeros((1, seq, head_dim), dtype=jnp.float32)
    v = v.at[0, 0, :].set(1.0)  # key/value 0: the only visible one for query 0
    v = v.at[0, 1:, :].set(999.0)  # future values: must NOT leak into query 0

    got = np.asarray(
        pallas_attention(q, k, v, causal=True, block_q=seq, block_k=seq, interpret=True)
    )

    # Query 0 sees only key 0, so its output is exactly value row 0 (all ones).
    assert np.allclose(got[0, 0], 1.0, atol=1e-5), (
        "causal mask leaked future values into query position 0"
    )
    # A later query (position 3) sees all keys, so its output is the mean of all
    # value rows and must include the large future values (sanity: mask is not
    # masking everything).
    assert got[0, -1].mean() > 100.0, "causal mask over-masked the last query row"


def test_pallas_attention_4d_batched_input() -> None:
    """A 4D (batch, heads, seq, dim) input matches the reference per (batch, head)."""
    batch, num_heads, seq, head_dim = 2, 3, 128, 64
    key = jax.random.PRNGKey(2026)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (batch, num_heads, seq, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (batch, num_heads, seq, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, num_heads, seq, head_dim), dtype=jnp.float32)
    scale = 1.0 / (head_dim**0.5)

    got = np.asarray(pallas_attention(q, k, v, causal=False, interpret=True))
    assert got.shape == (batch, num_heads, seq, head_dim)

    # Reference: fold batch and heads into one leading axis (exactly what the
    # kernel does internally) and compare.
    qf = q.reshape(batch * num_heads, seq, head_dim)
    kf = k.reshape(batch * num_heads, seq, head_dim)
    vf = v.reshape(batch * num_heads, seq, head_dim)
    ref = np.asarray(_reference_attention(qf, kf, vf, scale=scale, causal=False)).reshape(
        batch, num_heads, seq, head_dim
    )

    cos = _cosine_sim(got, ref)
    assert cos > 0.99, f"cosine {cos} below 0.99 parity bar for 4D input"
    assert np.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_pallas_attention_numerically_stable_on_large_logits() -> None:
    """Large scores must not overflow: the online softmax shifts by the max first.

    Without the shift-before-exp guard, exp of a large score overflows to inf
    and the output is NaN. We inflate the scale so raw scores are ~1e3 and check
    the output stays finite and matches the (equally shifted) reference.
    """
    num_heads, seq, head_dim = 1, 128, 64
    key = jax.random.PRNGKey(99)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (num_heads, seq, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (num_heads, seq, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (num_heads, seq, head_dim), dtype=jnp.float32)
    big_scale = 1000.0  # push raw scores far past the fp32 exp overflow point

    got = np.asarray(pallas_attention(q, k, v, scale=big_scale, interpret=True))
    ref = np.asarray(_reference_attention(q, k, v, scale=big_scale, causal=False))

    assert np.all(np.isfinite(got)), "attention produced non-finite values on large logits"
    cos = _cosine_sim(got, ref)
    assert cos > 0.99, f"cosine {cos} below 0.99 on large logits"
    assert np.allclose(got, ref, atol=1e-3, rtol=1e-3)


def test_pallas_attention_rejects_rank_mismatch() -> None:
    """q, k, v must share rank; a 3D/4D mix should raise, not misbehave."""
    q = jnp.ones((2, 128, 64), dtype=jnp.float32)
    k = jnp.ones((1, 2, 128, 64), dtype=jnp.float32)
    v = jnp.ones((1, 2, 128, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="rank"):
        pallas_attention(q, k, v, interpret=True)


def test_pallas_attention_rejects_indivisible_block_q() -> None:
    """seq_q must be divisible by block_q so every grid step sees a full tile."""
    q = jnp.ones((1, 130, 64), dtype=jnp.float32)
    k = jnp.ones((1, 128, 64), dtype=jnp.float32)
    v = jnp.ones((1, 128, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="divisible"):
        pallas_attention(q, k, v, block_q=128, block_k=128, interpret=True)


def test_pallas_attention_rejects_head_dim_mismatch() -> None:
    """q, k, v must share head_dim; a mismatch should raise."""
    q = jnp.ones((1, 128, 64), dtype=jnp.float32)
    k = jnp.ones((1, 128, 32), dtype=jnp.float32)
    v = jnp.ones((1, 128, 32), dtype=jnp.float32)
    with pytest.raises(ValueError, match="head_dim"):
        pallas_attention(q, k, v, interpret=True)
