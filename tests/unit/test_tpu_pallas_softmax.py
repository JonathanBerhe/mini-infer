"""Parity tests for the Pallas TPU row-wise softmax (M0 scaffold kernel).

The kernel runs in Pallas *interpret* mode (`interpret=True`), which executes
on the host CPU with no TPU hardware, so these tests run anywhere JAX is
installed. `pytest.importorskip("jax")` skips the whole module cleanly when the
optional `tpu` extra is absent (plain M1 / CI), matching how the CUDA-kernel
tests gate on their optional deps.

Parity is checked against `jax.nn.softmax` with two independent assertions, the
same bar ADR-023 sets for every hand-written kernel:

- cosine similarity > 0.99 (the project-wide kernel parity threshold), and
- elementwise `allclose` (softmax is exactly reproducible in fp32, so we can
  demand more than cosine alone).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (must follow the importorskip guard)

from mini_infer.backends.tpu.pallas_softmax import (  # noqa: E402
    pallas_softmax,
    supports_pallas_softmax,
)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Flattened cosine similarity, the project-wide kernel parity metric."""
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf)))


def test_supports_predicate_true_when_jax_present() -> None:
    """With JAX importable, the dispatch predicate must report supported.

    (This module was import-skipped if JAX were absent, so here it is present.)
    """
    assert supports_pallas_softmax() is True


@pytest.mark.parametrize(
    ("rows", "cols", "row_block"),
    [
        (8, 128, 8),  # single grid step, one 8x128 tile
        (16, 128, 8),  # two grid steps
        (32, 256, 16),  # wider rows, larger row-block
        (24, 64, 8),  # cols below a 128-lane tile (interpret mode pads)
    ],
)
def test_pallas_softmax_matches_jax_reference(rows: int, cols: int, row_block: int) -> None:
    """Kernel output matches jax.nn.softmax on the last axis, within the parity bar."""
    key = jax.random.PRNGKey(rows * 1000 + cols)
    x = jax.random.normal(key, (rows, cols), dtype=jnp.float32)

    got = np.asarray(pallas_softmax(x, row_block=row_block, interpret=True))
    ref = np.asarray(jax.nn.softmax(x, axis=-1))

    cos = _cosine_sim(got, ref)
    assert cos > 0.99, f"cosine {cos} below 0.99 parity bar for shape ({rows}, {cols})"
    assert np.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_pallas_softmax_rows_sum_to_one() -> None:
    """Each output row is a probability distribution (sums to 1)."""
    key = jax.random.PRNGKey(7)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)

    got = np.asarray(pallas_softmax(x, interpret=True))

    row_sums = got.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_pallas_softmax_numerically_stable_on_large_logits() -> None:
    """Large logits must not overflow: the kernel subtracts the row max first.

    Without the max-subtraction guard, exp(1000) overflows to inf and the
    result is NaN. This asserts the stable path in the kernel body.
    """
    x = jnp.array([[1000.0, 1001.0, 1002.0], [-500.0, 0.0, 500.0]], dtype=jnp.float32)

    got = np.asarray(pallas_softmax(x, row_block=1, interpret=True))
    ref = np.asarray(jax.nn.softmax(x, axis=-1))

    assert np.all(np.isfinite(got)), "softmax produced non-finite values on large logits"
    assert np.allclose(got, ref, atol=1e-6)


def test_pallas_softmax_rejects_non_2d_input() -> None:
    """The kernel is 2D-only; a 1D input should raise, not silently misbehave."""
    x = jnp.ones((8,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="2D"):
        pallas_softmax(x, interpret=True)


def test_pallas_softmax_rejects_indivisible_row_block() -> None:
    """rows must be divisible by row_block so every grid step sees a full tile."""
    x = jnp.ones((10, 16), dtype=jnp.float32)
    with pytest.raises(ValueError, match="divisible"):
        pallas_softmax(x, row_block=4, interpret=True)
