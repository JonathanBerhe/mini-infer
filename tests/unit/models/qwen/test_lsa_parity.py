"""Latent Sparse Attention selection, pinned against the authors' reference.

The oracle is `lsa.core.reference` from `Prefrontal-AI/latent-sparse-attention`
(Apache-2.0), vendored at a pinned revision by
`scripts/qwen/clone_lsa_reference.py`. It is the plain-PyTorch flat design the
authors test their kernels against. The tests skip when it is not vendored.

Both sides run on the same keys and queries. The buckets must be bit-equal
(members, sizes and centroids), and so must the selected key lists, slot for
slot, because our layout (expansion best first, window, sinks) is the
reference's.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.nn import functional

from mini_infer.models.qwen.lsa import LsaConfig, cluster_blocks, select_keys

REFERENCE_DIR = Path(__file__).resolve().parents[4] / "third_party" / "lsa_reference"

# The served flat design, and a small one whose budget runs out mid-bucket.
CONFIGS = {
    "served": LsaConfig(),
    "small": LsaConfig(
        window_size=32,
        sink_size=2,
        top_k=24,
        top_p=5,
        index_dim=8,
        pool_size=2,
        bucket_size=4,
        bucket_cap=8,
    ),
}


@pytest.fixture(scope="module")
def reference() -> Any:
    if not (REFERENCE_DIR / "lsa" / "core" / "reference.py").exists():
        pytest.skip(
            f"LSA reference not vendored at {REFERENCE_DIR}; "
            "run `uv run python scripts/qwen/clone_lsa_reference.py`"
        )
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    return importlib.import_module("lsa.core.reference")


def _keys(cfg: LsaConfig, length: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Unit index keys drawn around a few centers, so Ward has structure to find."""
    generator = torch.Generator().manual_seed(seed)
    centers = functional.normalize(torch.randn(12, cfg.index_dim, generator=generator), dim=-1)
    pick = torch.randint(0, 12, (length,), generator=generator)
    noise = 0.4 * torch.randn(length, cfg.index_dim, generator=generator)
    return functional.normalize(centers[pick] + noise, dim=-1).to(dtype)


def _ours(cfg: LsaConfig, keys: torch.Tensor):
    blocks = keys.shape[0] // cfg.window_size
    closed = keys[: blocks * cfg.window_size].view(blocks, cfg.window_size, cfg.index_dim)
    return cluster_blocks(closed, torch.arange(blocks) * cfg.window_size, cfg)


def _reference_buckets(reference: Any, cfg: LsaConfig, keys: torch.Tensor):
    return reference.build_buckets(
        keys, cfg.pool_size, cfg.bucket_size, cfg.bucket_cap, cfg.window_size
    )


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_buckets_match_reference(reference, name, dtype):
    cfg = CONFIGS[name]
    keys = _keys(cfg, 9 * cfg.window_size + cfg.window_size // 2, dtype, seed=0)
    expected = _reference_buckets(reference, cfg, keys)
    ours = _ours(cfg, keys)
    assert torch.equal(ours.members.flatten(0, 1), expected.bucket_members)
    assert torch.equal(ours.sizes.flatten(0, 1), expected.bucket_sizes)
    assert torch.equal(ours.centroids.flatten(0, 1), expected.bucket_centroids)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_selection_matches_reference(reference, name):
    cfg = CONFIGS[name]
    length = 12 * cfg.window_size + 37
    keys = _keys(cfg, length, torch.bfloat16, seed=1)
    buckets = _reference_buckets(reference, cfg, keys)
    queries = _keys(cfg, length, torch.float32, seed=2)
    # Every block boundary and its neighbors, plus a spread of other positions.
    edges = torch.arange(0, length, cfg.window_size)
    positions = torch.cat([edges - 1, edges, edges + 1, torch.arange(3, length, 97)])
    positions = positions.clamp(0, length - 1).unique()
    queries = queries[positions]

    expected, expected_lengths = reference.read(
        queries, positions, buckets, cfg.top_p, cfg.top_k, cfg.sink_size, cfg.window_size
    )
    ours, ours_lengths = select_keys(
        queries[None],
        positions[None],
        buckets.bucket_centroids[None],
        buckets.bucket_members[None],
        cfg,
    )
    assert torch.equal(ours[0], expected)
    assert torch.equal(ours_lengths[0], expected_lengths)


def test_end_to_end_matches_reference_with_a_tight_budget(reference):
    """Our buckets fed to our selection, with a budget small enough to cut buckets."""
    cfg = LsaConfig(top_k=300, top_p=50)
    length = 10 * cfg.window_size + 3
    keys = _keys(cfg, length, torch.bfloat16, seed=3)
    queries = _keys(cfg, length, torch.float32, seed=4)
    positions = torch.arange(cfg.window_size, length, 61)
    queries = queries[positions]

    expected, expected_lengths = reference.read(
        queries,
        positions,
        _reference_buckets(reference, cfg, keys),
        cfg.top_p,
        cfg.top_k,
        cfg.sink_size,
        cfg.window_size,
    )
    buckets = _ours(cfg, keys)
    ours, ours_lengths = select_keys(
        queries[None],
        positions[None],
        buckets.centroids.flatten(0, 1)[None],
        buckets.members.flatten(0, 1)[None],
        cfg,
    )
    assert torch.equal(ours[0], expected)
    assert torch.equal(ours_lengths[0], expected_lengths)
