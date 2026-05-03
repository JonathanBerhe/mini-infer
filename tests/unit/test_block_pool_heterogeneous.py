"""Heterogeneous-KV BlockPool: per-layer `(num_kv_heads, head_dim)`.

Stage C1 of the multi-model plan: each layer can declare its own KV
shape; the pool stores per-layer tensors of (potentially) different
sizes. Block IDs remain global — slot `block_id` reserves a chunk of
`block_size` tokens in EVERY layer's storage. The homogeneous default
preserves all current models' behavior.
"""

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool


def _make_pool(
    *,
    layer_kv_shape: list[tuple[int, int]] | None,
    num_layers: int = 2,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    num_blocks: int = 8,
    block_size: int = 4,
    kv_quant: str | None = None,
) -> BlockPool:
    return BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        kv_quant=kv_quant,
        layer_kv_shape=layer_kv_shape,
    )


def test_homogeneous_default_unchanged() -> None:
    """No `layer_kv_shape` -> homogeneous pool, legacy properties work."""
    pool = _make_pool(layer_kv_shape=None)
    assert pool.num_kv_heads == 4
    assert pool.head_dim == 16
    assert pool.num_kv_heads_for_layer(0) == 4
    assert pool.head_dim_for_layer(1) == 16
    # Storage is still the rectangular tensor view + per-layer accessor returns
    # the right slice into it.
    assert pool.storage.shape == (2, 2, 8, 4, 4, 16)
    k0, v0 = pool.storage_for_layer(0)
    assert k0.shape == (8, 4, 4, 16)
    assert v0.shape == (8, 4, 4, 16)


def test_heterogeneous_per_layer_shapes() -> None:
    """Per-layer `(num_kv_heads, head_dim)` produces per-layer tensors."""
    pool = _make_pool(
        layer_kv_shape=[(8, 32), (2, 64)],
        num_layers=2,
        num_kv_heads=4,
        head_dim=16,
    )
    assert pool.num_kv_heads_for_layer(0) == 8
    assert pool.head_dim_for_layer(0) == 32
    assert pool.num_kv_heads_for_layer(1) == 2
    assert pool.head_dim_for_layer(1) == 64
    k0, v0 = pool.storage_for_layer(0)
    k1, v1 = pool.storage_for_layer(1)
    assert k0.shape == (8, 4, 8, 32)
    assert v0.shape == (8, 4, 8, 32)
    assert k1.shape == (8, 4, 2, 64)
    assert v1.shape == (8, 4, 2, 64)


def test_legacy_properties_raise_on_heterogeneous() -> None:
    """`pool.num_kv_heads` / `head_dim` / `storage` only valid for homogeneous."""
    pool = _make_pool(layer_kv_shape=[(8, 32), (2, 64)])
    with pytest.raises(RuntimeError, match="heterogeneous"):
        _ = pool.num_kv_heads
    with pytest.raises(RuntimeError, match="heterogeneous"):
        _ = pool.head_dim
    with pytest.raises(RuntimeError, match="heterogeneous"):
        _ = pool.storage


def test_heterogeneous_round_trip_per_layer() -> None:
    """Write to layer 0 and layer 1 with their own shapes; read back."""
    pool = _make_pool(layer_kv_shape=[(4, 8), (2, 16)])
    k0, v0 = pool.storage_for_layer(0)
    k1, v1 = pool.storage_for_layer(1)

    # Layer 0 slot 3, position 1: (4 heads, 8 dim).
    block_id = pool.allocate()
    sentinel_k0 = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
    sentinel_v0 = sentinel_k0 * -1.0
    k0[block_id, 1] = sentinel_k0
    v0[block_id, 1] = sentinel_v0
    # Layer 1 same block_id: (2 heads, 16 dim).
    sentinel_k1 = torch.arange(2 * 16, dtype=torch.float32).view(2, 16) + 100
    sentinel_v1 = sentinel_k1 * 2.0
    k1[block_id, 1] = sentinel_k1
    v1[block_id, 1] = sentinel_v1

    # Re-read.
    k0_re, v0_re = pool.storage_for_layer(0)
    k1_re, v1_re = pool.storage_for_layer(1)
    assert torch.equal(k0_re[block_id, 1], sentinel_k0)
    assert torch.equal(v0_re[block_id, 1], sentinel_v0)
    assert torch.equal(k1_re[block_id, 1], sentinel_k1)
    assert torch.equal(v1_re[block_id, 1], sentinel_v1)


def test_validator_length_mismatch() -> None:
    with pytest.raises(ValueError, match="layer_kv_shape has"):
        _make_pool(
            layer_kv_shape=[(4, 16)],  # 1 entry but num_layers=2
            num_layers=2,
        )


def test_validator_non_positive_entries() -> None:
    with pytest.raises(ValueError, match="positive ints"):
        _make_pool(layer_kv_shape=[(0, 16), (4, 16)])
    with pytest.raises(ValueError, match="positive ints"):
        _make_pool(layer_kv_shape=[(4, -8), (4, 16)])


def test_validator_heterogeneous_with_quant_rejects() -> None:
    """Stage C1 rejects heterogeneous shape + any kv_quant."""
    with pytest.raises(ValueError, match="heterogeneous layer_kv_shape requires kv_quant=None"):
        _make_pool(
            layer_kv_shape=[(4, 16), (2, 32)],
            num_blocks=8,
            block_size=16,
            head_dim=16,
            kv_quant="fp8",
        )
