"""Per-layer stream-descriptor `BlockPool` (Stage C3).

Generalizes the heterogeneous-K/V pool: each layer carries a list of
named `StreamSpec`s instead of a single `(num_kv_heads, head_dim)`. The
legacy K/V layout is the special case `["k", "v"]` with identical shape;
MLA-style layouts (DeepSeek-V2/V3) carry `["kv_latent", "k_rope"]` with
different shapes per stream.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool, StreamSpec
from mini_infer.cache.paged_kv_cache import PagedKVCache


def _make_legacy_pool() -> BlockPool:
    return BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )


def _make_mla_pool() -> BlockPool:
    streams = [[StreamSpec("kv_latent", 1, 16), StreamSpec("k_rope", 1, 4)]] * 2
    return BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=2,
        num_kv_heads=1,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
        layer_streams=streams,
    )


def test_legacy_kv_layout_synthesized_when_layer_streams_omitted() -> None:
    """`layer_streams=None` synthesizes `[k, v]` per layer from `layer_kv_shape`."""
    pool = _make_legacy_pool()
    assert pool._is_legacy_kv_layout is True
    assert pool.stream_names(0) == ["k", "v"]
    k_spec = pool.stream_spec(0, "k")
    v_spec = pool.stream_spec(0, "v")
    assert (k_spec.num_kv_heads, k_spec.head_dim) == (2, 8)
    assert (v_spec.num_kv_heads, v_spec.head_dim) == (2, 8)


def test_legacy_kv_storage_aliases_rectangular_layout() -> None:
    """Stream storage for `k`/`v` aliases `_layer_storage[i][0/1]` (no extra memory)."""
    pool = _make_legacy_pool()
    layer_t = pool._layer_storage[0]
    assert pool.storage_for_stream(0, "k").data_ptr() == layer_t[0].data_ptr()
    assert pool.storage_for_stream(0, "v").data_ptr() == layer_t[1].data_ptr()


def test_legacy_accessors_still_work_on_legacy_kv_layout() -> None:
    """`storage_for_layer` / `num_kv_heads_for_layer` continue to work."""
    pool = _make_legacy_pool()
    k, v = pool.storage_for_layer(0)
    assert k.shape == (8, 4, 2, 8)
    assert v.shape == (8, 4, 2, 8)
    assert pool.num_kv_heads_for_layer(0) == 2
    assert pool.head_dim_for_layer(0) == 8


def test_mla_layout_per_stream_shapes() -> None:
    """MLA-style streams allocate per-layer per-stream tensors of differing shapes."""
    pool = _make_mla_pool()
    assert pool._is_legacy_kv_layout is False
    assert pool.stream_names(0) == ["kv_latent", "k_rope"]
    assert pool.storage_for_stream(0, "kv_latent").shape == (8, 4, 1, 16)
    assert pool.storage_for_stream(0, "k_rope").shape == (8, 4, 1, 4)
    assert pool.num_kv_heads_for_stream(0, "kv_latent") == 1
    assert pool.head_dim_for_stream(0, "kv_latent") == 16
    assert pool.head_dim_for_stream(0, "k_rope") == 4


def test_legacy_accessors_raise_on_mla_layout() -> None:
    """Legacy K/V accessors point migrators at the new stream API."""
    pool = _make_mla_pool()
    with pytest.raises(RuntimeError, match="non-standard stream layout"):
        pool.storage_for_layer(0)
    with pytest.raises(RuntimeError, match="non-standard stream layout"):
        pool.num_kv_heads_for_layer(0)
    with pytest.raises(RuntimeError, match="non-standard stream layout"):
        pool.head_dim_for_layer(0)


def test_unknown_stream_name_raises() -> None:
    pool = _make_mla_pool()
    with pytest.raises(KeyError, match="kv_latent"):
        pool.stream_spec(0, "nonsense")


def test_layer_streams_validation_length_mismatch() -> None:
    streams = [[StreamSpec("k", 2, 8), StreamSpec("v", 2, 8)]]
    with pytest.raises(ValueError, match="layer_streams has 1 entries"):
        BlockPool(
            num_blocks=4,
            block_size=4,
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            layer_streams=streams,
        )


def test_layer_streams_validation_duplicate_name() -> None:
    streams = [[StreamSpec("k", 2, 8), StreamSpec("k", 2, 8)]]
    with pytest.raises(ValueError, match="duplicate stream name"):
        BlockPool(
            num_blocks=4,
            block_size=4,
            num_layers=1,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            layer_streams=streams,
        )


def test_layer_streams_validation_non_positive_dims() -> None:
    streams = [[StreamSpec("k", 0, 8), StreamSpec("v", 2, 8)]]
    with pytest.raises(ValueError, match="must be positive"):
        BlockPool(
            num_blocks=4,
            block_size=4,
            num_layers=1,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            layer_streams=streams,
        )


def test_layer_streams_with_quant_rejects() -> None:
    """Stage C3 doesn't support per-stream + KV quantization."""
    streams = [[StreamSpec("kv_latent", 1, 16), StreamSpec("k_rope", 1, 4)]]
    with pytest.raises(ValueError, match="kv_quant=None"):
        BlockPool(
            num_blocks=4,
            block_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
            dtype=torch.float32,
            device="cpu",
            layer_streams=streams,
            kv_quant="fp8",
            attention_backend="flashinfer",
        )


def test_paged_cache_append_and_materialize_stream_round_trip() -> None:
    """Write per-stream packed tensors, read them back."""
    pool = _make_mla_pool()
    cache = PagedKVCache(pool)
    cache.add_request_slot()

    total_q = 5
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)
    kv_latent_packed = torch.arange(total_q * 1 * 16, dtype=torch.float32).view(total_q, 1, 16)
    k_rope_packed = torch.arange(total_q * 1 * 4, dtype=torch.float32).view(total_q, 1, 4) + 1000

    cache.append_stream_packed(kv_latent_packed, cu_seqlens_q, 0, "kv_latent")
    cache.append_stream_packed(k_rope_packed, cu_seqlens_q, 0, "k_rope")

    # Read back
    kv_full, cu_k, max_k = cache.materialize_packed_stream(0, "kv_latent")
    rope_full, _, _ = cache.materialize_packed_stream(0, "k_rope")
    assert kv_full.shape == (total_q, 1, 16)
    assert rope_full.shape == (total_q, 1, 4)
    assert torch.equal(kv_full, kv_latent_packed)
    assert torch.equal(rope_full, k_rope_packed)
    assert max_k == total_q
    assert cu_k.tolist() == [0, total_q]


def test_legacy_kv_round_trip_through_stream_api() -> None:
    """Legacy K/V cache works via `append_stream_packed` too (back-compat)."""
    pool = _make_legacy_pool()
    cache = PagedKVCache(pool)
    cache.add_request_slot()

    total_q = 3
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)
    k_packed = torch.arange(total_q * 2 * 8, dtype=torch.float32).view(total_q, 2, 8)
    v_packed = k_packed * -1.0

    cache.append_stream_packed(k_packed, cu_seqlens_q, 0, "k")
    cache.append_stream_packed(v_packed, cu_seqlens_q, 0, "v")

    k_full, _, _ = cache.materialize_packed_stream(0, "k")
    v_full, _, _ = cache.materialize_packed_stream(0, "v")
    assert torch.equal(k_full, k_packed)
    assert torch.equal(v_full, v_packed)
