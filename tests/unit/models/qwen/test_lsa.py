"""Latent Sparse Attention selection: the properties that hold for any input.

These tests state what `qwen/lsa.py` must do in its own terms: which dims
feed the indexer, that Ward partitions every closed block into a fixed number
of capped buckets, and that a query reads each causal key at most once, its
window and sinks always, and every earlier key when the context is short.
`test_lsa_parity.py` compares the same code against the authors' reference.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional

from mini_infer.models.qwen.lsa import (
    IndexProjection,
    LsaConfig,
    cluster_blocks,
    index_key_input,
    index_query_input,
    select_keys,
)

# Small sizes that still exercise every rule: several closed blocks, a
# budget that runs out mid-bucket, and buckets that straddle the window.
SMALL = LsaConfig(
    window_size=32,
    sink_size=2,
    top_k=24,
    top_p=5,
    index_dim=8,
    pool_size=2,
    bucket_size=4,
    bucket_cap=8,
)


def _unit(*shape: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return functional.normalize(torch.randn(*shape, generator=generator), dim=-1)


def _cache(cfg: LsaConfig, length: int, seed: int = 0):
    """Buckets of every closed block of `length` random unit index keys."""
    keys = _unit(length, cfg.index_dim, seed=seed)
    blocks = length // cfg.window_size
    closed = keys[: blocks * cfg.window_size].view(blocks, cfg.window_size, cfg.index_dim)
    return keys, cluster_blocks(closed, torch.arange(blocks) * cfg.window_size, cfg)


def _select(cfg: LsaConfig, buckets, queries: torch.Tensor, positions: torch.Tensor):
    selected, lengths = select_keys(
        queries[None],
        positions[None],
        buckets.centroids.flatten(0, 1)[None],
        buckets.members.flatten(0, 1)[None],
        cfg,
    )
    return selected[0], lengths[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"bucket_cap": 15},  # under twice bucket_size
        {"pool_size": 3},  # does not divide bucket_size
        {"bucket_size": 24},  # does not divide window_size
        {"top_k": 0},
        {"sink_size": -1},
    ],
)
def test_config_refuses_sizes_that_do_not_nest(overrides):
    with pytest.raises(ValueError):
        LsaConfig(**overrides)


def test_config_defaults_are_the_served_flat_design():
    cfg = LsaConfig()
    assert cfg.block_bucket_count == 64
    assert cfg.list_width == 4096 + 512 + 4


def test_index_inputs_drop_the_rotary_prefix_and_group_queries_by_kv_head():
    kv_heads, group, head_dim, rotary_dim = 4, 6, 16, 4
    key = torch.randn(3, kv_heads, head_dim)
    query = torch.randn(3, kv_heads * group, head_dim)

    key_input = index_key_input(key, rotary_dim)
    query_input = index_query_input(query, kv_heads, rotary_dim)

    free = head_dim - rotary_dim
    assert key_input.dtype == torch.bfloat16
    assert query_input.dtype == torch.float32
    for kv in range(kv_heads):
        span = slice(kv * free, (kv + 1) * free)
        torch.testing.assert_close(key_input[:, span], key[:, kv, rotary_dim:].bfloat16())
        heads = query[:, kv * group : (kv + 1) * group, rotary_dim:]
        torch.testing.assert_close(query_input[:, span], heads.sum(1))


def test_index_projection_gives_unit_rows_in_the_served_dtypes():
    projection = IndexProjection(torch.randn(24, 8), torch.randn(24, 8))
    keys = projection.index_keys(torch.randn(5, 24))
    queries = projection.index_queries(torch.randn(5, 24))
    assert keys.dtype == torch.bfloat16
    assert queries.dtype == torch.float32
    torch.testing.assert_close(keys.float().norm(dim=-1), torch.ones(5), atol=1e-2, rtol=0)
    torch.testing.assert_close(queries.norm(dim=-1), torch.ones(5))


def test_index_projection_refuses_maps_of_different_shapes():
    with pytest.raises(ValueError):
        IndexProjection(torch.randn(24, 8), torch.randn(24, 4))


def test_ward_partitions_each_block_into_capped_buckets_of_whole_pools():
    cfg = SMALL
    _, buckets = _cache(cfg, 4 * cfg.window_size)
    assert buckets.members.shape == (4, cfg.block_bucket_count, cfg.bucket_cap)
    for block in range(4):
        members = buckets.members[block]
        held = members[members >= 0].sort().values
        start = block * cfg.window_size
        # Every key of the block is in exactly one bucket.
        assert torch.equal(held, torch.arange(start, start + cfg.window_size, dtype=torch.int32))
        sizes = (members >= 0).sum(1)
        assert torch.equal(sizes.int(), buckets.sizes[block])
        assert int(sizes.min()) >= cfg.pool_size
        assert int(sizes.max()) <= cfg.bucket_cap
        for row, size in zip(members, sizes.tolist(), strict=True):
            keys_of_bucket = row[:size]
            # Members are listed in position order, and a pool is never split.
            assert torch.equal(keys_of_bucket, keys_of_bucket.sort().values)
            pools, counts = ((keys_of_bucket - start) // cfg.pool_size).unique(return_counts=True)
            assert torch.equal(counts, torch.full_like(pools, cfg.pool_size))


def test_a_centroid_is_the_normalized_sum_of_its_keys():
    cfg = SMALL
    keys, buckets = _cache(cfg, 2 * cfg.window_size)
    centroids = buckets.centroids.flatten(0, 1)
    for row, members in enumerate(buckets.members.flatten(0, 1)):
        expected = functional.normalize(keys[members[members >= 0].long()].sum(0), dim=0)
        torch.testing.assert_close(centroids[row], expected)


def test_a_block_clustered_alone_gives_the_buckets_it_gives_among_others():
    cfg = SMALL
    keys, buckets = _cache(cfg, 3 * cfg.window_size)
    middle = keys[cfg.window_size : 2 * cfg.window_size][None]
    alone = cluster_blocks(middle, torch.tensor([cfg.window_size]), cfg)
    assert torch.equal(alone.members[0], buckets.members[1])
    torch.testing.assert_close(alone.centroids[0], buckets.centroids[1])


def test_a_query_reads_each_causal_key_at_most_once_with_its_window_and_sinks():
    cfg = SMALL
    length = 7 * cfg.window_size + 5
    _, buckets = _cache(cfg, length)
    positions = torch.arange(length)
    selected, lengths = _select(cfg, buckets, _unit(length, cfg.index_dim, seed=1), positions)
    for position, row, count in zip(positions.tolist(), selected, lengths.tolist(), strict=True):
        keys = row[:count]
        assert bool((row[count:] == -1).all())
        assert bool((keys >= 0).all()) and bool((keys <= position).all())
        assert keys.unique().numel() == count
        window_start = max(0, position + 1 - cfg.window_size)
        must = set(range(window_start, position + 1)) | set(range(min(cfg.sink_size, window_start)))
        assert must <= set(keys.tolist())
        assert count - len(must) <= cfg.top_k


def test_a_short_context_reads_every_earlier_key():
    # `top_p` must cover every bucket a query at this length can see, as the
    # served sizes do (576 visible buckets at most, against 1024).
    cfg = replace(SMALL, top_p=8)
    length = cfg.list_width  # the longest context that reads everything
    assert (length - 1) // cfg.window_size * cfg.block_bucket_count <= cfg.top_p
    _, buckets = _cache(cfg, length)
    positions = torch.arange(length)
    selected, lengths = _select(cfg, buckets, _unit(length, cfg.index_dim, seed=2), positions)
    assert torch.equal(lengths, (positions + 1).int())
    for position, row in enumerate(selected):
        keys = row[: position + 1].sort().values
        assert torch.equal(keys, torch.arange(position + 1, dtype=torch.int32))


def test_a_query_never_sees_the_buckets_of_its_own_block():
    cfg = SMALL
    length = 4 * cfg.window_size
    _, buckets = _cache(cfg, length)
    # A query at the last position of block 2 has blocks 0 and 1 closed
    # before it. The cache also holds block 2 and 3, which it must not read
    # through their buckets.
    position = 3 * cfg.window_size - 1
    selected, lengths = _select(
        cfg, buckets, _unit(1, cfg.index_dim, seed=3), torch.tensor([position])
    )
    keys = selected[0, : int(lengths[0])]
    assert int(keys.max()) == position
    # The window is the last 32 positions and the sinks are 2, so the rest
    # came from buckets.
    expanded = keys[: int(lengths[0]) - cfg.window_size - cfg.sink_size]
    assert expanded.numel() > 0
    assert bool((expanded < 2 * cfg.window_size).all())


def test_queries_read_the_same_alone_as_in_a_chunk_or_a_batch():
    cfg = SMALL
    length = 6 * cfg.window_size
    _, buckets = _cache(cfg, length)
    queries = _unit(length, cfg.index_dim, seed=4)
    positions = torch.arange(length)
    chunk, chunk_lengths = _select(cfg, buckets, queries, positions)
    for position in (0, cfg.window_size, 3 * cfg.window_size + 7, length - 1):
        alone, alone_length = _select(
            cfg, buckets, queries[position : position + 1], positions[position : position + 1]
        )
        assert torch.equal(alone[0], chunk[position])
        assert int(alone_length[0]) == int(chunk_lengths[position])

    # Two requests in one call read what each reads alone.
    _, other = _cache(cfg, length, seed=5)
    stacked = select_keys(
        torch.stack([queries, queries]),
        torch.stack([positions, positions]),
        torch.stack([buckets.centroids.flatten(0, 1), other.centroids.flatten(0, 1)]),
        torch.stack([buckets.members.flatten(0, 1), other.members.flatten(0, 1)]),
        cfg,
    )
    second, second_lengths = _select(cfg, other, queries, positions)
    assert torch.equal(stacked[0][0], chunk) and torch.equal(stacked[0][1], second)
    assert torch.equal(stacked[1][1], second_lengths)


def test_a_cache_with_no_closed_block_reads_the_window_and_sinks():
    cfg = SMALL
    selected, lengths = select_keys(
        _unit(1, 3, cfg.index_dim),
        torch.tensor([[0, 10, cfg.window_size - 1]]),
        torch.zeros(1, 0, cfg.index_dim),
        torch.zeros(1, 0, cfg.bucket_cap, dtype=torch.int32),
        cfg,
    )
    assert lengths.tolist() == [[1, 11, cfg.window_size]]
    assert selected[0, 1, :11].tolist() == list(range(11))
