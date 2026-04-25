import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.exceptions import OutOfMemoryError


def make_pool(num_blocks: int = 4, block_size: int = 4) -> BlockPool:
    return BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def test_allocate_returns_unique_ids_in_range() -> None:
    pool = make_pool(num_blocks=4)
    ids = {pool.allocate() for _ in range(4)}
    assert len(ids) == 4
    assert all(0 <= i < 4 for i in ids)


def test_free_returns_block_to_pool() -> None:
    pool = make_pool(num_blocks=2)
    a = pool.allocate()
    pool.allocate()
    assert pool.num_free_blocks == 0
    pool.free(a)
    assert pool.num_free_blocks == 1
    assert pool.allocate() == a  # LIFO free list


def test_allocate_when_empty_raises() -> None:
    pool = make_pool(num_blocks=2)
    pool.allocate()
    pool.allocate()
    with pytest.raises(OutOfMemoryError):
        pool.allocate()


def test_num_free_blocks_tracks_allocate_and_free() -> None:
    pool = make_pool(num_blocks=4)
    assert pool.num_free_blocks == 4
    a = pool.allocate()
    b = pool.allocate()
    assert pool.num_free_blocks == 2
    pool.free(a)
    assert pool.num_free_blocks == 3
    pool.free(b)
    assert pool.num_free_blocks == 4


def test_view_shape_is_block_size_by_kv_heads_by_head_dim() -> None:
    pool = make_pool(num_blocks=2, block_size=4)
    block_id = pool.allocate()
    key, value = pool.view(block_id, layer_idx=0)
    assert key.shape == (4, 2, 4)
    assert value.shape == (4, 2, 4)


def test_views_share_storage_with_pool() -> None:
    pool = make_pool(num_blocks=2, block_size=4)
    block_id = pool.allocate()
    key, _ = pool.view(block_id, layer_idx=1)
    key[0, 0, 0] = 7.5
    key_again, _ = pool.view(block_id, layer_idx=1)
    assert key_again[0, 0, 0].item() == 7.5


def test_invalid_num_blocks_raises() -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        BlockPool(
            num_blocks=0,
            block_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )


def test_invalid_block_size_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        BlockPool(
            num_blocks=4,
            block_size=0,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
