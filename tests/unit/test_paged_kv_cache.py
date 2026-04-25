import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache


def make_pool(block_size: int = 4, num_blocks: int = 8) -> BlockPool:
    return BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def make_kv(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic K/V where each value encodes (token_idx, head, dim) for easy checking."""
    base = torch.arange(num_tokens * 2 * 4, dtype=torch.float32).reshape(1, num_tokens, 2, 4)
    # Convert to HF shape: (batch=1, num_kv_heads=2, seq_len, head_dim=4)
    key = base.permute(0, 2, 1, 3).contiguous()
    value = key + 1000.0  # distinguish value from key
    return key, value


def test_update_first_token_allocates_one_block() -> None:
    pool = make_pool(block_size=4)
    cache = PagedKVCache(pool)
    key, value = make_kv(1)
    cache.update(key, value, layer_idx=0)
    assert cache.get_seq_length() == 1
    assert len(cache.block_ids) == 1
    assert pool.num_free_blocks == pool.num_blocks - 1


def test_update_fills_block_then_overflows_to_second() -> None:
    pool = make_pool(block_size=4)
    cache = PagedKVCache(pool)
    # First call: 4 tokens -> exactly 1 full block
    key, value = make_kv(4)
    cache.update(key, value, layer_idx=0)
    assert cache.get_seq_length() == 4
    assert len(cache.block_ids) == 1

    # Decode-style: 1 more token -> needs a 2nd block
    key2, value2 = make_kv(1)
    cache.update(key2, value2, layer_idx=0)
    assert cache.get_seq_length() == 5
    assert len(cache.block_ids) == 2


def test_get_seq_length_advances_only_on_layer_zero() -> None:
    pool = make_pool(block_size=4)
    cache = PagedKVCache(pool)
    key, value = make_kv(3)
    cache.update(key, value, layer_idx=0)
    cache.update(key, value, layer_idx=1)  # second layer in same step
    # Token count should be 3, not 6
    assert cache.get_seq_length() == 3


def test_update_returns_full_history_with_correct_shape() -> None:
    pool = make_pool(block_size=4)
    cache = PagedKVCache(pool)
    # Prefill 5 tokens
    key, value = make_kv(5)
    out_key, out_value = cache.update(key, value, layer_idx=0)
    # Materialized output should be (batch=1, num_kv_heads=2, seq_len=5, head_dim=4)
    assert out_key.shape == (1, 2, 5, 4)
    assert out_value.shape == (1, 2, 5, 4)


def test_update_returned_values_match_what_was_written() -> None:
    pool = make_pool(block_size=4)
    cache = PagedKVCache(pool)
    key, value = make_kv(5)
    out_key, out_value = cache.update(key, value, layer_idx=0)
    assert torch.equal(out_key, key)
    assert torch.equal(out_value, value)


def test_decode_step_appends_to_history_correctly() -> None:
    pool = make_pool(block_size=4)
    cache = PagedKVCache(pool)
    # Prefill with 3 tokens
    prompt_key, prompt_value = make_kv(3)
    cache.update(prompt_key, prompt_value, layer_idx=0)
    # Decode one more token (synthetic; offset to be distinguishable)
    decode_key = torch.full((1, 2, 1, 4), 99.0)
    decode_value = torch.full((1, 2, 1, 4), -99.0)
    out_key, out_value = cache.update(decode_key, decode_value, layer_idx=0)
    assert out_key.shape == (1, 2, 4, 4)
    # First 3 positions match the prompt
    assert torch.equal(out_key[:, :, :3, :], prompt_key)
    # Last position matches the decode token
    assert torch.equal(out_key[:, :, 3:, :], decode_key)
    assert torch.equal(out_value[:, :, 3:, :], decode_value)


def test_layer_zero_and_layer_one_are_independent_storage() -> None:
    pool = make_pool(block_size=4)
    cache = PagedKVCache(pool)
    key0 = torch.full((1, 2, 2, 4), 1.0)
    val0 = torch.full((1, 2, 2, 4), 2.0)
    key1 = torch.full((1, 2, 2, 4), 3.0)
    val1 = torch.full((1, 2, 2, 4), 4.0)
    cache.update(key0, val0, layer_idx=0)
    cache.update(key1, val1, layer_idx=1)
    out_k0, out_v0 = cache._materialize(layer_idx=0)
    out_k1, out_v1 = cache._materialize(layer_idx=1)
    assert torch.equal(out_k0, key0)
    assert torch.equal(out_v0, val0)
    assert torch.equal(out_k1, key1)
    assert torch.equal(out_v1, val1)


def test_free_returns_blocks_to_pool() -> None:
    pool = make_pool(block_size=4, num_blocks=8)
    cache = PagedKVCache(pool)
    key, value = make_kv(10)
    cache.update(key, value, layer_idx=0)
    blocks_used = len(cache.block_ids)
    assert pool.num_free_blocks == 8 - blocks_used

    cache.free()
    assert pool.num_free_blocks == 8
    assert cache.get_seq_length() == 0
    assert cache.block_ids == []
