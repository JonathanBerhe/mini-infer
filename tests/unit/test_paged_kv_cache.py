import pytest
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


def make_kv(num_tokens: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic K/V where each value encodes (batch, token_idx, head, dim) for easy checking."""
    base = torch.arange(batch * num_tokens * 2 * 4, dtype=torch.float32).reshape(
        batch, num_tokens, 2, 4
    )
    # Convert to HF shape: (batch, num_kv_heads=2, seq_len, head_dim=4)
    key = base.permute(0, 2, 1, 3).contiguous()
    value = key + 1000.0  # distinguish value from key
    return key, value


def make_single_request_cache(pool: BlockPool) -> PagedKVCache:
    """Fresh cache with one empty request slot — the prefill starting state."""
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def test_update_first_token_allocates_one_block() -> None:
    pool = make_pool(block_size=4)
    cache = make_single_request_cache(pool)
    key, value = make_kv(1)
    cache.update(key, value, layer_idx=0)
    assert cache.get_seq_length() == 1
    assert len(cache.block_ids_for_request(0)) == 1
    assert pool.num_free_blocks == pool.num_blocks - 1


def test_update_fills_block_then_overflows_to_second() -> None:
    pool = make_pool(block_size=4)
    cache = make_single_request_cache(pool)
    # First call: 4 tokens -> exactly 1 full block
    key, value = make_kv(4)
    cache.update(key, value, layer_idx=0)
    assert cache.get_seq_length() == 4
    assert len(cache.block_ids_for_request(0)) == 1

    # Decode-style: 1 more token -> needs a 2nd block
    key2, value2 = make_kv(1)
    cache.update(key2, value2, layer_idx=0)
    assert cache.get_seq_length() == 5
    assert len(cache.block_ids_for_request(0)) == 2


def test_get_seq_length_advances_only_on_layer_zero() -> None:
    pool = make_pool(block_size=4)
    cache = make_single_request_cache(pool)
    key, value = make_kv(3)
    cache.update(key, value, layer_idx=0)
    cache.update(key, value, layer_idx=1)  # second layer in same step
    # Token count should be 3, not 6
    assert cache.get_seq_length() == 3


def test_update_returns_full_history_with_correct_shape() -> None:
    pool = make_pool(block_size=4)
    cache = make_single_request_cache(pool)
    # Prefill 5 tokens
    key, value = make_kv(5)
    out_key, out_value = cache.update(key, value, layer_idx=0)
    # Materialized output should be (batch=1, num_kv_heads=2, seq_len=5, head_dim=4)
    assert out_key.shape == (1, 2, 5, 4)
    assert out_value.shape == (1, 2, 5, 4)


def test_update_returned_values_match_what_was_written() -> None:
    pool = make_pool(block_size=4)
    cache = make_single_request_cache(pool)
    key, value = make_kv(5)
    out_key, out_value = cache.update(key, value, layer_idx=0)
    assert torch.equal(out_key, key)
    assert torch.equal(out_value, value)


def test_decode_step_appends_to_history_correctly() -> None:
    pool = make_pool(block_size=4)
    cache = make_single_request_cache(pool)
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
    cache = make_single_request_cache(pool)
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
    cache = make_single_request_cache(pool)
    key, value = make_kv(10)
    cache.update(key, value, layer_idx=0)
    blocks_used = len(cache.block_ids_for_request(0))
    assert pool.num_free_blocks == 8 - blocks_used

    cache.free()
    assert pool.num_free_blocks == 8
    assert cache.batch_size == 0


def test_add_request_slot_increases_batch_size() -> None:
    pool = make_pool()
    cache = PagedKVCache(pool)
    assert cache.batch_size == 0
    idx_a = cache.add_request_slot()
    idx_b = cache.add_request_slot()
    assert idx_a == 0
    assert idx_b == 1
    assert cache.batch_size == 2
    assert cache.seq_lens_list() == [0, 0]


def test_remove_request_frees_blocks_and_shifts_indices() -> None:
    pool = make_pool(block_size=4, num_blocks=8)
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache.add_request_slot()
    cache.add_request_slot()
    # Three batch_size=3 updates, each writing 2 tokens for a different request slot.
    # Build a (3, 2, 2, 4) K/V tensor and call append_kv with B=3.
    key = torch.full((3, 2, 2, 4), 0.0)
    value = torch.full((3, 2, 2, 4), 0.0)
    for b in range(3):
        key[b].fill_(float(b + 1))
        value[b].fill_(float(b + 1) * 10)
    cache.append_kv(key, value, layer_idx=0)

    assert cache.batch_size == 3
    assert cache.seq_lens_list() == [2, 2, 2]
    free_before_remove = pool.num_free_blocks

    cache.remove_request(1)  # remove the middle slot
    assert cache.batch_size == 2
    assert cache.seq_lens_list() == [2, 2]
    # Block freed: request 1 had 1 block (2 tokens, block_size=4), so +1 free.
    assert pool.num_free_blocks == free_before_remove + 1
    # The original slot 2 (last) is now at index 1.
    # Verify by reading its block IDs differ from what slot 0's would be.
    assert cache.block_ids_for_request(0) != cache.block_ids_for_request(1)


def test_remove_request_invalid_index_raises() -> None:
    pool = make_pool()
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    with pytest.raises(IndexError):
        cache.remove_request(5)


def test_merge_request_transfers_ownership() -> None:
    pool = make_pool(block_size=4, num_blocks=8)
    main = PagedKVCache(pool)

    # First request: prefill via a temp cache, then merge into main.
    temp_a = make_single_request_cache(pool)
    key_a, value_a = make_kv(3)
    temp_a.update(key_a, value_a, layer_idx=0)
    free_after_temp_a = pool.num_free_blocks

    idx = main.merge_request(temp_a)
    assert idx == 0
    assert main.batch_size == 1
    assert main.seq_lens_list() == [3]
    # Blocks did NOT free during merge — ownership transferred.
    assert pool.num_free_blocks == free_after_temp_a
    # `temp_a.free()` is now a no-op because ownership transferred.
    temp_a.free()
    assert pool.num_free_blocks == free_after_temp_a

    # Freeing main DOES return the blocks.
    main.free()
    assert pool.num_free_blocks == 8


def test_merge_request_rejects_multi_batch_source() -> None:
    pool = make_pool()
    main = PagedKVCache(pool)
    other = PagedKVCache(pool)
    other.add_request_slot()
    other.add_request_slot()
    with pytest.raises(ValueError, match="batch_size=1"):
        main.merge_request(other)


def test_append_kv_batched_writes_per_request() -> None:
    pool = make_pool(block_size=4, num_blocks=8)
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache.add_request_slot()

    # Per-request distinguishable values: req 0 -> 1.0, req 1 -> -1.0.
    key = torch.zeros(2, 2, 1, 4)
    value = torch.zeros(2, 2, 1, 4)
    key[0].fill_(1.0)
    value[0].fill_(1.0)
    key[1].fill_(-1.0)
    value[1].fill_(-1.0)
    cache.append_kv(key, value, layer_idx=0)
    assert cache.seq_lens_list() == [1, 1]

    # Materialize and check that each request's slot got the right value.
    out_key, _ = cache._materialize(layer_idx=0)
    assert out_key.shape == (2, 2, 1, 4)
    assert torch.allclose(out_key[0], torch.ones_like(out_key[0]))
    assert torch.allclose(out_key[1], -torch.ones_like(out_key[1]))


def test_append_kv_rejects_batch_mismatch() -> None:
    pool = make_pool()
    cache = make_single_request_cache(pool)  # batch_size=1
    # Pass B=2 input — should raise.
    key = torch.zeros(2, 2, 1, 4)
    value = torch.zeros(2, 2, 1, 4)
    with pytest.raises(ValueError, match="input batch=2 but cache batch_size=1"):
        cache.append_kv(key, value, layer_idx=0)


def test_append_kv_packed_uniform_matches_append_kv() -> None:
    """Calling append_kv_packed with uniform per-slot lengths matches append_kv."""
    pool_a = make_pool(block_size=4, num_blocks=8)
    pool_b = make_pool(block_size=4, num_blocks=8)

    cache_via_append = PagedKVCache(pool_a)
    cache_via_append.add_request_slot()
    cache_via_append.add_request_slot()
    key, value = make_kv(num_tokens=3, batch=2)
    cache_via_append.append_kv(key, value, layer_idx=0)

    cache_via_packed = PagedKVCache(pool_b)
    cache_via_packed.add_request_slot()
    cache_via_packed.add_request_slot()
    # Packed layout: (B*N, num_kv_heads, head_dim) with batch-major order.
    packed_k = key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[3])
    packed_v = value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[3])
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    cache_via_packed.append_kv_packed(packed_k, packed_v, cu_seqlens, layer_idx=0)

    assert cache_via_append.seq_lens_list() == cache_via_packed.seq_lens_list()
    assert pool_a.num_free_blocks == pool_b.num_free_blocks
    out_a, _ = cache_via_append._materialize(layer_idx=0)
    out_b, _ = cache_via_packed._materialize(layer_idx=0)
    assert torch.equal(out_a, out_b)


def test_append_kv_packed_ragged_writes_to_correct_slots() -> None:
    """Different per-slot lengths in one packed call write to the correct positions."""
    pool = make_pool(block_size=4, num_blocks=8)
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache.add_request_slot()
    cache.add_request_slot()

    # Per-slot distinguishable values: slot 0 -> 1.0 (3 tokens), slot 1 -> 2.0 (5 tokens),
    # slot 2 -> 3.0 (1 token). Total packed length = 9.
    packed_k = torch.cat(
        [
            torch.full((3, 2, 4), 1.0),
            torch.full((5, 2, 4), 2.0),
            torch.full((1, 2, 4), 3.0),
        ],
        dim=0,
    )
    packed_v = packed_k + 100.0
    cu_seqlens = torch.tensor([0, 3, 8, 9], dtype=torch.int32)
    cache.append_kv_packed(packed_k, packed_v, cu_seqlens, layer_idx=0)
    assert cache.seq_lens_list() == [3, 5, 1]

    out_key, out_value = cache._materialize(layer_idx=0)
    # max_seq=5; slot 0 has 3 valid positions then padding; slot 2 has 1 valid then padding.
    assert out_key.shape == (3, 2, 5, 4)
    assert torch.allclose(out_key[0, :, :3, :], torch.full_like(out_key[0, :, :3, :], 1.0))
    assert torch.allclose(out_key[1, :, :5, :], torch.full_like(out_key[1, :, :5, :], 2.0))
    assert torch.allclose(out_key[2, :, :1, :], torch.full_like(out_key[2, :, :1, :], 3.0))
    # Values use +100 offset.
    assert torch.allclose(out_value[1, :, :5, :], torch.full_like(out_value[1, :, :5, :], 102.0))


def test_append_kv_packed_zero_length_skips_slot() -> None:
    """Slots with diff=0 in cu_seqlens are not modified."""
    pool = make_pool(block_size=4, num_blocks=8)
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache.add_request_slot()

    # Pre-populate slot 0 with 2 tokens; leave slot 1 empty.
    initial_k = torch.full((1, 2, 2, 4), 7.0)
    initial_v = torch.full((1, 2, 2, 4), 7.0)
    # Use packed API directly to write only to slot 0.
    cache.append_kv_packed(
        initial_k.transpose(1, 2).reshape(-1, 2, 4),
        initial_v.transpose(1, 2).reshape(-1, 2, 4),
        torch.tensor([0, 2, 2], dtype=torch.int32),
        layer_idx=0,
    )
    assert cache.seq_lens_list() == [2, 0]
    free_after_init = pool.num_free_blocks

    # Now write to slot 1 only; slot 0 must be untouched.
    new_k = torch.full((3, 2, 4), 9.0)
    new_v = torch.full((3, 2, 4), 9.0)
    cache.append_kv_packed(new_k, new_v, torch.tensor([0, 0, 3], dtype=torch.int32), layer_idx=0)
    assert cache.seq_lens_list() == [2, 3]

    out_key, _ = cache._materialize(layer_idx=0)
    # Slot 0 still has 7.0 in its 2 valid positions.
    assert torch.allclose(out_key[0, :, :2, :], torch.full_like(out_key[0, :, :2, :], 7.0))
    # Slot 1 has 9.0 in its 3 valid positions.
    assert torch.allclose(out_key[1, :, :3, :], torch.full_like(out_key[1, :, :3, :], 9.0))
    # Slot 1 needed one block; slot 0 unchanged.
    assert pool.num_free_blocks == free_after_init - 1


def test_append_kv_packed_rejects_wrong_cu_seqlens_length() -> None:
    """cu_seqlens_q_new must have batch_size+1 entries."""
    pool = make_pool()
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache.add_request_slot()
    packed_k = torch.zeros(2, 2, 4)
    packed_v = torch.zeros(2, 2, 4)
    bad_cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)  # only 2 entries, need 3
    with pytest.raises(ValueError, match="cu_seqlens_q_new has 2 entries"):
        cache.append_kv_packed(packed_k, packed_v, bad_cu_seqlens, layer_idx=0)


def test_materialize_batched_pads_to_max_seq() -> None:
    """Two requests with different seq_lens: materialize pads shorter to max_seq with zeros."""
    pool = make_pool(block_size=4, num_blocks=8)

    # Build two single-request prefill caches with different seq_lens, merge into main.
    temp_short = make_single_request_cache(pool)
    short_key = torch.full((1, 2, 3, 4), 1.0)
    short_val = torch.full((1, 2, 3, 4), 1.0)
    temp_short.update(short_key, short_val, layer_idx=0)

    temp_long = make_single_request_cache(pool)
    long_key = torch.full((1, 2, 5, 4), 2.0)
    long_val = torch.full((1, 2, 5, 4), 2.0)
    temp_long.update(long_key, long_val, layer_idx=0)

    main = PagedKVCache(pool)
    main.merge_request(temp_short)
    main.merge_request(temp_long)
    assert main.seq_lens_list() == [3, 5]

    out_key, _ = main._materialize(layer_idx=0)
    # Padded to max_seq=5; row 0 has 1.0 in positions 0-2 and zeros in 3-4.
    assert out_key.shape == (2, 2, 5, 4)
    assert torch.allclose(out_key[0, :, :3, :], torch.full_like(out_key[0, :, :3, :], 1.0))
    assert torch.allclose(out_key[0, :, 3:, :], torch.zeros_like(out_key[0, :, 3:, :]))
    assert torch.allclose(out_key[1], torch.full_like(out_key[1], 2.0))
