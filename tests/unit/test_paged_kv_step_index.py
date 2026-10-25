"""The per-step KV index matches the old per-token write and per-request gather.

`PagedKVCache` builds its write and gather indices once per step and reads
`cu_seqlens_q` from the device once per step, so no layer waits on the GPU.
These tests run the same workload through the new code and through copies of
the old code, and require bit-equal K/V and attention output at every layer of
every step. Bit-equal attention means bit-equal logits, so greedy decoding
emits the same tokens.
"""

from collections.abc import Callable

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.packed_attention import packed_attention_forward, packed_attention_torch
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache

BLOCK_SIZE = 4
NUM_LAYERS = 3
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4
HEAD_DIM = 8


def make_pool(attention_backend: str = "flash_attn") -> BlockPool:
    return BlockPool(
        num_blocks=64,
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
        prefix_cache=PrefixCache(block_size=BLOCK_SIZE),
        attention_backend=attention_backend,
    )


def old_write(
    cache: PagedKVCache,
    layer_idx: int,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    cu_seqlens_q_new: torch.Tensor,
) -> None:
    """The per-token write that `_write_packed_kv_uncompressed` used before."""
    k_pool, v_pool = cache._pool.storage_for_layer(layer_idx)
    for batch_idx in range(cache.batch_size):
        packed_start = int(cu_seqlens_q_new[batch_idx])
        new_tokens = int(cu_seqlens_q_new[batch_idx + 1]) - packed_start
        start_token = cache.seq_lens_list()[batch_idx] - new_tokens
        for token_offset in range(new_tokens):
            position = start_token + token_offset
            block_id = cache.block_ids_for_request(batch_idx)[position // BLOCK_SIZE]
            k_pool[block_id, position % BLOCK_SIZE] = packed_k[packed_start + token_offset]
            v_pool[block_id, position % BLOCK_SIZE] = packed_v[packed_start + token_offset]


def old_gather(cache: PagedKVCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-request gather that `materialize_packed_kv` used before."""
    k_pool, v_pool = cache._pool.storage_for_layer(layer_idx)
    keys, values = [], []
    for batch_idx, seq_len in enumerate(cache.seq_lens_list()):
        positions = torch.arange(seq_len)
        table = torch.tensor(cache.block_ids_for_request(batch_idx), dtype=torch.long)
        block_ids = table[positions // BLOCK_SIZE]
        keys.append(k_pool[block_ids, positions % BLOCK_SIZE])
        values.append(v_pool[block_ids, positions % BLOCK_SIZE])
    return torch.cat(keys), torch.cat(values)


class Twin:
    """One cache on the new code and one on the old write, fed the same inputs."""

    def __init__(self, attention_backend: str) -> None:
        self.new = PagedKVCache(make_pool(attention_backend))
        self.ref = PagedKVCache(make_pool(attention_backend))
        self.ref._write_packed_kv_uncompressed = (  # type: ignore[method-assign]
            lambda layer_idx, k, v, cu: old_write(self.ref, layer_idx, k, v, cu)
        )
        self.generator = torch.Generator().manual_seed(0)

    def both(self, action: Callable[[PagedKVCache], object]) -> None:
        """Apply a layout change to both caches, then gather before any new step."""
        action(self.new)
        action(self.ref)
        assert self.new.seq_lens_list() == self.ref.seq_lens_list()
        for batch_idx in range(self.new.batch_size):
            assert self.new.block_ids_for_request(batch_idx) == self.ref.block_ids_for_request(
                batch_idx
            )
        if sum(self.new.seq_lens_list()) == 0:
            return
        for layer_idx in range(NUM_LAYERS):
            new_k, new_v, _, _ = self.new.materialize_packed_kv(layer_idx)
            ref_k, ref_v = old_gather(self.ref, layer_idx)
            assert torch.equal(new_k, ref_k)
            assert torch.equal(new_v, ref_v)

    def step(self, new_tokens: list[int], cu_seqlens: torch.Tensor | None = None) -> None:
        """One forward step. A caller can pass one `cu_seqlens` tensor it reuses across steps."""
        assert len(new_tokens) == self.new.batch_size
        cu_list = [0]
        for count in new_tokens:
            cu_list.append(cu_list[-1] + count)
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(cu_list, dtype=torch.int32)
        else:
            cu_seqlens.copy_(torch.tensor(cu_list, dtype=torch.int32))
        total = cu_list[-1]
        for layer_idx in range(NUM_LAYERS):
            k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=self.generator)
            v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=self.generator)
            q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, generator=self.generator)
            self.new.append_kv_packed(k, v, cu_seqlens, layer_idx)
            self.ref.append_kv_packed(k, v, torch.tensor(cu_list, dtype=torch.int32), layer_idx)

            new_k, new_v, new_cu_k, max_k = self.new.materialize_packed_kv(layer_idx)
            ref_k, ref_v = old_gather(self.ref, layer_idx)
            assert torch.equal(new_k, ref_k)
            assert torch.equal(new_v, ref_v)
            assert new_cu_k.tolist() == self.ref.cu_seqlens_k_list()
            assert max_k == max(self.ref.seq_lens_list())

            out_new = packed_attention_forward(q, self.new, layer_idx, cu_seqlens)
            # The old call: device-style tensors for both boundaries.
            out_ref = packed_attention_torch(
                q,
                ref_k,
                ref_v,
                torch.tensor(cu_list, dtype=torch.int32),
                torch.tensor(self.ref.cu_seqlens_k_list(), dtype=torch.int32),
                HEAD_DIM**-0.5,
            )
            assert torch.equal(out_new, out_ref)


@pytest.mark.parametrize("attention_backend", ["flash_attn", "torch"])
def test_step_index_matches_old_code_across_a_serving_workload(attention_backend: str) -> None:
    """Chunked prefill, decode, prefix-cache swap, rollback, removal, prefix hit and merge."""
    twin = Twin(attention_backend)
    prompt_a = list(range(100, 110))
    prompt_b = list(range(200, 207))
    # Slot 2 repeats prompt A. Both fill their first block in the same step, so
    # the last layer swaps slot 2's block for the one slot 0 published.
    for prompt in (prompt_a, prompt_b, prompt_a):
        twin.both(lambda cache, p=prompt: cache.add_request_slot(prompt_token_ids=p))

    twin.step([4, 7, 4])
    assert twin.new.block_ids_for_request(2)[0] == twin.new.block_ids_for_request(0)[0]
    twin.step([4, 1, 4])
    twin.step([2, 1, 2])
    for _ in range(3):
        twin.step([1, 1, 1])

    # Speculative-decoding style rollback, then decode past the cut.
    twin.both(lambda cache: cache.truncate_to(1, cache.seq_lens_list()[1] - 2))
    twin.step([1, 1, 1])

    twin.both(lambda cache: cache.remove_request(0))
    # Two cached blocks of prompt A give the new slot a prefix hit.
    twin.both(lambda cache: cache.add_request_slot(prompt_token_ids=prompt_a[:9]))
    assert twin.new.seq_lens_list()[-1] == 8
    twin.step([1, 1, 1])

    def merge_prefilled(cache: PagedKVCache) -> None:
        single = PagedKVCache(cache._pool)
        single.add_request_slot()
        k = torch.ones(5, NUM_KV_HEADS, HEAD_DIM)
        for layer_idx in range(NUM_LAYERS):
            single.append_kv_packed(k, k + 1, torch.tensor([0, 5], dtype=torch.int32), layer_idx)
        cache.merge_request(single)

    twin.both(merge_prefilled)
    twin.step([1, 1, 1, 1])
    twin.step([0, 3, 1, 0])


def test_cu_seqlens_tensor_updated_in_place_between_steps() -> None:
    """A static `cu_seqlens` buffer with new values each step still writes correctly."""
    twin = Twin("flash_attn")
    for _ in range(2):
        twin.both(lambda cache: cache.add_request_slot())
    static_cu = torch.zeros(3, dtype=torch.int32)
    twin.step([5, 3], cu_seqlens=static_cu)
    twin.step([1, 1], cu_seqlens=static_cu)
    twin.step([2, 0], cu_seqlens=static_cu)


def test_later_layer_with_other_lengths_writes_where_old_code_did() -> None:
    """A later layer that passes other lengths than layer 0 gets its own write index."""
    twin = Twin("flash_attn")
    twin.both(lambda cache: cache.add_request_slot())
    k = torch.randn(3, NUM_KV_HEADS, HEAD_DIM)
    for cache in (twin.new, twin.ref):
        cache.append_kv_packed(k, k, torch.tensor([0, 3], dtype=torch.int32), layer_idx=0)
        cache.append_kv_packed(k[:2], k[:2], torch.tensor([0, 2], dtype=torch.int32), layer_idx=1)
    new_k, _, _, _ = twin.new.materialize_packed_kv(1)
    ref_k, _ = old_gather(twin.ref, 1)
    assert torch.equal(new_k[1:], ref_k[1:])  # position 0 of layer 1 was never written


def test_layers_share_one_index_and_one_device_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each step reads `cu_seqlens_q` once and builds two indices, whatever the layer count."""
    reads: list[int] = []
    builds: list[int] = []
    read = PagedKVCache._read_cu_seqlens_q
    build = PagedKVCache._block_slot_index

    def counting_read(self: PagedKVCache, cu_seqlens_q: torch.Tensor) -> list[int]:
        reads.append(1)
        return read(self, cu_seqlens_q)

    def counting_build(
        self: PagedKVCache, ranges: list[tuple[int, int]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        builds.append(1)
        return build(self, ranges, device)

    monkeypatch.setattr(PagedKVCache, "_read_cu_seqlens_q", counting_read)
    monkeypatch.setattr(PagedKVCache, "_block_slot_index", counting_build)

    cache = PagedKVCache(make_pool())
    cache.add_request_slot()
    cache.add_request_slot()
    for new_tokens in ([6, 3], [1, 1], [1, 1]):
        reads.clear()
        builds.clear()
        cu_seqlens = torch.tensor([0, new_tokens[0], sum(new_tokens)], dtype=torch.int32)
        total = sum(new_tokens)
        for layer_idx in range(NUM_LAYERS):
            kv = torch.randn(total, NUM_KV_HEADS, HEAD_DIM)
            cache.append_kv_packed(kv, kv, cu_seqlens, layer_idx)
            q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM)
            packed_attention_forward(q, cache, layer_idx, cu_seqlens)
        assert len(reads) == 1
        assert len(builds) == 2  # one write index, one gather index


def test_write_casts_to_the_pool_dtype() -> None:
    """The indexed write keeps the old copy's implicit cast to the pool dtype."""
    cache = PagedKVCache(make_pool())
    cache.add_request_slot()
    k = torch.randn(3, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float64)
    cache.append_kv_packed(k, k, torch.tensor([0, 3], dtype=torch.int32), layer_idx=0)
    keys, _, _, _ = cache.materialize_packed_kv(0)
    assert keys.dtype == torch.float32
    assert torch.equal(keys, k.to(torch.float32))
