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


def old_page_index(cache: PagedKVCache) -> tuple[list[int], list[int], list[int]]:
    """The page index that `flashinfer_attention_forward` built for every layer before."""
    indptr = [0]
    indices: list[int] = []
    last_page_lens: list[int] = []
    running = 0
    for batch_idx, seq_len in enumerate(cache.seq_lens_list()):
        if seq_len == 0:
            indptr.append(running)
            last_page_lens.append(0)
            continue
        num_pages = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        indices.extend(cache.block_ids_for_request(batch_idx)[:num_pages])
        running += num_pages
        indptr.append(running)
        rem = seq_len % BLOCK_SIZE
        last_page_lens.append(BLOCK_SIZE if rem == 0 else rem)
    return indptr, indices, last_page_lens


def page_index_lists(cache: PagedKVCache) -> tuple[list[int], list[int], list[int]]:
    indptr, indices, last_page_len = cache.page_index(torch.device("cpu"))
    assert indptr.dtype == indices.dtype == last_page_len.dtype == torch.int32
    return indptr.tolist(), indices.tolist(), last_page_len.tolist()


def test_page_index_matches_old_code_across_a_serving_workload() -> None:
    """Empty slot, full last page, partial last page, prefix swap, rollback and removal."""
    cache = PagedKVCache(make_pool())
    prompt_a = list(range(100, 110))
    for prompt in (prompt_a, list(range(200, 207)), prompt_a):
        cache.add_request_slot(prompt_token_ids=prompt)
    cache.add_request_slot()  # stays empty
    empty = ([0, 0, 0, 0, 0], [], [0, 0, 0, 0])
    assert page_index_lists(cache) == old_page_index(cache) == empty

    def step(new_tokens: list[int]) -> None:
        cu_list = [0]
        for count in new_tokens:
            cu_list.append(cu_list[-1] + count)
        cu_seqlens = torch.tensor(cu_list, dtype=torch.int32)
        for layer_idx in range(NUM_LAYERS):
            kv = torch.randn(cu_list[-1], NUM_KV_HEADS, HEAD_DIM)
            cache.append_kv_packed(kv, kv, cu_seqlens, layer_idx)
            assert page_index_lists(cache) == old_page_index(cache)

    # Slot 0 fills one block exactly; the last layer swaps slot 2's block for it.
    step([4, 7, 4, 0])
    assert cache.block_ids_for_request(2)[0] == cache.block_ids_for_request(0)[0]
    assert page_index_lists(cache)[2] == [BLOCK_SIZE, 3, BLOCK_SIZE, 0]
    step([4, 1, 4, 0])
    step([1, 1, 1, 0])
    cache.truncate_to(1, cache.seq_lens_list()[1] - 2)
    assert page_index_lists(cache) == old_page_index(cache)
    cache.remove_request(0)
    assert page_index_lists(cache) == old_page_index(cache)
    step([1, 1, 2])


def test_page_index_is_built_once_per_step() -> None:
    """Every layer of a step gets the same tensors; the next step gets new ones."""
    cache = PagedKVCache(make_pool())
    cache.add_request_slot()
    cache.add_request_slot()
    seen: list[torch.Tensor] = []
    for new_tokens in ([6, 3], [1, 1], [1, 1]):
        cu_seqlens = torch.tensor([0, new_tokens[0], sum(new_tokens)], dtype=torch.int32)
        total = sum(new_tokens)
        step_indptrs = []
        for layer_idx in range(NUM_LAYERS):
            kv = torch.randn(total, NUM_KV_HEADS, HEAD_DIM)
            cache.append_kv_packed(kv, kv, cu_seqlens, layer_idx)
            step_indptrs.append(cache.page_index(torch.device("cpu"))[0])
        assert all(indptr is step_indptrs[0] for indptr in step_indptrs)
        assert all(indptr is not previous for previous in seen for indptr in step_indptrs)
        seen.append(step_indptrs[0])


def test_static_decode_step_matches_the_eager_append() -> None:
    """A static step writes the same K/V where the eager append writes it, advances the
    same counts, pads into the scratch block, and leaves the cache eager again."""
    from mini_infer.cache.paged_kv_cache import StaticDecodeStep

    eager = PagedKVCache(make_pool())
    static = PagedKVCache(make_pool())
    for cache in (eager, static):
        cache.add_request_slot()
        cache.add_request_slot()
    generator = torch.Generator().manual_seed(1)
    cu_prefill = torch.tensor([0, 6, 9], dtype=torch.int32)
    for layer_idx in range(NUM_LAYERS):
        k = torch.randn(9, NUM_KV_HEADS, HEAD_DIM, generator=generator)
        v = torch.randn(9, NUM_KV_HEADS, HEAD_DIM, generator=generator)
        eager.append_kv_packed(k, v, cu_prefill, layer_idx)
        static.append_kv_packed(k, v, cu_prefill, layer_idx)

    device = torch.device("cpu")
    scratch = static._pool.allocate()
    step = StaticDecodeStep.allocate(batch=4, scratch_block=scratch, num_blocks=64, device=device)
    cu_decode = torch.tensor([0, 1, 2], dtype=torch.int32)
    for _ in range(3):  # slot 0 crosses a block boundary at 8 tokens
        new_k = [
            torch.randn(2, NUM_KV_HEADS, HEAD_DIM, generator=generator) for _ in range(NUM_LAYERS)
        ]
        new_v = [
            torch.randn(2, NUM_KV_HEADS, HEAD_DIM, generator=generator) for _ in range(NUM_LAYERS)
        ]
        for layer_idx in range(NUM_LAYERS):
            eager.append_kv_packed(new_k[layer_idx], new_v[layer_idx], cu_decode, layer_idx)

        static.begin_static_decode(step, active=2)
        assert static.static_step is step and step.active == 2
        assert static.seq_lens_list() == eager.seq_lens_list()
        assert static.cu_seqlens_q_list(cu_decode) == [0, 1, 2, 3, 4]
        indptr, indices, last_page_len = static.page_index(device)
        assert indptr is step.indptr
        real_indptr, real_indices, real_last = old_page_index(static)
        last_page = real_indptr[-1]
        assert indptr.tolist() == [*real_indptr, last_page + 1, last_page + 2]
        assert indices[: len(real_indices) + 2].tolist() == [*real_indices, scratch, scratch]
        assert last_page_len.tolist() == [*real_last, 1, 1]
        padding = torch.ones(2, NUM_KV_HEADS, HEAD_DIM)
        for layer_idx in range(NUM_LAYERS):
            static.append_kv_packed(
                torch.cat([new_k[layer_idx], padding]),
                torch.cat([new_v[layer_idx], padding]),
                step.qo_indptr,
                layer_idx,
            )
        static.end_static_decode()
        assert static.static_step is None

        for layer_idx in range(NUM_LAYERS):
            eager_k, eager_v, _, _ = eager.materialize_packed_kv(layer_idx)
            static_k, static_v, _, _ = static.materialize_packed_kv(layer_idx)
            assert torch.equal(static_k, eager_k)
            assert torch.equal(static_v, eager_v)
            k_pool, _ = static._pool.storage_for_layer(layer_idx)
            assert torch.all(k_pool[scratch, 0] == 1)


def test_static_decode_step_out_of_memory_leaves_the_counts_in_place() -> None:
    """An allocation failure happens before any write, with no slot advanced."""
    from mini_infer.cache.paged_kv_cache import StaticDecodeStep

    pool = BlockPool(
        num_blocks=3,
        block_size=BLOCK_SIZE,
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache.add_request_slot()
    kv = torch.randn(8, NUM_KV_HEADS, HEAD_DIM)
    cache.append_kv_packed(kv, kv, torch.tensor([0, 4, 8], dtype=torch.int32), layer_idx=0)
    step = StaticDecodeStep.allocate(
        batch=2, scratch_block=pool.allocate(), num_blocks=3, device=torch.device("cpu")
    )
    with pytest.raises(Exception, match="no free blocks"):
        cache.begin_static_decode(step, active=2)
    assert cache.static_step is None
    assert cache.seq_lens_list() == [4, 4]


def test_static_decode_step_checks_its_batch() -> None:
    from mini_infer.cache.paged_kv_cache import StaticDecodeStep

    cache = PagedKVCache(make_pool())
    cache.add_request_slot()
    step = StaticDecodeStep.allocate(
        batch=1, scratch_block=cache._pool.allocate(), num_blocks=64, device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="must equal batch_size"):
        cache.begin_static_decode(step, active=0)
    cache.add_request_slot()
    with pytest.raises(ValueError, match="do not fit"):
        cache.begin_static_decode(step, active=2)
    cache.remove_request(1)
    cache.begin_static_decode(step, active=1)
    with pytest.raises(RuntimeError, match="already in progress"):
        cache.begin_static_decode(step, active=1)
    cache.end_static_decode()
    assert cache.seq_lens_list() == [1]


def _row_writer(
    k_pool: torch.Tensor, v_pool: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """The indexed store that `append_kv_packed` does, as a writer for `append_kv_with`."""

    def write(block_ids: torch.Tensor, slots: torch.Tensor) -> None:
        k_pool[block_ids, slots] = k
        v_pool[block_ids, slots] = v

    return write


def test_append_kv_with_matches_append_kv_packed() -> None:
    """A caller that stores the rows itself gets the same bookkeeping, row order and
    publication as `append_kv_packed`, across a serving workload."""
    packed = PagedKVCache(make_pool())
    custom = PagedKVCache(make_pool())
    prompt_a = list(range(100, 110))
    for cache in (packed, custom):
        for prompt in (prompt_a, list(range(200, 207)), prompt_a):
            cache.add_request_slot(prompt_token_ids=prompt)
    generator = torch.Generator().manual_seed(4)

    def step(new_tokens: list[int]) -> None:
        cu_list = [0]
        for count in new_tokens:
            cu_list.append(cu_list[-1] + count)
        cu_seqlens = torch.tensor(cu_list, dtype=torch.int32)
        total = cu_list[-1]
        for layer_idx in range(NUM_LAYERS):
            k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=generator)
            v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=generator)
            packed.append_kv_packed(k, v, cu_seqlens, layer_idx)
            k_pool, v_pool = custom._pool.storage_for_layer(layer_idx)
            custom.append_kv_with(cu_seqlens, layer_idx, _row_writer(k_pool, v_pool, k, v))
        assert custom.seq_lens_list() == packed.seq_lens_list()
        for batch_idx in range(custom.batch_size):
            assert custom.block_ids_for_request(batch_idx) == packed.block_ids_for_request(
                batch_idx
            )
        for layer_idx in range(NUM_LAYERS):
            custom_k, custom_v, _, _ = custom.materialize_packed_kv(layer_idx)
            packed_k, packed_v, _, _ = packed.materialize_packed_kv(layer_idx)
            assert torch.equal(custom_k, packed_k)
            assert torch.equal(custom_v, packed_v)

    # Slot 2's first block swaps for the one slot 0 publishes on the last layer.
    step([4, 7, 4])
    assert custom.block_ids_for_request(2)[0] == custom.block_ids_for_request(0)[0]
    step([4, 1, 4])
    step([1, 1, 1])
    step([0, 3, 1])


def test_append_kv_with_uses_the_static_step_index() -> None:
    from mini_infer.cache.paged_kv_cache import StaticDecodeStep

    cache = PagedKVCache(make_pool())
    cache.add_request_slot()
    kv = torch.randn(3, NUM_KV_HEADS, HEAD_DIM)
    cache.append_kv_packed(kv, kv, torch.tensor([0, 3], dtype=torch.int32), layer_idx=0)
    step = StaticDecodeStep.allocate(
        batch=2, scratch_block=cache._pool.allocate(), num_blocks=64, device=torch.device("cpu")
    )
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    cache.begin_static_decode(step, active=1)
    cache.append_kv_with(
        torch.zeros(3, dtype=torch.int32), 0, lambda ids, slots: seen.append((ids, slots))
    )
    cache.end_static_decode()
    assert len(seen) == 1
    assert seen[0][0] is step.write_block_ids and seen[0][1] is step.write_slots
    assert cache.seq_lens_list() == [4]
