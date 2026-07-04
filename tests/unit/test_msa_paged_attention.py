"""MSA block-sparse paged decode: the torch reference vs the dense-mask oracle.

`msa_paged_decode_torch` gathers ONLY the indexer-selected blocks from the
paged pool and runs softmax over them; the oracle materializes the full
history and masks the non-selected keys to -inf. Softmax over a subset equals
softmax over the full set with the complement at -inf, so the two must agree
to float rounding. The end-to-end wiring (selection + kernel path inside the
model) is pinned by `test_minimax_m3_parity.py::test_greedy_decode_kernel_path`.
"""

from __future__ import annotations

import torch

from mini_infer.cache.block_pool import BlockPool, StreamSpec
from mini_infer.cache.msa_paged_attention import msa_paged_decode_torch
from mini_infer.cache.paged_kv_cache import PagedKVCache

NUM_KV_HEADS = 2
NUM_Q_HEADS = 8
HEAD_DIM = 16
POOL_BLOCK = 4
INDEX_BLOCK = 8  # 2 pool blocks per index block


def _build_cache(seq_lens: list[int]) -> PagedKVCache:
    """One layer of M3-shaped k/v streams filled with random data."""
    streams = [
        [
            StreamSpec("k", NUM_KV_HEADS, HEAD_DIM),
            StreamSpec("v", NUM_KV_HEADS, HEAD_DIM),
        ]
    ]
    pool = BlockPool(
        num_blocks=64,
        block_size=POOL_BLOCK,
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
        layer_streams=streams,
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    for _ in seq_lens:
        cache.add_request_slot()
    # One packed append covering every slot (the API requires all slots in cu).
    total = sum(seq_lens)
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM)
    cache.append_stream_packed(k, cu, 0, "k")
    cache.append_stream_packed(v, cu, 0, "v")
    return cache


def _oracle(
    q: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    selected: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Dense-mask oracle for one request: full history + -inf on non-selected."""
    group = NUM_Q_HEADS // NUM_KV_HEADS
    k = k_full.repeat_interleave(group, dim=1).float()  # (seq, H, d)
    v = v_full.repeat_interleave(group, dim=1).float()
    scores = torch.einsum("hd,shd->hs", q.float(), k) * HEAD_DIM**-0.5
    keep = torch.zeros(seq_len, dtype=torch.bool)
    for b in selected.tolist():
        if b >= 0:
            keep[b * INDEX_BLOCK : min((b + 1) * INDEX_BLOCK, seq_len)] = True
    scores = scores.masked_fill(~keep[None, :], float("-inf"))
    return torch.einsum("hs,shd->hd", torch.softmax(scores, dim=-1), v)


def test_msa_paged_decode_matches_dense_mask_oracle() -> None:
    """Random selections (with -1 padding and a partial last block) match the oracle."""
    torch.manual_seed(0)
    seq_lens = [21, 8, 33]  # 21 and 33 end mid index block; 8 is exactly one
    cache = _build_cache(seq_lens)
    pool = cache._pool
    k_pool = pool.storage_for_stream(0, "k")
    v_pool = pool.storage_for_stream(0, "v")

    q = torch.randn(len(seq_lens), NUM_Q_HEADS, HEAD_DIM)
    block_tables = cache.block_tables_per_request_tensor("cpu")

    # Per-request selections: always the local (last) block plus some others,
    # padded with -1, mirroring `MiniMaxM3Indexer.select_cached` output.
    selections = [
        torch.tensor([2, 0, -1], dtype=torch.int64),  # seq 21 -> blocks 0..2, last partial
        torch.tensor([0, -1, -1], dtype=torch.int64),  # seq 8 -> single full block
        torch.tensor([4, 1, 3], dtype=torch.int64),  # seq 33 -> blocks 0..4, last has 1 tok
    ]

    got = msa_paged_decode_torch(
        q,
        k_pool,
        v_pool,
        block_tables,
        seq_lens,
        selections,
        index_block_size=INDEX_BLOCK,
    )

    for r, n in enumerate(seq_lens):
        k_full, _, _ = cache.materialize_packed_stream(0, "k")
        v_full, _, _ = cache.materialize_packed_stream(0, "v")
        # Slice this request's history out of the packed form.
        starts = [0]
        for m in seq_lens:
            starts.append(starts[-1] + m)
        k_r = k_full[starts[r] : starts[r] + n]
        v_r = v_full[starts[r] : starts[r] + n]
        want = _oracle(q[r], k_r, v_r, selections[r], n)
        assert torch.allclose(got[r].float(), want, atol=1e-5), (
            f"request {r}: max_abs={(got[r].float() - want).abs().max().item():.2e}"
        )


def test_msa_paged_decode_rejects_non_divisible_blocks() -> None:
    torch.manual_seed(1)
    cache = _build_cache([8])
    pool = cache._pool
    k_pool = pool.storage_for_stream(0, "k")
    v_pool = pool.storage_for_stream(0, "v")
    q = torch.randn(1, NUM_Q_HEADS, HEAD_DIM)
    try:
        msa_paged_decode_torch(
            q,
            k_pool,
            v_pool,
            cache.block_tables_per_request_tensor("cpu"),
            [8],
            [torch.tensor([0], dtype=torch.int64)],
            index_block_size=POOL_BLOCK + 1,  # not a multiple of the pool block
        )
    except ValueError as err:
        assert "multiple" in str(err)
    else:
        raise AssertionError("expected ValueError for non-divisible block sizes")
