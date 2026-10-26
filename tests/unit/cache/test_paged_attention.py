import pytest
import torch

from mini_infer.cache.paged_attention import (
    paged_attention_decode_torch,
    paged_attention_decode_torch_batched,
)
from tests.unit._kernel_test_utils import (
    cosine_sim,
    materialize_kv,
    populate_pool,
    sdpa_reference,
)


def test_torch_decode_matches_sdpa_at_block_aligned_length() -> None:
    seq_len = 8  # exactly two blocks of size 4
    k_pool, v_pool, block_table = populate_pool(seq_len=seq_len)
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 4, 4, generator=g)  # 4 q-heads, head_dim=4 (GQA group=2)

    out = paged_attention_decode_torch(q, k_pool, v_pool, block_table, seq_len=seq_len)

    k_full, v_full = materialize_kv(k_pool, v_pool, block_table, seq_len, group_size=2)
    expected = sdpa_reference(q, k_full, v_full)
    assert cosine_sim(out, expected) > 0.99
    assert torch.allclose(out, expected, atol=1e-5)


def test_torch_decode_handles_partially_filled_last_block() -> None:
    seq_len = 5  # one full block + one slot of the second
    k_pool, v_pool, block_table = populate_pool(seq_len=seq_len)
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, 4, 4, generator=g)

    out = paged_attention_decode_torch(q, k_pool, v_pool, block_table, seq_len=seq_len)
    k_full, v_full = materialize_kv(k_pool, v_pool, block_table, seq_len, group_size=2)
    expected = sdpa_reference(q, k_full, v_full)

    assert cosine_sim(out, expected) > 0.99
    assert torch.allclose(out, expected, atol=1e-5)


def test_torch_decode_gqa_head_grouping() -> None:
    """14 q heads, 2 kv heads, group_size=7 (Qwen2.5-0.5B's actual ratio)."""
    seq_len = 16
    k_pool, v_pool, block_table = populate_pool(
        num_blocks=8, block_size=4, num_kv_heads=2, head_dim=4, seq_len=seq_len, seed=3
    )
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, 14, 4, generator=g)

    out = paged_attention_decode_torch(q, k_pool, v_pool, block_table, seq_len=seq_len)
    k_full, v_full = materialize_kv(k_pool, v_pool, block_table, seq_len, group_size=7)
    expected = sdpa_reference(q, k_full, v_full)
    assert cosine_sim(out, expected) > 0.99
    assert torch.allclose(out, expected, atol=1e-5)


def test_torch_decode_seq_len_one() -> None:
    seq_len = 1
    k_pool, v_pool, block_table = populate_pool(seq_len=seq_len)
    g = torch.Generator().manual_seed(4)
    q = torch.randn(1, 4, 4, generator=g)

    out = paged_attention_decode_torch(q, k_pool, v_pool, block_table, seq_len=seq_len)
    k_full, v_full = materialize_kv(k_pool, v_pool, block_table, seq_len, group_size=2)
    expected = sdpa_reference(q, k_full, v_full)
    assert torch.allclose(out, expected, atol=1e-5)


def test_torch_batched_matches_per_request_outputs() -> None:
    """Batched reference output equals running each request through the single-request reference."""
    g = torch.Generator().manual_seed(11)

    # Three requests with different seq_lens and Qs, sharing the pool.
    seq_lens = [4, 7, 12]
    pool_args = {
        "num_blocks": 8,
        "block_size": 4,
        "num_kv_heads": 2,
        "head_dim": 4,
    }
    block_size = pool_args["block_size"]
    # Build one shared pool that has data for the longest request.
    k_pool, v_pool, _ = populate_pool(seq_len=max(seq_lens), seed=12, **pool_args)

    # Each request gets its own block table (just the prefix needed).
    block_tables = [
        torch.arange((seq_len + block_size - 1) // block_size, dtype=torch.int64)
        for seq_len in seq_lens
    ]
    qs = [torch.randn(1, 4, 4, generator=g) for _ in seq_lens]
    q_batch = torch.cat(qs, dim=0)

    out_batched = paged_attention_decode_torch_batched(
        q_batch, k_pool, v_pool, block_tables, seq_lens
    )

    expected = torch.cat(
        [
            paged_attention_decode_torch(
                qs[request_index],
                k_pool,
                v_pool,
                block_tables[request_index],
                seq_lens[request_index],
            )
            for request_index in range(len(seq_lens))
        ],
        dim=0,
    )
    assert torch.allclose(out_batched, expected, atol=1e-6)


def test_torch_batched_with_b_equals_one_matches_single_request() -> None:
    """Edge case: a 1-element batch must produce the same output as the single-request call."""
    seq_len = 8
    k_pool, v_pool, block_table = populate_pool(seq_len=seq_len, seed=21)
    g = torch.Generator().manual_seed(21)
    q = torch.randn(1, 4, 4, generator=g)

    out_single = paged_attention_decode_torch(q, k_pool, v_pool, block_table, seq_len=seq_len)
    out_batched = paged_attention_decode_torch_batched(q, k_pool, v_pool, [block_table], [seq_len])
    assert torch.allclose(out_single, out_batched, atol=1e-6)


def test_torch_batched_with_gqa_head_grouping() -> None:
    """Qwen2.5-0.5B-shaped batch (14 q heads, 2 kv heads, group_size=7) across 2 requests."""
    seq_lens = [8, 12]
    k_pool, v_pool, _ = populate_pool(
        num_blocks=8, block_size=4, num_kv_heads=2, head_dim=4, seq_len=max(seq_lens), seed=22
    )
    g = torch.Generator().manual_seed(22)
    block_size = 4
    block_tables = [
        torch.arange((seq_len + block_size - 1) // block_size, dtype=torch.int64)
        for seq_len in seq_lens
    ]
    q_batch = torch.cat([torch.randn(1, 14, 4, generator=g) for _ in seq_lens], dim=0)

    out = paged_attention_decode_torch_batched(q_batch, k_pool, v_pool, block_tables, seq_lens)
    assert out.shape == (2, 14, 4)
    # Each row corresponds to its own request; check that swapping rows doesn't match
    # (cheap proof the output is genuinely per-request, not duplicated).
    assert not torch.allclose(out[0], out[1], atol=1e-3)


def test_torch_batched_isolates_per_request_block_tables() -> None:
    """Two requests pointing at different (non-overlapping) blocks must read distinct K/V."""
    block_size = 4
    head_dim = 4
    num_kv_heads = 2
    # Hand-built pool: block 0 has K/V values = +1, block 1 has K/V values = -1.
    k_pool = torch.zeros(8, block_size, num_kv_heads, head_dim)
    v_pool = torch.zeros_like(k_pool)
    k_pool[0] = 1.0
    v_pool[0] = 1.0
    k_pool[1] = -1.0
    v_pool[1] = -1.0

    g = torch.Generator().manual_seed(23)
    q_batch = torch.cat([torch.randn(1, 2, head_dim, generator=g) for _ in range(2)], dim=0)
    block_tables = [torch.tensor([0], dtype=torch.int64), torch.tensor([1], dtype=torch.int64)]
    seq_lens = [block_size, block_size]

    out = paged_attention_decode_torch_batched(q_batch, k_pool, v_pool, block_tables, seq_lens)
    # Request 0 attends to all-ones V; output should be ~1. Request 1 attends to all-(-1) V; ~-1.
    assert torch.allclose(out[0], torch.ones_like(out[0]), atol=1e-5)
    assert torch.allclose(out[1], -torch.ones_like(out[1]), atol=1e-5)


def test_torch_batched_rejects_mismatched_lengths() -> None:
    """B inferred from q must match block_tables and seq_lens lengths."""
    g = torch.Generator().manual_seed(13)
    k_pool, v_pool, _ = populate_pool(seq_len=4, seed=13)
    q = torch.randn(2, 4, 4, generator=g)
    with pytest.raises(ValueError, match="B="):
        paged_attention_decode_torch_batched(q, k_pool, v_pool, [torch.tensor([0])], [4])


@pytest.mark.requires_cuda
def test_triton_matches_torch_reference() -> None:
    """Triton kernel output must match the torch reference within cosine sim > 0.99."""
    from mini_infer.cache.paged_attention import (
        paged_attention_decode_triton,
        supports_paged_kernel,
    )

    device = torch.device("cuda")
    if not supports_paged_kernel(device):
        pytest.skip("paged kernel not available on this device")

    seq_len = 16
    k_pool_cpu, v_pool_cpu, block_table_cpu = populate_pool(seq_len=seq_len, seed=7)
    g = torch.Generator().manual_seed(7)
    q_cpu = torch.randn(1, 4, 4, generator=g)

    q = q_cpu.to(device).to(torch.float16)
    k_pool = k_pool_cpu.to(device).to(torch.float16)
    v_pool = v_pool_cpu.to(device).to(torch.float16)
    block_table = block_table_cpu.to(device)

    out_triton = paged_attention_decode_triton(q, k_pool, v_pool, block_table, seq_len=seq_len)
    out_ref = paged_attention_decode_torch(
        q_cpu.float(),
        k_pool_cpu.float(),
        v_pool_cpu.float(),
        block_table_cpu,
        seq_len=seq_len,
    )
    sim = cosine_sim(out_triton.float().cpu(), out_ref)
    assert sim > 0.99, f"cosine sim {sim} below threshold"
