import pytest
import torch

from mini_infer.cache.paged_attention import paged_attention_decode_torch
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
