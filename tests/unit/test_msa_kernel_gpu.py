"""MSA Triton decode kernel vs the torch reference (CUDA, `-m gpu`).

The CPU suite pins `msa_paged_decode_torch` against the dense-mask oracle;
this pins the Triton kernel against that torch reference on GPU, at M3's real
decode shape (GQA group 16, head_dim 128) with ragged batches, -1 selection
padding, and a partial last block. Complements the Modal bench script
(`scripts/modal_msa_kernel_bench.py`) with a pytest any-GPU-runner form.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.msa_paged_attention import (
    msa_paged_decode_torch,
    msa_paged_decode_triton,
    supports_msa_kernel,
)

NUM_KV_HEADS = 4
NUM_Q_HEADS = 64  # group 16, M3's decode shape (tl.dot needs group >= 16)
HEAD_DIM = 128
POOL_BLOCK = 16
INDEX_BLOCK = 128  # 8 pool blocks per index block
TOPK = 4


def _random_pool_and_requests(
    seq_lens: list[int], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """A shared random pool plus per-request block tables and selections.

    Selections are per KV head (`(NUM_KV_HEADS, TOPK)`, transformers 5.14
    semantics) and drawn independently per head so a program reading another
    head's block list would fail the parity check."""
    total_blocks = sum(-(-s // POOL_BLOCK) for s in seq_lens) + 2
    k_pool = torch.randn(total_blocks, POOL_BLOCK, NUM_KV_HEADS, HEAD_DIM, device=device).to(dtype)
    v_pool = torch.randn(total_blocks, POOL_BLOCK, NUM_KV_HEADS, HEAD_DIM, device=device).to(dtype)
    block_tables = []
    selections = []
    next_block = 0
    for s in seq_lens:
        n_pool = -(-s // POOL_BLOCK)
        block_tables.append(
            torch.arange(next_block, next_block + n_pool, device=device, dtype=torch.int32)
        )
        next_block += n_pool
        n_index = -(-s // INDEX_BLOCK)
        local = n_index - 1  # the query's own block, always selected
        others = [b for b in range(n_index) if b != local]
        per_head = []
        for _ in range(NUM_KV_HEADS):
            perm = torch.randperm(len(others))[: TOPK - 1].tolist()
            sel = [local] + [others[i] for i in perm]
            sel += [-1] * (TOPK - len(sel))  # pad like the indexer does
            per_head.append(sel)
        selections.append(torch.tensor(per_head, device=device, dtype=torch.int64))
    return k_pool, v_pool, block_tables, selections


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_msa_kernel_matches_torch_reference(dtype: torch.dtype) -> None:
    if not (torch.cuda.is_available() and supports_msa_kernel("cuda")):
        pytest.skip("CUDA + Triton required for the MSA decode kernel")

    torch.manual_seed(0)
    device = torch.device("cuda")
    # Ragged batch; the last length exercises a partial final index block.
    seq_lens = [1, INDEX_BLOCK, 3 * INDEX_BLOCK + 37, 6 * INDEX_BLOCK + 1]
    k_pool, v_pool, block_tables, selections = _random_pool_and_requests(seq_lens, device, dtype)
    q = torch.randn(len(seq_lens), NUM_Q_HEADS, HEAD_DIM, device=device).to(dtype)

    ref = msa_paged_decode_torch(
        q, k_pool, v_pool, block_tables, seq_lens, selections, index_block_size=INDEX_BLOCK
    )
    out = msa_paged_decode_triton(
        q, k_pool, v_pool, block_tables, seq_lens, selections, index_block_size=INDEX_BLOCK
    )

    assert out.shape == ref.shape
    cs = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert float(cs) > 0.999, f"kernel vs torch reference cosine {float(cs):.6f}"
    atol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(out.float(), ref.float(), atol=atol), (
        f"max_abs_diff={(out.float() - ref.float()).abs().max().item():.6f}"
    )


@pytest.mark.gpu
def test_msa_kernel_small_group_falls_back_to_torch() -> None:
    """GQA group < 16 cannot use tl.dot; the launcher must return the torch result."""
    if not (torch.cuda.is_available() and supports_msa_kernel("cuda")):
        pytest.skip("CUDA + Triton required for the MSA decode kernel")

    torch.manual_seed(1)
    device = torch.device("cuda")
    seq_lens = [40]
    # group 2: under the MMA minimum.
    k_pool = torch.randn(4, POOL_BLOCK, 2, 32, device=device)
    v_pool = torch.randn(4, POOL_BLOCK, 2, 32, device=device)
    block_tables = [torch.arange(3, device=device, dtype=torch.int32)]
    selections = [torch.tensor([[0, -1], [0, 1]], device=device, dtype=torch.int64)]
    q = torch.randn(1, 4, 32, device=device)

    ref = msa_paged_decode_torch(
        q, k_pool, v_pool, block_tables, seq_lens, selections, index_block_size=POOL_BLOCK
    )
    out = msa_paged_decode_triton(
        q, k_pool, v_pool, block_tables, seq_lens, selections, index_block_size=POOL_BLOCK
    )
    assert torch.allclose(out, ref, atol=1e-6)
