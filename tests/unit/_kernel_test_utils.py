"""Test helpers for paged-attention numerical correctness checks.

Module name starts with `_` so pytest doesn't collect it as a test file. Other
tests import from here directly.

"SDPA" = Scaled Dot-Product Attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`. The
canonical attention computation that PyTorch exposes as
`torch.nn.functional.scaled_dot_product_attention`. We use a hand-written SDPA
here as the math oracle our paged kernel must match.
"""

import math

import torch


def sdpa_reference(
    q: torch.Tensor,  # (batch, num_q_heads, head_dim)
    k_full: torch.Tensor,  # (seq_len, num_q_heads, head_dim) after GQA broadcast
    v_full: torch.Tensor,  # same shape as k_full
) -> torch.Tensor:
    """Plain Scaled Dot-Product Attention over an already-materialized contiguous K/V."""
    head_dim = q.shape[-1]
    q_f = q.float()
    k_f = k_full.float()
    v_f = v_full.float()
    scores = torch.einsum("bhd,shd->bhs", q_f, k_f) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,shd->bhd", weights, v_f)
    return out.to(q.dtype)


def materialize_kv(
    k_pool_layer: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    v_pool_layer: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manually rebuild contiguous K/V from blocks; broadcast KV heads to Q heads (GQA)."""
    block_size = k_pool_layer.shape[1]
    positions = torch.arange(seq_len, device=k_pool_layer.device)
    block_ids = block_table[positions // block_size]
    slots = positions % block_size
    k = k_pool_layer[block_ids, slots]
    v = v_pool_layer[block_ids, slots]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    return k, v


def populate_pool(
    *,
    num_blocks: int = 8,
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 4,
    seq_len: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill `seq_len` worth of K/V across the first ceil(seq_len/block_size) blocks."""
    g = torch.Generator().manual_seed(seed)
    k_pool = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    v_pool = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    num_blocks_used = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(num_blocks_used, dtype=torch.int64)
    for pos in range(seq_len):
        b = pos // block_size
        s = pos % block_size
        k_pool[b, s] = torch.randn(num_kv_heads, head_dim, generator=g)
        v_pool[b, s] = torch.randn(num_kv_heads, head_dim, generator=g)
    return k_pool, v_pool, block_table


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
