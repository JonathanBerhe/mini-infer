"""Varlen-packed Multi-head Latent Attention SDPA reference.

Standard `packed_attention_torch` assumes Q, K, V share a head dim. MLA
breaks that assumption: Q / K carry `qk_head_dim = qk_nope_head_dim +
qk_rope_head_dim` (=192 for V2-Lite) while V carries `v_head_dim` (=128).
Attention scores need the qk dim; output uses the v dim.

flash-attn 2 and FlashInfer's prefill kernel both assume symmetric Q/K/V
head_dim and reject this layout. vLLM and SGLang route MLA layers
through their Triton unified-attention kernel; we don't ship a Triton
kernel, so this PyTorch reference is the path. Per-request loop over
the cu_seqlens_q boundaries — slow but trivially correct, used as the
oracle for any future fast kernel.
"""

from __future__ import annotations

import torch


def mla_packed_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Per-request causal SDPA with asymmetric V head_dim.

    Shapes:
      q   : `(total_q, num_heads, qk_head_dim)`
      k   : `(total_k, num_heads, qk_head_dim)`  -- broadcast already applied
      v   : `(total_k, num_heads, v_head_dim)`
      cu_seqlens_q, cu_seqlens_k : `(batch + 1,)` int32

    Returns `(total_q, num_heads, v_head_dim)`.

    Causal mask is identical to `packed_attention_torch`: query at
    intra-request position `i` (where the request has `q_len` queries
    and `k_len` cached keys, `q_len <= k_len`) attends to keys
    `0..(k_len - q_len + i)`.
    """
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(
            f"q/k/v must share num_heads; got q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}"
        )
    if q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k must share qk_head_dim; got q={q.shape[2]}, k={k.shape[2]}")
    if cu_seqlens_q.shape[0] != cu_seqlens_k.shape[0]:
        raise ValueError(
            f"cu_seqlens_q has {cu_seqlens_q.shape[0]} entries but "
            f"cu_seqlens_k has {cu_seqlens_k.shape[0]}"
        )
    batch_size = cu_seqlens_q.shape[0] - 1
    out = torch.empty(q.shape[0], q.shape[1], v.shape[2], dtype=q.dtype, device=q.device)

    for batch_idx in range(batch_size):
        q_start = int(cu_seqlens_q[batch_idx])
        q_end = int(cu_seqlens_q[batch_idx + 1])
        k_start = int(cu_seqlens_k[batch_idx])
        k_end = int(cu_seqlens_k[batch_idx + 1])
        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len == 0:
            continue
        if q_len > k_len:
            raise ValueError(
                f"request {batch_idx}: q_len={q_len} > k_len={k_len}; "
                "queries must fit inside their own K/V history"
            )

        q_b = q[q_start:q_end]
        k_b = k[k_start:k_end]
        v_b = v[k_start:k_end]
        # fp32 attention math for numerical stability.
        scores = torch.einsum("qhd,khd->qhk", q_b.float(), k_b.float()) * softmax_scale
        # Causal mask: intra-request position i attends to keys 0..(k_len - q_len + i).
        q_offset = k_len - q_len
        q_abs_positions = torch.arange(q_len, device=q.device) + q_offset
        k_positions = torch.arange(k_len, device=q.device)
        invalid = k_positions[None, :] > q_abs_positions[:, None]
        scores = scores.masked_fill(invalid[:, None, :], -float("inf"))
        weights = torch.softmax(scores, dim=-1)
        # einsum on V uses v's distinct head_dim (different from q/k).
        out[q_start:q_end] = torch.einsum("qhk,khd->qhd", weights, v_b.float()).to(q.dtype)

    return out
