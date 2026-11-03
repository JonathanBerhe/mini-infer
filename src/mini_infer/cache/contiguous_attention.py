"""Attention over contiguous per-row K/V buffers, read only up to each row's length.

The state-cache families keep a full-attention layer's history as one
contiguous `(B, max_seq_len, num_kv_heads, head_dim)` buffer per row (see
`Qwen35StateCache`). FlashAttention's `flash_attn_with_kvcache` reads such a
buffer in place: it stops at each row's `cache_seqlens` and splits a long
context across thread blocks. The shapes do not depend on the lengths, so a
decode step that calls it can still be captured in a CUDA graph.

CPU and MPS have no FlashAttention. There the model keeps its own PyTorch
path, which `supports_contiguous_kernel` tells it to do.
"""

from __future__ import annotations

import torch

from mini_infer.device import is_cuda_device, require_cuda_device

try:
    from flash_attn import flash_attn_with_kvcache

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False
    flash_attn_with_kvcache = None

# FlashAttention runs in half precision only.
_HALF_DTYPES = (torch.float16, torch.bfloat16)


def supports_contiguous_kernel(k_buffer: torch.Tensor) -> bool:
    """Whether `contiguous_attention` can run on this buffer's device and dtype.

    Single source of truth for the check, so the model does not test devices
    itself. Today: CUDA with flash-attn installed and a half-precision cache.
    """
    if not _FLASH_ATTN_AVAILABLE:
        return False
    return is_cuda_device(k_buffer.device) and k_buffer.dtype in _HALF_DTYPES


def contiguous_attention(
    query: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Causal grouped-query attention of a span over each row's cached keys.

    The span's own K/V must already be in the buffers. Row `b` attends to its
    first `lengths[b]` keys, and the span's last query sits at the last of
    them, so query `i` of a span of `T` sees keys up to `lengths[b] - T + i`.

    Args:
        query: `(B, T, num_attention_heads, head_dim)`.
        k_buffer: `(B, max_seq_len, num_kv_heads, head_dim)` stored keys.
        v_buffer: `(B, max_seq_len, num_kv_heads, head_dim)` stored values.
        lengths: `(B,)` integer cache length of each row after the span, on
            the buffers' device.
        scale: softmax scale.

    Returns:
        `(B, T, num_attention_heads, head_dim)` in the query's dtype.
    """
    if not _FLASH_ATTN_AVAILABLE:
        raise RuntimeError("flash-attn not installed; install via the [cuda] extra")
    require_cuda_device(query.device, "flash_attn_with_kvcache")
    out: torch.Tensor = flash_attn_with_kvcache(
        query.to(k_buffer.dtype),
        k_buffer,
        v_buffer,
        cache_seqlens=lengths.to(torch.int32),
        softmax_scale=scale,
        causal=True,
    )
    return out.to(query.dtype)
