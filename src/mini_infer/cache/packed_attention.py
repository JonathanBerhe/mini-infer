"""Packed-sequence attention for chunked prefill + batched decode in one forward.

Each engine step packs q-tokens from all in-flight requests into a single packed
sequence (a prefilling request contributes `chunk_size` tokens; a decoding
request contributes 1). Attention is varlen-aware: per-request boundaries are
defined by `cu_seqlens_q` / `cu_seqlens_k`, and each request's queries can only
attend to keys within its own request, with causal ordering inside the request.

Two backends:
- `packed_attention_flash`: CUDA + FlashAttention's `flash_attn_varlen_func`. Fast.
- `packed_attention_torch`: pure PyTorch reference looping per request through
  a standard SDPA call. Slow but device-agnostic and the numerical oracle the
  flash backend is validated against.

Use `packed_attention_forward(...)` as the entry point; it dispatches to the
fastest available implementation for the device.
"""

import math

import torch

try:
    from flash_attn import flash_attn_varlen_func

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_func = None


def supports_packed_kernel(device: torch.device | str) -> bool:
    """Whether `flash_attn_varlen_func` can run on this device.

    Single source of truth so we don't sprinkle device checks across modules.
    Today: CUDA only (FlashAttention requires Ampere+ for FA2/3, Hopper+ for FA4).
    """
    if not _FLASH_ATTN_AVAILABLE:
        return False
    if isinstance(device, str):
        return device == "cuda"
    return device.type == "cuda"


def packed_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Varlen attention over packed-q and packed-k/v with causal masking per request.

    Shapes:
        q             : (total_q, num_q_heads, head_dim)  — packed across requests
        k             : (total_k, num_kv_heads, head_dim) — packed across requests
        v             : (total_k, num_kv_heads, head_dim)
        cu_seqlens_q  : (B+1,) int — cumulative q lengths
        cu_seqlens_k  : (B+1,) int — cumulative k lengths (full per-request K/V history)
        max_seqlen_q  : int — max query length in the batch
        max_seqlen_k  : int — max key length in the batch
        softmax_scale : float, defaults to 1/sqrt(head_dim)

    Returns: (total_q, num_q_heads, head_dim) attention output.

    GQA is handled implicitly: when `num_q_heads != num_kv_heads`, the K/V are
    broadcast to match the Q head count (group_size = num_q_heads / num_kv_heads).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if supports_packed_kernel(q.device):
        return _packed_attention_flash(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale
        )
    return packed_attention_torch(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)


def packed_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Pure PyTorch reference: per-request SDPA over the packed sequences.

    Loops over the batch dim, slices each request's q and k/v out of the packed
    tensors, computes causal SDPA with GQA broadcasting, and writes back to the
    output's per-request slice. Slow (Python loop) but trivially correct;
    serves as the oracle for the flash backend.

    Causal-mask convention for chunked prefill: a request's `q_len` tokens
    occupy positions `[k_len - q_len, k_len)` within its full K/V history (the
    new tokens are the most recently appended). Each query at intra-request
    position `i` attends to keys `0..(k_len - q_len + i)`.
    """
    out = torch.empty_like(q)
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads={num_q_heads} not a multiple of num_kv_heads={num_kv_heads}")
    group_size = num_q_heads // num_kv_heads

    batch_size = cu_seqlens_q.shape[0] - 1
    if cu_seqlens_k.shape[0] - 1 != batch_size:
        raise ValueError(
            f"cu_seqlens_q has {batch_size + 1} entries but cu_seqlens_k has "
            f"{cu_seqlens_k.shape[0]} entries"
        )

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
                f"queries must fit inside their own K/V history"
            )

        q_b = q[q_start:q_end]
        k_b = k[k_start:k_end]
        v_b = v[k_start:k_end]
        if group_size > 1:
            k_b = k_b.repeat_interleave(group_size, dim=1)
            v_b = v_b.repeat_interleave(group_size, dim=1)

        # Compute attention in fp32 for numerical stability.
        scores = torch.einsum("qhd,khd->qhk", q_b.float(), k_b.float()) * softmax_scale

        # Causal mask: query at intra-request position i = (k_len - q_len + i)
        # in absolute K terms. It attends to K positions [0, k_len - q_len + i].
        q_offset = k_len - q_len
        q_abs_positions = torch.arange(q_len, device=q.device) + q_offset
        k_positions = torch.arange(k_len, device=q.device)
        invalid = k_positions[None, :] > q_abs_positions[:, None]  # (q_len, k_len)
        scores = scores.masked_fill(invalid[:, None, :], -float("inf"))

        weights = torch.softmax(scores, dim=-1)
        out[q_start:q_end] = torch.einsum("qhk,khd->qhd", weights, v_b.float()).to(q.dtype)

    return out


def _packed_attention_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """FlashAttention varlen call. CUDA only.

    Inputs must be on a CUDA device with a half-precision dtype (fp16 / bf16 —
    FA's varlen API doesn't accept fp32). cu_seqlens are int32; FlashAttention
    requires this exact dtype.
    """
    if not _FLASH_ATTN_AVAILABLE:
        raise RuntimeError(
            "flash-attn not installed; install via `uv add flash-attn` or use "
            "packed_attention_torch instead"
        )
    if q.device.type != "cuda":
        raise RuntimeError("flash_attn_varlen_func requires CUDA tensors")
    out: torch.Tensor = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
    )
    return out
