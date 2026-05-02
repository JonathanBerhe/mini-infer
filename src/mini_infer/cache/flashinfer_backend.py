"""FlashInfer paged-attention backend for mini-infer.

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the kernel
library that powers vLLM and SGLang. It exposes paged attention with
the same `(num_pages, page_size, num_kv_heads, head_dim)` NHD layout
mini-infer's bf16 `BlockPool._storage` already uses, and is selectable
via `attention_backend="flashinfer"` on the engine.

A single `BatchPrefillWithPagedKVCacheWrapper` and
`BatchDecodeWithPagedKVCacheWrapper` share one persistent 128 MiB
uint8 workspace buffer per process, lazy-initialized on first use.
`plan()` runs once per attention call (per layer per step) and is
cheap; `run()` executes the attention math against the paged K/V
cache.

CUDA-only. Imports are guarded so non-CUDA installs (M1, CI) still
load this module without flashinfer-python installed.
"""

from __future__ import annotations

from typing import Any

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.device import is_cuda_device, require_cuda_device

try:
    import flashinfer  # type: ignore[import-not-found]

    _FLASHINFER_AVAILABLE = True
except ImportError:  # macOS / no-CUDA / linux without the wheel installed
    _FLASHINFER_AVAILABLE = False
    flashinfer = None


# Runtime toggle for A/B benchmarking on the same model load. When True,
# `supports_flashinfer_backend` reports False even on CUDA, forcing
# `packed_attention_forward` through the flash-attn paths. Mirrors the
# `_FUSED_DISABLED_FOR_BENCH` pattern in `turbo_kernel.py`.
_FLASHINFER_DISABLED_FOR_BENCH = False

# Persistent workspace shared by the prefill and decode wrappers. 128 MiB
# is FlashInfer's recommended default for moderate batch sizes; we
# allocate lazily on first use rather than at module load so non-CUDA
# imports stay free.
_WORKSPACE_BYTES = 128 * 1024 * 1024
_workspace_buffer: torch.Tensor | None = None
_prefill_wrapper: Any = None
_decode_wrapper: Any = None


def supports_flashinfer_backend(device: torch.device | str) -> bool:
    """Whether the FlashInfer paged-attention backend can run on this device.

    Today: CUDA + flashinfer-python only. Non-CUDA falls through to the
    PyTorch reference path inside `packed_attention_forward`.
    `_FLASHINFER_DISABLED_FOR_BENCH` overrides to False for A/B benchmarks.
    """
    if _FLASHINFER_DISABLED_FOR_BENCH:
        return False
    if not _FLASHINFER_AVAILABLE:
        return False
    return is_cuda_device(device)


def _ensure_wrappers(device: torch.device) -> tuple[Any, Any]:
    """Lazy-initialize the persistent workspace + wrapper instances.

    Returns `(prefill_wrapper, decode_wrapper)` for use in the
    dispatcher. Both share a single 128 MiB `uint8` buffer because their
    plan/run lifecycles don't overlap — only one is "current" per
    forward pass.
    """
    global _workspace_buffer, _prefill_wrapper, _decode_wrapper
    if _workspace_buffer is None:
        _workspace_buffer = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    assert flashinfer is not None  # _FLASHINFER_AVAILABLE checked at call site
    if _prefill_wrapper is None:
        _prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace_buffer, kv_layout="NHD"
        )
    if _decode_wrapper is None:
        _decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _workspace_buffer, kv_layout="NHD"
        )
    return _prefill_wrapper, _decode_wrapper


def flashinfer_attention_forward(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Paged attention via FlashInfer.

    Same contract as `_packed_attention_paged_flash` in `packed_attention.py`:
    given `q` packed across requests and a `PagedKVCache`, produce the
    causal-masked attention output. Routes through FlashInfer's
    decode-only wrapper if all q_lens == 1, otherwise through the
    prefill wrapper (which handles mixed prefill/decode batches via
    `qo_indptr`).

    Args:
        q: `(total_q, num_q_heads, head_dim)` bf16/fp16 packed across
            requests. Must be CUDA-resident and contiguous.
        cache: `PagedKVCache` whose underlying pool is uncompressed
            (FlashInfer reads K/V directly from the bf16 storage).
        layer_idx: which layer's K/V to attend over.
        cu_seqlens_q: `(B+1,)` int cumulative q boundaries.
        softmax_scale: typically `1/sqrt(head_dim)`.

    Returns:
        `(total_q, num_q_heads, head_dim)` attention output.
    """
    if not _FLASHINFER_AVAILABLE:
        raise RuntimeError(
            "flashinfer-python not installed; install via the [cuda] extra "
            "or use a different attention_backend"
        )
    require_cuda_device(q.device, "FlashInfer paged attention")

    if cache._pool.kv_quant is not None:
        raise RuntimeError(
            "FlashInfer backend (Stage 1) only supports the uncompressed "
            f"bf16 pool layout; got kv_quant={cache._pool.kv_quant!r}. "
            "Use the materialized path until FP8/NVFP4 modes ship."
        )

    device = q.device
    num_q_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = cache._pool.num_kv_heads
    page_size = cache._pool.block_size

    prefill_wrapper, decode_wrapper = _ensure_wrappers(device)

    # CSR-style page index triple. FlashInfer wants:
    #   paged_kv_indptr     (B+1,) int32 — cumulative pages per request
    #   paged_kv_indices    (sum_pages,) int32 — flat block IDs
    #   paged_kv_last_page_len (B,) int32 — tokens valid in each
    #       request's last page (must be in [1, page_size]; pages with
    #       seq_len % page_size == 0 report page_size, not 0).
    seq_lens = cache.seq_lens_list()
    indptr = [0]
    indices: list[int] = []
    last_page_lens: list[int] = []
    running = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        if seq_len == 0:
            indptr.append(running)
            last_page_lens.append(0)
            continue
        block_ids = cache._block_ids[batch_idx]
        num_pages = (seq_len + page_size - 1) // page_size
        indices.extend(block_ids[:num_pages])
        running += num_pages
        indptr.append(running)
        rem = seq_len % page_size
        last_page_lens.append(page_size if rem == 0 else rem)

    paged_kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor(indices, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor(last_page_lens, dtype=torch.int32, device=device)
    qo_indptr = cu_seqlens_q.to(dtype=torch.int32, device=device)

    # Layer slice from the bf16 pool. For uncompressed turbo=None pools,
    # storage shape is `(num_layers, 2, num_blocks, block_size,
    # num_kv_heads, head_dim)`. The per-layer slice is `(2, num_blocks,
    # ...)`; FlashInfer's wrapper accepts K and V as a `(k, v)` tuple
    # (each `(num_blocks, page_size, num_kv_heads, head_dim)` NHD), so
    # we slice without copying.
    k_cache = cache._pool.storage[layer_idx, 0]
    v_cache = cache._pool.storage[layer_idx, 1]
    kv_cache: tuple[torch.Tensor, torch.Tensor] | torch.Tensor = (k_cache, v_cache)

    # Always go through the prefill wrapper. FlashInfer's decode wrapper
    # is faster but only supports power-of-2 GQA group sizes (1, 2, 4,
    # 8, 16); models like Qwen2.5 use group_size=7 which the decode
    # kernel rejects. The prefill wrapper handles arbitrary group_size
    # via `qo_indptr` (q_len=1 per request collapses to the decode case
    # internally).
    _ = decode_wrapper  # kept for future re-enable on power-of-2 GQA models
    prefill_wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        sm_scale=softmax_scale,
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
    )
    out: torch.Tensor = prefill_wrapper.run(q, kv_cache)
    return out
