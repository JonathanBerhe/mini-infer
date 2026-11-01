"""FlashInfer paged-attention backend for mini-infer.

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the kernel
library that powers vLLM and SGLang. It exposes paged attention with
the same `(num_pages, page_size, num_kv_heads, head_dim)` NHD layout
mini-infer's bf16 `BlockPool._storage` already uses, and is selectable
via `attention_backend="flashinfer"` on the engine.

A single `BatchPrefillWithPagedKVCacheWrapper` and
`BatchDecodeWithPagedKVCacheWrapper` share one persistent 128 MiB
uint8 workspace buffer per process, lazy-initialized on first use.
`plan()` runs once per step: the paged cache builds its page index once
per step, and `flashinfer_attention_forward` plans again only when that
index, the query boundaries or the attention parameters changed. A plan
per layer made the host wait for the GPU several times per layer.
`run()` executes the attention math against the paged K/V cache, once
per layer.

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
# One prefill wrapper for each KV storage class (the pool's `kv_quant`). FlashInfer
# `plan()` resolves backend="auto" once, from the dtypes of its first plan, and stores
# the choice on the wrapper. A shared wrapper thus pins every later pool to that backend.
# On Hopper, a bf16 plan pins fa3 (FlashAttention 3). A later fp8 plan then JIT-builds
# an fa3 template for bf16 q and e4m3 KV, which does not exist, and fails with cutlass
# "No eligible GMMA operator" (H100, 0.6.16rc4, 2026-07-29). In the reverse order, bf16
# pools lose fa3 silently. Per-class wrappers give fa2 for fp8 KV and fa3 for bf16.
_prefill_wrappers: dict[str, Any] = {}
_decode_wrapper: Any = None
# The last plan: (wrapper, page indptr tensor, query boundaries, attention parameters).
# `flashinfer_attention_forward` skips `plan()` while these are unchanged, so one plan
# per step serves every layer. The engine runs one forward at a time, so one slot is
# enough.
_last_plan: tuple[Any, torch.Tensor, tuple[int, ...], tuple[Any, ...]] | None = None


def _planned(
    wrapper: Any, indptr: torch.Tensor, qo_indptr: tuple[int, ...], params: tuple[Any, ...]
) -> bool:
    """Whether `wrapper` already holds the plan for this page index and these parameters.

    The cache rebuilds its page index when the layout changes, so the same `indptr`
    object means the same step of the same cache. `_last_plan` keeps a reference to
    the tensor, so its identity cannot pass to another tensor while it is the key.
    """
    last = _last_plan
    return (
        last is not None
        and last[0] is wrapper
        and last[1] is indptr
        and last[2] == qo_indptr
        and last[3] == params
    )


def _wrapper_key(kv_quant: str | None) -> str:
    """Return the prefill-wrapper key: the KV storage class of the pool.

    In one class the (q, kv) dtypes never change, so the first-plan backend stays correct.
    """
    return kv_quant if kv_quant is not None else "default"


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


def _ensure_wrappers(device: torch.device, kv_quant: str | None) -> tuple[Any, Any]:
    """Lazy-initialize the persistent workspace + wrapper instances.

    Returns `(prefill_wrapper, decode_wrapper)` for use in the
    dispatcher. All wrappers share one 128 MiB `uint8` buffer. Their uses never overlap,
    because the engine runs one forward at a time and each attention call runs one wrapper.
    Prefill wrappers are cached for each KV storage class (see `_prefill_wrappers`).
    """
    global _workspace_buffer, _decode_wrapper
    if _workspace_buffer is None:
        _workspace_buffer = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    assert flashinfer is not None  # _FLASHINFER_AVAILABLE checked at call site
    key = _wrapper_key(kv_quant)
    if key not in _prefill_wrappers:
        _prefill_wrappers[key] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace_buffer, kv_layout="NHD"
        )
    if _decode_wrapper is None:
        _decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _workspace_buffer, kv_layout="NHD"
        )
    return _prefill_wrappers[key], _decode_wrapper


def flashinfer_attention_forward(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window: int | None = None,
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
        window: when set, enables sliding-window attention via
            FlashInfer's `window_left=window-1` plan arg (each query
            attends to the most recent `window` keys). `None` =
            full causal.

    Returns:
        `(total_q, num_q_heads, head_dim)` attention output.
    """
    global _last_plan
    if not _FLASHINFER_AVAILABLE:
        raise RuntimeError(
            "flashinfer-python not installed; install via the [cuda] extra "
            "or use a different attention_backend"
        )
    require_cuda_device(q.device, "FlashInfer paged attention")

    if cache._pool.kv_quant not in (None, "fp8", "nvfp4"):
        raise RuntimeError(
            f"FlashInfer backend supports kv_quant in (None, 'fp8', 'nvfp4'); "
            f"got {cache._pool.kv_quant!r}."
        )

    device = q.device
    num_q_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = cache._pool.num_kv_heads_for_layer(layer_idx)
    page_size = cache._pool.block_size

    prefill_wrapper, decode_wrapper = _ensure_wrappers(device, cache._pool.kv_quant)

    # The CSR page index triple that FlashInfer wants, built by the cache once
    # per step and shared by every layer (see `PagedKVCache.page_index`).
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = cache.page_index(device)

    # Layer slice. For uncompressed pools the storage is bf16; for fp8
    # pools we read from `_fp8_storage` and pass per-head scales to
    # FlashInfer's run() so the kernel dequants on-the-fly. For nvfp4
    # pools we additionally pass per-block FP8 scales via `kv_cache_sf`
    # (the V side has TRT-LLM's 4-token swizzle baked in by the writer).
    kv_cache_sf: tuple[torch.Tensor, torch.Tensor] | None = None
    if cache._pool.kv_quant == "fp8":
        assert cache._pool._fp8_storage is not None
        assert cache._pool._fp8_scales is not None
        k_cache = cache._pool._fp8_storage[layer_idx, 0]
        v_cache = cache._pool._fp8_storage[layer_idx, 1]
        kv_data_type: torch.dtype = torch.float8_e4m3fn
        # FlashInfer's `run()` takes k_scale and v_scale as Python floats
        # (per-tensor scale). Collapse our per-head scales to the head-wise
        # max so no head saturates on dequant; precision for low-magnitude
        # heads is slightly worse than a true per-head path would give.
        k_scale: float | None = float(cache._pool._fp8_scales[layer_idx, 0].max().item())
        v_scale: float | None = float(cache._pool._fp8_scales[layer_idx, 1].max().item())
    elif cache._pool.kv_quant == "nvfp4":
        assert cache._pool._nvfp4_storage is not None
        assert cache._pool._nvfp4_block_scales is not None
        assert cache._pool._nvfp4_global_sf is not None
        assert cache._pool._nvfp4_initialized is not None
        k_cache = cache._pool._nvfp4_storage[layer_idx, 0]
        v_cache = cache._pool._nvfp4_storage[layer_idx, 1]
        kv_cache_sf = (
            cache._pool._nvfp4_block_scales[layer_idx, 0],
            cache._pool._nvfp4_block_scales[layer_idx, 1],
        )
        # FlashInfer's plan() reads the packed-FP4 storage as uint8.
        kv_data_type = torch.uint8
        # k_scale / v_scale carry the per-(layer, side) global scale as
        # Python floats. Convention from `nvfp4_quantize_paged_kv_cache`:
        # k_scale = 1 / k_global_sf so the kernel multiplies on dequant.
        if not bool(cache._pool._nvfp4_initialized[layer_idx, 0]) or not bool(
            cache._pool._nvfp4_initialized[layer_idx, 1]
        ):
            raise RuntimeError(
                f"NVFP4 attention before any K/V written for layer {layer_idx}; "
                "the writer initializes per-(layer, side) global scales on first append."
            )
        k_scale = 1.0 / float(cache._pool._nvfp4_global_sf[layer_idx, 0].item())
        v_scale = 1.0 / float(cache._pool._nvfp4_global_sf[layer_idx, 1].item())
    else:
        k_cache, v_cache = cache._pool.storage_for_layer(layer_idx)
        kv_data_type = q.dtype
        k_scale = None
        v_scale = None
    kv_cache: tuple[torch.Tensor, torch.Tensor] | torch.Tensor = (k_cache, v_cache)

    # Always go through the prefill wrapper. FlashInfer's decode wrapper
    # is faster but only supports power-of-2 GQA group sizes (1, 2, 4,
    # 8, 16); models like Qwen2.5 use group_size=7 which the decode
    # kernel rejects. The prefill wrapper handles arbitrary group_size
    # via `qo_indptr` (q_len=1 per request collapses to the decode case
    # internally).
    _ = decode_wrapper  # kept for future re-enable on power-of-2 GQA models
    # One plan per step. The page index and the query boundaries are the same
    # for every layer, and `plan()` makes the host wait for the GPU several
    # times, so the layers after the first reuse the plan. A layer with other
    # parameters (a sliding window, other head counts) plans again.
    qo_indptr_list = tuple(cache.cu_seqlens_q_list(cu_seqlens_q))
    params = (
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q.dtype,
        kv_data_type,
        window,
        softmax_scale,
    )
    if not _planned(prefill_wrapper, paged_kv_indptr, qo_indptr_list, params):
        prefill_wrapper.plan(
            cu_seqlens_q.to(dtype=torch.int32, device=device),
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
            kv_data_type=kv_data_type,
            window_left=(window - 1) if window is not None else -1,
        )
        _last_plan = (prefill_wrapper, paged_kv_indptr, qo_indptr_list, params)
    if kv_cache_sf is not None:
        out: torch.Tensor = prefill_wrapper.run(
            q, kv_cache, k_scale=k_scale, v_scale=v_scale, kv_cache_sf=kv_cache_sf
        )
    elif k_scale is not None:
        out = prefill_wrapper.run(q, kv_cache, k_scale=k_scale, v_scale=v_scale)
    else:
        out = prefill_wrapper.run(q, kv_cache)
    return out
