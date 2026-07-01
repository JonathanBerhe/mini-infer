"""Packed-sequence attention for chunked prefill + batched decode in one forward.

Each engine step packs q-tokens from all in-flight requests into a single packed
sequence (a prefilling request contributes `chunk_size` tokens; a decoding
request contributes 1). Attention is varlen-aware: per-request boundaries are
defined by `cu_seqlens_q` / per-request `seq_lens`, and each request's queries
can only attend to keys within its own request, with causal ordering inside
the request.

Two backends, both selected via `packed_attention_forward(...)`:

- **CUDA (FlashAttention 2.8+)**: paged varlen call. The kernel reads K/V
  directly from `BlockPool` storage via `block_table` — no per-layer gather.
- **CPU / MPS (PyTorch reference)**: per-request SDPA loop after gathering
  per-request K/V into a packed buffer. Slow but correct, and the numerical
  oracle that the FA backend is validated against.
"""

import math

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.device import is_cuda_device, require_cuda_device

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
    return is_cuda_device(device)


# FlashAttention's paged varlen API (`flash_attn_varlen_func` + `block_table`)
# requires the K/V cache block size to be a multiple of this value. Smaller
# block sizes still work, but go through the materialized varlen path (no
# direct paged read; one gather per layer per step).
_FA_PAGED_BLOCK_SIZE_MULTIPLE = 256


def packed_attention_forward(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float | None = None,
    block_mask: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Varlen attention over packed-q with K/V read from a `PagedKVCache`.

    The cache is expected to have already had this step's new K/V appended via
    `cache.append_kv_packed(...)` before this call — i.e. for every request,
    `cache.seq_lens_list()[batch_idx]` is the post-append K-length.

    Per-layer sliding-window attention is honored automatically via
    `cache._pool.layer_attention[layer_idx]`: `"full"` runs unrestricted
    causal attention; `("sliding", w)` clamps each query at position `q`
    to attend only to keys at positions `[q - w + 1, q]`.

    Three backends are picked automatically:

    - **CUDA + `block_size % 256 == 0`**: paged FA varlen via `block_table`.
      No per-layer gather; reads K/V directly from `BlockPool` storage.
    - **CUDA + other block_size**: materialized FA varlen. Per-layer gather
      into a contiguous packed buffer, then `flash_attn_varlen_func` without
      `block_table`. Compatible with any block size; this is the original
      Slice B path.
    - **CPU / MPS**: PyTorch reference (per-request SDPA loop after gather).

    Args:
        q: `(total_q, num_q_heads, head_dim)` packed across requests.
        cache: shared `PagedKVCache` with `batch_size == B`.
        layer_idx: which layer's K/V to attend over.
        cu_seqlens_q: `(B+1,)` int cumulative q boundaries; `cu_seqlens_q[-1]`
            equals `total_q`.
        softmax_scale: defaults to `1/sqrt(head_dim)`.
        block_mask: optional per-request additive attention bias, one
            `(q_len_r, 1, k_len_r)` tensor per request (0 where a key is kept,
            `finfo.min` where masked). Used by MiniMax-M3 MSA: the bias folds the
            block selection AND causality together, so it REPLACES the built-in
            causal fill. Only the `torch` backend can consume it (the CUDA fast
            paths take no per-query mask); passing it with another backend raises.

    Returns:
        `(total_q, num_q_heads, head_dim)` attention output. GQA is handled by
        broadcasting on the K head count when `num_q_heads != num_kv_heads`.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if block_mask is not None and cache._pool.attention_backend != "torch":
        raise ValueError(
            "block_mask (MSA block-sparse bias) requires the 'torch' attention "
            f"backend; got {cache._pool.attention_backend!r}"
        )

    # Per-layer attention pattern. `None` window = full causal (every
    # current model except Gemma family); positive int = sliding window
    # of that size. The dispatcher passes it through to whichever backend
    # runs, each of which has a native SWA arg.
    layer_spec = cache._pool.layer_attention[layer_idx]
    window: int | None = layer_spec[1] if isinstance(layer_spec, tuple) else None

    # Forced-torch backend: skip every CUDA fast path and run the
    # materialized SDPA reference. The reference path handles arbitrary
    # head_dim (no kernel-tile constraints), which is what models like
    # Gemma 4 31B (head_dim=512 on full layers) need today — neither
    # FlashInfer's prefill nor flash-attn 2 supports head_dim > 256.
    # vLLM and SGLang reach the same conclusion via their Triton unified
    # attention kernel; we don't ship a Triton kernel, so we accept the
    # materialized-SDPA tax for the whole model. See ADR / model file
    # `Gemma4ForCausalLM.required_attention_backend`.
    if cache._pool.attention_backend == "torch":
        keys_packed, values_packed, cu_seqlens_k, _ = cache.materialize_packed_kv(layer_idx)
        return packed_attention_torch(
            q,
            keys_packed,
            values_packed,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            window=window,
            block_mask=block_mask,
        )

    # FlashInfer paged-attention backend (opt-in via the pool's
    # `attention_backend="flashinfer"`). Handles bf16, fp8 e4m3fn, and
    # nvfp4 paged storage (FlashInfer fuses dequant into the tensor-core
    # kernel for the quantized modes). Hoisted ahead of the flash-attn
    # gate because FlashInfer is independent of flash-attn — Blackwell
    # builds ship without flash-attn (no torch-2.6 wheel) but FlashInfer
    # still works.
    from mini_infer.cache.flashinfer_backend import (
        flashinfer_attention_forward,
        supports_flashinfer_backend,
    )

    if (
        cache._pool.attention_backend == "flashinfer"
        and supports_flashinfer_backend(q.device)
        and cache._pool.kv_quant in (None, "fp8", "nvfp4")
    ):
        return flashinfer_attention_forward(
            q, cache, layer_idx, cu_seqlens_q, softmax_scale, window=window
        )

    if supports_packed_kernel(q.device):
        from mini_infer.cache.turbo_kernel import (
            fused_turbo_attention_decode,
            supports_fused_attn_kernel,
        )

        block_size = cache._pool.block_size
        # Fully-fused TurboQuant attention path: only fires for turbo3
        # decode-only batches (one Q token per request). Reads compressed
        # K/V directly, dequants in registers, runs online softmax — never
        # materializes bf16 K/V in HBM. Multi-token Q (chunked prefill)
        # falls through to the V2a materialized path below.
        if (
            cache._pool.kv_quant == "turbo3"
            and supports_fused_attn_kernel(q.device)
            and q.shape[0] == cu_seqlens_q.shape[0] - 1
        ):
            seq_lens = cache.seq_lens_tensor(q.device)
            block_tables = cache.block_table_padded(q.device)
            return fused_turbo_attention_decode(
                cache._pool,
                layer_idx,
                q=q,
                seq_lens=seq_lens,
                block_tables=block_tables,
                softmax_scale=softmax_scale,
            )
        # Paged FA varlen reads K/V directly from bf16 pool storage; not
        # compatible with compressed (TurboQuant) storage. Compressed always
        # routes through the materialized path which knows how to dequant.
        if cache._pool.kv_quant is None and block_size % _FA_PAGED_BLOCK_SIZE_MULTIPLE == 0:
            return _packed_attention_paged_flash(
                q, cache, layer_idx, cu_seqlens_q, softmax_scale, window=window
            )
        return _packed_attention_materialized_flash(
            q, cache, layer_idx, cu_seqlens_q, softmax_scale, window=window
        )
    # PyTorch reference path: gather K/V from blocks into a packed buffer, then
    # use the per-request SDPA reference. Slower but device-agnostic.
    keys_packed, values_packed, cu_seqlens_k, _ = cache.materialize_packed_kv(layer_idx)
    return packed_attention_torch(
        q, keys_packed, values_packed, cu_seqlens_q, cu_seqlens_k, softmax_scale, window=window
    )


def packed_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    window: int | None = None,
    block_mask: list[torch.Tensor] | None = None,
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

    Sliding-window: when `window` is set, additionally clamp each query at
    absolute position `q` to attend only to keys at `[q - window + 1, q]`.
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
    if block_mask is not None and len(block_mask) != batch_size:
        raise ValueError(
            f"block_mask has {len(block_mask)} entries but there are {batch_size} requests"
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

        if block_mask is not None:
            # MiniMax-M3 MSA: the per-request additive bias already folds block
            # selection AND causality together (build_block_mask re-applies the
            # token-level causal mask), so it REPLACES the built-in causal fill.
            # Shape (q_len, 1, k_len) broadcasts over the query heads.
            scores = scores + block_mask[batch_idx].to(scores.dtype)
        else:
            # Causal mask: query at intra-request position i = (k_len - q_len + i)
            # in absolute K terms. It attends to K positions [0, k_len - q_len + i].
            q_offset = k_len - q_len
            q_abs_positions = torch.arange(q_len, device=q.device) + q_offset
            k_positions = torch.arange(k_len, device=q.device)
            invalid = k_positions[None, :] > q_abs_positions[:, None]  # (q_len, k_len)
            if window is not None:
                # Sliding window: also forbid keys older than `q - window + 1`.
                too_old = k_positions[None, :] < (q_abs_positions[:, None] - window + 1)
                invalid = invalid | too_old
            scores = scores.masked_fill(invalid[:, None, :], -float("inf"))

        weights = torch.softmax(scores, dim=-1)
        out[q_start:q_end] = torch.einsum("qhk,khd->qhd", weights, v_b.float()).to(q.dtype)

    return out


def _packed_attention_materialized_flash(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window: int | None = None,
) -> torch.Tensor:
    """Materialized FA varlen call: gather K/V from blocks into a packed buffer.

    Used when the cache's block_size doesn't satisfy FA's paged constraint
    (must be divisible by 256). One gather per layer per step. Slower than the
    paged path but works for any block size, and faster than the PyTorch
    reference because the attention math itself is still done by FA.

    `window` enables causal sliding-window via flash-attn's `window_size=(left, right)`
    argument; for causal SWA `right=0` and `left=window-1`.
    """
    if not _FLASH_ATTN_AVAILABLE:
        raise RuntimeError(
            "flash-attn not installed; install via the [cuda] extra or use "
            "packed_attention_torch instead"
        )
    require_cuda_device(q.device, "flash_attn_varlen_func")

    keys_packed, values_packed, cu_seqlens_k, max_seqlen_k = cache.materialize_packed_kv(layer_idx)
    max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    window_size = (window - 1, 0) if window is not None else (-1, -1)

    out: torch.Tensor = flash_attn_varlen_func(
        q,
        keys_packed,
        values_packed,
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window_size,
    )
    return out


def _packed_attention_paged_flash(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window: int | None = None,
) -> torch.Tensor:
    """Paged varlen FlashAttention call: reads K/V directly from `BlockPool`.

    Requires flash-attn 2.8+ (`flash_attn_varlen_func` with `block_table`).
    Inputs must be on a CUDA device with a half-precision dtype (fp16 / bf16);
    FA's varlen API doesn't accept fp32.

    The kernel walks `block_table[batch_idx]` to find each request's blocks,
    reads `cache_seqlens[batch_idx]` valid positions out of each, and runs
    causal varlen attention. No per-layer gather, no contiguous K/V buffer.

    `window` enables causal sliding-window via flash-attn's `window_size=(left, right)`.
    """
    if not _FLASH_ATTN_AVAILABLE:
        raise RuntimeError(
            "flash-attn not installed; install via the [cuda] extra or use "
            "packed_attention_torch instead"
        )
    require_cuda_device(q.device, "flash_attn_varlen_func")

    device = q.device
    keys_pool, values_pool = cache.pool_storage_for_layer(layer_idx)
    block_table = cache.block_table_padded(device)
    cache_seqlens = cache.seq_lens_tensor(device)
    max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    max_seqlen_k = int(cache_seqlens.max().item())
    cu_seqlens_k = torch.zeros(cache_seqlens.shape[0] + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.cumsum(cache_seqlens, dim=0)
    window_size = (window - 1, 0) if window is not None else (-1, -1)

    out: torch.Tensor = flash_attn_varlen_func(
        q,
        keys_pool,
        values_pool,
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        block_table=block_table,
        window_size=window_size,
    )
    return out
