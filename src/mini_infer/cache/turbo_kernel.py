"""Fused dequant kernel for TurboQuant V3 (`turbo3`) KV cache.

V3 today (`PagedKVCache.materialize_packed_kv` compressed branch in
`paged_kv_cache.py:301-324`) calls `pool.read_compressed_block` twice per
block per layer per decode step. Each call goes through
`polar_dequantize_block` (`turbo_quant.py:303-369`) plus an
inverse-rotation matmul, which fires multiple CUDA-kernel launches per
block. With moderate context and 24-28 layers that's hundreds-to-thousands
of launches per decode step at ~10 µs floor each. The arithmetic is cheap;
**Python launch overhead is what crushes throughput** to 0.04-0.18x of bf16.

This kernel collapses one full layer's worth of K (or V) dequantization
into a single launch:

1. Each program processes one `(task, kv_head)` pair, where a "task" is
   one block in some request's slot. The kernel writes directly into the
   final packed `(total_k, num_kv_heads, head_dim)` buffer at the
   pre-computed token offset, so no host-side scatter is needed.
2. Inside the program: unpack 4-bit nibbles, look up Lloyd-Max codebook
   value, apply optional QJL residual sign nudge (K side, 3-bit codebook
   + 1-bit sign), multiply by `1/sqrt(head_dim)` and per-vector radius,
   then inverse-rotate via `tl.dot` against the per-layer rotation
   matrix's transpose.

CUDA-only. Calls on non-CUDA devices fall back via the dispatcher in
`PagedKVCache.materialize_packed_kv`.
"""

from __future__ import annotations

import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.turbo_quant import _LLOYD_MAX_GAUSSIAN_3BIT
from mini_infer.device import is_cuda_device, require_cuda_device

# QJL residual-sign nudge step for the K-side 3-bit codebook. The reference
# Python form recomputes this from the codebook tensor on every dequant call
# (`turbo_quant.py:357`); the fused kernel needs it as a runtime float arg, so
# we precompute from the constants once at module load instead of doing a
# `.item()` H2D sync on the cached device tensor every layer.
_QJL_STEP_K = float((_LLOYD_MAX_GAUSSIAN_3BIT[1] - _LLOYD_MAX_GAUSSIAN_3BIT[0]) / 4.0)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


# Runtime toggle for benchmarking. Mirrors `int8_kernel._FUSED_DISABLED_FOR_BENCH`.
# When True, `supports_fused_kernel` reports False even on CUDA, forcing
# `materialize_packed_kv` to take the Python-loop path. Used for A/B
# benchmarking on the same model load.
_FUSED_DISABLED_FOR_BENCH = False


def supports_fused_kernel(device: torch.device | str) -> bool:
    """Whether the fused TurboQuant dequant kernel can run on this device.

    Today: CUDA + Triton only. Non-CUDA falls back to the Python-loop
    path inside `PagedKVCache.materialize_packed_kv`.
    `_FUSED_DISABLED_FOR_BENCH` overrides to False for A/B benchmarking.
    """
    if _FUSED_DISABLED_FOR_BENCH:
        return False
    if not _TRITON_AVAILABLE:
        return False
    return is_cuda_device(device)


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _turbo_dequant_kernel(  # type: ignore[no-untyped-def]
        # Storage pointers
        packed_ptr,
        radii_ptr,
        rotation_ptr,
        codebook_ptr,
        # Per-task pointers
        task_block_ids_ptr,
        task_offsets_ptr,
        task_valid_ptr,
        # Output pointer
        out_ptr,
        # Strides (in elements; for int8 storage this equals bytes)
        packed_stride_block,
        radii_stride_block,
        # Runtime scalars
        sqrt_hd_inv,
        qjl_step,
        # Constexpr shape
        BLOCK_SIZE: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_QJL: tl.constexpr,
    ) -> None:
        """One program per (task, kv_head). Writes one block's worth of
        dequantized + inverse-rotated bf16 values directly into the
        packed output buffer.

        A "task" is one (batch_slot, block_idx_in_slot) pair: the host
        wrapper enumerates these and packs `(physical_block_id,
        out_token_offset, valid_count)` into the three task tensors. The
        kernel reads its task triple, materializes a full `(BLOCK_SIZE,
        HEAD_DIM)` slice for this kv_head, then stores only the first
        `valid_count` rows into `out_ptr[offset : offset + valid_count,
        kv_head, :]`. The store mask is the only place the variable
        `valid_count` per block is used.
        """
        pid_task = tl.program_id(0)
        pid_head = tl.program_id(1)

        block_id = tl.load(task_block_ids_ptr + pid_task)
        out_token_offset = tl.load(task_offsets_ptr + pid_task)
        valid = tl.load(task_valid_ptr + pid_task)

        offs_b = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_d_byte = tl.arange(0, HEAD_DIM // 2)

        # Load (BLOCK_SIZE, HEAD_DIM // 2) packed bytes for this kv_head.
        # Layout: packed[block_id, b, h, d_byte] is at byte offset:
        #   block_id * packed_stride_block
        #     + b * (NUM_KV_HEADS * HEAD_DIM // 2)
        #     + h * (HEAD_DIM // 2)
        #     + d_byte
        base_byte = block_id * packed_stride_block + pid_head * (HEAD_DIM // 2)
        byte_ptrs = (
            packed_ptr
            + base_byte
            + offs_b[:, None] * (NUM_KV_HEADS * HEAD_DIM // 2)
            + offs_d_byte[None, :]
        )
        bytes_tile = tl.load(byte_ptrs)  # (BLOCK_SIZE, HEAD_DIM // 2) int8

        # Cast to int32 and mask to uint8 semantics. Triton's `>>` on int8
        # is arithmetic (sign-extending), so the explicit & 0xFF + cast to
        # int32 keeps the bit pattern clean across the shift.
        bytes_u = bytes_tile.to(tl.int32) & 0xFF
        low_nib = bytes_u & 0x0F
        high_nib = (bytes_u >> 4) & 0x0F

        # Codebook index (and QJL sign for K).
        if HAS_QJL:
            low_idx = low_nib & 0x07
            low_sign = (low_nib >> 3) & 0x01
            high_idx = high_nib & 0x07
            high_sign = (high_nib >> 3) & 0x01
        else:
            low_idx = low_nib
            high_idx = high_nib
            low_sign = tl.zeros_like(low_nib)
            high_sign = tl.zeros_like(high_nib)

        low_val = tl.load(codebook_ptr + low_idx)
        high_val = tl.load(codebook_ptr + high_idx)

        if HAS_QJL:
            # Quarter-step nudge based on the residual sign bit.
            # See `polar_dequantize_block` at turbo_quant.py:357 for the
            # reference Python form.
            low_val = low_val + (low_sign.to(tl.float32) * 2.0 - 1.0) * qjl_step
            high_val = high_val + (high_sign.to(tl.float32) * 2.0 - 1.0) * qjl_step

        # Undo the sqrt(head_dim) rescale (matches turbo_quant.py:361).
        low_val = low_val * sqrt_hd_inv
        high_val = high_val * sqrt_hd_inv

        # Per-vector radius. `radii` shape: (num_blocks, block_size,
        # num_kv_heads) — one scalar per (token, kv_head). Broadcast over
        # head_dim, same value for the (low, high) pair sharing a byte.
        radii_offset = block_id * radii_stride_block + pid_head
        radii_ptrs = radii_ptr + radii_offset + offs_b * NUM_KV_HEADS
        radii_tile = tl.load(radii_ptrs).to(tl.float32)  # (BLOCK_SIZE,)

        low_val = low_val * radii_tile[:, None]  # (BLOCK_SIZE, HEAD_DIM // 2)
        high_val = high_val * radii_tile[:, None]

        # Interleave low (even d) with high (odd d) -> (BLOCK_SIZE, HEAD_DIM).
        # tl.join stacks along a new last axis; reshape merges it back into d.
        rotated_tile = tl.join(low_val, high_val).reshape(BLOCK_SIZE, HEAD_DIM)

        # Inverse rotation: rotated @ R^T. Load R, transpose for the dot.
        rot_ptrs = rotation_ptr + offs_d[:, None] * HEAD_DIM + offs_d[None, :]
        rotation_tile = tl.load(rot_ptrs)  # (HEAD_DIM, HEAD_DIM) bf16
        rotation_t = tl.trans(rotation_tile)

        # Cast both operands to bf16 for tl.dot's tensor-core path; fp32
        # accumulator is implicit in tl.dot.
        rotated_bf16 = rotated_tile.to(tl.bfloat16)
        output_f32 = tl.dot(rotated_bf16, rotation_t, allow_tf32=False)
        output_bf16 = output_f32.to(tl.bfloat16)

        # Store to packed output at (out_token_offset + offs_b, pid_head, :).
        # out shape: (total_k, NUM_KV_HEADS, HEAD_DIM); inner two are constexpr.
        out_ptrs = (
            out_ptr
            + (out_token_offset + offs_b)[:, None] * (NUM_KV_HEADS * HEAD_DIM)
            + pid_head * HEAD_DIM
            + offs_d[None, :]
        )
        store_mask = offs_b[:, None] < valid
        tl.store(out_ptrs, output_bf16, mask=store_mask)


def fused_materialize_packed_kv(
    pool: BlockPool,
    layer_idx: int,
    *,
    task_block_ids: torch.Tensor,
    task_offsets: torch.Tensor,
    task_valid: torch.Tensor,
    key_out: torch.Tensor,
    value_out: torch.Tensor,
) -> None:
    """Fused dequantization for one layer's worth of compressed K/V.

    Replaces the per-block Python loop in
    `PagedKVCache.materialize_packed_kv` (`paged_kv_cache.py:301-324`) with
    one kernel launch per side (K and V). Writes results into the
    caller-provided `key_out` / `value_out` packed buffers in place.

    Args:
        pool: a `BlockPool` constructed with `kv_quant="turbo3"`.
        layer_idx: which layer's compressed storage to read.
        task_block_ids: int32 `(num_tasks,)`. Physical block ids; one
            entry per (batch_slot, block_idx_in_slot) pair the caller
            wants materialized.
        task_offsets: int32 `(num_tasks,)`. Token offset within the
            packed output where this task's block writes.
        task_valid: int32 `(num_tasks,)`. How many tokens of this block
            are valid (≤ block_size); the kernel masks the rest.
        key_out, value_out: `(total_k, num_kv_heads, head_dim)` in
            `pool.dtype`. Filled in place.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available; cannot run fused TurboQuant kernel")
    if pool.kv_quant != "turbo3":
        raise RuntimeError(
            f"fused_materialize_packed_kv only supports kv_quant='turbo3'; got {pool.kv_quant!r}"
        )
    if pool._compressed_storage is None or pool._radii_storage is None:
        raise RuntimeError("compressed/radii storage missing; pool not compressed")
    if pool._rotation is None or pool._k_codebook is None or pool._v_codebook is None:
        raise RuntimeError("rotation or codebooks missing; pool not compressed")

    device = pool._compressed_storage.device
    require_cuda_device(device, "fused TurboQuant dequant kernel")

    num_tasks = task_block_ids.shape[0]
    if num_tasks == 0:
        return

    block_size = pool.block_size
    num_kv_heads = pool.num_kv_heads
    head_dim = pool.head_dim

    # Layer + side slices.
    packed_k = pool._compressed_storage[layer_idx, 0]  # (num_blocks, packed_bytes)
    packed_v = pool._compressed_storage[layer_idx, 1]
    radii_k = pool._radii_storage[layer_idx, 0]  # (num_blocks, block_size, num_kv_heads)
    radii_v = pool._radii_storage[layer_idx, 1]
    rotation = pool._rotation[layer_idx]  # (head_dim, head_dim)

    # Stride between consecutive blocks in the (num_blocks, ...) views.
    # int8 stride == bytes; bf16 stride == elements (= 2 bytes).
    packed_stride_block = packed_k.stride(0)
    radii_stride_block = radii_k.stride(0)

    # 1/sqrt(head_dim) undoes the V3 polar rescale; QJL step (K side
    # only) is precomputed at module load — see `_QJL_STEP_K`.
    sqrt_hd_inv = float(1.0 / (head_dim**0.5))
    k_codebook = pool._k_codebook
    v_codebook = pool._v_codebook

    grid = (num_tasks, num_kv_heads)

    # K side: 3-bit Lloyd-Max codebook + 1-bit QJL residual sign.
    _turbo_dequant_kernel[grid](
        packed_k,
        radii_k,
        rotation,
        k_codebook,
        task_block_ids,
        task_offsets,
        task_valid,
        key_out,
        packed_stride_block,
        radii_stride_block,
        sqrt_hd_inv,
        _QJL_STEP_K,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HAS_QJL=True,
        num_warps=4,
    )

    # V side: 4-bit Lloyd-Max codebook only, no QJL.
    _turbo_dequant_kernel[grid](
        packed_v,
        radii_v,
        rotation,
        v_codebook,
        task_block_ids,
        task_offsets,
        task_valid,
        value_out,
        packed_stride_block,
        radii_stride_block,
        sqrt_hd_inv,
        0.0,  # qjl_step ignored under HAS_QJL=False
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HAS_QJL=False,
        num_warps=4,
    )
