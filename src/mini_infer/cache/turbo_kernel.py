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


# Runtime toggles for A/B benchmarking on the same model load.
#
# - `_FUSED_DISABLED_FOR_BENCH = True`: disables BOTH V2a (dequant) and
#   V2b (attention) paths. Forces `materialize_packed_kv` through the
#   per-block Python loop and `packed_attention_forward` through the
#   bf16 materialized path. This is the "what would V3 look like with
#   no Triton" comparison.
# - `_FUSED_ATTN_DISABLED_FOR_BENCH = True` (DEFAULT): disables only the
#   V2b attention path. Materialize stays fused (V2a) and FA varlen
#   handles the attention math. V2b is off by default because at 7B+
#   our custom Triton softmax loses to FlashAttention's hand-tuned
#   kernel by ~12% throughput (validated 2026-05-02 on A10 + Qwen2.5-7B
#   in `docs/benchmarks/2026-05-02-turboquant-v2b.md`); the kernel is
#   kept opt-in for memory-constrained scenarios where the avoided
#   transient buffer matters more than throughput.
_FUSED_DISABLED_FOR_BENCH = False
_FUSED_ATTN_DISABLED_FOR_BENCH = True


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


def supports_fused_attn_kernel(device: torch.device | str) -> bool:
    """Whether the fully-fused TurboQuant attention kernel (V2b) can run.

    Stricter than `supports_fused_kernel`: V2b additionally requires
    `_FUSED_ATTN_DISABLED_FOR_BENCH` to be False. **The default is True**
    (V2b off) because V2a is faster on 7B+ — see the toggle's docstring
    above. Set the flag to False to opt into V2b for memory-constrained
    workloads or A/B comparisons.
    """
    if _FUSED_ATTN_DISABLED_FOR_BENCH:
        return False
    return supports_fused_kernel(device)


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

    @triton.jit  # type: ignore[untyped-decorator]
    def _turbo_fused_attn_decode_kernel(  # type: ignore[no-untyped-def]
        # Q / output
        Q_ptr,
        Out_ptr,
        # Compressed K/V storage (one layer slice each)
        packed_k_ptr,
        packed_v_ptr,
        radii_k_ptr,
        radii_v_ptr,
        rotation_ptr,
        k_codebook_ptr,
        v_codebook_ptr,
        # Per-request
        block_tables_ptr,
        seq_lens_ptr,
        # Strides
        q_stride_token,
        q_stride_head,
        out_stride_token,
        out_stride_head,
        packed_stride_block,
        radii_stride_block,
        # Runtime scalars
        max_blocks_per_req,
        sqrt_hd_inv,
        qjl_step_k,
        softmax_scale,
        # Constexpr
        BLOCK_SIZE: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ) -> None:
        """One program per (request, q_head). Reads compressed K/V tiles
        directly from the pool, dequantizes them in registers using the
        V2a codec body, and runs FlashAttention-2 online softmax over
        them — never materializing bf16 K/V in HBM.

        Decode-only: assumes Q has exactly one token per request
        (`total_q == batch_size`). The dispatcher gates on this; multi-
        token Q (chunked prefill) routes through the V2a materialized
        path.

        Mirrors the structure of
        `paged_attention.py:_paged_attention_decode_batched_kernel`
        but with two changes:

        1. K/V load is replaced by the V2a codec — unpack 4-bit nibbles,
           Lloyd-Max codebook lookup, optional QJL nudge (K only),
           multiply by `1/sqrt(head_dim)` and per-vector radius, then
           inverse-rotate via `tl.dot` against the per-layer rotation
           matrix's transpose. Same arithmetic as
           `polar_dequantize_block` + `inverse_rotate`, fused.
        2. Output is written to a packed `(total_q, num_q_heads,
           head_dim)` buffer instead of the per-batch (B, ...) buffer
           the paged decode kernel uses.
        """
        pid_req = tl.program_id(0)
        pid_qh = tl.program_id(1)

        kv_h = pid_qh // (NUM_Q_HEADS // NUM_KV_HEADS)
        seq_len = tl.load(seq_lens_ptr + pid_req)

        # Decode-only contract: q_len == 1 per request, so the Q row for
        # request pid_req is row pid_req in the packed Q tensor.
        out_row = pid_req

        d_offsets = tl.arange(0, HEAD_DIM)
        q_ptrs = Q_ptr + out_row * q_stride_token + pid_qh * q_stride_head + d_offsets
        q = tl.load(q_ptrs).to(tl.float32)

        # Load the per-layer rotation matrix once. Reused across every
        # K-block iteration for both K and V dequant. At head_dim=128
        # this is 32 KB bf16, comfortably within A10's 96 KB SMEM.
        offs_d = tl.arange(0, HEAD_DIM)
        rot_ptrs = rotation_ptr + offs_d[:, None] * HEAD_DIM + offs_d[None, :]
        rotation_tile = tl.load(rot_ptrs)
        rotation_t = tl.trans(rotation_tile)

        # Online-softmax accumulators.
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        num_blocks_used = tl.cdiv(seq_len, BLOCK_SIZE)
        pos_in_block = tl.arange(0, BLOCK_SIZE)
        offs_d_byte = tl.arange(0, HEAD_DIM // 2)
        block_table_offset = pid_req * max_blocks_per_req

        for block_index in range(0, num_blocks_used):
            block_id = tl.load(block_tables_ptr + block_table_offset + block_index)
            global_pos = block_index * BLOCK_SIZE + pos_in_block
            pos_mask = global_pos < seq_len

            # ── Dequant K (3-bit Lloyd-Max + 1-bit QJL) ──
            base_byte = block_id * packed_stride_block + kv_h * (HEAD_DIM // 2)
            byte_ptrs = (
                base_byte
                + pos_in_block[:, None] * (NUM_KV_HEADS * HEAD_DIM // 2)
                + offs_d_byte[None, :]
            )
            bytes_k = tl.load(packed_k_ptr + byte_ptrs)
            bytes_k_u = bytes_k.to(tl.int32) & 0xFF
            low_nib_k = bytes_k_u & 0x0F
            high_nib_k = (bytes_k_u >> 4) & 0x0F
            low_idx_k = low_nib_k & 0x07
            low_sign_k = (low_nib_k >> 3) & 0x01
            high_idx_k = high_nib_k & 0x07
            high_sign_k = (high_nib_k >> 3) & 0x01
            low_val_k = tl.load(k_codebook_ptr + low_idx_k)
            high_val_k = tl.load(k_codebook_ptr + high_idx_k)
            low_val_k = low_val_k + (low_sign_k.to(tl.float32) * 2.0 - 1.0) * qjl_step_k
            high_val_k = high_val_k + (high_sign_k.to(tl.float32) * 2.0 - 1.0) * qjl_step_k
            low_val_k = low_val_k * sqrt_hd_inv
            high_val_k = high_val_k * sqrt_hd_inv

            radii_k_offset = block_id * radii_stride_block + kv_h
            radii_k_ptrs = radii_k_ptr + radii_k_offset + pos_in_block * NUM_KV_HEADS
            radii_k_tile = tl.load(radii_k_ptrs).to(tl.float32)
            low_val_k = low_val_k * radii_k_tile[:, None]
            high_val_k = high_val_k * radii_k_tile[:, None]

            rotated_k = tl.join(low_val_k, high_val_k).reshape(BLOCK_SIZE, HEAD_DIM)
            rotated_k_bf16 = rotated_k.to(tl.bfloat16)
            k_tile_f32 = tl.dot(rotated_k_bf16, rotation_t, allow_tf32=False)

            # ── Scores: q · K row, scaled, with seq_len mask ──
            scores = tl.sum(q[None, :] * k_tile_f32, axis=1) * softmax_scale
            scores = tl.where(pos_mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores))
            scale_factor = tl.exp(m_i - m_new)
            scores_norm = tl.exp(scores - m_new)
            l_new = l_i * scale_factor + tl.sum(scores_norm)

            # ── Dequant V (4-bit Lloyd-Max only) ──
            bytes_v = tl.load(packed_v_ptr + byte_ptrs)
            bytes_v_u = bytes_v.to(tl.int32) & 0xFF
            low_nib_v = bytes_v_u & 0x0F
            high_nib_v = (bytes_v_u >> 4) & 0x0F
            low_val_v = tl.load(v_codebook_ptr + low_nib_v)
            high_val_v = tl.load(v_codebook_ptr + high_nib_v)
            low_val_v = low_val_v * sqrt_hd_inv
            high_val_v = high_val_v * sqrt_hd_inv

            radii_v_ptrs = radii_v_ptr + radii_k_offset + pos_in_block * NUM_KV_HEADS
            radii_v_tile = tl.load(radii_v_ptrs).to(tl.float32)
            low_val_v = low_val_v * radii_v_tile[:, None]
            high_val_v = high_val_v * radii_v_tile[:, None]

            rotated_v = tl.join(low_val_v, high_val_v).reshape(BLOCK_SIZE, HEAD_DIM)
            rotated_v_bf16 = rotated_v.to(tl.bfloat16)
            v_tile_f32 = tl.dot(rotated_v_bf16, rotation_t, allow_tf32=False)

            # ── Online softmax accumulator update ──
            acc = acc * scale_factor + tl.sum(scores_norm[:, None] * v_tile_f32, axis=0)
            m_i = m_new
            l_i = l_new

        out = acc / l_i
        out_ptrs = Out_ptr + out_row * out_stride_token + pid_qh * out_stride_head + d_offsets
        tl.store(out_ptrs, out.to(tl.bfloat16))


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


def fused_turbo_attention_decode(
    pool: BlockPool,
    layer_idx: int,
    *,
    q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """V2b: fully-fused dequant + online softmax attention for decode.

    Replaces the V2a path's `materialize_packed_kv` + `flash_attn_varlen_func`
    pair with a single Triton kernel that walks compressed K/V blocks for
    each request, dequants tiles in registers using the V2a codec, runs
    FlashAttention-2 online softmax against the request's Q vector, and
    writes the attention output. K/V are never materialized to HBM —
    that's the peak-memory win the ADR-013 follow-up promised.

    Decode-only contract: each request has exactly one Q token. The
    dispatcher in `packed_attention.py` enforces this; multi-token Q
    (chunked prefill) routes through the V2a path.

    Args:
        pool: a `BlockPool` with `kv_quant="turbo3"`.
        layer_idx: which layer's compressed storage to attend.
        q: `(batch_size, num_q_heads, head_dim)` bf16. Rows align 1:1
            with requests (decode contract).
        seq_lens: int32 `(batch_size,)`. Current K-history length per
            request (post-append).
        block_tables: int32 `(batch_size, max_blocks_per_req)` padded
            block-id table; rows past `cdiv(seq_len, block_size)` are
            masked out by the kernel via `seq_len`.
        softmax_scale: typically `1/sqrt(head_dim)`.

    Returns:
        `(batch_size, num_q_heads, head_dim)` bf16 attention output.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available; cannot run fused TurboQuant attention")
    if pool.kv_quant != "turbo3":
        raise RuntimeError(
            f"fused_turbo_attention_decode only supports kv_quant='turbo3'; got {pool.kv_quant!r}"
        )
    if pool._compressed_storage is None or pool._radii_storage is None:
        raise RuntimeError("compressed/radii storage missing; pool not compressed")
    if pool._rotation is None or pool._k_codebook is None or pool._v_codebook is None:
        raise RuntimeError("rotation or codebooks missing; pool not compressed")

    require_cuda_device(q.device, "fused TurboQuant attention kernel")

    batch_size, num_q_heads, head_dim = q.shape
    if head_dim != pool.head_dim:
        raise ValueError(f"q head_dim={head_dim} doesn't match pool.head_dim={pool.head_dim}")
    if num_q_heads % pool.num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads={num_q_heads} not a multiple of num_kv_heads={pool.num_kv_heads}"
        )
    if seq_lens.shape[0] != batch_size:
        raise ValueError(
            f"seq_lens has {seq_lens.shape[0]} entries but q has batch_size={batch_size}"
        )
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            f"block_tables has {block_tables.shape[0]} rows but q has batch_size={batch_size}"
        )

    packed_k = pool._compressed_storage[layer_idx, 0]  # (num_blocks, packed_bytes)
    packed_v = pool._compressed_storage[layer_idx, 1]
    radii_k = pool._radii_storage[layer_idx, 0]  # (num_blocks, block_size, num_kv_heads)
    radii_v = pool._radii_storage[layer_idx, 1]
    rotation = pool._rotation[layer_idx]

    sqrt_hd_inv = float(1.0 / (head_dim**0.5))
    out = torch.empty_like(q)

    grid = (batch_size, num_q_heads)
    _turbo_fused_attn_decode_kernel[grid](
        q,
        out,
        packed_k,
        packed_v,
        radii_k,
        radii_v,
        rotation,
        pool._k_codebook,
        pool._v_codebook,
        block_tables,
        seq_lens,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        packed_k.stride(0),
        radii_k.stride(0),
        block_tables.shape[1],
        sqrt_hd_inv,
        _QJL_STEP_K,
        softmax_scale,
        BLOCK_SIZE=pool.block_size,
        NUM_KV_HEADS=pool.num_kv_heads,
        NUM_Q_HEADS=num_q_heads,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out
