"""Paged attention for the decode path: read K/V directly from BlockPool storage.

Two implementations:
- `paged_attention_decode_torch`: pure PyTorch, hardware-agnostic. Our numerical
  oracle and the fallback when CUDA + Triton aren't available.
- `paged_attention_decode_triton`: CUDA-only Triton kernel. Fast path. Validated
  against the torch reference within cosine similarity > 0.99 (CLAUDE.md Stretch D).

Use `paged_attention_decode(...)` for the dispatcher; it picks the fastest
available implementation for the device.
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


def supports_paged_kernel(device: torch.device | str) -> bool:
    """Whether our paged attention kernel can run on this device.

    Single source of truth so we don't sprinkle `device == "cuda"` checks
    across the codebase. Today: CUDA only (requires Triton). Future devices
    (ROCm, etc.) plug in here.
    """
    if not _TRITON_AVAILABLE:
        return False
    if isinstance(device, str):
        return device == "cuda"
    return device.type == "cuda"


def paged_attention_decode_torch(
    q: torch.Tensor,
    k_pool_layer: torch.Tensor,
    v_pool_layer: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Pure PyTorch reference: gather K/V from blocks, compute attention.

    Shapes:
      q             : (batch, num_q_heads, head_dim) — single decode token's queries
      k_pool_layer  : (num_blocks, block_size, num_kv_heads, head_dim) — this layer's K
      v_pool_layer  : same shape — this layer's V
      block_table   : (num_blocks_used,) int — block IDs in order
      seq_len       : current cached length (number of valid positions)

    Returns: (batch, num_q_heads, head_dim) attention output for the decode position.
    """
    _, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_pool_layer.shape
    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads

    # Gather K/V for the cached positions: shape (seq_len, num_kv_heads, head_dim).
    positions = torch.arange(seq_len, device=q.device)
    block_ids_per_pos = block_table[positions // block_size]
    slots_per_pos = positions % block_size
    k_full = k_pool_layer[block_ids_per_pos, slots_per_pos]  # (seq_len, num_kv_heads, head_dim)
    v_full = v_pool_layer[block_ids_per_pos, slots_per_pos]

    # Broadcast KV heads to Q heads (GQA): (seq_len, num_q_heads, head_dim).
    k_full = k_full.repeat_interleave(group_size, dim=1)
    v_full = v_full.repeat_interleave(group_size, dim=1)

    # Compute attention in fp32 for stability.
    q_f = q.float()
    k_f = k_full.float()
    v_f = v_full.float()

    # Scores: (batch, num_q_heads, seq_len). q is (batch, h, d), k is (s, h, d).
    scores = torch.einsum("bhd,shd->bhs", q_f, k_f) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,shd->bhd", weights, v_f)

    return out.to(q.dtype)


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _paged_attention_decode_kernel(  # type: ignore[no-untyped-def]
        Q_ptr,
        K_pool_ptr,
        V_pool_ptr,
        block_table_ptr,
        Out_ptr,
        seq_len,
        num_q_heads,
        num_kv_heads,
        stride_q_b,
        stride_q_h,
        stride_q_d,
        stride_kv_blk,
        stride_kv_pos,
        stride_kv_h,
        stride_kv_d,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
    ) -> None:
        # One program per (batch, q_head).
        pid = tl.program_id(0)
        b = pid // num_q_heads
        h = pid % num_q_heads
        kv_h = h // (num_q_heads // num_kv_heads)

        # Load Q for this (batch, q_head). Q is small (just HEAD_DIM floats per head)
        # so we pull the whole vector into registers once. fp32 cast for stability:
        # softmax / online-max accumulation is sensitive to precision.
        d_offsets = tl.arange(0, HEAD_DIM)
        q_offsets = b * stride_q_b + h * stride_q_h + d_offsets * stride_q_d
        q = tl.load(Q_ptr + q_offsets).to(tl.float32)

        # Online-softmax state. We never materialize the full attention scores;
        # instead we update a running max (m_i), running normalizer (l_i), and
        # weighted-V accumulator (acc) one block at a time. Same trick FlashAttention
        # uses; produces the same output as a single softmax over all positions.
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        num_blocks_used = tl.cdiv(seq_len, BLOCK_SIZE)
        pos_in_block = tl.arange(0, BLOCK_SIZE)

        for block_index in range(0, num_blocks_used):
            # Indirection: physical block ID lives in the request's block table.
            # The block stores K/V for `BLOCK_SIZE` consecutive logical positions.
            block_id = tl.load(block_table_ptr + block_index)
            global_pos = block_index * BLOCK_SIZE + pos_in_block
            mask = global_pos < seq_len  # last block may be partially filled

            # Gather a (BLOCK_SIZE, HEAD_DIM) tile of K from the pool. The 2D offset
            # arithmetic picks: which physical block (block_id), which slot inside
            # it (pos_in_block, broadcast as a column), which KV head (kv_h, GQA),
            # and all HEAD_DIM elements (d_offsets, broadcast as a row).
            k_offsets = (
                block_id * stride_kv_blk
                + pos_in_block[:, None] * stride_kv_pos
                + kv_h * stride_kv_h
                + d_offsets[None, :] * stride_kv_d
            )
            k = tl.load(K_pool_ptr + k_offsets, mask=mask[:, None], other=0.0).to(tl.float32)

            # Scores: q . k_row, scaled. Masked positions become -inf so softmax
            # gives them zero weight without polluting the running max below.
            scores = tl.sum(q[None, :] * k, axis=1) * SCALE
            scores = tl.where(mask, scores, -float("inf"))

            # Online softmax update: when this block's scores exceed the prior
            # running max, rescale our accumulators (acc, l_i) by exp(m_i - m_new)
            # so they're consistent with the new max. Then add this block's
            # contribution. Same numerical answer as one big softmax, no overflow,
            # no need to keep all scores around.
            m_new = tl.maximum(m_i, tl.max(scores))
            scale_factor = tl.exp(m_i - m_new)
            scores_norm = tl.exp(scores - m_new)
            l_new = l_i * scale_factor + tl.sum(scores_norm)

            # V uses identical indexing as K (same layer, same block, same slots,
            # same KV head). The `acc * scale_factor` matches the rescale we did
            # for l_i so the running weighted sum stays consistent. tl.sum over
            # axis=0 collapses the BLOCK_SIZE positions into a single HEAD_DIM
            # output vector.
            v_offsets = (
                block_id * stride_kv_blk
                + pos_in_block[:, None] * stride_kv_pos
                + kv_h * stride_kv_h
                + d_offsets[None, :] * stride_kv_d
            )
            v = tl.load(V_pool_ptr + v_offsets, mask=mask[:, None], other=0.0).to(tl.float32)
            acc = acc * scale_factor + tl.sum(scores_norm[:, None] * v, axis=0)

            m_i = m_new
            l_i = l_new

        # Final normalize: acc currently holds sum(softmax_unnormalized * V); divide
        # by l_i (the normalizer) once, at the end. Then store at the right offset.
        out = acc / l_i
        out_offsets = b * stride_o_b + h * stride_o_h + d_offsets * stride_o_d
        tl.store(Out_ptr + out_offsets, out)


def paged_attention_decode_triton(
    q: torch.Tensor,
    k_pool_layer: torch.Tensor,
    v_pool_layer: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """CUDA-only Triton kernel launcher. Same contract as the torch reference."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton not available; use paged_attention_decode_torch instead")
    if q.device.type != "cuda":
        raise RuntimeError("Triton paged attention requires a CUDA tensor for q")

    batch, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_pool_layer.shape
    assert k_pool_layer.shape == v_pool_layer.shape
    assert num_q_heads % num_kv_heads == 0

    # Output buffer in fp32; cast back at the end.
    out_fp32 = torch.empty((batch, num_q_heads, head_dim), dtype=torch.float32, device=q.device)

    grid = (batch * num_q_heads,)
    # Strides describe how many elements to advance in memory per step along each
    # tensor dimension. Triton kernels compute pointer offsets manually (there's no
    # ndarray indexing inside a kernel), so we hand them to the kernel explicitly.
    # That way the kernel works regardless of how the tensor was laid out (a fresh
    # contiguous tensor vs a sliced/transposed view of a bigger one). Format below:
    # one stride per dim, in the order the kernel expects.
    _paged_attention_decode_kernel[grid](
        q.contiguous(),
        k_pool_layer,
        v_pool_layer,
        block_table.to(torch.int32),
        out_fp32,
        seq_len,
        num_q_heads,
        num_kv_heads,
        q.stride(0),  # batch stride
        q.stride(1),  # q-head stride
        q.stride(2),  # head_dim stride
        k_pool_layer.stride(0),  # block stride
        k_pool_layer.stride(1),  # within-block position stride
        k_pool_layer.stride(2),  # kv-head stride
        k_pool_layer.stride(3),  # head_dim stride
        out_fp32.stride(0),  # batch stride
        out_fp32.stride(1),  # q-head stride
        out_fp32.stride(2),  # head_dim stride
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        SCALE=1.0 / math.sqrt(head_dim),
    )
    return out_fp32.to(q.dtype)


def paged_attention_decode(
    q: torch.Tensor,
    k_pool_layer: torch.Tensor,
    v_pool_layer: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Dispatcher: Triton on CUDA, pure PyTorch otherwise."""
    if supports_paged_kernel(q.device):
        return paged_attention_decode_triton(q, k_pool_layer, v_pool_layer, block_table, seq_len)
    return paged_attention_decode_torch(q, k_pool_layer, v_pool_layer, block_table, seq_len)
