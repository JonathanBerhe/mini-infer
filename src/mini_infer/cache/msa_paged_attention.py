"""Block-sparse paged attention for the MSA decode path (MiniMax-M3).

The torch oracle for M3 materializes the FULL per-request K/V history each step
and applies a dense additive block mask (`packed_attention_torch(block_mask=...)`),
so its per-step traffic grows with context length. At decode the query only
attends to the indexer's selected index blocks (top-k + local), so the kernel
path reads exactly those blocks from the paged pool via the block table and
never materializes the rest: per-step traffic is O(topk * index_block_size)
regardless of context.

Two implementations:
- `msa_paged_decode_torch`: pure PyTorch, hardware-agnostic. Validated CPU-side
  against the dense-mask oracle (`tests/unit/test_msa_paged_attention.py`).
- `msa_paged_decode_triton`: CUDA-only Triton kernel, one program per
  (request, q_head), online softmax in fp32 over the selected blocks only.
  Mirrors `paged_attention.py`'s decode kernel plus the selection indirection.

Selection comes from `MiniMaxM3Indexer.select_cached` (the same routine that
builds the oracle's dense mask, so both paths share one selection). The
selection is PER INDEX HEAD — `index_n_heads == num_key_value_heads`, one
independent block set per KV / GQA group (transformers 5.14 semantics) — so
each request carries a `(num_kv_heads, topk)` id tensor and each KV head's
program walks its own block list. At decode the query is the newest token
(position `seq_len - 1`), so every cached position in a selected block is
causally visible; the only masking needed is `position < seq_len` in the
(possibly partial) last block.
"""

from __future__ import annotations

import math

import torch

from mini_infer.device import is_cuda_device, require_cuda_device

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


def supports_msa_kernel(device: torch.device | str) -> bool:
    """Whether the MSA block-sparse decode kernel can run on this device."""
    if not _TRITON_AVAILABLE:
        return False
    return is_cuda_device(device)


def _selected_pool_entries(
    selected_blocks: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    index_block_size: int,
    pool_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate selected INDEX-block ids into (physical pool block id, base pos) pairs.

    An index block spans `index_block_size` tokens; the pool stores
    `pool_block_size` tokens per physical block. Each selected index block
    therefore expands to `index_block_size // pool_block_size` pool entries
    (the sizes must divide; enforced by the caller). `-1` selection padding
    expands to `-1` base positions, which the consumers mask out.

    Selected ids are sorted ASCENDING first (mirroring the reference kernel's
    contract): it fixes the accumulation order to a deterministic position
    order, matching the oracle's natural key order.

    Returns `(pool_ids, base_positions)`, both 1D int32 of length
    `num_selected * ratio`, where entry i covers token positions
    `[base_positions[i], base_positions[i] + pool_block_size)`.
    """
    device = selected_blocks.device
    ratio = index_block_size // pool_block_size
    # Ascending sort; -1 padding sorts to the FRONT, which is fine (dead entries
    # are masked wherever they land).
    sel = selected_blocks.to(torch.int64).sort().values  # (topk,)
    chunk = torch.arange(ratio, device=device, dtype=torch.int64)  # (ratio,)
    # Logical pool-block index of each (selected index block, chunk) pair.
    logical = sel[:, None] * ratio + chunk[None, :]  # (topk, ratio)
    base = logical * pool_block_size  # token base position per entry
    valid = (sel[:, None] >= 0) & (base < seq_len)
    # Clamp invalid logical ids to 0 so the block-table gather stays in range;
    # the -1 base positions mark them dead for the consumers.
    logical_safe = logical.clamp(min=0).clamp(max=max(block_table.numel() - 1, 0))
    pool_ids = block_table.to(torch.int64)[logical_safe]
    pool_ids = torch.where(valid, pool_ids, torch.zeros_like(pool_ids))
    base = torch.where(valid, base, torch.full_like(base, -1))
    return pool_ids.reshape(-1).to(torch.int32), base.reshape(-1).to(torch.int32)


def msa_paged_decode_torch(
    q: torch.Tensor,
    k_pool_layer: torch.Tensor,
    v_pool_layer: torch.Tensor,
    block_tables: list[torch.Tensor],
    seq_lens: list[int],
    selected_blocks: list[torch.Tensor],
    *,
    index_block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Pure PyTorch reference: gather ONLY the selected blocks, then attention.

    Shapes:
      q               : (B, num_q_heads, head_dim), one decode token per request
      k_pool_layer    : (num_blocks, pool_block_size, num_kv_heads, head_dim)
      v_pool_layer    : same shape
      block_tables    : per request, 1D int tensor of pool block ids in order
      seq_lens        : per request, cached length INCLUDING the current token
      selected_blocks : per request, (num_kv_heads, topk) int64 selected
                        index-block ids, -1 padded — one row per KV / GQA
                        group (the decode row of
                        `MiniMaxM3Indexer.select_cached`)

    Returns: (B, num_q_heads, head_dim).
    """
    batch, num_q_heads, head_dim = q.shape
    _, pool_block_size, num_kv_heads, _ = k_pool_layer.shape
    if index_block_size % pool_block_size != 0:
        raise ValueError(
            f"index_block_size={index_block_size} must be a multiple of the pool "
            f"block_size={pool_block_size} for the block-sparse decode path"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads={num_q_heads} not divisible by num_kv_heads={num_kv_heads}")
    group_size = num_q_heads // num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    outs = []
    offs = torch.arange(pool_block_size, device=q.device, dtype=torch.int32)
    for r in range(batch):
        if selected_blocks[r].ndim != 2 or selected_blocks[r].shape[0] != num_kv_heads:
            raise ValueError(
                f"request {r}: selected_blocks must be (num_kv_heads={num_kv_heads}, topk); "
                f"got {tuple(selected_blocks[r].shape)}"
            )
        head_outs = []
        for kv_h in range(num_kv_heads):
            pool_ids, base = _selected_pool_entries(
                selected_blocks[r][kv_h],
                block_tables[r],
                seq_lens[r],
                index_block_size,
                pool_block_size,
            )
            # Expand entries to token positions; drop dead entries and the tail
            # beyond seq_len (partial last block).
            positions = base[:, None] + offs[None, :]  # (entries, pool_block_size)
            keep = (base[:, None] >= 0) & (positions < seq_lens[r])
            entry_ids = pool_ids[:, None].expand_as(positions)[keep].long()
            slot = (positions % pool_block_size)[keep].long()
            k_sel = k_pool_layer[entry_ids, slot, kv_h].float()  # (n_sel, d)
            v_sel = v_pool_layer[entry_ids, slot, kv_h].float()

            q_g = q[r, kv_h * group_size : (kv_h + 1) * group_size].float()  # (group, d)
            scores = torch.einsum("gd,sd->gs", q_g, k_sel) * scale
            weights = torch.softmax(scores, dim=-1)
            head_outs.append(torch.einsum("gs,sd->gd", weights, v_sel))
        outs.append(torch.cat(head_outs, dim=0))
    return torch.stack(outs, dim=0).to(q.dtype)


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _msa_paged_decode_kernel(  # type: ignore[no-untyped-def]
        Q_ptr,
        K_pool_ptr,
        V_pool_ptr,
        pool_ids_ptr,  # (B, num_kv_heads, max_entries) int32 physical pool block ids
        base_pos_ptr,  # (B, num_kv_heads, max_entries) int32 token base positions, -1 = dead
        seq_lens_ptr,  # (B,) int32
        Out_ptr,
        max_entries,
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
        GROUP: tl.constexpr,  # q heads per kv head (16 for M3)
        HEAD_DIM: tl.constexpr,
        POOL_BLOCK_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
    ) -> None:
        # One program per (request, KV head): a (GROUP, HEAD_DIM) output tile
        # covering the whole GQA group. Each selected K/V block is read ONCE
        # per group (vs once per q head in a per-head grid), and QK / PV are
        # real tl.dot MMAs. The selection is one block set PER KV head
        # (index_n_heads == num_kv_heads, transformers 5.14 semantics), so
        # each program walks its own head's entry row; dead entries carry
        # base position -1.
        req_idx = tl.program_id(0)
        kv_h = tl.program_id(1)
        num_kv_heads = tl.num_programs(1)

        seq_len = tl.load(seq_lens_ptr + req_idx)

        g_offsets = tl.arange(0, GROUP)  # q heads kv_h*GROUP .. +GROUP-1
        d_offsets = tl.arange(0, HEAD_DIM)
        q_offsets = (
            req_idx * stride_q_b
            + (kv_h * GROUP + g_offsets[:, None]) * stride_q_h
            + d_offsets[None, :] * stride_q_d
        )
        q = tl.load(Q_ptr + q_offsets)  # (GROUP, HEAD_DIM), pool dtype

        # Online-softmax accumulators, one lane per q head; fp32 throughout,
        # same as the torch oracle (scores fp32, softmax fp32, fp32 acc).
        m_i = tl.full([GROUP], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([GROUP], dtype=tl.float32)
        acc = tl.zeros([GROUP, HEAD_DIM], dtype=tl.float32)

        pos_in_block = tl.arange(0, POOL_BLOCK_SIZE)
        entry_offset = (req_idx * num_kv_heads + kv_h) * max_entries

        for entry in range(0, max_entries):
            base = tl.load(base_pos_ptr + entry_offset + entry)
            # Dead entries (selection -1 padding, sorted to the front) are
            # skipped OUTSIDE the softmax update: running them through would
            # compute exp(-inf - -inf) = nan in the rescale while the running
            # max is still -inf. The branch is a uniform scalar, so the whole
            # program skips together (no divergence cost).
            if base >= 0:
                block_id = tl.load(pool_ids_ptr + entry_offset + entry)
                global_pos = base + pos_in_block
                # The tail past seq_len (partial last block) masks to -inf.
                mask = global_pos < seq_len

                k_offsets = (
                    block_id * stride_kv_blk
                    + pos_in_block[:, None] * stride_kv_pos
                    + kv_h * stride_kv_h
                    + d_offsets[None, :] * stride_kv_d
                )
                k = tl.load(K_pool_ptr + k_offsets, mask=mask[:, None], other=0.0)

                # QK in the storage dtype with fp32 accumulation: bf16*bf16
                # products are exact in fp32, so only the reduction order
                # differs from the oracle's fp32 einsum.
                scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE
                scores = tl.where(mask[None, :], scores, -float("inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new[:, None])  # (GROUP, POOL_BLOCK_SIZE) fp32
                l_i = l_i * alpha + tl.sum(p, axis=1)

                v_offsets = (
                    block_id * stride_kv_blk
                    + pos_in_block[:, None] * stride_kv_pos
                    + kv_h * stride_kv_h
                    + d_offsets[None, :] * stride_kv_d
                )
                v = tl.load(V_pool_ptr + v_offsets, mask=mask[:, None], other=0.0)
                # PV in fp32 (weights are fp32; upcast V) to stay closest to
                # the oracle; the kernel is bandwidth-bound at these sizes, so
                # the slower fp32 MMA is not on the critical path.
                acc = acc * alpha[:, None] + tl.dot(
                    p, v.to(tl.float32), input_precision="ieee", out_dtype=tl.float32
                )
                m_i = m_new

        out = acc / l_i[:, None]
        out_offsets = (
            req_idx * stride_o_b
            + (kv_h * GROUP + g_offsets[:, None]) * stride_o_h
            + d_offsets[None, :] * stride_o_d
        )
        tl.store(Out_ptr + out_offsets, out)


def msa_paged_decode_triton(
    q: torch.Tensor,
    k_pool_layer: torch.Tensor,
    v_pool_layer: torch.Tensor,
    block_tables: list[torch.Tensor],
    seq_lens: list[int],
    selected_blocks: list[torch.Tensor],
    *,
    index_block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """CUDA-only Triton launcher. Same contract as `msa_paged_decode_torch`."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton not available; use msa_paged_decode_torch instead")
    require_cuda_device(q.device, "MSA block-sparse decode kernel")

    batch, num_q_heads, head_dim = q.shape
    _, pool_block_size, num_kv_heads, _ = k_pool_layer.shape
    if index_block_size % pool_block_size != 0:
        raise ValueError(
            f"index_block_size={index_block_size} must be a multiple of the pool "
            f"block_size={pool_block_size} for the block-sparse decode path"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    group = num_q_heads // num_kv_heads
    if group < 16 or pool_block_size < 16 or head_dim < 16:
        # tl.dot needs every MMA dim >= 16; tiny shapes take the torch reference.
        return msa_paged_decode_torch(
            q,
            k_pool_layer,
            v_pool_layer,
            block_tables,
            seq_lens,
            selected_blocks,
            index_block_size=index_block_size,
            scale=scale,
        )

    num_kv_heads_local = k_pool_layer.shape[2]
    per_req_head = [
        [
            _selected_pool_entries(
                selected_blocks[r][kv_h],
                block_tables[r],
                seq_lens[r],
                index_block_size,
                pool_block_size,
            )
            for kv_h in range(num_kv_heads_local)
        ]
        for r in range(batch)
    ]
    max_entries = max(ids.numel() for heads in per_req_head for ids, _ in heads)
    pool_ids = torch.zeros(
        (batch, num_kv_heads_local, max_entries), dtype=torch.int32, device=q.device
    )
    base_pos = torch.full(
        (batch, num_kv_heads_local, max_entries), -1, dtype=torch.int32, device=q.device
    )
    for r, heads in enumerate(per_req_head):
        for kv_h, (ids, base) in enumerate(heads):
            pool_ids[r, kv_h, : ids.numel()] = ids
            base_pos[r, kv_h, : base.numel()] = base
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)

    q_c = q.contiguous()
    out_fp32 = torch.empty((batch, num_q_heads, head_dim), dtype=torch.float32, device=q.device)
    grid = (batch, num_kv_heads)
    _msa_paged_decode_kernel[grid](
        q_c,
        k_pool_layer,
        v_pool_layer,
        pool_ids,
        base_pos,
        seq_lens_t,
        out_fp32,
        max_entries,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        k_pool_layer.stride(0),
        k_pool_layer.stride(1),
        k_pool_layer.stride(2),
        k_pool_layer.stride(3),
        out_fp32.stride(0),
        out_fp32.stride(1),
        out_fp32.stride(2),
        GROUP=group,
        HEAD_DIM=head_dim,
        POOL_BLOCK_SIZE=pool_block_size,
        SCALE=scale,
    )
    return out_fp32.to(q.dtype)


def msa_paged_decode(
    q: torch.Tensor,
    k_pool_layer: torch.Tensor,
    v_pool_layer: torch.Tensor,
    block_tables: list[torch.Tensor],
    seq_lens: list[int],
    selected_blocks: list[torch.Tensor],
    *,
    index_block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Dispatcher: Triton on CUDA, pure PyTorch otherwise."""
    if supports_msa_kernel(q.device):
        return msa_paged_decode_triton(
            q,
            k_pool_layer,
            v_pool_layer,
            block_tables,
            seq_lens,
            selected_blocks,
            index_block_size=index_block_size,
            scale=scale,
        )
    return msa_paged_decode_torch(
        q,
        k_pool_layer,
        v_pool_layer,
        block_tables,
        seq_lens,
        selected_blocks,
        index_block_size=index_block_size,
        scale=scale,
    )
