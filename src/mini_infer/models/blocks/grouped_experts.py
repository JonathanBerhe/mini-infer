"""Routed MoE experts as two grouped matmuls over all (token, expert) pairs.

The per-expert loop in the MoE blocks gathers one expert's tokens, runs its
MLP and scatters the result back. That is about ten small launches per
expert, plus a host sync to learn which experts got any tokens. Here the
(token, expert) pairs are sorted by expert on the GPU and each projection
runs once over every expert's slice of the sorted rows. Nothing reads back
to the host, so the same code also runs inside a captured CUDA graph.

The weights must be stacked per expert: `gate_up` `(E, G, in)` and `down`
`(E, out, I)`. `G` is `I` for a plain MLP and `2I` for a gated one with the
gate rows first. Plain tensors (bf16) go through `functional.grouped_mm`;
`BlockFp8Weight`s (e4m3 with 128x128 block scales) go through
`fp8_grouped_kernel.block_fp8_grouped_mm`, a Triton kernel or, for many rows
on sm90 and sm100, a Gluon kernel. `BlockFp4Weight`s (packed e2m1 with one
scale per row and 32 columns) go through the Triton kernel of
`fp4_grouped_kernel.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional

from mini_infer.device import is_hopper_or_newer_device
from mini_infer.models.blocks.fp4_grouped_kernel import block_fp4_grouped_mm
from mini_infer.models.blocks.fp8_grouped_kernel import block_fp8_grouped_mm

# Runtime toggle for benchmarking. When True, `supports_grouped_experts`
# reports False, so the blocks run their per-expert path. The bench uses it to
# A/B the two paths on one model load. Not a public API.
_DISABLED_FOR_BENCH = False


def supports_grouped_experts(device: torch.device | str) -> bool:
    """Whether the MoE blocks route their experts through `grouped_expert_mlp`.

    `functional.grouped_mm` runs CUTLASS grouped GEMMs on Hopper and
    Blackwell. Older GPUs and the CPU get a per-group fallback inside PyTorch
    that is no faster than the blocks' own loop, so they keep the loop.
    """
    if _DISABLED_FOR_BENCH or not hasattr(functional, "grouped_mm"):
        return False
    return is_hopper_or_newer_device(device)


@dataclass(frozen=True)
class BlockFp8Weight:
    """Stacked block-FP8 expert weights for `grouped_expert_mlp`.

    `weight` is `(E, N, K)` `float8_e4m3fn` and `scale` its fp32 scale per
    `block x block` tile, `(E, ceil(N / block), ceil(K / block))`.
    """

    weight: torch.Tensor
    scale: torch.Tensor
    block: int

    @property
    def shape(self) -> torch.Size:
        return self.weight.shape


@dataclass(frozen=True)
class BlockFp4Weight:
    """Stacked packed-FP4 expert weights for `grouped_expert_mlp`.

    `packed` is `(E, N, K / 2)` bytes, each holding two e2m1 values (the low
    nibble first), and `scale` its fp32 scale per row and `block` columns,
    `(E, N, K / block)`.
    """

    packed: torch.Tensor
    scale: torch.Tensor
    block: int

    @property
    def shape(self) -> torch.Size:
        """`(E, N, K)`, the shape of the dequantized weights."""
        experts, rows, half_cols = self.packed.shape
        return torch.Size((experts, rows, half_cols * 2))


ExpertWeight = torch.Tensor | BlockFp8Weight | BlockFp4Weight


def _grouped_matmul(
    rows: torch.Tensor,
    weight: ExpertWeight,
    offs: torch.Tensor,
    row_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """`(M, K)` sorted rows times their expert's `(N, K)` weight, `-> (M, N)`."""
    if isinstance(weight, BlockFp8Weight):
        return block_fp8_grouped_mm(
            rows,
            weight.weight,
            weight.scale,
            block=weight.block,
            row_experts=row_experts,
            group_ends=offs,
            num_experts=num_experts,
        )
    if isinstance(weight, BlockFp4Weight):
        return block_fp4_grouped_mm(
            rows,
            weight.packed,
            weight.scale,
            block=weight.block,
            row_experts=row_experts,
            group_ends=offs,
            num_experts=num_experts,
        )
    return functional.grouped_mm(rows, weight.transpose(-2, -1), offs=offs)


def grouped_expert_mlp(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up: ExpertWeight,
    down: ExpertWeight,
    activation: Callable[[torch.Tensor], torch.Tensor],
    *,
    num_experts: int,
    expert_start: int = 0,
    accumulate_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Weighted sum of each token's routed experts, `(T, in)` -> `(T, out)`.

    `topk_indices` and `topk_weights` are `(T, k)`: global expert ids and
    their routing weights. `gate_up` and `down` hold this rank's experts,
    global ids `[expert_start, expert_start + E)`. When `num_experts` is
    larger than `E` (expert parallelism), pairs routed to other ranks add
    zero, as in the per-expert loop. `activation` maps the first projection
    `(rows, G)` to `(rows, I)`. The weighted sum over a token's k experts
    runs in `accumulate_dtype` (default `x.dtype`). The result has
    `out_dtype` (default `x.dtype`).
    """
    tokens, top_k = topk_indices.shape
    num_local = gate_up.shape[0]
    result_dtype = out_dtype or x.dtype
    if tokens == 0:
        return x.new_zeros(0, down.shape[1], dtype=result_dtype)
    acc_dtype = accumulate_dtype or x.dtype

    local_ids = topk_indices.reshape(-1)
    if expert_start:
        local_ids = local_ids - expert_start
    partial = num_local != num_experts
    if partial:
        # Off-rank pairs get id E: they sort after every local group.
        on_rank = (local_ids >= 0) & (local_ids < num_local)
        local_ids = torch.where(on_rank, local_ids, num_local)
    sorted_ids, order = torch.sort(local_ids, stable=True)
    # offs[e] is the end row of expert e's group in the sorted order.
    bounds = torch.arange(num_local, device=x.device, dtype=sorted_ids.dtype)
    offs = torch.searchsorted(sorted_ids, bounds, right=True).to(torch.int32)

    rows = x.index_select(0, order // top_k)
    hidden = activation(_grouped_matmul(rows, gate_up, offs, sorted_ids, num_experts))
    out_rows = _grouped_matmul(hidden, down, offs, sorted_ids, num_experts)

    weighted = out_rows.to(acc_dtype) * topk_weights.reshape(-1)[order].to(acc_dtype)[:, None]
    if partial:
        # Rows past the last group are never written by the grouped matmul.
        weighted = torch.where((sorted_ids < num_local)[:, None], weighted, 0)
    per_pair = torch.empty_like(weighted).index_copy_(0, order, weighted)
    return per_pair.view(tokens, top_k, -1).sum(dim=1).to(result_dtype)
