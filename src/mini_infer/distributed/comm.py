"""Collective communication wrappers used by the TP layer.

These are very thin: they exist mainly to provide *no-op fast paths when
`world_size == 1`* so the rest of the codebase can call them
unconditionally. Without that, every TP-aware module would have to guard
each collective with `if world_size > 1` — which is noisy and easy to get
wrong (forgetting one such guard means the single-device test path quietly
calls into `torch.distributed` and crashes).

We mirror the names from `torch.distributed` rather than introducing new
names, so anyone familiar with NCCL / Megatron can read this module without
a glossary.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from mini_infer.distributed.group import get_world_size, is_initialized


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place sum-reduce across all ranks.

    Used for: row-parallel linear output, vocab-parallel embedding output.

    No-op (returns input unchanged) when world_size == 1. This is the path
    single-device tests take, which is why we don't allocate a temporary.
    """
    if get_world_size() == 1:
        return tensor
    if not is_initialized():
        # Defensive: if a caller asks for a collective with no PG, that's a
        # programming error, not a silent no-op.
        raise RuntimeError(
            "all_reduce_sum called with world_size > 1 but no process group is initialised"
        )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_along_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Gather `tensor` from every rank and concatenate along `dim`.

    Used for: column-parallel linear output when the next op needs the full
    activation; lm-head logits before sampling.

    Returns a fresh tensor (input is not modified in place). On
    `world_size == 1` returns the input unchanged.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    if not is_initialized():
        raise RuntimeError(
            "all_gather_along_dim called with world_size > 1 but no process group is initialised"
        )

    # We use the list-based `dist.all_gather` API rather than
    # `all_gather_into_tensor` because the latter requires the output
    # tensor to be flat-stacked along dim 0; concatenating along an
    # arbitrary `dim` afterwards would need extra movedim/reshape
    # gymnastics that the list API avoids. The cost is one extra `cat`
    # call, which is cheap relative to the collective itself.
    gathered_per_rank: list[torch.Tensor] = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_per_rank, tensor.contiguous())
    return torch.cat(gathered_per_rank, dim=dim)


def all_to_all_single(
    output: torch.Tensor,
    input_: torch.Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
) -> None:
    """Send `input_split_sizes[i]` rows to rank `i`, receive `output_split_sizes[i]`.

    Used for: expert-parallel MoE token dispatch (Phase 3). The split-size
    arguments are how each rank says "I'm sending this many tokens to each
    other rank"; the symmetry comes from each rank computing its own
    output_split_sizes from the dispatched expert IDs.

    No-op (`output.copy_(input_)`) when `world_size == 1`.
    """
    if get_world_size() == 1:
        if output.shape != input_.shape:
            raise ValueError(
                "all_to_all_single at world_size=1 requires output.shape == input_.shape, "
                f"got {tuple(output.shape)} vs {tuple(input_.shape)}"
            )
        output.copy_(input_)
        return
    if not is_initialized():
        raise RuntimeError(
            "all_to_all_single called with world_size > 1 but no process group is initialised"
        )
    dist.all_to_all_single(
        output,
        input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast `tensor` from rank `src` to all ranks.

    Used for: shipping the same input prompt token IDs to every rank during
    prefill (every rank needs them since embedding is vocab-parallel).
    No-op when world_size == 1.
    """
    if get_world_size() == 1:
        return tensor
    if not is_initialized():
        raise RuntimeError(
            "broadcast called with world_size > 1 but no process group is initialised"
        )
    dist.broadcast(tensor, src=src)
    return tensor


def barrier() -> None:
    """Block until every rank reaches this point. No-op at world_size == 1."""
    if get_world_size() == 1:
        return
    if not is_initialized():
        raise RuntimeError("barrier called with world_size > 1 but no process group is initialised")
    dist.barrier()
