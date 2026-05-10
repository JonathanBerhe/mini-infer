"""Process-group lifecycle for tensor-parallel inference.

mini-infer's TP layer is built on top of `torch.distributed`. This module is
the single point that owns *initialisation* and *teardown* of the global
process group, plus tiny accessors so the rest of the code never has to
guard `dist.is_initialized()` itself.

Contract
--------
- Every TP-aware module (`ColumnParallelLinear`, `VocabParallelEmbedding`, ...)
  reads `get_world_size()` / `get_rank()` at construction time and slices its
  weight buffer accordingly.
- Outside a multi-process launch, `get_world_size()` returns 1 and
  `get_rank()` returns 0. The TP-aware modules then behave as ordinary
  `nn.Linear` / `nn.Embedding` modules with no collective calls. This is the
  invariant that keeps the existing single-device unit tests bit-identical.
- Multi-process tests / production launches call `init_distributed(...)`
  exactly once at startup; the matching `destroy_distributed()` is called at
  shutdown (or in a `finally:` block in test harnesses).

Backend selection
-----------------
`backend="auto"` picks `nccl` when CUDA is available (production), else
`gloo` (CPU multi-process tests on macOS / CI). Callers can force either.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed(
    world_size: int,
    rank: int,
    *,
    backend: str = "auto",
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    """Initialise the global process group.

    Idempotent under "already initialised" only when the existing world_size /
    rank match; mismatched re-init raises so a misconfigured test fails loudly
    rather than silently using stale collective topology.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    if dist.is_available() and dist.is_initialized():
        existing_world = dist.get_world_size()
        existing_rank = dist.get_rank()
        if existing_world != world_size or existing_rank != rank:
            raise RuntimeError(
                f"init_distributed called with world_size={world_size}, rank={rank} "
                f"but process group already initialised with "
                f"world_size={existing_world}, rank={existing_rank}"
            )
        return

    # `world_size=1` is a useful no-op: the same code path the multi-rank
    # launcher takes, but skipped here so single-device tests don't have to
    # spin up a real PG (NCCL on a 1-rank world is wasteful; gloo on macOS
    # is finicky).
    if world_size == 1:
        return

    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # `torch.distributed` reads these from the environment. We set them here
    # so callers don't have to remember the magic variable names.
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


def destroy_distributed() -> None:
    """Tear down the global process group if one was created.

    Safe to call from a `finally:` block whether or not init succeeded.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_initialized() -> bool:
    """True iff a real (multi-rank) process group is live."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Number of ranks; 1 when no PG is active.

    Returning 1 (instead of raising) is what lets TP-aware modules build
    transparently in single-device tests.
    """
    if is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """This rank's index in the PG; 0 when no PG is active."""
    if is_initialized():
        return dist.get_rank()
    return 0
