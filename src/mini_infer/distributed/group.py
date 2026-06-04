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
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

# When True, `get_world_size()` / `get_rank()` report (1, 0) regardless of the
# active process group. This is for ranks that participate in a multi-rank PG
# for some OTHER purpose than tensor parallelism, and need their model built +
# run as a full, un-sharded replica with no TP collectives.
#
# The motivating case is prefill/decode (PD) disaggregation: the prefill rank
# and decode rank form a 2-rank PG, but each runs a COMPLETE model and they
# communicate only via an explicit `kv_transfer` send/recv handoff, never via
# TP all-reduces. Without this, the model's TP-aware layers see world_size=2,
# insert all-reduce collectives, and rank 0's prefill deadlocks on the first
# collective (the token embedding) waiting for rank 1, which is parked in
# `recv_handoff`. The handoff itself uses `dist.send`/`dist.recv` + the torch
# `dist.get_rank()` directly, so it is unaffected by this override.
#
# It must stay active for the worker's whole lifetime (construction AND every
# forward): some layers capture world_size at construction (embedding), but
# others call `all_reduce_sum` unconditionally and rely on its live
# world_size==1 no-op (MoE). A construction-only scope would miss the latter.
_FORCE_REPLICA = False


@contextmanager
def replica_scope() -> Iterator[None]:
    """Force single-replica (no-TP) topology for the duration of the scope.

    Within the scope `get_world_size()` returns 1 and `get_rank()` returns 0,
    so TP-aware modules build + run as plain replicas with no collective calls.
    Wrap a PD worker's entire model lifetime (load + prefill / decode) in this;
    the `kv_transfer` handoff inside the scope still uses the real process group
    via `dist.send`/`dist.recv`. Nesting is supported.
    """
    global _FORCE_REPLICA
    previous = _FORCE_REPLICA
    _FORCE_REPLICA = True
    try:
        yield
    finally:
        _FORCE_REPLICA = previous


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
    """Number of TP ranks; 1 when no PG is active or inside `replica_scope()`.

    Returning 1 (instead of raising) is what lets TP-aware modules build
    transparently in single-device tests. Inside `replica_scope()` it returns
    1 even under a live multi-rank PG, so a PD worker builds a full replica.
    """
    if _FORCE_REPLICA:
        return 1
    if is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """This rank's TP index; 0 when no PG is active or inside `replica_scope()`."""
    if _FORCE_REPLICA:
        return 0
    if is_initialized():
        return dist.get_rank()
    return 0
