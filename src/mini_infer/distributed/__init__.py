"""Tensor-parallel infrastructure for mini-infer.

This package provides Megatron-style TP primitives:

  - `init_distributed` / `destroy_distributed` / `get_world_size` /
    `get_rank` for process-group lifecycle.
  - `all_reduce_sum`, `all_gather_along_dim`, `all_to_all_single`,
    `broadcast`, `barrier` for the collectives we use.
  - `ColumnParallelLinear`, `RowParallelLinear` for sharded linear layers.
  - `VocabParallelEmbedding` for sharded embedding tables.

All TP-aware modules degrade to their plain-PyTorch equivalents when
`world_size == 1`, so the existing single-device unit tests stay
bit-identical and don't need to know about TP.
"""

from mini_infer.distributed.comm import (
    all_gather_along_dim,
    all_reduce_sum,
    all_to_all_single,
    barrier,
    broadcast,
)
from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.group import (
    destroy_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_initialized,
)
from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mini_infer.distributed.loader import load_state_dict_with_tp

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "all_gather_along_dim",
    "all_reduce_sum",
    "all_to_all_single",
    "barrier",
    "broadcast",
    "destroy_distributed",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_initialized",
    "load_state_dict_with_tp",
]
