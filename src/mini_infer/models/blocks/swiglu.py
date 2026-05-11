"""SwiGLU FFN block (Llama / Qwen2 / Mistral).

Three projections: `gate_proj` (silu-gated), `up_proj` (passed through),
`down_proj` (back to hidden_size). Names align with HF so weight loading
is identity.

Tensor parallelism
------------------
Standard Megatron pairing: `gate_proj` and `up_proj` are column-parallel
along `intermediate_size`; `down_proj` is row-parallel along its input.
SiLU and elementwise multiply commute with the column-sharding (they're
per-feature), so the column-then-row chain produces exactly one
all-reduce per FFN. At `world_size=1` it reduces to plain `nn.Linear`.
"""

import torch
from torch import nn
from torch.nn import functional

from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # Column-parallel: each rank holds `intermediate_size // world_size`
        # output features. SiLU + elementwise mul operate per-feature so they
        # work on the sharded activation as-is.
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        # Row-parallel: input is the sharded gated activation, output is the
        # all-reduced full hidden state. One all-reduce per FFN.
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return out
