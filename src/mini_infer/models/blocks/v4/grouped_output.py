"""Grouped Output Projection (V4 paper §2.3.2, "Shared KV MQA" paragraph).

Standard `o_proj` is one linear from `n_h * c` to `d`. With V4-Pro's
`n_h=128`, that's a (32k -> 7k) matmul on every token. The grouped
form splits the `n_h` query heads into `g` groups, projects each group
independently through a low-rank `(c·n_h/g) -> r` matrix, concatenates
the `g` group outputs, then a final `(g·r) -> d` projection. Same
parameter count as the monolithic version, but each per-group GEMM
has lower compute when `n_h` is large.

Layout matches the reference V4 inference code:
- `wo_a`: `(g * r, n_h/g * c)`. Reshaped to `(g, r, n_h/g * c)` and
  applied via `einsum("bsgd,grd->bsgr", per_group_input, wo_a)`.
- `wo_b`: standard `nn.Linear((g*r) -> d)` for the final projection.

Edge case `g == 1` reduces to a normal low-rank `o_proj` (one
big matmul); the parity test sets g=1 to verify monolithic
equivalence.

Tensor parallelism
------------------
Sharded by group: each rank owns `num_groups // world_size` groups.
`wo_a`'s first axis (`num_groups * o_lora_rank`) is sliced — each rank
holds the wo_a entries that match its local groups. `wo_b` is
row-parallel along its input (`num_groups * o_lora_rank`) so the final
projection emits the full hidden state via one all-reduce. Constraint:
`num_groups` must be divisible by `world_size`. For V4-Pro with
`num_groups=2` this caps useful TP at 2 ranks for a single layer's
output projection; future Phase-2 work can also support sharding
within a group (along `heads_per_group`) for higher TP factors.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.distributed.linear import RowParallelLinear, _split_size


class GroupedOutputProjection(nn.Module):
    """Per-group low-rank output projection."""

    def __init__(
        self,
        *,
        num_heads: int,
        kv_head_dim: int,
        num_groups: int,
        o_lora_rank: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if num_heads % num_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups})"
            )
        from mini_infer.distributed.group import get_rank, get_world_size

        world_size = get_world_size()
        num_groups_per_rank = _split_size(num_groups, world_size, "num_groups")
        self.num_heads = num_heads
        self.kv_head_dim = kv_head_dim
        self.num_groups = num_groups
        self.num_groups_per_rank = num_groups_per_rank
        self.heads_per_group = num_heads // num_groups
        self.o_lora_rank = o_lora_rank
        self.hidden_size = hidden_size
        self.world_size = world_size
        self.rank = get_rank()

        # `wo_a` is logically `(num_groups, o_lora_rank, heads_per_group * kv_head_dim)`
        # stored flat as `(num_groups * o_lora_rank, heads_per_group * kv_head_dim)`.
        # Sharded by group: each rank holds `num_groups_per_rank * o_lora_rank` rows.
        self.wo_a = nn.Parameter(
            torch.empty(num_groups_per_rank * o_lora_rank, self.heads_per_group * kv_head_dim)
        )
        # `wo_b` is row-parallel: takes the (sharded) `num_groups * o_lora_rank`
        # input and all-reduces to produce the full hidden state.
        self.wo_b = RowParallelLinear(num_groups * o_lora_rank, hidden_size, bias=False)

    def load_full_wo_a(
        self,
        full_wo_a: torch.Tensor,
        *,
        target_device: torch.device | str | None = None,
    ) -> None:
        """Slice this rank's group rows from the full `wo_a`.

        Full shape: `(num_groups * o_lora_rank, heads_per_group * kv_head_dim)`.
        """
        expected_shape = (
            self.num_groups * self.o_lora_rank,
            self.heads_per_group * self.kv_head_dim,
        )
        if full_wo_a.shape != expected_shape:
            raise ValueError(
                f"full_wo_a shape {tuple(full_wo_a.shape)} does not match expected {expected_shape}"
            )
        rows_per_rank = self.num_groups_per_rank * self.o_lora_rank
        start = self.rank * rows_per_rank
        end = start + rows_per_rank
        if self.wo_a.is_meta:
            sliced = full_wo_a[start:end].contiguous()
            if target_device is not None:
                sliced = sliced.to(device=target_device)
            self.wo_a = nn.Parameter(sliced, requires_grad=False)
        else:
            sliced = full_wo_a[start:end].to(self.wo_a.dtype).contiguous()
            with torch.no_grad():
                self.wo_a.copy_(sliced)

    def forward(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Project `(B, T, num_heads_local, kv_head_dim)` to `(B, T, hidden_size)`.

        At `world_size=1`, `num_heads_local == num_heads` and the whole
        chain is bit-identical to the un-sharded grouped-output formula.
        """
        bsz, seqlen, n_h_local, d = attn_out.shape
        expected_n_h_local = self.num_groups_per_rank * self.heads_per_group
        if n_h_local != expected_n_h_local or d != self.kv_head_dim:
            raise ValueError(
                f"attn_out shape {attn_out.shape} does not match "
                f"(num_heads_local={expected_n_h_local}, kv_head_dim={self.kv_head_dim})"
            )
        # Group the heads: each group holds `heads_per_group` head outputs flattened.
        per_group = attn_out.view(
            bsz,
            seqlen,
            self.num_groups_per_rank,
            self.heads_per_group * self.kv_head_dim,
        )
        # Per-group projection: (B, T, g_local, in_features) @ (g_local, r, in_features)
        # -> (B, T, g_local, r). Each rank only computes its own groups.
        wo_a = self.wo_a.view(self.num_groups_per_rank, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", per_group, wo_a)
        out = out.flatten(2)  # (B, T, num_groups_per_rank * o_lora_rank)
        # `wo_b` is row-parallel: takes the sharded input and all-reduces.
        result: torch.Tensor = self.wo_b(out)
        return result
