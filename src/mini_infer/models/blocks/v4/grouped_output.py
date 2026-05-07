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
"""

from __future__ import annotations

import torch
from torch import nn


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
        self.num_heads = num_heads
        self.kv_head_dim = kv_head_dim
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.o_lora_rank = o_lora_rank
        self.hidden_size = hidden_size

        # `wo_a` is logically `(num_groups, o_lora_rank, heads_per_group * kv_head_dim)`
        # but we store it flat as `(num_groups * o_lora_rank, heads_per_group * kv_head_dim)`
        # so it loads from a checkpoint via a single `nn.Linear`-shaped weight tensor
        # (matching the reference `wo_a.weight` exactly).
        self.wo_a = nn.Parameter(
            torch.empty(num_groups * o_lora_rank, self.heads_per_group * kv_head_dim)
        )
        # Final per-token projection.
        self.wo_b = nn.Linear(num_groups * o_lora_rank, hidden_size, bias=False)

    def forward(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Project `(B, T, num_heads, kv_head_dim)` to `(B, T, hidden_size)`."""
        bsz, seqlen, n_h, d = attn_out.shape
        if n_h != self.num_heads or d != self.kv_head_dim:
            raise ValueError(
                f"attn_out shape {attn_out.shape} does not match "
                f"(num_heads={self.num_heads}, kv_head_dim={self.kv_head_dim})"
            )
        # Group the heads: each group holds `heads_per_group` head outputs flattened.
        per_group = attn_out.view(
            bsz, seqlen, self.num_groups, self.heads_per_group * self.kv_head_dim
        )
        # Per-group projection: (B, T, g, in_features) @ (g, r, in_features) -> (B, T, g, r).
        wo_a = self.wo_a.view(self.num_groups, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", per_group, wo_a)
        # Concat groups + final projection.
        out = out.flatten(2)  # (B, T, num_groups * o_lora_rank)
        result: torch.Tensor = self.wo_b(out)
        return result
