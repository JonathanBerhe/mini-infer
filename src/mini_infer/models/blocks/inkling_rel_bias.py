"""Inkling relative position bias: learned, query-conditioned, distance-indexed.

Inkling uses no RoPE and no absolute position embedding. Instead each
attention layer adds a bias to the attention logits (transformers 5.14
`InklingRelativeLogits`):

    rel_states = r_proj(hidden)                # (q, heads, d_rel)
    profiles   = rel_states @ proj             # (q, heads, rel_extent)
    bias[q, h, k] = profiles[q, h, q_pos - k_pos]   if 0 <= distance < rel_extent
                    0                                otherwise

`proj` is a trained bank of `d_rel` bias-vs-distance profiles; each query
token mixes them into one bias value per backward distance. Causality and
the sliding window are NOT part of the bias (they stay in the attention
mask); the bias is simply zero outside the learned extent. Sliding layers
use `rel_extent = sliding_window_size`; global layers use the config's
`rel_extent` (1024 for the released checkpoints), so a global layer's bias
influences only the most recent `rel_extent` keys even at 1M context.
"""

from __future__ import annotations

import torch
from torch import nn


class InklingRelativeLogits(nn.Module):
    """The learned profile bank. `proj` is `(d_rel, rel_extent)`, HF name
    `rel_logits_proj.proj` (a bare Parameter, not a Linear)."""

    def __init__(self, d_rel: int, rel_extent: int) -> None:
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent))

    def forward(
        self,
        relative_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Bias for one request.

        Args:
            relative_states: `(q_len, num_heads, d_rel)` from `r_proj`.
            query_positions: `(q_len,)` absolute positions of the queries.
            key_positions: `(k_len,)` absolute positions of the keys.

        Returns:
            `(q_len, num_heads, k_len)` additive bias, zero outside
            `0 <= q_pos - k_pos < rel_extent`.
        """
        # (q, heads, extent): per-token bias profile over backward distance.
        rel_logits = relative_states @ self.proj
        distance = query_positions[:, None] - key_positions[None, :]  # (q, k)
        gather_index = distance.clamp(0, self.rel_extent - 1)
        gather_index = gather_index[:, None, :].expand(-1, rel_logits.shape[1], -1)
        position_bias = rel_logits.gather(-1, gather_index)
        out_of_extent = (distance < 0) | (distance >= self.rel_extent)
        return position_bias.masked_fill(out_of_extent[:, None, :], 0.0)
