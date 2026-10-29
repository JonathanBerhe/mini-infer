"""AttnRes: Kimi K3's attention over the residual stream's depth.

A plain pre-norm transformer feeds each sublayer the running sum of every
earlier sublayer's output (the residual stream). K3 (`attn_res_block_size`)
keeps that stream as a short list of **blocks** instead: every
`attn_res_block_size` layers the running sum is snapshotted into the list
and restarted from zero. Before each attention and each FFN, and once more
before the final norm, the layer picks its input by a per-token softmax
attention over the snapshots plus the current running sum, where the single
learned query is the product of an RMSNorm weight and a `hidden -> 1`
projection (`self_attention_res_norm/proj`, `mlp_res_norm/proj`,
`output_attn_res_norm/proj` in the checkpoint).

Pinned from the reference `_apply_attn_res`: the candidates are RMS-normalized
(no affine) only for SCORING; the softmax mixes the UNNORMALIZED candidates;
everything runs in fp32 and casts back to the stream's dtype. The mix is
per token, so decode carries no extra cross-token state; the per-token cost
is `num_blocks` extra hidden vectors during the forward.
"""

from __future__ import annotations

import torch
from torch import nn


class AttnResQueryNorm(nn.Module):
    """The RMSNorm affine weight of one AttnRes query, kept under the checkpoint name.

    The reference stores it as a `KimiRMSNorm` (`*_res_norm.weight`), but
    never calls that norm: `_apply_attn_res` only multiplies its `weight`
    into the scoring query, and the eps it uses is `config.rms_norm_eps`.
    So this module holds the weight and has no forward, which keeps anyone
    from normalizing the candidates twice by calling it.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


def attn_res_mix(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Softmax-mix the stored blocks and the current running sum, per token.

    Args:
        prefix_sum: `(..., hidden)` the running residual sum.
        block_residual: `(..., num_blocks, hidden)` earlier snapshots
            (`num_blocks` may be 0, in which case the result is `prefix_sum`).
        proj_weight: `(1, hidden)` or `(hidden,)` learned query projection.
        norm_weight: `(hidden,)` RMSNorm affine weight folded into the query.
        eps: RMSNorm epsilon.

    Returns:
        `(..., hidden)` in `prefix_sum.dtype`.
    """
    candidates = torch.cat([block_residual, prefix_sum.unsqueeze(-2)], dim=-2)
    values = candidates.float()
    keys = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps)
    query = norm_weight.float() * proj_weight.float().reshape(-1)
    scores = (keys * query).sum(dim=-1)  # (..., num_blocks + 1)
    probs = scores.softmax(dim=-1)
    mixed = (probs.unsqueeze(-1) * values).sum(dim=-2)
    return mixed.to(prefix_sum.dtype)
