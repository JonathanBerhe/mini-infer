"""Hyper-Connections (DeepSeek-V4 paper §2.5).

Replaces the standard `x = x + sublayer(norm(x))` residual with a
multi-residual mixing scheme: the hidden state carries `hc_mult` copies,
and each sublayer call reduces those copies into one (for the sublayer
input) then expands the sublayer output back into `hc_mult` copies
mixed against the residual via a Sinkhorn-normalized combination matrix.

Structure of one HC-mediated residual (one attention OR one FFN call):

    residual: (B, T, hc_mult, dim)
        |
        +-- hc_pre  -> sublayer_input: (B, T, dim)        + post: (B, T, hc_mult)
                                                          + comb: (B, T, hc_mult, hc_mult)
                                                                 (doubly-stochastic after Sinkhorn)
        |
        sublayer_input -> RMSNorm -> attn / ffn -> sublayer_output: (B, T, dim)
        |
        +-- hc_post(sublayer_output, residual, post, comb) -> output: (B, T, hc_mult, dim)

`hc_pre` and `hc_post` together replace `x + sublayer(norm(x))`. With
`hc_mult=1` the design degenerates to a standard residual.

The math comes directly from the V4 reference's `Block.hc_pre` /
`Block.hc_post` plus the `hc_split_sinkhorn` tilelang kernel
(`third_party/deepseek_v4_reference/kernel.py`). `hc_split_sinkhorn`
is transcribed line-by-line into PyTorch here:

    pre[i,j]    = sigmoid(mixes[i,j] * scale[0] + base[j]) + eps
    post[i,j]   = 2 * sigmoid(mixes[i,j+hc] * scale[1] + base[j+hc])
    comb[i,j,k] = mixes[i,j*hc+k+2*hc] * scale[2] + base[j*hc+k+2*hc]
    # Sinkhorn-normalize comb to (approximately) doubly stochastic:
    comb = softmax_row(comb) + eps
    comb = comb / (sum_row(comb) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (sum_col(comb) + eps)
        comb = comb / (sum_row(comb) + eps)
    # (the kernel's column normalize swap order is the first iteration's
    #  second step; subsequent iterations alternate row-then-column.)

After all iterations comb's row sums and column sums are both
approximately 1, modulo the additive-eps regularization.

The `hc_eps` parameter is added at every normalization step to keep
the gradient finite when row/col sums are tiny — same guard the
kernel's `eps` argument supplies.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import linear


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split `mixes` into (pre, post, comb) and Sinkhorn-normalize comb.

    Pure-PyTorch transcription of the V4 reference's `hc_split_sinkhorn`
    tilelang kernel. Used as the kernel's PyTorch shim in test environments.

    Args:
        mixes: `(..., (2 + hc_mult) * hc_mult)` raw scores from the linear
            preceding the split.
        hc_scale: `(3,)` per-chunk scaling factor (pre / post / comb).
        hc_base: `((2 + hc_mult) * hc_mult,)` per-feature additive bias.
        hc_mult: Number of residual copies (`hc` in the kernel).
        sinkhorn_iters: How many alternating row+col normalization
            iterations to run on the comb matrix. The reference uses 20.
        eps: Additive regularizer used in two places: appended to the
            sigmoid output for `pre`, and added to the row/col sums in
            the denominator of every Sinkhorn normalization step.

    Returns:
        `(pre, post, comb)` where:
          - pre: `(..., hc_mult)` weights to reduce hc_mult copies → 1.
          - post: `(..., hc_mult)` weights to expand sublayer output → hc_mult copies.
          - comb: `(..., hc_mult, hc_mult)` Sinkhorn-normalized combination
            matrix (approximately doubly stochastic after the iterations).
    """
    if hc_scale.shape != (3,):
        raise ValueError(f"hc_scale must have shape (3,); got {tuple(hc_scale.shape)}")
    expected_mix_hc = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != expected_mix_hc:
        raise ValueError(
            f"mixes.shape[-1]={mixes.shape[-1]} must equal (2 + hc_mult) * hc_mult"
            f" = {expected_mix_hc}"
        )
    if hc_base.shape != (expected_mix_hc,):
        raise ValueError(
            f"hc_base must have shape ({expected_mix_hc},); got {tuple(hc_base.shape)}"
        )

    # Slice the mixes / base into the three chunks: pre (size hc_mult),
    # post (size hc_mult), comb (size hc_mult^2 → reshape to hc_mult x hc_mult).
    pre_features = mixes[..., :hc_mult]
    post_features = mixes[..., hc_mult : 2 * hc_mult]
    comb_features = mixes[..., 2 * hc_mult :].reshape(*mixes.shape[:-1], hc_mult, hc_mult)

    base_pre = hc_base[:hc_mult]
    base_post = hc_base[hc_mult : 2 * hc_mult]
    base_comb = hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)

    # ---- pre: per-copy reduction weight, sigmoid + eps ----
    pre = torch.sigmoid(pre_features * hc_scale[0] + base_pre) + eps

    # ---- post: per-copy expansion weight, 2 * sigmoid (no eps, scaled by 2) ----
    post = 2.0 * torch.sigmoid(post_features * hc_scale[1] + base_post)

    # ---- comb: linear combination + Sinkhorn-normalize ----
    comb = comb_features * hc_scale[2] + base_comb

    # First iteration step 1: row-softmax-with-eps. The kernel does
    #   comb = softmax(comb, dim=-1)
    #   comb = comb + eps
    # before the column normalization.
    comb = comb.softmax(dim=-1) + eps
    # First iteration step 2: column normalize, with eps in the denominator.
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    # Remaining iterations: row-then-column normalize, eps in each denominator.
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


class HyperConnections(nn.Module):
    """One HC-mediated residual (one sublayer's worth of multi-residual mixing).

    A `Block` (one transformer decoder layer) typically owns TWO instances
    of this — one for the attention sublayer, one for the FFN sublayer —
    so each has its own learnable `(fn, base, scale)` parameters.

    Args:
        hidden_size: Per-copy feature dim (the model's `dim`).
        hc_mult: Number of residual copies. With `hc_mult=1` the design
            collapses to a standard residual.
        sinkhorn_iters: Iterations of alternating row+col normalization
            in `hc_split_sinkhorn`. V4 reference uses 20.
        hc_eps: Additive regularizer in normalization. V4 reference uses 1e-6.
        rms_norm_eps: Used by the rsqrt normalization inside `hc_pre`
            (the kernel's `norm_eps` — distinct from `hc_eps`).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hc_mult <= 0:
            raise ValueError(f"hc_mult must be positive, got {hc_mult}")
        if sinkhorn_iters <= 0:
            raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_norm_eps = rms_norm_eps

        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_size
        # `fn`: linear projection from `(B, T, hc_mult * dim)` to `(B, T, mix_hc)`.
        # The split-into-(pre,post,comb) happens inside `hc_split_sinkhorn`.
        self.fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        # `base`: per-feature additive bias for the (pre, post, comb) chunks.
        self.base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        # `scale`: 3 scalars, one per chunk (pre, post, comb).
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, hc_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce `(B, T, hc_mult, dim)` → `(B, T, dim)` for sublayer input.

        Returns `(sublayer_input, post, comb)` where post + comb get fed
        into `hc_post` after the sublayer runs.

        Math (mirrors `Block.hc_pre` in the reference):
            x_flat = hc_state.flatten(2)            # (B, T, hc_mult * dim)
            rsqrt = rsqrt(x_flat^2.mean(-1) + rms_norm_eps)
            mixes = (x_flat @ fn.T) * rsqrt          # (B, T, mix_hc)
            pre, post, comb = hc_split_sinkhorn(mixes, scale, base, ...)
            sublayer_input = sum_h(pre[h] * hc_state[h])  # (B, T, dim)
        """
        original_dtype = hc_state.dtype
        hc_state_flat = hc_state.flatten(2).float()  # (B, T, hc_mult * dim)
        # RMS-norm-style rsqrt — this is the same `norm_eps` the kernel
        # threads through the rsqrt call, NOT `hc_eps`.
        rsqrt_factor = torch.rsqrt(
            hc_state_flat.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps
        )
        mixes = linear(hc_state_flat, self.fn) * rsqrt_factor

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=self.hc_eps,
        )

        # Reduce hc_mult copies into one via `pre` weighted sum along the hc axis.
        # `pre.unsqueeze(-1)` shape: (B, T, hc_mult, 1) broadcasts against
        # `hc_state` shape (B, T, hc_mult, dim) → sum over the hc axis.
        sublayer_input = torch.sum(pre.unsqueeze(-1) * hc_state, dim=2)
        return sublayer_input.to(original_dtype), post, comb

    def hc_post(
        self,
        sublayer_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand `(B, T, dim)` sublayer output back to `(B, T, hc_mult, dim)`.

        Args:
            sublayer_output: `(B, T, dim)` — the attn/ffn output for the
                reduced single-copy state.
            residual: `(B, T, hc_mult, dim)` — the hc_state from BEFORE
                this sublayer's `hc_pre` call.
            post: `(B, T, hc_mult)` — per-copy weight for the sublayer
                output (from `hc_pre`).
            comb: `(B, T, hc_mult, hc_mult)` — Sinkhorn-normalized
                combination matrix (from `hc_pre`).

        Math (mirrors `Block.hc_post`):
            out[h] = post[h] * sublayer_output + sum_k(comb[h, k] * residual[k])

        Returns:
            `(B, T, hc_mult, dim)` — the new hc_state passed to the next sublayer.
        """
        # post[h] * sublayer_output expands sublayer output into hc_mult copies.
        # post.unsqueeze(-1): (B, T, hc_mult, 1)
        # sublayer_output.unsqueeze(-2): (B, T, 1, dim)
        # product: (B, T, hc_mult, dim)
        scaled_sublayer = post.unsqueeze(-1) * sublayer_output.unsqueeze(-2)
        # Mix residual's hc_mult copies into the output's hc_mult copies.
        # The reference's convention: comb's COLUMN index labels the output
        # copy, ROW index labels the residual copy being read. So
        # `out[..., j, d] = sum_i(comb[..., i, j] * residual[..., i, d])`,
        # which is `comb^T @ residual`.
        # comb.unsqueeze(-1): (B, T, hc_mult, hc_mult, 1)  — last axis broadcasts to dim
        # residual.unsqueeze(-2): (B, T, hc_mult, 1, dim)  — middle 1 broadcasts to hc_mult
        # product:                (B, T, hc_mult, hc_mult, dim)
        # sum over the FIRST hc_mult axis (the row index of comb / residual's own axis).
        residual_mix = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
        return (scaled_sublayer + residual_mix).type_as(sublayer_output)
