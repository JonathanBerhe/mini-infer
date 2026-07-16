"""Inkling short convolution (SConv): depthwise causal conv + residual.

Inkling (thinkingmachines/inkling, transformers 5.14 `modeling_inkling.py`)
inserts four of these per decoder layer: after the K and V projections and
on the attention/MLP branch outputs, just before the residual add. Each is
a per-channel (depthwise) causal 1D convolution of kernel size 4 with no
bias and no activation, computed in fp32, with the residual folded INSIDE
the module:

    out = (conv1d_causal(x.float()) + x.float()).to(x.dtype)

HF keeps the conv weights fp32-strict (`_keep_in_fp32_modules_strict`); the
bf16 checkpoint values are exactly representable, so casting our parameter
to fp32 at compute time reproduces the reference bit-for-bit.

The serving-side statefulness (a decode step needs the previous
`kernel_size - 1` PRE-conv inputs) is the caller's problem: the model
stores pre-conv inputs as per-token streams in the `PagedKVCache` and
hands this module the tail it gathered. This module is pure math.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class InklingShortConv(nn.Module):
    """Depthwise causal conv1d + residual, fp32 compute.

    `weight` is `(channels, kernel_size)`: HF's `nn.Conv1d` weight
    `(channels, 1, kernel_size)` with the singleton squeezed; the weight
    remap in `models/inkling.py` does the squeeze.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))

    def forward(self, x: torch.Tensor, history_tail: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the causal conv to `x` for ONE request's new tokens.

        Args:
            x: `(new_tokens, channels)` pre-conv inputs for this step.
            history_tail: `(tail_len, channels)` the request's most recent
                pre-conv inputs from BEFORE this step, `tail_len <=
                kernel_size - 1`. `None` (or empty) means the sequence
                starts here; missing left context is zero-padded, matching
                HF's `padding=kernel_size - 1` prefill conv.

        Returns:
            `(new_tokens, channels)` post-conv (residual included) in
            `x.dtype`.
        """
        new_tokens = x.shape[0]
        x32 = x.float()
        if history_tail is not None and history_tail.shape[0] > 0:
            full = torch.cat([history_tail.float(), x32], dim=0)
        else:
            full = x32
        # (tokens, C) -> (1, C, tokens); depthwise causal conv with full left
        # zero-padding, then keep only the new positions.
        seq = full.transpose(0, 1).unsqueeze(0)
        out = functional.conv1d(
            seq,
            self.weight.float().unsqueeze(1),
            bias=None,
            padding=self.kernel_size - 1,
            groups=self.channels,
        )[:, :, : full.shape[0]]
        out = out[0, :, -new_tokens:].transpose(0, 1)
        return (out + x32).to(x.dtype)
