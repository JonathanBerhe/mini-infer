"""Gemma's RMSNorm variant: `(1 + weight) * x / rms(x)`.

Differs from the standard `RMSNorm` only by the `+1` offset on the
weight. Matches HF Gemma 2 / 3 / 4 numerically and shares the same
parameter name (`weight`) so weight loading is identity rename.
"""

import torch
from torch import nn


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Gemma's checkpoints store the weight pre-shifted: the value on
        # disk is what's added to 1, not the final scale. So the parameter
        # initializes to zero (final scale = 1).
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)
