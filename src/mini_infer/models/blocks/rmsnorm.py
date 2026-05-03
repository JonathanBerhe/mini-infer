"""RMSNorm: root-mean-square layer norm used by Llama / Qwen2 / Mistral / DeepSeek."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with the HF-canonical compute order.

    `weight` parameter aligns with HF's name so weight loading is identity.
    Compute happens in fp32 then casts back, matching HF's numerical
    behavior across model families that use this layer.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
