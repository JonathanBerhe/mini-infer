"""RMSNorm: root-mean-square layer norm used by Llama / Qwen2 / Mistral / DeepSeek / Gemma 4."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with the HF-canonical compute order.

    `weight` parameter aligns with HF's name so weight loading is identity.
    Compute happens in fp32 then casts back, matching HF's numerical
    behavior across model families that use this layer.

    `with_scale=False` skips the learnable weight entirely and returns the
    pure normalization `x * rsqrt(mean(x²) + eps)`. Gemma 4's `v_norm` uses
    this mode so V is RMS-normalized to unit RMS without an affine rescale.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.with_scale:
            return self.weight * x.to(input_dtype)
        return x.to(input_dtype)
