"""RMSNorm: root-mean-square layer norm used by Llama / Qwen2 / Mistral / DeepSeek / Gemma 4."""

import torch
from torch import nn

from mini_infer.models.blocks.rms_norm_kernel import fused_rms_norm, rms_norm_rows


class RMSNorm(nn.Module):
    """RMSNorm with the HF-canonical compute order.

    `weight` parameter aligns with HF's name so weight loading is identity.
    Compute happens in fp32 then casts back, matching HF's numerical
    behavior across model families that use this layer.

    `with_scale=False` skips the learnable weight entirely and returns the
    pure normalization `x * rsqrt(mean(x²) + eps)`. Gemma 4's `v_norm` uses
    this mode so V is RMS-normalized to unit RMS without an affine rescale.

    `fused=True` runs the norm as one Triton kernel on CUDA
    (`rms_norm_kernel.py`, ADR-045). Its sum runs in another order, so a few
    outputs differ from the PyTorch path in the last bit. DeepSeek-V4 sets it
    (`DeepseekV4Config.fused_rms_norm`); other models keep the PyTorch path.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True, fused: bool = False
    ) -> None:
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.fused = fused

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.with_scale else None
        if self.fused:
            rows = rms_norm_rows(hidden_states, weight)
            if rows is not None:
                return fused_rms_norm(rows, hidden_states.shape, weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.with_scale:
            return self.weight * x.to(input_dtype)
        return x.to(input_dtype)


def head_rms_norm(x: torch.Tensor, eps: float, *, fused: bool = False) -> torch.Tensor:
    """DeepSeek-V4's per-head q-norm: `x * rsqrt(mean(x²) + eps)`, no weight.

    The scale is cast to x's dtype before the multiply, as in the reference.
    `fused=True` runs it as one Triton kernel on CUDA, as `RMSNorm(fused=True)`.
    """
    if fused:
        rows = rms_norm_rows(x)
        if rows is not None:
            return fused_rms_norm(rows, x.shape, None, eps, scale_first=True)
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)
