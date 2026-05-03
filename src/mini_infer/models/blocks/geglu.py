"""GeGLU FFN used by the Gemma family.

Same shape as `SwiGLU` (gate, up, down projections) but with the
`gelu_pytorch_tanh` activation on the gate branch instead of silu.
Parameter names align with HF Gemma's `Gemma{2,3,4}MLP` so weight
loading is identity rename.
"""

import torch
from torch import nn
from torch.nn import functional


class GeGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = functional.gelu(self.gate_proj(x), approximate="tanh")
        out: torch.Tensor = self.down_proj(gate * self.up_proj(x))
        return out
