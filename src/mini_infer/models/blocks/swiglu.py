"""SwiGLU FFN block (Llama / Qwen2 / Mistral).

Three projections: `gate_proj` (silu-gated), `up_proj` (passed through),
`down_proj` (back to hidden_size). Names align with HF so weight loading
is identity.
"""

import torch
from torch import nn
from torch.nn import functional


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return out
