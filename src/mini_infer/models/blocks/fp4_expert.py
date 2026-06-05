"""FP4-resident MoE expert: weights stay packed NVFP4, dequantized per call.

`MixtralExpert` holds its w1/w2/w3 as BF16 `nn.Linear` weights. For
DeepSeek-V4-Flash that's a non-starter: the routed experts dominate the
parameter count (~277B params), and dequantizing them to BF16 is a 4x
storage blow-up (~130 GiB FP4 -> ~518 GiB BF16) that does not fit 2x B200
(see `scripts/profile_v4_dequant.py`). The model only fits while the
experts stay FP4.

`FP4Expert` keeps each weight as the packed NVFP4 form it ships in:

  - `{w}_packed`: int8, shape `(out, in // 2)` (two FP4 nibbles per byte).
  - `{w}_scale` : per-block scale, shape `(out, in // block_size)`.

stored as buffers (not Parameters: inference-only, and packed int8 is not
a valid autograd dtype). `forward` dequantizes each weight to the
activation dtype transiently, runs the SwiGLU, and lets the BF16 copy
free. This is the correctness-first path: it materializes the full weight
per call, so it is slower than a fused FP4 GEMM (the performance
follow-up) but never holds more than one expert's BF16 weights at once.

Math is identical to `MixtralExpert` once dequantized: matching
`MixtralExpert(x)` on the dequantized weights is the parity contract.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16

_FP4_BLOCK_SIZE = 32


class FP4Expert(nn.Module):
    """One MoE expert (SwiGLU) with FP4-resident, dequant-per-call weights.

    Drop-in for `MixtralExpert` at the `forward(x) -> x` interface; differs
    only in storage (packed FP4 buffers vs BF16 `nn.Linear` weights). Weight
    names mirror Mixtral's `w1` (gate), `w2` (down), `w3` (up).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        block_size: int = _FP4_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.block_size = block_size
        # w1 (gate) and w3 (up): (intermediate, hidden). w2 (down): (hidden, intermediate).
        self._register_packed("w1", intermediate_size, hidden_size)
        self._register_packed("w3", intermediate_size, hidden_size)
        self._register_packed("w2", hidden_size, intermediate_size)

    def _register_packed(self, name: str, out_dim: int, in_dim: int) -> None:
        if in_dim % self.block_size != 0:
            raise ValueError(
                f"{name}: in_dim={in_dim} must be divisible by block_size={self.block_size}"
            )
        # `(out, in // 2)` packed bytes + `(out, in // block_size)` scale.
        self.register_buffer(
            f"{name}_packed",
            torch.zeros(out_dim, in_dim // 2, dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            f"{name}_scale",
            torch.zeros(out_dim, in_dim // self.block_size, dtype=torch.float32),
            persistent=True,
        )

    def _dequant(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize one weight to `dtype` (transient; freed after the matmul)."""
        packed: torch.Tensor = getattr(self, f"{name}_packed")
        scale: torch.Tensor = getattr(self, f"{name}_scale")
        return dequantize_nvfp4_to_bf16(packed, scale, block_size=self.block_size).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequant cast to x.dtype so CPU fp32 tests and GPU bf16 both work
        # (dequantize_nvfp4_to_bf16 always returns bf16).
        w1 = self._dequant("w1", x.dtype)
        w3 = self._dequant("w3", x.dtype)
        w2 = self._dequant("w2", x.dtype)
        out: torch.Tensor = functional.linear(
            functional.silu(functional.linear(x, w1)) * functional.linear(x, w3), w2
        )
        return out
