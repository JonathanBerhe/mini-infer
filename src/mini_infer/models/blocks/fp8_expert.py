"""FP8-resident MoE expert: weights stay block-FP8 e4m3, dequantized per call.

The block-FP8 (`e4m3`, `[128,128]`) counterpart of `FP4Expert`, for the format
GLM-5.2-FP8 ships in. The routed experts are ~96% of the 753 GB checkpoint, so
dequantizing them to BF16 doubles that to ~1.5 TB and overflows a single 8-GPU
node. Keeping them e4m3-resident (and dequantizing only the much smaller
attention / dense / shared / indexer weights to BF16) fits ~785 GB on 8xH200.

Each weight is `float8_e4m3fn` `(out, in)` plus a ceil-block scale
`(ceil(out/128), ceil(in/128))`, stored as buffers (inference-only; FP8 is not a
valid autograd dtype). `forward` dequantizes one weight at a time to the
activation dtype, runs the SwiGLU, and lets the transient BF16 copy free. Math
matches `MixtralExpert` on the dequantized weights (the parity contract); a
fused FP8 GEMM is the performance follow-up.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from mini_infer.models.blocks.activations import GateUpActivation, swiglu
from mini_infer.quant.nvfp4 import dequantize_block_fp8_to_bf16_partial

_FP8_BLOCK = 128


class Fp8Expert(nn.Module):
    """One MoE expert (SwiGLU) with block-FP8-resident, dequant-per-call weights.

    Drop-in for `MixtralExpert` at the `forward(x) -> x` interface; differs only
    in storage (e4m3 weight + block scale buffers vs BF16 `nn.Linear`). Weight
    names mirror Mixtral's `w1` (gate), `w2` (down), `w3` (up); each `w{n}` is the
    e4m3 tensor and `w{n}_scale` its per-block scale.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        block: int = _FP8_BLOCK,
        activation: GateUpActivation = swiglu,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.block = block
        # Default `swiglu` matches MixtralExpert; M3's fp8 experts pass `swigluoai`.
        self._activation = activation
        self._register("w1", intermediate_size, hidden_size)  # gate
        self._register("w3", intermediate_size, hidden_size)  # up
        self._register("w2", hidden_size, intermediate_size)  # down

    def _register(self, name: str, out_dim: int, in_dim: int) -> None:
        self.register_buffer(
            name, torch.zeros(out_dim, in_dim, dtype=torch.float8_e4m3fn), persistent=True
        )
        self.register_buffer(
            f"{name}_scale",
            torch.zeros(
                math.ceil(out_dim / self.block),
                math.ceil(in_dim / self.block),
                dtype=torch.float32,
            ),
            persistent=True,
        )

    def _dequant(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize one weight to `dtype` (transient; freed after the matmul)."""
        weight: torch.Tensor = getattr(self, name)
        scale: torch.Tensor = getattr(self, f"{name}_scale")
        return dequantize_block_fp8_to_bf16_partial(
            weight, scale, block_size=(self.block, self.block)
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self._dequant("w1", x.dtype)
        w3 = self._dequant("w3", x.dtype)
        w2 = self._dequant("w2", x.dtype)
        gated = self._activation(functional.linear(x, w1), functional.linear(x, w3))
        out: torch.Tensor = functional.linear(gated, w2)
        return out
