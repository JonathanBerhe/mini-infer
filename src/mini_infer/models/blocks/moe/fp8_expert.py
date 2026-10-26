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
matches `MixtralExpert` on the dequantized weights (the parity contract).

On Hopper and newer GPUs, `PackedFp8Experts` stacks a layer's experts and runs
them through `fp8_grouped_kernel.block_fp8_grouped_mm`. Its Triton and Gluon
kernels dequantize with the same rounding as `forward`.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from mini_infer.models.blocks.activations import GateUpActivation, swiglu
from mini_infer.models.blocks.moe.grouped_experts import BlockFp8Weight, grouped_expert_mlp
from mini_infer.quant.nvfp4 import dequantize_block_fp8_to_bf16_partial

_FP8_BLOCK = 128


class Fp8Expert(nn.Module):
    """One MoE expert (SwiGLU) with block-FP8-resident, dequant-per-call weights.

    Drop-in for `MixtralExpert` at the `forward(x) -> x` interface; differs only
    in storage (e4m3 weight + block scale buffers vs BF16 `nn.Linear`). Weight
    names mirror Mixtral's `w1` (gate), `w2` (down), `w3` (up); each `w{n}` is the
    e4m3 tensor and `w{n}_scale` its per-block scale.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    w3_scale: torch.Tensor

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


class PackedFp8Experts:
    """The weights of a list of `Fp8Expert`s, stacked for `grouped_expert_mlp`.

    The FP8 counterpart of `PackedSwigluExperts`: building one copies every
    expert's `w1` and `w3` (with their block scales) into one `(E, 2I, in)`
    e4m3 tensor and its `w2` into one `(E, out, I)` tensor, then points the
    experts' own buffers at slices of the stacks. The weights stay e4m3, the
    experts still work on their own, and `state_dict` keys do not change.
    Stacking `w1` over `w3` keeps their scale blocks aligned only when `I` is
    a multiple of the block size, so `can_pack` asks for that. When a loader
    gives a weight or a scale new storage, `matches` detects it and `current`
    packs again.
    """

    def __init__(self, experts: nn.ModuleList) -> None:
        members = [cast(Fp8Expert, expert) for expert in experts]
        first = members[0]
        block, inter, hidden = first.block, first.intermediate_size, first.hidden_size
        inter_blocks = inter // block
        device = first.w1.device
        # Outside inference mode, as in `PackedSwigluExperts`.
        with torch.no_grad(), torch.inference_mode(False):
            gate_up = torch.empty(
                len(members), 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=device
            )
            gate_up_scale = torch.empty(
                len(members), 2 * inter_blocks, math.ceil(hidden / block), device=device
            )
            down = torch.empty(
                len(members), hidden, inter, dtype=torch.float8_e4m3fn, device=device
            )
            down_scale = torch.empty(
                len(members), math.ceil(hidden / block), math.ceil(inter / block), device=device
            )
            for idx, expert in enumerate(members):
                gate_up[idx, :inter].copy_(expert.w1)
                gate_up[idx, inter:].copy_(expert.w3)
                gate_up_scale[idx, :inter_blocks].copy_(expert.w1_scale)
                gate_up_scale[idx, inter_blocks:].copy_(expert.w3_scale)
                down[idx].copy_(expert.w2)
                down_scale[idx].copy_(expert.w2_scale)
                expert.w1 = gate_up[idx, :inter]
                expert.w3 = gate_up[idx, inter:]
                expert.w1_scale = gate_up_scale[idx, :inter_blocks]
                expert.w3_scale = gate_up_scale[idx, inter_blocks:]
                expert.w2 = down[idx]
                expert.w2_scale = down_scale[idx]
        self.gate_up = BlockFp8Weight(gate_up, gate_up_scale, block)
        self.down = BlockFp8Weight(down, down_scale, block)
        self._inter = inter
        self._combine = first._activation
        self._pointers = self._storage(experts)

    @staticmethod
    def can_pack(intermediate_size: int, block: int = _FP8_BLOCK) -> bool:
        """Whether experts of this width can be stacked gate-over-up."""
        return intermediate_size % block == 0

    @staticmethod
    def _storage(experts: nn.ModuleList) -> tuple[int, ...]:
        first = cast(Fp8Expert, experts[0])
        last = cast(Fp8Expert, experts[-1])
        return (
            first.w1.data_ptr(),
            first.w1_scale.data_ptr(),
            last.w2.data_ptr(),
            last.w2_scale.data_ptr(),
        )

    @staticmethod
    def current(pack: PackedFp8Experts | None, experts: nn.ModuleList) -> PackedFp8Experts:
        """`pack` if it still backs `experts`, else a fresh pack of them."""
        if pack is not None and pack.matches(experts):
            return pack
        return PackedFp8Experts(experts)

    def matches(self, experts: nn.ModuleList) -> bool:
        """Whether the experts' buffers still live in this pack's stacks."""
        return self._storage(experts) == self._pointers

    def _activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self._combine(gate_up[:, : self._inter], gate_up[:, self._inter :])
        return out

    def run(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_experts: int,
        expert_start: int,
    ) -> torch.Tensor:
        """`grouped_expert_mlp` over the packed experts; see that function."""
        return grouped_expert_mlp(
            x,
            topk_indices,
            topk_weights,
            self.gate_up,
            self.down,
            self._activation,
            num_experts=num_experts,
            expert_start=expert_start,
        )
