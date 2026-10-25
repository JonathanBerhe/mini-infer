"""FP4-resident MoE expert: weights stay packed NVFP4, dequantized per call.

`MixtralExpert` holds its w1/w2/w3 as BF16 `nn.Linear` weights. For
DeepSeek-V4-Flash that's a non-starter: the routed experts dominate the
parameter count (~277B params), and dequantizing them to BF16 is a 4x
storage blow-up (~130 GiB FP4 -> ~518 GiB BF16) that does not fit 2x B200
(see `scripts/deepseek_v4/profile_v4_dequant.py`). The model only fits while the
experts stay FP4.

`FP4Expert` keeps each weight as the packed NVFP4 form it ships in:

  - `{w}_packed`: int8, shape `(out, in // 2)` (two FP4 nibbles per byte).
  - `{w}_scale` : per-block scale, shape `(out, in // block_size)`.

stored as buffers (not Parameters: inference-only, and packed int8 is not
a valid autograd dtype). `forward` dequantizes each weight to the
activation dtype transiently, runs the SwiGLU, and lets the BF16 copy
free. It materializes the full weight per call, but never holds more than
one expert's BF16 weights at once.

Math is identical to `MixtralExpert` once dequantized: matching
`MixtralExpert(x)` on the dequantized weights is the parity contract.

On Hopper and newer GPUs, `PackedFp4Experts` stacks a layer's experts and runs
them through the Triton grouped kernel of `fp4_grouped_kernel.py`, which
dequantizes in registers with the same rounding as `forward`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional

from mini_infer.models.blocks.activations import expert_swiglu
from mini_infer.models.blocks.moe.fp4_grouped_kernel import e8m0_scales, to_e8m0
from mini_infer.models.blocks.moe.grouped_experts import BlockFp4Weight, grouped_expert_mlp
from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16

if TYPE_CHECKING:
    from mini_infer.models.deepseek_v4.act_quant import SimulatedActQuant

_FP4_BLOCK_SIZE = 32


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class FP4Expert(nn.Module):
    """One MoE expert (SwiGLU) with FP4-resident, dequant-per-call weights.

    Drop-in for `MixtralExpert` at the `forward(x) -> x` interface; differs
    only in storage (packed FP4 buffers vs BF16 `nn.Linear` weights). Weight
    names mirror Mixtral's `w1` (gate), `w2` (down), `w3` (up).

    `act_quant` with `fp8_gemm_inputs` set round-trips each GEMM input
    through FP8 (block 128), like the reference's `fp4_gemm` path.
    `swiglu_limit > 0` clamps the gate and up projections before the SiLU,
    like DeepSeek-V4's experts (see `activations.clamped_swiglu`).
    """

    w1_packed: torch.Tensor
    w2_packed: torch.Tensor
    w3_packed: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    w3_scale: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        block_size: int = _FP4_BLOCK_SIZE,
        act_quant: SimulatedActQuant | None = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.block_size = block_size
        self.swiglu_limit = swiglu_limit
        self._activation = expert_swiglu(swiglu_limit)
        if act_quant is not None and act_quant.fp8_gemm_inputs:
            act_quant.check_gemm_input_dim(hidden_size, "FP4Expert w1/w3")
            act_quant.check_gemm_input_dim(intermediate_size, "FP4Expert w2")
        self.act_quant = act_quant
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

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor | None = None) -> torch.Tensor:
        """`(N, hidden)` -> `(N, hidden)`; `routing_weights` as in `MixtralExpert`."""
        # Dequant cast to x.dtype so CPU fp32 tests and GPU bf16 both work
        # (dequantize_nvfp4_to_bf16 always returns bf16).
        w1 = self._dequant("w1", x.dtype)
        w3 = self._dequant("w3", x.dtype)
        w2 = self._dequant("w2", x.dtype)
        # Identity unless the FP8 GEMM-input round trip is enabled.
        gemm_input = self.act_quant.gemm_input if self.act_quant is not None else _identity
        x_in = gemm_input(x)
        hidden = self._activation(functional.linear(x_in, w1), functional.linear(x_in, w3))
        if routing_weights is not None:
            hidden = hidden * routing_weights
        out: torch.Tensor = functional.linear(gemm_input(hidden), w2)
        return out


class PackedFp4Experts:
    """The weights of a list of `FP4Expert`s, stacked for `grouped_expert_mlp`.

    The FP4 counterpart of `PackedFp8Experts`: building one copies every
    expert's packed `w1` and `w3` (with their scales) into one `(E, 2I, in/2)`
    byte tensor and its `w2` into one `(E, out, I/2)` tensor, then points the
    experts' own buffers at slices of the stacks. The weights stay packed, the
    experts still work on their own, and `state_dict` keys do not change. The
    scales cover one row each, so `w1` and `w3` always stack. When a loader
    gives a weight or a scale new storage, `matches` detects it and `current`
    packs again.

    With `fp8_activations`, the pack also keeps the scales as e8m0 bytes, so
    the matmuls run W4A8 (ADR-043). That needs every scale to be a power of
    two, as in DeepSeek-V4's checkpoints, and costs one byte per 32 weights.
    True raises if a scale is not one. None runs W4A8 only if every scale is
    one, and else keeps bf16 inputs. `self.fp8_activations` says which.
    """

    def __init__(self, experts: nn.ModuleList, *, fp8_activations: bool | None = False) -> None:
        members = [cast(FP4Expert, expert) for expert in experts]
        first = members[0]
        block, inter = first.block_size, first.intermediate_size
        hidden = first.hidden_size
        device = first.w1_packed.device
        count = len(members)
        # Outside inference mode, as in `PackedSwigluExperts`.
        with torch.no_grad(), torch.inference_mode(False):
            gate_up = torch.empty(count, 2 * inter, hidden // 2, dtype=torch.int8, device=device)
            gate_up_scale = torch.empty(count, 2 * inter, hidden // block, device=device)
            down = torch.empty(count, hidden, inter // 2, dtype=torch.int8, device=device)
            down_scale = torch.empty(count, hidden, inter // block, device=device)
            for idx, expert in enumerate(members):
                gate_up[idx, :inter].copy_(expert.w1_packed)
                gate_up[idx, inter:].copy_(expert.w3_packed)
                gate_up_scale[idx, :inter].copy_(expert.w1_scale)
                gate_up_scale[idx, inter:].copy_(expert.w3_scale)
                down[idx].copy_(expert.w2_packed)
                down_scale[idx].copy_(expert.w2_scale)
                expert.w1_packed = gate_up[idx, :inter]
                expert.w3_packed = gate_up[idx, inter:]
                expert.w1_scale = gate_up_scale[idx, :inter]
                expert.w3_scale = gate_up_scale[idx, inter:]
                expert.w2_packed = down[idx]
                expert.w2_scale = down_scale[idx]
        gate_up_e8m0 = down_e8m0 = None
        if fp8_activations:
            gate_up_e8m0, down_e8m0 = to_e8m0(gate_up_scale), to_e8m0(down_scale)
        elif fp8_activations is None:
            gate_up_e8m0, down_e8m0 = e8m0_scales(gate_up_scale), e8m0_scales(down_scale)
            if gate_up_e8m0 is None or down_e8m0 is None:
                gate_up_e8m0 = down_e8m0 = None
        self.requested = fp8_activations
        self.fp8_activations = gate_up_e8m0 is not None
        self.gate_up = BlockFp4Weight(gate_up, gate_up_scale, block, gate_up_e8m0)
        self.down = BlockFp4Weight(down, down_scale, block, down_e8m0)
        self._inter = inter
        self._combine = first._activation
        self._pointers = self._storage(experts)

    @staticmethod
    def _storage(experts: nn.ModuleList) -> tuple[int, ...]:
        first = cast(FP4Expert, experts[0])
        last = cast(FP4Expert, experts[-1])
        return (
            first.w1_packed.data_ptr(),
            first.w1_scale.data_ptr(),
            last.w2_packed.data_ptr(),
            last.w2_scale.data_ptr(),
        )

    @staticmethod
    def current(
        pack: PackedFp4Experts | None,
        experts: nn.ModuleList,
        *,
        fp8_activations: bool | None = False,
    ) -> PackedFp4Experts:
        """`pack` if it still backs `experts` in the same mode, else a fresh pack of them."""
        if pack is not None and pack.matches(experts) and pack.requested == fp8_activations:
            return pack
        return PackedFp4Experts(experts, fp8_activations=fp8_activations)

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
        accumulate_dtype: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
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
            accumulate_dtype=accumulate_dtype,
            out_dtype=out_dtype,
        )
