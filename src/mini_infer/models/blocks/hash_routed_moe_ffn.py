"""Hash-routed sparse MoE FFN (DeepSeek-V4 paper §2.2).

Composes a `HashRoutedGate` (hash-table routing for the first
`n_hash_layers` of V4, score-topk for the rest) with the same
gather → per-expert MLP → scatter pipeline as `MoEFFN`. Two structural
differences vs the Mixtral-style `MoEFFN`:

  - **`forward` takes `input_ids`**: required when the gate uses hash
    routing. Score-topk gates ignore it. The decoder layer that owns
    this FFN must thread `input_ids` down — V4's `Block` already does.
  - **fp32 routed accumulator**: matches the V4 reference's
    `MoE.forward` exactly. With softmax / sigmoid weights summed across
    `num_activated` experts, the fp32 accumulator keeps the
    weighted-sum precise across long sequences before casting back to
    the model's working dtype.

Shared-expert handling mirrors `MoEFFN`: V4's reference asserts
`n_shared_experts == 1` and uses one shared MLP; we keep the
"collapse N shared experts into one MLP with N x intermediate width"
shortcut from `MoEFFN` so the same primitive serves both V2/V3
(`n_shared_experts == 2`) and V4 (`n_shared_experts == 1`) layouts.

Expert parallelism: same scheme as `MoEFFN`. Gate (including the
hash-routing tid2eid table) replicated; routed experts sharded across
ranks; partial routed sums all-reduced; shared expert replicated and
added once after the reduce.

Bit-parity vs the reference's `MoE.forward` on synthetic input across
both routing modes and the supported score functions.

On Hopper and newer GPUs the routed experts run as grouped matmuls
(`grouped_experts.py`): FP4 experts through `PackedFp4Experts`, bf16 ones
through `PackedSwigluExperts`, both summed in fp32 like the loop. With the
reference's FP8 GEMM-input round trip on, the block keeps the per-expert loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.distributed.comm import all_reduce_sum
from mini_infer.distributed.linear import _split_size
from mini_infer.models.blocks.activations import expert_swiglu
from mini_infer.models.blocks.fp4_expert import FP4Expert, PackedFp4Experts
from mini_infer.models.blocks.fp4_grouped_kernel import supports_block_fp4_grouped_mm
from mini_infer.models.blocks.grouped_experts import supports_grouped_experts
from mini_infer.models.blocks.hash_routed_gate import (
    HashRoutedGate,
    RoutingMode,
    ScoreFunction,
)
from mini_infer.models.blocks.mixtral_moe import MixtralExpert, PackedSwigluExperts

if TYPE_CHECKING:
    from mini_infer.models.blocks.v4.act_quant import SimulatedActQuant


class HashRoutedMoEFFN(nn.Module):
    """V4-style MoE FFN: hash-or-score gate + per-expert MLP + shared expert(s)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_routed_experts: int,
        num_activated_experts: int,
        routing_mode: RoutingMode,
        score_func: ScoreFunction = "softmax",
        route_scale: float = 1.0,
        vocab_size: int | None = None,
        n_shared_experts: int = 0,
        shared_intermediate_size: int | None = None,
        expert_dtype: str = "bf16",
        act_quant: SimulatedActQuant | None = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        from mini_infer.distributed.group import get_rank, get_world_size

        world_size = get_world_size()
        num_experts_per_rank = _split_size(num_routed_experts, world_size, "num_routed_experts")
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts  # global
        self.num_experts_per_rank = num_experts_per_rank
        self.num_activated_experts = num_activated_experts
        self.routing_mode = routing_mode
        self.world_size = world_size
        self.rank = get_rank()
        # Range of *global* expert indices this rank owns.
        self.local_expert_start = self.rank * num_experts_per_rank
        self.local_expert_end = self.local_expert_start + num_experts_per_rank

        # Gate handles all (mode, score_func, route_scale, renorm) variants.
        # Replicated under TP — every rank routes from the same hidden state
        # and picks the same activated experts.
        self.gate = HashRoutedGate(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            routing_mode=routing_mode,
            score_func=score_func,
            route_scale=route_scale,
            vocab_size=vocab_size,
        )

        # Local experts only. `self.experts[local_idx]` has global index
        # `local_expert_start + local_idx`. With `expert_dtype="fp4"` the
        # routed experts stay NVFP4-resident (`FP4Expert`: packed int8 +
        # per-block scale buffers, dequantized per call) instead of holding
        # full-width BF16 weights -- the only way V4-Flash's routed experts
        # fit in HBM. Shared experts below are unaffected (they ship FP8 and
        # dequantize to BF16 at load).
        if expert_dtype not in ("bf16", "fp4"):
            raise ValueError(f"expert_dtype must be 'bf16' or 'fp4'; got {expert_dtype!r}")
        self.expert_dtype = expert_dtype
        # V4's `swiglu_limit`: every expert (routed and shared) clamps its gate
        # and up projections before the SiLU. 0 keeps plain SwiGLU.
        self.swiglu_limit = swiglu_limit
        activation = expert_swiglu(swiglu_limit)
        routed_experts: list[nn.Module] = []
        for _ in range(num_experts_per_rank):
            if expert_dtype == "fp4":
                routed_experts.append(
                    FP4Expert(
                        hidden_size,
                        intermediate_size,
                        act_quant=act_quant,
                        swiglu_limit=swiglu_limit,
                    )
                )
            else:
                expert = MixtralExpert(hidden_size, intermediate_size, activation=activation)
                if act_quant is not None:
                    act_quant.quantize_gemm_inputs(expert.w1, expert.w2, expert.w3)
                routed_experts.append(expert)
        self.experts = nn.ModuleList(routed_experts)
        # The reference's experts all hold FP8 (shared) or FP4 (routed) weights,
        # so with the FP8 GEMM-input round trip on, the routing weight must
        # scale the activation before `w2` (the reference's order) rather than
        # the expert output: `w2` then quantizes the scaled input.
        self.routing_weights_before_down_proj = act_quant is not None and act_quant.fp8_gemm_inputs
        # Packed on the first grouped forward, see `_grouped_pack`.
        self._expert_pack: PackedSwigluExperts | PackedFp4Experts | None = None

        # Shared expert(s): always-on MLP added to the routed sum. V4 uses one;
        # V2/V3 use two and we collapse them into a single MLP with a wider
        # intermediate (math is identical to running them in parallel + summing,
        # one matmul instead of two).
        self.n_shared_experts = n_shared_experts
        if n_shared_experts > 0:
            shared_intermediate = (
                shared_intermediate_size
                if shared_intermediate_size is not None
                else intermediate_size
            )
            self.shared_experts: MixtralExpert | None = MixtralExpert(
                hidden_size, shared_intermediate * n_shared_experts, activation=activation
            )
            if act_quant is not None:
                act_quant.quantize_gemm_inputs(
                    self.shared_experts.w1, self.shared_experts.w2, self.shared_experts.w3
                )
        else:
            self.shared_experts = None

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Route each token to its activated experts; sum the weighted outputs.

        Args:
            hidden_states: `(B, T, hidden_size)` or `(N, hidden_size)`.
            input_ids: Token ids matching `hidden_states`'s leading dims.
                Required iff `routing_mode == "hash"` (see `HashRoutedGate`).

        Returns:
            Same shape as `hidden_states`. dtype matches the input.
        """
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, original_shape[-1])
        flat_input_ids = input_ids.reshape(-1) if input_ids is not None else None

        # Gate gives us per-token (weights, expert indices) for the activated experts.
        per_token_weights, per_token_expert_indices = self.gate(flat_hidden_states, flat_input_ids)
        # Cast weights to the working dtype for the per-expert weighted sum.
        per_token_weights = per_token_weights.to(flat_hidden_states.dtype)

        pack = self._grouped_pack(flat_hidden_states)
        if pack is not None:
            routed_accumulator = pack.run(
                flat_hidden_states,
                per_token_expert_indices,
                per_token_weights,
                num_experts=self.num_routed_experts,
                expert_start=self.local_expert_start,
                accumulate_dtype=torch.float32,
                out_dtype=torch.float32,
            )
        else:
            routed_accumulator = self._loop_experts(
                flat_hidden_states, per_token_expert_indices, per_token_weights
            )

        # All-reduce the routed sum across ranks; no-op at world_size=1.
        routed_accumulator = all_reduce_sum(routed_accumulator)
        result = routed_accumulator.to(flat_hidden_states.dtype)

        # Shared expert is replicated and produces the same value on every
        # rank; add it AFTER the reduce so it isn't scaled by world_size.
        if self.shared_experts is not None:
            result = result + self.shared_experts(flat_hidden_states)

        return result.view(original_shape)

    def runs_grouped(self, device: torch.device | str, dtype: torch.dtype) -> bool:
        """Whether a forward on `device` with `dtype` activations takes the grouped path.

        The reference's FP8 GEMM-input round trip quantizes each expert's
        inputs, which the grouped matmuls do not do, so it keeps the loop. The
        FP4 kernel rounds the dequantized weight to bf16 and multiplies in the
        activation dtype, which matches `FP4Expert` for 16-bit inputs only.
        """
        if self.routing_weights_before_down_proj or not supports_grouped_experts(device):
            return False
        if self.expert_dtype == "bf16":
            return True
        return dtype in (torch.bfloat16, torch.float16) and supports_block_fp4_grouped_mm(device)

    def _grouped_pack(self, flat: torch.Tensor) -> PackedSwigluExperts | PackedFp4Experts | None:
        """The packed experts when this forward takes the grouped path, else `None`."""
        if not self.runs_grouped(flat.device, flat.dtype):
            return None
        pack = self._expert_pack
        if self.expert_dtype == "bf16":
            bf16_pack = pack if isinstance(pack, PackedSwigluExperts) else None
            self._expert_pack = PackedSwigluExperts.current(bf16_pack, self.experts)
            return self._expert_pack
        fp4_pack = pack if isinstance(pack, PackedFp4Experts) else None
        self._expert_pack = PackedFp4Experts.current(fp4_pack, self.experts)
        return self._expert_pack

    def _loop_experts(
        self,
        flat_hidden_states: torch.Tensor,
        per_token_expert_indices: torch.Tensor,
        per_token_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert dispatch into an fp32 `(N, hidden)` sum of the local experts."""
        # fp32 accumulator, as in the reference: it keeps precision across
        # the per-expert weighted sums.
        routed_accumulator = torch.zeros_like(flat_hidden_states, dtype=torch.float32)

        # Loop only over experts this rank owns; the all-reduce in `forward`
        # accumulates partial sums across ranks.
        for local_idx in range(self.num_experts_per_rank):
            global_idx = self.local_expert_start + local_idx
            token_positions, top_k_slots = torch.where(per_token_expert_indices == global_idx)
            if token_positions.numel() == 0:
                continue
            tokens_for_this_expert = flat_hidden_states[token_positions]
            weights_for_this_expert = per_token_weights[token_positions, top_k_slots, None]
            if self.routing_weights_before_down_proj:
                weighted_expert_output = self.experts[local_idx](
                    tokens_for_this_expert, routing_weights=weights_for_this_expert
                )
            else:
                # `MixtralExpert(x) * w` is equivalent to the reference's
                # `expert(x, weights=w)` which folds `w` between SwiGLU and w2 —
                # `w2` is linear and `w` is a per-row scalar, so the scalar
                # commutes through the down-projection.
                weighted_expert_output = (
                    self.experts[local_idx](tokens_for_this_expert) * weights_for_this_expert
                )
            routed_accumulator.index_add_(0, token_positions, weighted_expert_output.float())
        return routed_accumulator
