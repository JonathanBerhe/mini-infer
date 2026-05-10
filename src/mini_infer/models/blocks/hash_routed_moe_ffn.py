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

Bit-parity vs the reference's `MoE.forward` on synthetic input across
both routing modes and the supported score functions.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.models.blocks.hash_routed_gate import (
    HashRoutedGate,
    RoutingMode,
    ScoreFunction,
)
from mini_infer.models.blocks.mixtral_moe import MixtralExpert


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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_activated_experts = num_activated_experts
        self.routing_mode = routing_mode

        # Gate handles all (mode, score_func, route_scale, renorm) variants.
        self.gate = HashRoutedGate(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            routing_mode=routing_mode,
            score_func=score_func,
            route_scale=route_scale,
            vocab_size=vocab_size,
        )

        # One MLP per routed expert, named to match Mixtral's safetensors layout
        # so the eventual V4 weight loader can do an identity rename.
        self.experts = nn.ModuleList(
            [MixtralExpert(hidden_size, intermediate_size) for _ in range(num_routed_experts)]
        )

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
                hidden_size, shared_intermediate * n_shared_experts
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

        # fp32 accumulator — matches the reference and keeps precision across
        # the per-expert weighted sums.
        routed_accumulator = torch.zeros_like(flat_hidden_states, dtype=torch.float32)

        for expert_index in range(self.num_routed_experts):
            # `(token_pos, top_k_slot)` pairs that selected `expert_index`.
            token_positions, top_k_slots = torch.where(per_token_expert_indices == expert_index)
            if token_positions.numel() == 0:
                continue
            tokens_for_this_expert = flat_hidden_states[token_positions]
            weights_for_this_expert = per_token_weights[token_positions, top_k_slots, None]
            # `MixtralExpert(x) * w` is equivalent to the reference's
            # `expert(x, weights=w)` which folds `w` between SwiGLU and w2 —
            # `w2` is linear and `w` is a per-row scalar, so the scalar
            # commutes through the down-projection.
            weighted_expert_output = (
                self.experts[expert_index](tokens_for_this_expert) * weights_for_this_expert
            )
            routed_accumulator.index_add_(0, token_positions, weighted_expert_output.float())

        result = routed_accumulator.to(flat_hidden_states.dtype)

        if self.shared_experts is not None:
            result = result + self.shared_experts(flat_hidden_states)

        return result.view(original_shape)
