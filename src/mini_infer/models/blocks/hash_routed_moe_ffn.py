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
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.distributed.comm import all_reduce_sum
from mini_infer.distributed.linear import _split_size
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
        from mini_infer.distributed.group import get_rank, get_world_size

        world_size = get_world_size()
        num_experts_per_rank = _split_size(
            num_routed_experts, world_size, "num_routed_experts"
        )
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
        # `local_expert_start + local_idx`.
        self.experts = nn.ModuleList(
            [MixtralExpert(hidden_size, intermediate_size) for _ in range(num_experts_per_rank)]
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

        # Loop only over experts this rank owns; the all-reduce below
        # accumulates partial sums across ranks.
        for local_idx in range(self.num_experts_per_rank):
            global_idx = self.local_expert_start + local_idx
            token_positions, top_k_slots = torch.where(per_token_expert_indices == global_idx)
            if token_positions.numel() == 0:
                continue
            tokens_for_this_expert = flat_hidden_states[token_positions]
            weights_for_this_expert = per_token_weights[token_positions, top_k_slots, None]
            # `MixtralExpert(x) * w` is equivalent to the reference's
            # `expert(x, weights=w)` which folds `w` between SwiGLU and w2 —
            # `w2` is linear and `w` is a per-row scalar, so the scalar
            # commutes through the down-projection.
            weighted_expert_output = (
                self.experts[local_idx](tokens_for_this_expert) * weights_for_this_expert
            )
            routed_accumulator.index_add_(0, token_positions, weighted_expert_output.float())

        # All-reduce the routed sum across ranks; no-op at world_size=1.
        routed_accumulator = all_reduce_sum(routed_accumulator)
        result = routed_accumulator.to(flat_hidden_states.dtype)

        # Shared expert is replicated and produces the same value on every
        # rank; add it AFTER the reduce so it isn't scaled by world_size.
        if self.shared_experts is not None:
            result = result + self.shared_experts(flat_hidden_states)

        return result.view(original_shape)
