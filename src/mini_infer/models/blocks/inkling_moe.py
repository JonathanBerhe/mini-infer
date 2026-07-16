"""Inkling MoE: sigmoid router with a shared-expert sink + gamma'd shared experts.

Matches transformers 5.14 `modeling_inkling.py` (`InklingTopkRouter`,
`InklingExperts`, `InklingSharedExperts`, `InklingMLP`) op-for-op.

The gate differs from the DeepSeek-V3 `noaux_tc` family (`glm_moe_gate.py`)
in two load-bearing ways, which is why it gets its own implementation:

1. **Selection vs weighting are decoupled differently.** Selection uses
   `sigmoid(routed_logits) + e_score_correction_bias` (aux-loss-free, like
   DeepSeek-V3), but the expert WEIGHTS are not the unbiased sigmoid
   scores. They come from a log-sigmoid softmax over the top-k routed
   logits concatenated with the shared experts' logits.
2. **Shared experts sit inside the normalization** ("shared expert sink"):
   the router weight has `n_routed + n_shared` rows, and the shared
   experts' normalized weights (gammas) shrink when routed experts score
   high. Every weight is then scaled by `route_scale * global_scale`
   (route_scale = 8.0 for the released checkpoints; global_scale is a
   learned scalar).

The dense MLP (`InklingDenseMLP`, used for `mlp_layer_types == "dense"`
layers) is a plain SwiGLU times a learned `global_scale` scalar.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional

from mini_infer.models.blocks.mixtral_moe import MixtralExpert


class InklingDenseMLP(nn.Module):
    """SwiGLU with a learned output scale. HF names: gate/up/down_proj + global_scale."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.global_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return out * self.global_scale


class InklingGate(nn.Module):
    """Router over `n_routed + n_shared` experts.

    Returns `(topk_weights, topk_indices, shared_gammas)` for a flat
    `(tokens, hidden)` input. `e_score_correction_bias` affects SELECTION
    only; the weights are normalized log-sigmoid probabilities over the
    chosen routed logits plus the (always-on) shared logits.
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        n_shared_experts: int,
        top_k: int,
        route_scale: float,
    ) -> None:
        super().__init__()
        if n_shared_experts < 1:
            raise ValueError(
                f"InklingGate requires n_shared_experts >= 1 (the shared sink); "
                f"got {n_shared_experts}"
            )
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts + n_shared_experts, hidden_size))
        self.global_scale = nn.Parameter(torch.ones(1))
        self.e_score_correction_bias = nn.Parameter(torch.empty(n_routed_experts))

    def forward(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = functional.linear(flat, self.weight)
        scores = router_logits.sigmoid()
        routed_scores = scores[..., : -self.n_shared_experts]
        scores_for_choice = routed_scores + self.e_score_correction_bias
        topk_indices = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)[1]

        routed_logits = router_logits[..., : -self.n_shared_experts]
        shared_logits = router_logits[..., -self.n_shared_experts :]
        topk_logits = torch.cat([routed_logits.gather(-1, topk_indices), shared_logits], dim=-1)
        topk_log_probs = functional.logsigmoid(topk_logits)
        topk_weights = torch.exp(topk_log_probs - torch.logsumexp(topk_log_probs, -1, keepdim=True))
        topk_weights = topk_weights * self.route_scale * self.global_scale

        shared_gammas = topk_weights[..., -self.n_shared_experts :].contiguous()
        topk_weights = topk_weights[..., : self.top_k].contiguous()
        return topk_weights, topk_indices, shared_gammas


class InklingMoE(nn.Module):
    """Gate + routed `MixtralExpert`s + gamma-weighted shared experts.

    Shared experts multiply their gamma into the ACTIVATED intermediate
    (before down-proj) and accumulate across experts in fp32, mirroring
    HF's `InklingSharedExperts.forward` op order so parity holds in low
    precision too.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        n_shared_experts: int,
        top_k: int,
        route_scale: float,
    ) -> None:
        super().__init__()
        self.gate = InklingGate(hidden_size, n_routed_experts, n_shared_experts, top_k, route_scale)
        self.experts = nn.ModuleList(
            MixtralExpert(hidden_size, moe_intermediate_size) for _ in range(n_routed_experts)
        )
        self.shared_experts = nn.ModuleList(
            MixtralExpert(hidden_size, moe_intermediate_size) for _ in range(n_shared_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, input_shape[-1])
        topk_weights, topk_indices, shared_gammas = self.gate(flat)

        routed_out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, top_k_pos = torch.where(topk_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat[token_idx]) * topk_weights[token_idx, top_k_pos, None]
            routed_out.index_add_(0, token_idx, expert_out.to(routed_out.dtype))

        # Shared experts: gamma scales the activated intermediate; the
        # cross-expert sum runs in fp32 (HF: `down.float().sum(dim=0)`).
        shared_out = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
        for shared_idx, module in enumerate(self.shared_experts):
            expert = cast(MixtralExpert, module)
            activated = functional.silu(expert.w1(flat)) * expert.w3(flat)
            activated = activated * shared_gammas[:, shared_idx, None]
            shared_out += expert.w2(activated).float()

        out = routed_out + shared_out.to(flat.dtype)
        return out.view(input_shape)
