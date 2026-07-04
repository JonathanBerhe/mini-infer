"""GLM-MoE-DSA MoE: DeepSeek-V3-style `noaux_tc` sigmoid gate + sparse FFN.

The routing is the one piece mini-infer's `MoEFFN` (softmax) does not cover, so
it lives here as `GlmNoAuxTcGate`, faithful to HF `GlmMoeDsaMoE.route_tokens_to_experts`:

    scores       = sigmoid(x @ Wᵀ)                 # per-expert affinity
    choice       = scores + e_score_correction_bias # bias used for SELECTION only
    group_score  = choice.view(G groups).topk(2).sum(-1)   # DeepSeek group score
    keep groups  = topk(group_score, topk_group)           # mask other groups -inf
    idx          = topk(choice_masked, top_k)
    weights      = scores.gather(idx)              # UNBIASED scores weight the experts
    weights     /= weights.sum() + 1e-20           # if norm_topk_prob
    weights     *= routed_scaling_factor

The aux-loss-free bias (`noaux_tc`) only tilts *selection*; the expert weighting
uses the raw sigmoid scores. `n_group == topk_group == 1` (the real config)
makes the grouping a no-op, but the general path mirrors HF for any grouping.

`GlmMoeFFN` reuses `MixtralExpert` for the per-expert SwiGLU and the shared
expert, plus the same expert-parallel dispatch shape as `MoEFFN` (each rank owns
a contiguous expert range; partial routed sums are all-reduced; the shared
expert is replicated and added after the reduce).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from mini_infer.distributed.comm import all_reduce_sum
from mini_infer.distributed.linear import _split_size
from mini_infer.models.blocks.activations import GateUpActivation, swiglu
from mini_infer.models.blocks.fp8_expert import Fp8Expert
from mini_infer.models.blocks.mixtral_moe import MixtralExpert


class GlmNoAuxTcGate(nn.Module):
    """DeepSeek-V3 `noaux_tc` sigmoid router with grouped top-k selection."""

    e_score_correction_bias: torch.Tensor

    def __init__(
        self,
        *,
        hidden_size: int,
        n_routed_experts: int,
        top_k: int,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if n_routed_experts % n_group != 0:
            raise ValueError(
                f"n_routed_experts={n_routed_experts} must be divisible by n_group={n_group}"
            )
        if not 0 < top_k <= n_routed_experts:
            raise ValueError(f"top_k={top_k} must be in (0, n_routed_experts={n_routed_experts}]")
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        # Gate is replicated (every rank routes identically). Named/shaped to
        # match HF `mlp.gate.weight` so weight loading is a direct copy.
        # A real checkpoint always overwrites this via load_weights, but an
        # un-loaded model (tests that run a freshly-constructed reference
        # model directly) must not depend on whatever garbage `torch.empty`
        # happens to return: init it the same way `nn.Linear`'s default
        # `reset_parameters` would, so a fresh instance is a well-defined,
        # finite router rather than uninitialized memory.
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Aux-loss-free selection bias (HF `mlp.gate.e_score_correction_bias`).
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(n_routed_experts, dtype=torch.float32)
        )

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route `(T, hidden)` tokens. Returns `(top_k_indices, top_k_weights)`,
        both `(T, top_k)`; weights are fp32 (unbiased sigmoid, normalized, scaled).
        """
        # fp32 routing for stability and bit-parity with HF (gate runs in fp32).
        logits = functional.linear(x_flat.float(), self.weight.float())
        scores = logits.sigmoid()
        choice = scores + self.e_score_correction_bias

        # Grouped top-k (DeepSeek): score each group by its top-2 experts, keep
        # the best `topk_group` groups, mask the rest before the final top-k.
        n_per_group = self.n_routed_experts // self.n_group
        group_scores = choice.view(-1, self.n_group, n_per_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, n_per_group)
            .reshape(-1, self.n_routed_experts)
        )
        choice_masked = choice.masked_fill(~score_mask.bool(), float("-inf"))

        topk_indices = torch.topk(choice_masked, k=self.top_k, dim=-1, sorted=False)[1]
        # Expert weights come from the UNBIASED sigmoid scores.
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class GlmMoeFFN(nn.Module):
    """GLM-MoE-DSA sparse FFN: `noaux_tc` gate + routed experts + shared expert.

    Mirrors HF `GlmMoeDsaMoE`. Reuses `MixtralExpert` for the SwiGLU experts
    and the (collapsed) shared expert, and the expert-parallel dispatch shape
    from `MoEFFN`: each rank owns a contiguous expert range, partial routed
    sums are all-reduced, and the replicated shared expert is added afterwards.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        top_k: int,
        n_shared_experts: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        expert_dtype: str = "bf16",
        activation: GateUpActivation = swiglu,
    ) -> None:
        super().__init__()
        from mini_infer.distributed.group import get_rank, get_world_size

        world_size = get_world_size()
        num_experts_per_rank = _split_size(n_routed_experts, world_size, "n_routed_experts")
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_rank = num_experts_per_rank
        self.rank = get_rank()
        self.local_expert_start = self.rank * num_experts_per_rank
        self.gate = GlmNoAuxTcGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
        # Local routed experts only (this rank's contiguous slice). "fp8" keeps
        # them e4m3-resident (dequant per-call); "bf16" is the dequantized path.
        if expert_dtype not in ("bf16", "fp8"):
            raise ValueError(f"expert_dtype must be 'bf16' or 'fp8'; got {expert_dtype!r}")
        expert_cls = Fp8Expert if expert_dtype == "fp8" else MixtralExpert
        self.experts = nn.ModuleList(
            [
                expert_cls(hidden_size, moe_intermediate_size, activation=activation)
                for _ in range(num_experts_per_rank)
            ]
        )
        # Shared expert fires on every token. DeepSeek collapses N shared
        # experts into one MLP of width `N * moe_intermediate_size`; replicated.
        self.n_shared_experts = n_shared_experts
        self.shared_experts: MixtralExpert | None = (
            MixtralExpert(
                hidden_size, moe_intermediate_size * n_shared_experts, activation=activation
            )
            if n_shared_experts > 0
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, hidden_states.shape[-1])

        top_k_indices, top_k_weights = self.gate(flat)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        out = torch.zeros_like(flat)
        # expert_mask[expert, top_k_pos, token] = 1 iff token picked expert at
        # that top-k slot. Permute so a per-expert lookup is an O(1) slice.
        expert_mask = functional.one_hot(top_k_indices, num_classes=self.n_routed_experts).permute(
            2, 1, 0
        )
        for local_idx in range(self.num_experts_per_rank):
            global_idx = self.local_expert_start + local_idx
            top_k_pos, token_idx = torch.where(expert_mask[global_idx])
            if token_idx.numel() == 0:
                continue
            tokens = flat[token_idx]
            weights = top_k_weights[token_idx, top_k_pos, None]
            out.index_add_(0, token_idx, self.experts[local_idx](tokens) * weights)
        # Sum partial routed contributions across ranks (no-op at world_size=1).
        out = all_reduce_sum(out)
        # Shared expert is replicated; add after the reduce so it isn't scaled
        # by world_size.
        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(input_shape)
