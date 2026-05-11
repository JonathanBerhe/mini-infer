"""Mixtral-style top-k sparse Mixture-of-Experts FFN.

A `gate` linear projects the hidden state to per-expert logits; the
top-k experts are selected (Mixtral defaults to top-2 of 8), their
softmax weights are renormalized to sum to 1, and each chosen expert
processes only the tokens routed to it. Per-expert outputs are
weighted-summed back into the token's final hidden state.

Parameter layout matches HF Mixtral's safetensors:
- `block_sparse_moe.gate.weight`            (router, no bias)
- `block_sparse_moe.experts.<j>.w1.weight`  (gate proj of expert j)
- `block_sparse_moe.experts.<j>.w2.weight`  (down proj)
- `block_sparse_moe.experts.<j>.w3.weight`  (up proj)

Naming convention is Mixtral's `w1/w2/w3` instead of Llama's
`gate_proj/up_proj/down_proj`. Mapping: `w1 = gate`, `w2 = down`,
`w3 = up`. Same SiLU-gated FFN math as `SwiGLU`.

V1 dispatch is the simple per-expert loop: gather tokens that picked
this expert, run them through the expert's MLP, scatter weighted
outputs back via `index_add_`. A fused grouped-GEMM is a follow-up.

Expert parallelism
------------------
Each rank owns `num_experts // world_size` *contiguous* experts (rank
`r` holds experts `[r*N/ws, (r+1)*N/ws)`). The gate is replicated so
every rank computes the same top-k. Each rank's per-expert loop runs
only over its local experts; tokens routed to off-rank experts produce
no contribution on this rank. A final all-reduce sums the partial
routed accumulators to recover the full result. Shared experts are
replicated on every rank.

This is the simplest correct form of EP; the all-to-all dispatch
optimisation (lower comm cost when tokens cluster on a few experts) is
a follow-up.
"""

import torch
from torch import nn
from torch.nn import functional

from mini_infer.distributed.comm import all_reduce_sum
from mini_infer.distributed.linear import _split_size


class MixtralExpert(nn.Module):
    """One expert's MLP. SwiGLU-shaped, but with Mixtral's w1/w2/w3 names.

    Inside an expert-parallel MoE this is built only on the rank that owns
    the expert; the un-sharded `nn.Linear`s here are *local* (no further
    sharding), so the per-rank expert acts as a normal `SwiGLU`-shape FFN.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.w2(functional.silu(self.w1(x)) * self.w3(x))
        return out


class MoEFFN(nn.Module):
    """Mixtral-style top-k sparse MoE: router + expert dispatch + weighted sum.

    Each token's hidden state goes through:
    1. `gate` produces per-expert routing logits `(T, num_experts)`.
    2. Softmax + top-k selects the `top_k` experts per token; weights are
       renormalized so they sum to 1 over the chosen experts.
    3. For each expert that received any tokens, run those tokens through
       its MLP and scatter the weighted output back into the result tensor
       at the original token positions.

    Used by Mixtral 8x7B / 8x22B and is the FFN primitive every Phase 4+
    MoE family (DeepSeek-V2/V3, Kimi-K2, V4) reuses.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        n_shared_experts: int = 0,
        shared_intermediate_size: int | None = None,
        renormalize_topk: bool = True,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
        from mini_infer.distributed.group import get_rank, get_world_size

        world_size = get_world_size()
        num_experts_per_rank = _split_size(num_experts, world_size, "num_experts")
        self.num_experts = num_experts  # global, used for top-k correctness
        self.num_experts_per_rank = num_experts_per_rank
        self.world_size = world_size
        self.rank = get_rank()
        # Range of *global* expert indices this rank owns.
        self.local_expert_start = self.rank * num_experts_per_rank
        self.local_expert_end = self.local_expert_start + num_experts_per_rank
        self.top_k = top_k
        # Mixtral renormalizes the top-k softmax probs to sum to 1; DeepSeek
        # keeps the raw probs and multiplies by `routed_scaling_factor`.
        self.renormalize_topk = renormalize_topk
        self.routed_scaling_factor = routed_scaling_factor
        # Gate is replicated: every rank computes the same routing decisions
        # from the (replicated) hidden state.
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        # Local experts only — each rank owns `num_experts_per_rank` of them.
        # `self.experts[local_idx]` is the expert with global index
        # `local_expert_start + local_idx`.
        self.experts = nn.ModuleList(
            [MixtralExpert(hidden_size, intermediate_size) for _ in range(num_experts_per_rank)]
        )
        # Shared experts (DeepSeek-V2/V3): MLPs that fire on every token,
        # added to the routed output. `n_shared_experts=0` keeps Mixtral's
        # routed-only behavior.
        self.n_shared_experts = n_shared_experts
        if n_shared_experts > 0:
            shared_inter = (
                shared_intermediate_size
                if shared_intermediate_size is not None
                else intermediate_size
            )
            # DeepSeek collapses N shared experts into a single MLP with
            # `N * intermediate_size` hidden width — bit-identical to
            # running N separate experts in parallel and summing, but a
            # single matmul. Match HF's `DeepseekV2MLP(intermediate_size *
            # n_shared_experts)` shape so weight loading is identity.
            self.shared_experts: MixtralExpert | None = MixtralExpert(
                hidden_size, shared_inter * n_shared_experts
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (1, total_q, hidden) — flatten to (T, hidden) for routing.
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Router: full-precision softmax for stability. Both Mixtral and
        # DeepSeek-V2 use softmax-over-all-experts then top-k; what differs
        # is whether they renormalize and what scaling factor they apply.
        # Gate is replicated, so every rank picks the same top-k globally.
        router_logits = self.gate(flat).float()
        router_probs = functional.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.renormalize_topk:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            top_k_weights = top_k_weights * self.routed_scaling_factor
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        out = torch.zeros_like(flat)
        # `expert_mask[expert_idx, top_k_pos, token_idx] = 1` iff token `token_idx`
        # picked `expert_idx` at position `top_k_pos` in its top-k. Permuting to
        # `(num_experts, top_k, T)` makes the per-expert lookup an O(1) slice.
        expert_mask = functional.one_hot(top_k_indices, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        # Loop only over experts this rank owns. Tokens routed to experts on
        # other ranks contribute zero here — the all-reduce below fills them in.
        for local_idx in range(self.num_experts_per_rank):
            global_idx = self.local_expert_start + local_idx
            top_k_pos, token_idx = torch.where(expert_mask[global_idx])
            if token_idx.numel() == 0:
                continue
            tokens = flat[token_idx]
            weights = top_k_weights[token_idx, top_k_pos, None]
            expert_out = self.experts[local_idx](tokens) * weights
            out.index_add_(0, token_idx, expert_out)
        # All-reduce the partial routed sums across ranks. At world_size=1
        # this is a no-op; existing single-device behaviour is bit-identical.
        out = all_reduce_sum(out)
        # Shared experts are replicated; their output is the same on every
        # rank, so it's added AFTER the routed all-reduce (otherwise the
        # reduce would multiply it by world_size).
        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(input_shape)
