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
"""

import torch
from torch import nn
from torch.nn import functional


class MixtralExpert(nn.Module):
    """One expert's MLP. SwiGLU-shaped, but with Mixtral's w1/w2/w3 names."""

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
    ) -> None:
        super().__init__()
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MixtralExpert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (1, total_q, hidden) — flatten to (T, hidden) for routing.
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Router: full-precision softmax for stability.
        router_logits = self.gate(flat).float()
        router_probs = functional.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize so the chosen `top_k` weights sum to 1 per token.
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        out = torch.zeros_like(flat)
        # `expert_mask[expert_idx, top_k_pos, token_idx] = 1` iff token `token_idx`
        # picked `expert_idx` at position `top_k_pos` in its top-k. Permuting to
        # `(num_experts, top_k, T)` makes the per-expert lookup an O(1) slice.
        expert_mask = functional.one_hot(top_k_indices, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            tokens = flat[token_idx]
            weights = top_k_weights[token_idx, top_k_pos, None]
            expert_out = self.experts[expert_idx](tokens) * weights
            out.index_add_(0, token_idx, expert_out)
        return out.view(input_shape)
