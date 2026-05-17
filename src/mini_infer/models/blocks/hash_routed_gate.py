"""Hash-routed MoE gate (DeepSeek-V4 paper §2.2, also softmax-topk variant).

V4's first `n_hash_layers` layers route each token to a fixed set of
experts via a per-token-id lookup table (`tid2eid: (vocab_size,
num_activated_experts) -> expert_indices`). Later layers route via the
standard "score the experts, pick top-k" path. Both modes share the
same expert-weighting math; only the index-selection differs.

This module is a single class with two configurations chosen at
construction time:

  - `routing_mode="hash"`: indices come from the lookup table; no
    learnable bias (the per-token routing IS the parameter).
  - `routing_mode="score_topk"`: indices come from `topk(scores + bias)`;
    bias is a learnable per-expert scalar that lets the model adjust
    the topk selection without affecting the weights it produces.

Three score functions are supported (V4 paper §2.2 + reference's
`Gate.score_func`):

  - `"softmax"`: standard softmax over the expert dimension. Weights
    are NOT renormalized after gathering top-k indices (softmax of all
    experts is the implicit "global" weighting).
  - `"sigmoid"`: per-expert independent sigmoid. Gathered weights ARE
    renormalized to sum to 1 across the activated experts.
  - `"softplus_sqrt"`: `sqrt(softplus(scores))`. Same renorm rule as
    sigmoid.

Final weights are multiplied by a `route_scale` scalar (paper's
`route_scale`; default 1.0). Bit-parity with the V4 reference's `Gate`
class on synthetic input across all six (mode x score-function) cells.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn.functional import linear, softplus

ScoreFunction = Literal["softmax", "sigmoid", "softplus_sqrt"]
RoutingMode = Literal["hash", "score_topk"]


class HashRoutedGate(nn.Module):
    """MoE gate with hash-table routing OR score-based top-k routing.

    Args:
        hidden_size: Input feature dim.
        num_routed_experts: Total number of routed experts the gate scores.
        num_activated_experts: How many experts each token routes to (top-k).
        routing_mode: `"hash"` (per-token-id lookup) or `"score_topk"`
            (top-k of scores + bias).
        score_func: Activation applied to the raw scores.
        route_scale: Scalar applied to the final weights (paper §2.2).
        vocab_size: Required iff `routing_mode == "hash"`. Sizes the
            `tid2eid` lookup table's first dim.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_routed_experts: int,
        num_activated_experts: int,
        routing_mode: RoutingMode,
        score_func: ScoreFunction = "softmax",
        route_scale: float = 1.0,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_routed_experts <= 0:
            raise ValueError(f"num_routed_experts must be positive, got {num_routed_experts}")
        if num_activated_experts <= 0 or num_activated_experts > num_routed_experts:
            raise ValueError(
                f"num_activated_experts must be in [1, num_routed_experts]; "
                f"got {num_activated_experts} of {num_routed_experts}"
            )
        if score_func not in ("softmax", "sigmoid", "softplus_sqrt"):
            raise ValueError(f"unknown score_func={score_func!r}")
        if routing_mode not in ("hash", "score_topk"):
            raise ValueError(f"unknown routing_mode={routing_mode!r}")
        if routing_mode == "hash" and (vocab_size is None or vocab_size <= 0):
            raise ValueError(
                "routing_mode='hash' requires a positive vocab_size for the "
                f"tid2eid lookup; got {vocab_size}"
            )
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_activated_experts = num_activated_experts
        self.routing_mode = routing_mode
        self.score_func = score_func
        self.route_scale = route_scale

        # `weight`: per-expert scoring projection. Always present (the score
        # function output drives the per-expert weights even in hash mode,
        # where only the index selection comes from `tid2eid`).
        self.weight = nn.Parameter(
            torch.empty(num_routed_experts, hidden_size, dtype=torch.float32)
        )
        if routing_mode == "hash":
            assert vocab_size is not None  # narrowed by the check above
            # `tid2eid`: per-token-id expert routing. Non-trainable — the
            # routing is fixed at training time per the V4 paper.
            self.tid2eid = nn.Parameter(
                torch.empty(vocab_size, num_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias: nn.Parameter | None = None
        else:
            # `bias`: per-expert top-k-selection bias. Shifts the score for
            # selection but is NOT added to the gathered weights — the model
            # uses bias to learn "which experts to prefer" while keeping the
            # weighting math driven by the unbiased score function.
            self.bias = nn.Parameter(torch.empty(num_routed_experts, dtype=torch.float32))
        # Initialize all parameters. In production, `load_state_dict` from
        # a checkpoint overwrites these immediately; the init exists to
        # make synthetic/random-weight uses (tests, demos) well-defined.
        # Skipping it left `torch.empty` allocations carrying whatever
        # bytes the allocator returned — values that happened to look
        # sane on macOS dev hardware but produced wildly out-of-range
        # int32s on Linux CI (e.g. -1095055515), crashing the hash-routed
        # path with index-out-of-bounds errors.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learned + non-learned buffers to well-defined values.

        Mirrors `nn.Linear.reset_parameters` for the score projection. The
        `tid2eid` table is randomly assigned across `num_routed_experts`
        (the paper's fix-at-train-time routing is captured at load time
        from the checkpoint; this init is for the synthetic / no-checkpoint
        case).
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.routing_mode == "hash":
            with torch.no_grad():
                self.tid2eid.copy_(
                    torch.randint(
                        low=0,
                        high=self.num_routed_experts,
                        size=self.tid2eid.shape,
                        dtype=torch.int32,
                    )
                )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute `(weights, indices)` for the experts each token routes to.

        Args:
            hidden_states: `(N, hidden_size)` flat token embeddings (the caller
                is expected to flatten `(B, T, hidden_size)` before passing).
            input_ids: `(N,)` int64 token ids — required iff `routing_mode == "hash"`.

        Returns:
            weights: `(N, num_activated_experts)` float — per-token weights for
                each chosen expert, scaled by `route_scale`. Sum-to-1 across the
                expert dim for non-softmax score functions.
            indices: `(N, num_activated_experts)` int64 — chosen expert indices.

        Math mirrors the V4 reference's `Gate.forward` exactly:
            scores_pre = score_func(linear(x.float(), weight.float()))
            indices    = tid2eid[input_ids]              if hash else
                         topk(scores_pre + bias).indices  if score_topk
            weights    = scores_pre.gather(1, indices)
            weights   /= weights.sum(-1, keepdim=True)    if score_func != "softmax"
            weights   *= route_scale
        """
        num_tokens = hidden_states.shape[0]
        # ---- Score the experts (always in fp32 for stability) ----
        raw_scores = linear(hidden_states.float(), self.weight.float())
        if self.score_func == "softmax":
            normalized_scores = raw_scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            normalized_scores = raw_scores.sigmoid()
        else:  # softplus_sqrt
            normalized_scores = softplus(raw_scores).sqrt()

        # ---- Pick the activated experts ----
        if self.routing_mode == "hash":
            if input_ids is None:
                raise ValueError(
                    "HashRoutedGate(routing_mode='hash') requires input_ids in forward"
                )
            if input_ids.shape[0] != num_tokens:
                raise ValueError(
                    f"input_ids has {input_ids.shape[0]} tokens but hidden_states has {num_tokens}"
                )
            # tid2eid is int32 buffer; gather requires int64 for indices below.
            indices = self.tid2eid[input_ids].to(torch.int64)
        else:
            # `score_topk`: bias affects ONLY the top-k selection, not the
            # weights returned to the caller.
            assert self.bias is not None
            biased_scores = normalized_scores + self.bias
            indices = biased_scores.topk(self.num_activated_experts, dim=-1).indices.to(torch.int64)

        # ---- Gather weights from the unbiased scores ----
        weights = normalized_scores.gather(1, indices)

        # ---- Per-token renormalization for non-softmax score functions ----
        # Softmax already produces a normalized distribution over ALL experts,
        # so the gathered subset shouldn't be renormalized — the missing mass
        # represents the unselected experts. Sigmoid / softplus_sqrt produce
        # independent per-expert scores, so renormalizing is required to make
        # them a proper combining-weight distribution.
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        weights = weights * self.route_scale
        return weights, indices
