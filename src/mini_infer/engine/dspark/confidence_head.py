"""Confidence head: predicts each draft token's conditional survival probability.

A single linear projection, no sigmoid inside the module. DeepSpec applies
sigmoid only at each call site (BCE-with-logits during training, an explicit
`.sigmoid()` at inference truncation), so this module returns a raw logit,
matching `deepspec/modeling/dspark/common.py::AcceptRatePredictor` exactly.

When `confidence_head_with_markov` is set, the caller concatenates the
Markov head's own previous-token embedding (`VanillaMarkovHead.get_prev_embeddings`)
onto the backbone hidden state before calling this head; that's why
`input_dim` is `hidden_size` or `hidden_size + markov_rank`, decided by
`Qwen3DSparkDrafter`, not by this module.
"""

from __future__ import annotations

import torch
from torch import nn


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logit: torch.Tensor = self.proj(features).squeeze(-1)
        return logit
