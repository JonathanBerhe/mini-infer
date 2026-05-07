"""Per-head learnable attention-sink logit (V4 paper §2.3.3, formula 27).

The sink trick (originating from StreamingLLM, OpenAI 2025): give each
head one extra "key" with a learnable logit `z'_h` and a value of zero.
Concretely the softmax denominator gains an `exp(z'_h)` term per head:

    sum_exp_h = sum_k exp(score_{q,k,h}) + exp(z'_h)

This lets a head dump unwanted attention probability into the sink
instead of distributing it across "real" keys (which corrupts the
output). It's a no-op at the value side because the sink's value is 0.

We expose it as a tiny `nn.Module` whose only state is `sink_logits`
of shape `(num_heads,)`. The actual softmax-incorporation lives in
`hca_mqa_with_sink` (the dispatcher) — keeping the parameter and the
math-that-uses-it separate matches how the reference V4 code organizes
the same idea.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionSink(nn.Module):
    """One learnable scalar logit per query head, added to the softmax denominator."""

    sink_logits: nn.Parameter

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        # Reference initializes from `torch.empty` and trains; for inference-time
        # block construction we start at zero (a near-no-op sink, ~1/Nk weight).
        self.sink_logits = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
