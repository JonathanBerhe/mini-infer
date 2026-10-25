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

Tensor parallelism
------------------
Sharded by head: each rank owns `num_heads // world_size` sink logits,
indexed in lock-step with that rank's Q heads. At `world_size=1` the
parameter is the full `(num_heads,)` tensor.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.distributed.group import get_rank, get_world_size


class AttentionSink(nn.Module):
    """One learnable scalar logit per query head, added to the softmax denominator."""

    sink_logits: nn.Parameter

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        world_size = get_world_size()
        if num_heads % world_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by world_size={world_size}")
        # Reference initializes from `torch.empty` and trains; for inference-time
        # block construction we start at zero (a near-no-op sink, ~1/Nk weight).
        self.num_heads = num_heads
        self.num_heads_per_rank = num_heads // world_size
        self.world_size = world_size
        self.rank = get_rank()
        self.sink_logits = nn.Parameter(torch.zeros(self.num_heads_per_rank, dtype=torch.float32))

    def load_full_logits(
        self,
        full_logits: torch.Tensor,
        *,
        target_device: torch.device | str | None = None,
    ) -> None:
        """Slice this rank's per-head logit range out of the full vector."""
        if full_logits.shape != (self.num_heads,):
            raise ValueError(
                f"full_logits shape {tuple(full_logits.shape)} does not match expected "
                f"({self.num_heads},)"
            )
        start = self.rank * self.num_heads_per_rank
        end = start + self.num_heads_per_rank
        if self.sink_logits.is_meta:
            sliced = full_logits[start:end].contiguous()
            if target_device is not None:
                sliced = sliced.to(device=target_device)
            self.sink_logits = nn.Parameter(sliced, requires_grad=False)
        else:
            sliced = full_logits[start:end].to(self.sink_logits.dtype).contiguous()
            with torch.no_grad():
                self.sink_logits.copy_(sliced)
