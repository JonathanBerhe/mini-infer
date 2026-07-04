"""Gate-up activation combiners for SwiGLU-family FFNs.

A gated FFN computes `down(act(gate(x), up(x)))`. The combiner `act` takes the
gate and up projections and returns the gated activation:

- `swiglu` is the standard SiLU form (Llama / Qwen / Mixtral / GLM):
  `silu(gate) * up`.
- `swigluoai` is the clamped variant (GPT-OSS, MiniMax-M3's `swigluoai`
  `hidden_act`). It clamps both branches and adds a unit bias to the up branch,
  so it is a JOINT function of gate and up, not a pointwise activation on the
  gate alone, and cannot be expressed by swapping the SiLU in `swiglu`.

Blocks that were SiLU-only (`SwiGLU`, `MixtralExpert`, `Fp8Expert`, `GlmMoeFFN`)
take an `activation` combiner defaulting to `swiglu`, so existing families are
numerically unchanged; M3 passes `swigluoai`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional

# A gate-up combiner: (gate_proj(x), up_proj(x)) -> gated activation.
GateUpActivation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Standard SwiGLU: `silu(gate) * up`."""
    return functional.silu(gate) * up


def swigluoai(
    gate: torch.Tensor, up: torch.Tensor, *, alpha: float = 1.702, limit: float = 7.0
) -> torch.Tensor:
    """Clamped SwiGLU (`swigluoai`, GPT-OSS / MiniMax-M3).

    `gate` is clamped to `max=limit` (no lower bound); `up` is clamped
    symmetrically to `[-limit, limit]`; the `+1` bias is on the up branch:

        out = (clamp(up, -limit, limit) + 1) * g * sigmoid(alpha * g),  g = clamp(gate, max=limit)

    `g` appears in both the linear factor and the sigmoid argument (so the gated
    term is `g * sigmoid(alpha * g)`, a SiLU with a temperature `alpha`). Defaults
    `alpha=1.702`, `limit=7.0` are M3's config values.
    """
    g = gate.clamp(max=limit)
    u = up.clamp(min=-limit, max=limit)
    return (u + 1.0) * g * torch.sigmoid(alpha * g)
