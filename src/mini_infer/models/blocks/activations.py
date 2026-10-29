"""Gate-up activation combiners for SwiGLU-family FFNs.

A gated FFN computes `down(act(gate(x), up(x)))`. The combiner `act` takes the
gate and up projections and returns the gated activation:

- `swiglu` is the standard SiLU form (Llama / Qwen / Mixtral / GLM):
  `silu(gate) * up`.
- `swigluoai` is the clamped variant (GPT-OSS, MiniMax-M3's `swigluoai`
  `hidden_act`). It clamps both branches and adds a unit bias to the up branch,
  so it is a JOINT function of gate and up, not a pointwise activation on the
  gate alone, and cannot be expressed by swapping the SiLU in `swiglu`.
- `clamped_swiglu` is DeepSeek-V4's experts (`swiglu_limit`): the same two
  clamps as `swigluoai`, then plain `silu(gate) * up`.
- `situ` is Kimi K3's SiTU-GLU (`hidden_act = "situ"`): a tanh-bounded SiLU on
  the gate and a tanh-bounded identity on the up branch, computed in fp32.

Blocks that were SiLU-only (`SwiGLU`, `MixtralExpert`, `Fp8Expert`, `GlmMoeFFN`)
take an `activation` combiner defaulting to `swiglu`, so existing families are
numerically unchanged; M3 passes `swigluoai`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
from torch.nn import functional

# A gate-up combiner: (gate_proj(x), up_proj(x)) -> gated activation.
GateUpActivation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Standard SwiGLU: `silu(gate) * up`."""
    return functional.silu(gate) * up


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, *, limit: float) -> torch.Tensor:
    """DeepSeek-V4 expert SwiGLU with `swiglu_limit`.

        out = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)

    Same clamps as the reference `Expert` and HF `DeepseekV4MLP` / `Experts`:
    `gate` only from above, `up` symmetrically. `limit` must be positive;
    callers keep plain `swiglu` for a limit of 0 (no clamp).
    """
    return functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def expert_swiglu(swiglu_limit: float) -> GateUpActivation:
    """The SwiGLU combiner for a DeepSeek-V4 expert: plain if `swiglu_limit` is 0."""
    if swiglu_limit < 0:
        raise ValueError(f"swiglu_limit must be >= 0, got {swiglu_limit}")
    if swiglu_limit == 0:
        return swiglu
    return partial(clamped_swiglu, limit=swiglu_limit)


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


def situ(
    gate: torch.Tensor, up: torch.Tensor, *, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """SiTU-GLU (Kimi K3's `hidden_act = "situ"`; reference `SituAndMul`).

        a   = beta * tanh(gate / beta) * sigmoid(gate)
        u   = linear_beta * tanh(up / linear_beta)   if linear_beta is set, else up
        out = a * u

    Both branches run in fp32 and the product is cast back to `gate.dtype`,
    as the reference does on its concatenated gate-up tensor. `beta` bounds
    the gate's SiLU to `(-beta, beta)`; `linear_beta` bounds the up branch
    the same way. K3 ships `beta = 4.0`, `linear_beta = 25.0`.
    """
    g = gate.float()
    u = up.float()
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (a * u).to(gate.dtype)


def situ_activation(beta: float | None, linear_beta: float | None) -> GateUpActivation:
    """The SiTU combiner for a config's `activation_situ_beta` /
    `activation_situ_linear_beta` (a falsy `beta` falls back to 1.0, the
    reference's `_get_situ_activation_params` rule)."""
    return partial(situ, beta=beta or 1.0, linear_beta=linear_beta)
