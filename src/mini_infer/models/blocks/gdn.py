"""Gated DeltaNet (GDN) core math, the linear-attention layer of Qwen3.5 / Qwen3.8.

GDN is the DeltaNet-lineage recurrence that three out of every four Qwen
hybrid layers use in place of softmax attention. Per sequence it carries a
fixed-size matrix state `S` of shape `(num_v_heads, head_k_dim, head_v_dim)`
updated by a gated delta rule:

    S_t = S_{t-1} * exp(g_t)                             # decay, g_t <= 0
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T         # rank-1 correction
    o_t = S_t^T q_t

This is the SAME recurrence `blocks/kda.py` already implements for Kimi
Delta Attention, with one difference: KDA's decay is per KEY CHANNEL
(the paper's `Diag(a_t)`), while GDN's is a single SCALAR per head. A
scalar is the special case of a diagonal whose entries are all equal, so
broadcasting the GDN scalar across the key channels makes `kda_recurrent`
and `kda_chunkwise` compute the GDN update exactly. That equivalence is not
an approximation: measured against the reference implementations in
transformers (`torch_recurrent_gated_delta_rule`, `torch_chunk_gated_delta_rule`)
in fp32, the max absolute error is at round-off level (~1e-7) across square
and non-square head dims, sequences crossing the chunk boundary, ragged
padding, and non-zero initial states. `tests/unit/test_gdn_block.py` pins it.

So this module deliberately holds ONLY what genuinely differs from KDA:

  - `gdn_gate`: the per-head scalar log-decay, from flat `(num_v_heads,)`
    parameters, where KDA's `kda_gate` reads per-channel ones.
  - `broadcast_gate`: the scalar-to-diagonal expansion that lets the KDA
    recurrence consume a GDN gate.
  - `gated_rmsnorm_silu`: the output norm, which gates with SiLU where KDA
    gates with sigmoid, and applies its affine weight in the INPUT dtype
    rather than in fp32.

Everything else (the delta rule itself, the short causal convolution, the
L2 normalization) is imported from `kda.py` rather than reimplemented.

Numerical conventions pinned from the reference:
  - the recurrence runs in fp32 and outputs cast back,
  - q and k are L2-normalized per head (eps 1e-6) and q is scaled by
    `head_k_dim ** -0.5` BEFORE the recurrence (the reference does this
    inside its kernels via `use_qk_l2norm_in_kernel=True`; we keep it at
    the call site so these functions stay the bare delta rule),
  - the decay gate is `g = -exp(A_log) * softplus(a + dt_bias)` in fp32,
  - `beta = sigmoid(b)`,
  - q and k are broadcast to the value-head count by `repeat_interleave`
    (adjacent duplication) before the recurrence.
"""

from __future__ import annotations

import torch
from torch.nn import functional


def gdn_gate(
    a_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Per-head scalar log-decay: `-exp(A_log) * softplus(a_raw + dt_bias)`.

    Mirrors the reference's
    `-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`.
    The `.float()` on `A_log` is load-bearing: in fp16 `exp` of the stored
    parameter can overflow to `-inf` before the multiply.

    Args:
        a_raw: `(batch, seq_len, num_v_heads)` gate features, the layer's
            `in_proj_a(hidden_states)` output, any float dtype.
        a_log: `(num_v_heads,)` per-head log decay rate.
        dt_bias: `(num_v_heads,)` bias added before softplus.

    Returns:
        `(batch, seq_len, num_v_heads)` fp32 log-decays, all `<= 0` so that
        `exp(g)` is a true decay.
    """
    return -a_log.float().exp() * functional.softplus(a_raw.float() + dt_bias.float())


def broadcast_gate(gate: torch.Tensor, head_k_dim: int) -> torch.Tensor:
    """Expand a per-head scalar log-decay to the per-channel form KDA expects.

    GDN decays the whole state matrix of a head by one scalar; KDA decays it
    by a diagonal. Repeating the scalar across the key channels expresses the
    former in the latter's shape, which is what lets this family reuse
    `kda_recurrent` / `kda_chunkwise` unchanged.

    Args:
        gate: `(batch, seq_len, num_v_heads)` fp32 log-decays.
        head_k_dim: key head dimension to broadcast across.

    Returns:
        `(batch, seq_len, num_v_heads, head_k_dim)` fp32, contiguous.
    """
    batch, seq_len, num_heads = gate.shape
    return gate.unsqueeze(-1).expand(batch, seq_len, num_heads, head_k_dim).contiguous()


def gated_rmsnorm_silu(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Per-head output norm of the GDN layer (the reference's gated RMSNorm).

    Pins the reference's dtype ordering exactly, which matters in bf16 and is
    invisible in fp32: RMS-normalize in fp32, cast back to the INPUT dtype,
    apply the affine weight there, then multiply by a fp32 SiLU of the gate,
    and only then cast the result.

    Note the activation: this family gates with SiLU, where KDA's
    `gated_rmsnorm` gates with sigmoid. The two are different functions, so
    `kda.gated_rmsnorm` must not be substituted here.

    Args:
        x: `(..., head_v_dim)` delta-rule output.
        gate: `(..., head_v_dim)` the layer's `in_proj_z` output.
        weight: `(head_v_dim,)` affine weight, head-shared and ones-init
            (plain RMSNorm affine, NOT the `1 + w` form the surrounding
            decoder norms use).
        eps: variance epsilon.

    Returns:
        Same shape and dtype as `x`.
    """
    input_dtype = x.dtype
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = weight * normed.to(input_dtype)
    return (normed * functional.silu(gate.float())).to(input_dtype)
