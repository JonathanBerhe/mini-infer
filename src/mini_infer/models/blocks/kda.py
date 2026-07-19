"""Kimi Delta Attention (KDA) core math — Kimi Linear (arXiv:2510.26692), Kimi K3.

KDA is a linear-attention layer: instead of a per-token KV cache it carries a
fixed-size matrix state `S` of shape `(num_heads, head_dim_k, head_dim_v)` per
sequence, updated by a gated delta rule (DeltaNet lineage) with a PER-CHANNEL
decay `a_t` (the paper's `Diag(a_t)`; Gated DeltaNet uses one scalar per head,
KDA refines it to one per key channel):

    S_t = S_{t-1} * Diag(exp(g_t))                       # decay, g_t <= 0 in log space
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T         # delta-rule rank-1 correction
    o_t = S_t^T q_t

The reference semantics are Moonshot's `modeling_kimi.py` driving the FLA
(`fla-org/flash-linear-attention`) `chunk_kda` / `fused_recurrent_kda` kernels;
the naive reference for those kernels is `fla/ops/kda/naive.py`. Numerical
conventions pinned from that stack (see docs/plans/kimi-k3-spec.md):

  - everything runs in fp32 inside the recurrence; outputs cast back,
  - q and k are L2-normalized per head (eps 1e-6) and q is scaled by
    `head_dim_k ** -0.5` BEFORE the recurrence (the kernels do this inside
    when `use_qk_l2norm_in_kernel=True`; we keep it explicit at the call
    site in the layer, so these functions are the bare delta rule),
  - the decay gate is `g = -exp(A_log) * softplus(raw + dt_bias)` in fp32,
  - the layer's inputs pass through short causal convolutions (kernel 4,
    SiLU) whose rolling per-request state is `(channels, kernel)` holding
    the last `kernel` RAW inputs, newest last (FLA cache layout).

Two equivalent evaluation orders are provided: `kda_recurrent` (one token at
a time; the decode step and the correctness oracle) and `kda_chunkwise`
(the prefill form: the paper's chunked WY/UT algorithm over its specialized
DPLR transition). The unit tests pin chunkwise == recurrent.
"""

from __future__ import annotations

import torch
from torch.nn import functional


def kda_gate(
    raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Per-channel log-space decay gate: `-exp(A_log) * softplus(raw + dt_bias)`.

    Mirrors FLA's `fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)`.

    Args:
        raw: `(..., num_heads * head_dim)` gate features (the layer's
            low-rank `f_b_proj(f_a_proj(x))` output), any float dtype.
        a_log: per-head log-decay-rate parameter; any shape with
            `num_heads` elements (the checkpoint ships `(1, 1, H, 1)`).
        dt_bias: `(num_heads * head_dim,)` bias added before softplus.
        head_dim: per-head gate width (the KDA key head dim).

    Returns:
        `(..., num_heads, head_dim)` fp32 log-decays, all `<= 0` so
        `exp(g)` is a true decay.
    """
    num_heads = raw.shape[-1] // head_dim
    gate = raw.float().view(*raw.shape[:-1], num_heads, head_dim)
    gate = gate + dt_bias.float().view(num_heads, head_dim)
    return -a_log.float().view(num_heads, 1).exp() * functional.softplus(gate)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last dim in fp32 (FLA `l2norm` convention, eps inside sqrt)."""
    x32 = x.float()
    return x32 * torch.rsqrt(x32.pow(2).sum(dim=-1, keepdim=True) + eps)


def gated_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """FLA `FusedRMSNormGated(activation='sigmoid')` equivalent.

    Per-head output norm of the KDA layer: RMS-normalize the last dim in
    fp32, apply the (head_dim-wide, head-shared) affine weight, THEN the
    sigmoid gate, and only then cast back — the kernel keeps everything in
    fp32 through the gate multiply.
    """
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = normed * weight.float()
    return (normed * torch.sigmoid(gate.float())).to(x.dtype)


def causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv + SiLU over a packed span, with optional left context.

    Args:
        x: `(batch, seq_len, channels)` RAW pre-conv inputs.
        weight: `(channels, kernel)` depthwise taps, `weight[:, -1]` applied
            to the newest position.
        conv_state: `(batch, channels, kernel)` rolling state holding the
            last `kernel` raw inputs (newest last), or None for a fresh
            sequence (zero left-padding).

    Returns:
        `(y, new_conv_state)` where `y` is `(batch, seq_len, channels)` in
        `x.dtype` (conv + SiLU computed in fp32) and `new_conv_state` is the
        updated `(batch, channels, kernel)` raw-input tail.
    """
    batch, _seq_len, channels = x.shape
    kernel = weight.shape[-1]
    x32 = x.float().transpose(1, 2)  # (batch, channels, seq_len)
    if conv_state is None:
        left = x32.new_zeros(batch, channels, kernel - 1)
    else:
        # The state holds the last `kernel` inputs; a new token's receptive
        # field needs only the most recent `kernel - 1` of them.
        left = conv_state.float()[:, :, 1:]
    padded = torch.cat([left, x32], dim=-1)
    y = functional.conv1d(padded, weight.float().unsqueeze(1), groups=channels)
    y = functional.silu(y)
    # New state: the last `kernel` raw inputs, zero-filled on the left when
    # the sequence (plus carried context) is shorter than the kernel.
    if conv_state is None:
        history = torch.cat([x32.new_zeros(batch, channels, kernel), x32], dim=-1)
    else:
        history = torch.cat([conv_state.float(), x32], dim=-1)
    new_state = history[:, :, -kernel:]
    return y.transpose(1, 2).to(x.dtype), new_state.to(x.dtype)


def causal_conv1d_step(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decode step of the depthwise causal conv + SiLU.

    Args:
        x: `(batch, channels)` the new token's RAW pre-conv input.
        weight: `(channels, kernel)` depthwise taps.
        conv_state: `(batch, channels, kernel)` raw-input tail, newest last.

    Returns:
        `(y, new_conv_state)`: `y` is `(batch, channels)` in `x.dtype`;
        the state is rolled left with `x` written at the newest slot.
    """
    new_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1).to(conv_state.dtype)], dim=-1)
    y = (new_state.float() * weight.float().unsqueeze(0)).sum(dim=-1)
    return functional.silu(y).to(x.dtype), new_state


def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta rule, one token at a time (decode step / correctness oracle).

    Inputs are PRE-normalized: the caller applies `l2norm` to q and k and the
    `head_dim_k ** -0.5` scale to q (see module docstring).

    Args:
        q, k: `(batch, seq_len, num_heads, head_dim_k)`.
        v: `(batch, seq_len, num_heads, head_dim_v)`.
        g: `(batch, seq_len, num_heads, head_dim_k)` fp32 log-decays.
        beta: `(batch, seq_len, num_heads)` fp32 delta-rule step sizes.
        initial_state: `(batch, num_heads, head_dim_k, head_dim_v)` fp32, or
            None for a zero state.

    Returns:
        `(o, final_state)`: `o` is `(batch, seq_len, num_heads, head_dim_v)`
        in `v.dtype`; `final_state` is fp32.
    """
    batch, seq_len, num_heads, dim_k = q.shape
    dim_v = v.shape[-1]
    q32, k32, v32, g32, beta32 = (t.float() for t in (q, k, v, g, beta))
    state = q32.new_zeros(batch, num_heads, dim_k, dim_v)
    if initial_state is not None:
        state = state + initial_state
    out = torch.zeros_like(v32)
    for t in range(seq_len):
        state = state * g32[:, t].exp().unsqueeze(-1)
        # v - S^T k: what the state currently predicts for this key, corrected.
        predicted = (k32[:, t].unsqueeze(-1) * state).sum(dim=-2)
        update = torch.einsum(
            "bhk,bhv->bhkv", beta32[:, t].unsqueeze(-1) * k32[:, t], v32[:, t] - predicted
        )
        state = state + update
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
    return out.to(v.dtype), state


def kda_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked-parallel evaluation of the same recurrence (the prefill form).

    Same contract as `kda_recurrent` (pre-normalized q/k, fp32 state in/out).
    Within each chunk the per-token rank-1 updates are folded into one matrix
    correction via the WY representation of the paper's specialized DPLR
    transition; across chunks the state advances once per chunk. Sequences
    are zero-padded to a chunk multiple: padded positions carry `beta = 0`
    and `g = 0`, which provably leave the state untouched.

    Deviation from FLA's `naive_chunk_kda` (documented, math-equivalent): the
    unit-lower-triangular inverse `(I + B)^{-1}` is computed with
    `torch.linalg.solve_triangular` instead of an in-place forward
    substitution loop. Same matrix, different fp op order; the unit tests pin
    chunkwise == recurrent to fp32 tolerance.
    """
    batch, seq_len, num_heads, dim_k = q.shape
    dim_v = v.shape[-1]
    out_dtype = v.dtype

    pad = (-seq_len) % chunk_size
    if pad:
        q, k, v, g, beta = (
            functional.pad(t.float(), (0, 0, 0, 0, 0, pad))
            if t.dim() == 4
            else functional.pad(t.float(), (0, 0, 0, pad))
            for t in (q, k, v, g, beta)
        )
    total = seq_len + pad
    num_chunks = total // chunk_size

    # (batch, heads, chunk_idx, chunk_pos, feature)
    def to_chunks(t: torch.Tensor) -> torch.Tensor:
        return (
            t.float()
            .view(batch, num_chunks, chunk_size, num_heads, -1)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )

    qc, kc, vc, gc = to_chunks(q), to_chunks(k), to_chunks(v), to_chunks(g)
    betac = (
        beta.float().view(batch, num_chunks, chunk_size, num_heads).permute(0, 3, 1, 2).contiguous()
    )
    # In-chunk cumulative log-decay; `exp(g_cum[c] - g_cum[i])` is the decay
    # applied between positions i and c of the same chunk.
    g_cum = gc.cumsum(dim=-2)

    # B[c, i] = beta_c * <k_c * exp(g_cum_c - g_cum_i), k_i> for c > i:
    # the strictly-lower Gram matrix coupling each token's delta-rule update
    # to the earlier in-chunk keys it partially overwrites.
    gram = qc.new_zeros(batch, num_heads, num_chunks, chunk_size, chunk_size)
    for i in range(chunk_size):
        decayed_keys = kc * (g_cum - g_cum[..., i : i + 1, :]).exp()
        gram[..., i] = torch.einsum("bhnck,bhnk->bhnc", decayed_keys, kc[..., i, :])
    gram = gram * betac.unsqueeze(-1)
    strict_lower = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=qc.device), diagonal=-1
    )
    gram = gram.masked_fill(~strict_lower, 0.0)

    # T = (I + B)^{-1} diag(beta): folds the chunk's sequential rank-1
    # corrections into one linear map (WY form).
    identity = torch.eye(chunk_size, dtype=torch.float32, device=qc.device)
    transform = torch.linalg.solve_triangular(
        gram + identity, torch.diag_embed(betac), upper=False, unitriangular=True
    )
    w = transform @ (g_cum.exp() * kc)  # decay-adjusted keys entering the state
    u = transform @ vc  # values with in-chunk overwrites folded in

    state = qc.new_zeros(batch, num_heads, dim_k, dim_v)
    if initial_state is not None:
        state = state + initial_state.float()
    out = torch.zeros_like(vc)
    causal_incl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=qc.device))
    for n in range(num_chunks):
        q_n, k_n, g_n = qc[..., n, :, :], kc[..., n, :, :], g_cum[..., n, :, :]
        # Aqk[c, i] = <q_c * exp(g_cum_c - g_cum_i), k_i> for c >= i: the
        # in-chunk attention of each query over earlier (and own) positions.
        attn = qc.new_zeros(batch, num_heads, chunk_size, chunk_size)
        for i in range(chunk_size):
            decayed_queries = q_n * (g_n - g_n[..., i : i + 1, :]).exp()
            attn[..., i] = torch.einsum("bhck,bhk->bhc", decayed_queries, k_n[..., i, :])
        attn = attn.masked_fill(~causal_incl, 0.0)

        v_adjusted = u[..., n, :, :] - w[..., n, :, :] @ state
        out[..., n, :, :] = (q_n * g_n.exp()) @ state + attn @ v_adjusted
        # Advance the state one chunk: decay to the chunk end, then absorb
        # each position's correction decayed from its position to the end.
        end_decay = g_n[..., -1:, :]
        state = state * end_decay[..., 0, :].exp().unsqueeze(-1)
        state = state + ((end_decay - g_n).exp() * k_n).transpose(-1, -2) @ v_adjusted

    out_flat = out.permute(0, 2, 3, 1, 4).reshape(batch, total, num_heads, dim_v)[:, :seq_len]
    return out_flat.to(out_dtype), state
