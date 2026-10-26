"""Mamba-2 building blocks: short conv, SSD scan, gated group RMSNorm.

A Mamba-2 layer is a selective state-space model (SSM): each head keeps a
fixed-size `(head_dim, state_size)` state `h`, and for every token

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * x_t B_t^T
    y_t = h_t C_t + D * x_t

where `dt_t > 0` is a per-head step size, `A < 0` a per-head decay rate, and
`B_t`, `C_t` input-dependent vectors of length `state_size` shared by the
heads of a group. It is linear attention with a scalar decay per head and no
delta rule: `B` plays the key, `C` the query, `dt * x` the value.

Two equivalent evaluations, pinned against each other in
`tests/unit/models/nemotron_h/test_mamba2_block.py`:

  - `ssd_recurrent`, one token at a time. The decode step uses it.
  - `ssd_chunked`, the state-space-duality (SSD) form: within a chunk the
    output is a masked matrix product, and only one state per chunk is
    carried forward. Prefill uses it.

All SSM math runs in fp32, as the `mamba_ssm` kernels do. The short conv
computes in fp32 and returns the input dtype, like `causal-conv1d`.
"""

from __future__ import annotations

import torch
from torch.nn import functional


def mamba_conv_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv + bias + SiLU over a span, with optional left context.

    Args:
        x: `(batch, seq_len, channels)` raw pre-conv inputs.
        weight: `(channels, kernel)` depthwise taps, `weight[:, -1]` on the
            newest position.
        bias: `(channels,)` or None.
        conv_state: `(batch, channels, kernel)` the last `kernel` raw inputs
            (newest last), or None for a fresh sequence.

    Returns:
        `(y, new_conv_state)`: `y` is `(batch, seq_len, channels)` in
        `x.dtype`; the new state holds the last `kernel` raw inputs.
    """
    batch, _seq_len, channels = x.shape
    kernel = weight.shape[-1]
    x32 = x.float().transpose(1, 2)
    if conv_state is None:
        left = x32.new_zeros(batch, channels, kernel - 1)
        history = torch.cat([x32.new_zeros(batch, channels, kernel), x32], dim=-1)
    else:
        left = conv_state.float()[:, :, 1:]
        history = torch.cat([conv_state.float(), x32], dim=-1)
    y = functional.conv1d(
        torch.cat([left, x32], dim=-1),
        weight.float().unsqueeze(1),
        None if bias is None else bias.float(),
        groups=channels,
    )
    y = functional.silu(y)
    return y.transpose(1, 2).to(x.dtype), history[:, :, -kernel:].to(x.dtype)


def mamba_conv_step(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decode step of the conv: `x` is `(batch, channels)`, returns `(y, new_state)`."""
    new_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1).to(conv_state.dtype)], dim=-1)
    y = (new_state.float() * weight.float().unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        y = y + bias.float()
    return functional.silu(y).to(x.dtype), new_state


def ssd_recurrent(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SSM one token at a time.

    Args:
        x: `(batch, seq_len, heads, head_dim)` fp32 inputs (after the conv).
        dt: `(batch, seq_len, heads)` fp32 step sizes (after softplus).
        a: `(heads,)` fp32 decay rates, negative.
        b, c: `(batch, seq_len, heads, state_size)` fp32, already broadcast
            from groups to heads.
        initial_state: `(batch, heads, head_dim, state_size)` fp32 or None.

    Returns:
        `(y, final_state)`: `y` is `(batch, seq_len, heads, head_dim)` fp32
        without the `D * x` skip term.
    """
    batch, seq_len, heads, head_dim = x.shape
    state_size = b.shape[-1]
    if initial_state is None:
        state = x.new_zeros(batch, heads, head_dim, state_size)
    else:
        state = initial_state.float()
    out = x.new_empty(batch, seq_len, heads, head_dim)
    for t in range(seq_len):
        decay = torch.exp(dt[:, t] * a)
        update = (dt[:, t, :, None] * x[:, t]).unsqueeze(-1) * b[:, t].unsqueeze(-2)
        state = state * decay[:, :, None, None] + update
        out[:, t] = torch.einsum("bhpn,bhn->bhp", state, c[:, t])
    return out, state


def _segment_sum(log_decay: torch.Tensor) -> torch.Tensor:
    """`out[..., l, s] = sum(log_decay[..., s + 1 : l + 1])` for `l >= s`, else `-inf`.

    Built from a cumulative sum of masked copies rather than a difference of
    two cumulative sums, which loses precision when the sums grow large.
    """
    length = log_decay.size(-1)
    expanded = log_decay[..., None].expand(*log_decay.shape, length)
    strictly_lower = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=log_decay.device), diagonal=-1
    )
    summed = torch.cumsum(expanded.masked_fill(~strictly_lower, 0.0), dim=-2)
    lower = torch.tril(torch.ones(length, length, dtype=torch.bool, device=log_decay.device))
    return summed.masked_fill(~lower, float("-inf"))


def ssd_chunked(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same SSM in chunks (state-space duality). Arguments as `ssd_recurrent`.

    Within a chunk, `y_l = sum_{s <= l} exp(sum of log decays in (s, l]) *
    (C_l . B_s) * dt_s x_s`, a masked matrix product. Across chunks only the
    state at each chunk end is carried, and `y_l` adds `exp(decay since the
    chunk start) * h_start C_l`. Padding positions get `dt = 0`, so they
    neither decay nor add to the state.
    """
    batch, seq_len, heads, head_dim = x.shape
    state_size = b.shape[-1]
    pad = (-seq_len) % chunk_size
    if pad:
        x = functional.pad(x, (0, 0, 0, 0, 0, pad))
        dt = functional.pad(dt, (0, 0, 0, pad))
        b = functional.pad(b, (0, 0, 0, 0, 0, pad))
        c = functional.pad(c, (0, 0, 0, 0, 0, pad))
    n_chunks = (seq_len + pad) // chunk_size
    shape = (batch, n_chunks, chunk_size, heads)
    values = (x * dt[..., None]).reshape(*shape, head_dim)
    keys = b.reshape(*shape, state_size)
    queries = c.reshape(*shape, state_size)
    log_decay = (dt * a).reshape(shape).permute(0, 3, 1, 2)  # (batch, heads, chunks, L)
    log_decay_cum = torch.cumsum(log_decay, dim=-1)

    # Intra-chunk: masked "attention" of C against B, applied to dt * x.
    decay_matrix = torch.exp(_segment_sum(log_decay))  # (batch, heads, chunks, L, L)
    scores = torch.einsum("bclhn,bcshn->bhcls", queries, keys) * decay_matrix
    y_diag = torch.einsum("bhcls,bcshp->bclhp", scores, values)

    # Each chunk's own contribution to the state at its end.
    decay_to_end = torch.exp(log_decay_cum[..., -1:] - log_decay_cum)
    chunk_states = torch.einsum("bclhn,bhcl,bclhp->bchpn", keys, decay_to_end, values)

    # Carry the state across chunks.
    chunk_decay = torch.exp(log_decay_cum[..., -1])  # (batch, heads, chunks)
    if initial_state is None:
        state = x.new_zeros(batch, heads, head_dim, state_size)
    else:
        state = initial_state.float()
    starts = []
    for chunk in range(n_chunks):
        starts.append(state)
        state = state * chunk_decay[:, :, chunk, None, None] + chunk_states[:, chunk]
    start_states = torch.stack(starts, dim=1)  # (batch, chunks, heads, head_dim, state)

    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", queries, start_states, torch.exp(log_decay_cum))
    y = (y_diag + y_off).reshape(batch, n_chunks * chunk_size, heads, head_dim)
    return y[:, :seq_len], state


def gated_group_rmsnorm(
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    """Mamba-2 output norm: gate first, then RMS-normalize each group.

    `y * silu(gate)` in fp32, normalized over consecutive groups of
    `group_size` channels, then scaled by `weight`. This is the
    `norm_before_gate=False` form of the reference.

    Args:
        y: `(..., channels)` SSM output, fp32.
        gate: `(..., channels)` gate projection, any float dtype.
        weight: `(channels,)` affine weight.

    Returns:
        `(..., channels)` in `y.dtype` promoted with `weight.dtype`.
    """
    input_dtype = y.dtype
    gated = y.float() * functional.silu(gate.float())
    grouped = gated.view(*gated.shape[:-1], -1, group_size)
    grouped = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    return weight * grouped.view_as(gated).to(input_dtype)
