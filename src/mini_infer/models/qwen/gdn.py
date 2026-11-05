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
padding, and non-zero initial states. `tests/unit/models/qwen/test_gdn_block.py` pins it.

So this module deliberately holds ONLY what genuinely differs from KDA:

  - `gdn_gate`: the per-head scalar log-decay, from flat `(num_v_heads,)`
    parameters, where KDA's `kda_gate` reads per-channel ones.
  - `broadcast_gate`: the scalar-to-diagonal expansion that lets the KDA
    recurrence consume a GDN gate.
  - `gdn_chunkwise`: the chunked prefill form for a scalar gate. With one
    decay per head, the decay between two positions is one number, so each
    in-chunk interaction matrix is a plain matmul times a decay mask. The
    per-channel form in `kda_chunkwise` instead loops over the 64 positions
    of a chunk, which launches about 20k small kernels per layer for 4096
    tokens.
  - `gated_rmsnorm_silu`: the output norm, which gates with SiLU where KDA
    gates with sigmoid, and applies its affine weight in the INPUT dtype
    rather than in fp32.

Everything else (the one-token delta rule, the short causal convolution,
the L2 normalization) is imported from `kda.py` rather than reimplemented.

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


def gdn_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked-parallel gated delta rule for a per-head SCALAR gate (the prefill form).

    The same WY / UT chunk algorithm as `kda_chunkwise`, which the unit
    tests pin it against, but with the decay as one number per head and
    position. Inside a chunk the decay from position `i` to position `c` is
    `exp(G_c - G_i)` for every key channel, with `G` the in-chunk cumulative
    log-decay. So the key-key and query-key couplings are a matmul times a
    `(chunk, chunk)` decay mask, computed for all chunks at once. Only the
    state hand-off from one chunk to the next stays sequential, at about
    ten kernels per chunk.

    Sequences are zero-padded to a chunk multiple. Padded positions carry
    `beta = 0` and `g = 0`, which leave the state untouched.

    Args:
        q, k: `(batch, seq_len, num_heads, head_dim_k)`, pre-normalized (see
            the module docstring).
        v: `(batch, seq_len, num_heads, head_dim_v)`.
        g: `(batch, seq_len, num_heads)` fp32 log-decays, `<= 0`.
        beta: `(batch, seq_len, num_heads)` fp32 delta-rule step sizes.
        initial_state: `(batch, num_heads, head_dim_k, head_dim_v)` fp32, or
            None for a zero state.
        chunk_size: positions per chunk.

    Returns:
        `(o, final_state)`: `o` is `(batch, seq_len, num_heads, head_dim_v)`
        in `v.dtype`; `final_state` is fp32.
    """
    batch, seq_len, num_heads, dim_k = q.shape
    dim_v = v.shape[-1]
    pad = (-seq_len) % chunk_size
    total = seq_len + pad
    num_chunks = total // chunk_size

    def per_head(t: torch.Tensor) -> torch.Tensor:
        """`(batch, seq_len, heads, dim)` -> `(batch, heads, chunks, chunk, dim)` fp32."""
        padded = functional.pad(t.float(), (0, 0, 0, 0, 0, pad))
        chunked = padded.view(batch, num_chunks, chunk_size, num_heads, -1)
        return chunked.permute(0, 3, 1, 2, 4).contiguous()

    def per_head_scalar(t: torch.Tensor) -> torch.Tensor:
        """`(batch, seq_len, heads)` -> `(batch, heads, chunks, chunk)` fp32."""
        padded = functional.pad(t.float(), (0, 0, 0, pad))
        return padded.view(batch, num_chunks, chunk_size, num_heads).permute(0, 3, 1, 2)

    qc, kc, vc = per_head(q), per_head(k), per_head(v)
    beta_c = per_head_scalar(beta)
    g_cum = per_head_scalar(g).cumsum(dim=-1)

    # decay[c, i] = exp(G_c - G_i) for c >= i, else 0. The upper triangle is
    # masked before the exp, where its positive differences could overflow.
    device = qc.device
    causal = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device))
    decay = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).masked_fill(~causal, float("-inf")).exp()

    # B[c, i] = beta_c * exp(G_c - G_i) * <k_c, k_i> for c > i, and
    # T = (I + B)^{-1} diag(beta), as in `kda_chunkwise`.
    gram = (kc @ kc.transpose(-1, -2)) * decay * beta_c.unsqueeze(-1)
    gram = gram.tril(diagonal=-1)
    identity = torch.eye(chunk_size, dtype=torch.float32, device=device)
    transform = torch.linalg.solve_triangular(
        gram + identity, torch.diag_embed(beta_c), upper=False, unitriangular=True
    )
    w = transform @ (g_cum.exp().unsqueeze(-1) * kc)  # decay-adjusted keys entering the state
    u = transform @ vc  # values with in-chunk overwrites folded in

    # Everything the chunk loop reads that does not depend on the state.
    attn = (qc @ kc.transpose(-1, -2)) * decay  # in-chunk attention, c >= i
    q_decayed = qc * g_cum.exp().unsqueeze(-1)
    end = g_cum[..., -1:]
    k_to_end = kc * (end - g_cum).exp().unsqueeze(-1)  # each key decayed to the chunk end
    end_decay = end.exp().unsqueeze(-1)  # (batch, heads, chunks, 1, 1)

    state = qc.new_zeros(batch, num_heads, dim_k, dim_v)
    if initial_state is not None:
        state = state + initial_state.float()
    out = torch.empty_like(vc)
    for n in range(num_chunks):
        v_adjusted = u[:, :, n] - w[:, :, n] @ state
        out[:, :, n] = q_decayed[:, :, n] @ state + attn[:, :, n] @ v_adjusted
        state = state * end_decay[:, :, n] + k_to_end[:, :, n].transpose(-1, -2) @ v_adjusted

    out_flat = out.permute(0, 2, 3, 1, 4).reshape(batch, total, num_heads, dim_v)[:, :seq_len]
    return out_flat.to(v.dtype), state


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
