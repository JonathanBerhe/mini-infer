"""Rotary Position Embeddings shared by Llama-shape models.

Both the cos/sin table generation (`RotaryEmbedding`) and the rotation
helper (`apply_rotary_pos_emb`) match HF's Llama/Qwen2 implementation
exactly. Mirroring the math is important because per-token logits must
match across the boundary between owned-model-code and the HF reference
parity tests.
"""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Computes `(cos, sin)` tables for RoPE on demand.

    Holds only the inverse-frequency buffer (no parameters). Each forward
    builds cos/sin for the given `position_ids` so we never cache against
    HF's static-cache assumption.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError(
                f"partial_rotary_factor must be in (0, 1]; got {partial_rotary_factor}"
            )
        # Standard RoPE: rotate every dim. Partial RoPE (Gemma 4 global
        # layers): only the first `head_dim * partial_rotary_factor` dims
        # rotate; the remaining dims keep zero inv_freq so cos=1, sin=0
        # and the rotation is a no-op for them. Same math as HF's
        # `_compute_proportional_rope_parameters` with `factor=1.0`.
        rope_angles = int(partial_rotary_factor * head_dim) // 2
        rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
        )
        nope_pad = head_dim // 2 - rope_angles
        if nope_pad > 0:
            inv_freq = torch.cat([rotated, torch.zeros(nope_pad, dtype=torch.float32)])
        else:
            inv_freq = rotated
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (1, total_q) absolute positions.
        # Output cos/sin: (1, total_q, head_dim) ready for apply_rotary_pos_emb.
        device = hidden_states.device
        inv_freq = self.inv_freq.to(device=device)
        # (1, head_dim/2, 1) @ (1, 1, total_q) -> (1, head_dim/2, total_q) -> transpose.
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_float = position_ids[:, None, :].to(torch.float32)
        # autocast off: rope tables must be fp32 to avoid drift on long contexts.
        with torch.autocast(device_type=device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_float).transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(dtype=hidden_states.dtype)
            sin = emb.sin().to(dtype=hidden_states.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-rotation used by Llama-shape RoPE: `(-x_high, x_low)` along last dim."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. Same math as HF's `apply_rotary_pos_emb`."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_interleaved_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE with interleaved (real, imag) pairing — DeepSeek-V2 / V3 / Kimi-K2.

    Matches HF's `apply_rotary_emb` (complex-number form) bit-for-bit but
    expressed in real arithmetic. For each consecutive pair `(q[2i],
    q[2i+1])` at position `p` and frequency `f_i`:
        q[2i]_new   = q[2i]   * cos(p*f_i) - q[2i+1] * sin(p*f_i)
        q[2i+1]_new = q[2i]   * sin(p*f_i) + q[2i+1] * cos(p*f_i)

    Note `cos` / `sin` here have last dim `head_dim` (matching our
    `RotaryEmbedding` output, which produces `cat([freqs, freqs])`); we
    take the first half since the second half mirrors it. The pairing
    convention differs from `apply_rotary_pos_emb`'s "stacked halves"
    layout — DeepSeek's checkpoint stores rope-dim weights interleaved.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Take only the first half — second half is identical (cat-of-self).
    rope_dim_half = q.shape[-1] // 2
    cos_half = cos[..., :rope_dim_half]
    sin_half = sin[..., :rope_dim_half]

    def _rotate(t: torch.Tensor) -> torch.Tensor:
        # Split into even / odd indices along the last dim.
        t_even = t[..., 0::2]
        t_odd = t[..., 1::2]
        rot_even = t_even * cos_half - t_odd * sin_half
        rot_odd = t_even * sin_half + t_odd * cos_half
        # Interleave back: stack along a new dim, then flatten.
        return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)

    return _rotate(q), _rotate(k)


def apply_partial_rope_last_n_dims(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    last_n_rotated: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate only the LAST `last_n_rotated` dims of `x` via interleaved RoPE.

    DeepSeek-V4 splits each KV head into `[nope_dims | rope_dims]` along the
    last axis; only the rope tail rotates. The reference V4 code expresses
    this as `apply_rotary_emb(x[..., -rd:], freqs_cis)` (in-place mutation
    of a slice). We return a new tensor for safety.

    Difference vs V2's `apply_interleaved_rotary_pos_emb`: V2 rotates the
    whole input (which is already the rope-only slice); V4 takes the full
    head and selectively rotates the tail. Centralizing the slice-and-rotate
    keeps callers from re-implementing the split.

    `inverse=True` rotates by the conjugate frequency. The reference uses
    this on the attention output to recover relative position encoding —
    `q · k^T` carries the position phase from both Q and K rotations, but
    after attention the output keeps Q's phase; rotating output by `-i`
    cancels it so subsequent layers see relative positions only.

    Shapes:
        x:   `(..., kv_head_dim)` where `kv_head_dim >= last_n_rotated`.
        cos, sin: `(B, T, last_n_rotated)` (full-dim form, halves
                  duplicated — matches our `RotaryEmbedding` output).
        Returns: `x` with the last `last_n_rotated` dims rotated.
    """
    if last_n_rotated <= 0 or last_n_rotated > x.shape[-1]:
        raise ValueError(
            f"last_n_rotated={last_n_rotated} out of range for x.shape[-1]={x.shape[-1]}"
        )
    if last_n_rotated % 2 != 0:
        raise ValueError(f"last_n_rotated must be even (interleaved RoPE), got {last_n_rotated}")

    rope_part = x[..., -last_n_rotated:]
    rest_part = x[..., :-last_n_rotated]

    # Broadcast cos/sin over leading dims of rope_part.
    while cos.ndim < rope_part.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    # Take half — `RotaryEmbedding` emits `cat([freqs, freqs])` so the
    # second half mirrors the first. Interleaved pairing only needs one half.
    half = last_n_rotated // 2
    cos_half = cos[..., :half]
    sin_half = sin[..., :half] if not inverse else -sin[..., :half]

    real = rope_part[..., 0::2]
    imag = rope_part[..., 1::2]
    rot_real = real * cos_half - imag * sin_half
    rot_imag = real * sin_half + imag * cos_half
    rotated = torch.stack([rot_real, rot_imag], dim=-1).flatten(-2)

    if rest_part.shape[-1] == 0:
        return rotated
    return torch.cat([rest_part, rotated], dim=-1)
