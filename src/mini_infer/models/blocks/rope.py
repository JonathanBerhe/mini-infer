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

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
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
