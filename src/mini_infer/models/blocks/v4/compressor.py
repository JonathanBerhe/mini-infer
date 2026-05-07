"""Token-level KV compressor (V4 paper §2.3.2, formulas 20-23).

Compresses every `m` consecutive tokens into one KV entry via a learned
softmax-weighted sum over the block. Used twice in V4:

  - HCA layers: heavy compression `m'=128`. Only this branch.
  - CSA layers: light compression `m=4` plus a separate Lightning Indexer
    that picks the top-k compressed entries per query.

Math (one block of `m` tokens at a time):
    KV   = x · W^{KV}        # (m, c)  -- per-token KV entries
    Z    = x · W^{Z}          # (m, c)  -- per-token compression weights
    S    = softmax_row(Z + B) # (m, c)  -- B is a learnable per-block-position bias
    C    = sum(S ⊙ KV, dim=0) # (c,)    -- one compressed entry
    C    = RMSNorm(C)
    C    = partial_rope(C, position=block_idx * m)  # rotate last `rope_head_dim` dims

The reference (`deepseek_v4_reference/model.py::Compressor`) carries
extra plumbing for decode-phase incremental compression (state buffers,
overlap mode for `m=4`). Our standalone block forward handles prefill
only and assumes `seqlen` is a multiple of `m` — the parity test enforces
this. CSA's overlap=True case is left for Stage C4b.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims


class TokenLevelCompressor(nn.Module):
    """Compress every `m` tokens into one KV entry via softmax-weighted sum."""

    def __init__(
        self,
        *,
        hidden_size: int,
        kv_head_dim: int,
        rope_head_dim: int,
        compression_ratio: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if compression_ratio <= 0:
            raise ValueError(f"compression_ratio must be positive, got {compression_ratio}")
        if rope_head_dim < 0 or rope_head_dim > kv_head_dim:
            raise ValueError(
                f"rope_head_dim={rope_head_dim} must be in [0, kv_head_dim={kv_head_dim}]"
            )
        if rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim must be even, got {rope_head_dim}")
        self.hidden_size = hidden_size
        self.kv_head_dim = kv_head_dim
        self.rope_head_dim = rope_head_dim
        self.compression_ratio = compression_ratio
        # `wkv` in the reference: x -> KV entries.
        self.kv_proj = nn.Linear(hidden_size, kv_head_dim, bias=False)
        # `wgate` in the reference: x -> compression-weight logits (pre-softmax).
        self.weight_proj = nn.Linear(hidden_size, kv_head_dim, bias=False)
        # `ape` in the reference: per-block-position learnable bias added to logits before softmax.
        # Shape `(m, kv_head_dim)`: each of the `m` positions inside a block gets its own
        # bias vector, broadcast across all blocks. Distinguishes "first token of block"
        # from "last token of block" so the softmax can learn position-dependent weighting.
        self.position_bias = nn.Parameter(torch.zeros(compression_ratio, kv_head_dim))
        # RMSNorm applied to the compressed entry before partial RoPE.
        self.norm = RMSNorm(kv_head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compress `(B, T, hidden_size)` -> `(B, T // m, kv_head_dim)`.

        `compressed_position_embeddings` is `(cos, sin)` for the
        `T // m` compressed positions (block index `i` -> token position
        `i * m`). The caller pre-computes these from the model's RoPE
        table; the compressor only applies them.
        """
        _, seqlen, _ = hidden_states.shape
        m = self.compression_ratio
        if seqlen % m != 0:
            raise ValueError(f"seqlen={seqlen} must be a multiple of compression_ratio={m}")

        # Stay in fp32 for the softmax math — the reference does this and
        # the bf16 path is meaningfully lossy on small absolute values.
        x = hidden_states.float()
        kv = self.kv_proj(x)  # (B, T, c)
        score = self.weight_proj(x)  # (B, T, c)

        # Reshape to (B, n_blocks, m, c). Add the per-block-position bias
        # `B[k, j]`: the bias at the k-th block-internal position, j-th
        # feature dim, broadcast across the batch and across blocks.
        kv = kv.unflatten(1, (-1, m))  # (B, n_blocks, m, c)
        score = score.unflatten(1, (-1, m)) + self.position_bias

        # Softmax along the m-axis — each compressed entry is a convex
        # combination of the `m` tokens that fed into it.
        weights = score.softmax(dim=2)
        compressed: torch.Tensor = (kv * weights).sum(dim=2)  # (B, n_blocks, c)

        # RMSNorm in the model's working dtype.
        compressed = self.norm(compressed.to(hidden_states.dtype))

        # Partial RoPE on the last `rope_head_dim` dims of the compressed entry.
        # Position for block i is `i * m` — handled by the caller producing the
        # right cos/sin tables.
        cos, sin = compressed_position_embeddings
        if self.rope_head_dim > 0:
            compressed = apply_partial_rope_last_n_dims(compressed, cos, sin, self.rope_head_dim)
        return compressed
