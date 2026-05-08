"""Lightning Indexer (V4 paper §2.3.2, formulas 13-17) — top-k selector for CSA.

CSA's compressor produces one entry per `m=4` tokens, but a 1M-context
prefill still has ~250k compressed entries. Attending to all of them
defeats the cost win. The indexer scores `(query, compressed_entry)`
pairs cheaply and keeps the top-k per query.

Math (one query at token `t`, scoring against all compressed entries
`c_j` from this layer):
    q_t        = W^{IQ} · h_t                # (n_h_idx, c_idx)  per-head queries
    K^I        = LightningCompressor(h)       # (n_compressed, c_idx)  shared across heads
    score_h_j  = ReLU(q_t,h · K^I_j)          # per-head, per-(query, key) raw score
    w_t,h      = (W^W · h_t)_h                # per-head, per-token weighting scalar
    score_t,j  = sum_h(score_h_j · w_t,h)     # collapse heads
    topk(t)    = argmax_k(score_t, k=top_k)   # indices into compressed entries

The ReLU before the head-sum is the key trick: it suppresses negative
contributions so the indexer behaves as a "vote" across heads — a
compressed entry survives only if at least one head likes it. Without
ReLU, a strongly-negative score on one head could cancel a positive on
another.

The reference (`Indexer` in `model.py`) wraps Q in a Hadamard rotation
and FP4 quantization for compute precision. We keep the math at fp32
for parity testing; the rotation cancels in `q · k^T` because Hadamard
is unitary, so dropping it leaves the dot products identical.

Compressor for the indexer is its OWN module (separate weights), uses
`overlap_mode=True` because CSA's compression ratio is `m=4`. Its
output dim `c_idx` is independent of and typically smaller than the
main attention's `c` (paper: 128 vs 512).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import relu

from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims
from mini_infer.models.blocks.v4.compressor import TokenLevelCompressor


class LightningIndexer(nn.Module):
    """Top-k compressed-entry selector for CSA.

    Args:
        hidden_size: Input feature dim of the hidden states.
        q_lora_rank: Rank of the shared low-rank Q latent (the indexer's `wq_b`
            up-projects from this latent, not from raw `hidden_states`).
        num_heads: Number of indexer heads. Independent of the main attention's
            head count; paper uses `n_h_idx = 64` regardless of `n_h`.
        head_dim: Per-head feature dim for the indexer Q and its compressed K.
            Independent of the main attention's `kv_head_dim`.
        rope_head_dim: Trailing dims of `head_dim` to rotate via partial RoPE.
        compression_ratio: Indexer's compression ratio (`m=4` in V4).
        top_k: How many compressed entries to keep per query.
        rms_norm_eps: Forwarded to the indexer's compressor norm.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        rope_head_dim: int,
        compression_ratio: int,
        top_k: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if rope_head_dim < 0 or rope_head_dim > head_dim:
            raise ValueError(f"rope_head_dim={rope_head_dim} must be in [0, head_dim={head_dim}]")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compression_ratio = compression_ratio
        self.top_k = top_k

        # Q up-projection from the SHARED q_lora latent (the main attention's
        # `q_a_layernorm` output). Reusing the latent saves one large matmul
        # vs projecting from raw `hidden_states`.
        self.wq_b = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        # Per-token, per-head weighting scalar: `(B, T, hidden_size) -> (B, T, n_h_idx)`.
        # Used to weight the per-head ReLU'd dot products before summing across heads.
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # Indexer's own compressor — separate weights from the main attention's compressor.
        # Always overlap-mode (m=4 in V4). Uses `head_dim` as the compressed-entry width.
        self.compressor = TokenLevelCompressor(
            hidden_size=hidden_size,
            kv_head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compression_ratio=compression_ratio,
            rms_norm_eps=rms_norm_eps,
            overlap_mode=True,
        )
        # Score scaling: 1/sqrt(d) AND a 1/sqrt(n_h_idx) factor pre-applied to the weights.
        # Reference: `weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)`.
        self._weight_scale = (head_dim**-0.5) * (num_heads**-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora_latent: torch.Tensor,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_offset: int,
    ) -> torch.Tensor:
        """Pick top-k compressed indices per query.

        Args:
            hidden_states: `(B, T, hidden_size)`.
            q_lora_latent: `(B, T, q_lora_rank)`. The output of the main
                attention's `q_a_layernorm(q_a_proj(hidden_states))` — passed
                in by the parent block to avoid recomputing the latent.
            token_position_embeddings: `(cos, sin)` for raw token positions
                `[0, T)`; each `(B, T, rope_head_dim)`. Used to rotate Q.
            compressed_position_embeddings: `(cos, sin)` for compressed
                positions `[0, m, 2m, ...]`; each `(B, T // m, rope_head_dim)`.
                Forwarded to the indexer's compressor.
            compressed_offset: Offset into the parent attention's concatenated
                KV tensor where its compressed branch begins. Returned indices
                are absolute (`local_idx + offset`) so the parent can pass
                them straight into `hca_mqa_with_sink`.

        Returns:
            `(B, T, top_k)` int64 indices. Entries that map to a causally
            invalid compressed block are returned as `-1` (the dispatcher
            masks them to `-inf` in the softmax).
        """
        bsz, seqlen, _ = hidden_states.shape
        n_h = self.num_heads
        d = self.head_dim
        m = self.compression_ratio

        # ---- Q: shared latent -> per-head + partial RoPE ----
        q = self.wq_b(q_lora_latent).view(bsz, seqlen, n_h, d)
        cos_t, sin_t = token_position_embeddings
        if self.rope_head_dim > 0:
            q = apply_partial_rope_last_n_dims(q, cos_t, sin_t, self.rope_head_dim)

        # ---- K: indexer's own compressed KV (shared across heads) ----
        compressed_k = self.compressor(hidden_states, compressed_position_embeddings)
        # compressed_k: (B, n_compressed, head_dim)
        n_compressed = compressed_k.shape[1]

        # ---- Per-head per-token weighting scalars ----
        weights = self.weights_proj(hidden_states) * self._weight_scale  # (B, T, n_h)

        # ---- Score: per-head dot-product, ReLU'd, weighted-summed across heads ----
        # einsum: (B, T, n_h, d) x (B, n_compressed, d) -> (B, T, n_h, n_compressed)
        per_head_score = torch.einsum("bthd,bkd->bthk", q.float(), compressed_k.float())
        per_head_score = relu(per_head_score)
        # Weighted sum across heads: (B, T, n_h, n_compressed) * (B, T, n_h, 1) summed over n_h.
        index_score = (per_head_score * weights.unsqueeze(-1)).sum(dim=2)
        # index_score: (B, T, n_compressed)

        # ---- Causal mask: query at position t can only see compressed blocks j < (t+1)/m ----
        cutoff = (torch.arange(1, seqlen + 1, device=hidden_states.device) // m).unsqueeze(
            1
        )  # (T, 1)
        block_idxs = torch.arange(n_compressed, device=hidden_states.device).unsqueeze(
            0
        )  # (1, n_compressed)
        invalid_block = block_idxs >= cutoff  # (T, n_compressed) — True for future blocks
        index_score = index_score.masked_fill(invalid_block.unsqueeze(0), float("-inf"))

        # ---- Top-k selection ----
        actual_k = min(self.top_k, n_compressed)
        topk_idxs = index_score.topk(actual_k, dim=-1)[1]  # (B, T, actual_k)

        # ---- Re-mask: when fewer than `actual_k` valid compressed entries exist
        # for a given query, top-k still returns indices into the masked region.
        # Replace those with -1 (the dispatcher reads -1 as padding).
        topk_invalid = topk_idxs >= cutoff  # (B, T, actual_k) via broadcasting
        topk_idxs = torch.where(topk_invalid, -1, topk_idxs + compressed_offset)
        return topk_idxs.to(torch.int64)
