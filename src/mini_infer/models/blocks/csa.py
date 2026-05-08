"""Compressed Sparse Attention (CSA) — DeepSeek-V4 §2.3.

CSA is HCA's three-branch design (compressor + sliding window + sink)
plus a Lightning Indexer that decides *which* compressed entries each
query gets to see — top-k sparse selection instead of HCA's
"all compressed entries are visible." This is what gives V4-Pro its
1M-context cost win: at heavy compression you can attend to everything
cheaply (HCA `m'=128`); at light compression you have ~30x more
compressed entries (CSA `m=4`), so you have to pick.

Architecture vs HCA:
    - Compressor uses `m=4` and `overlap_mode=True` (each compressed
      block also softmax-pools the previous block's tokens).
    - Lightning Indexer picks the top-k compressed entries per query.
      Its own compressor + per-head Q + per-head weighting; reuses the
      main attention's `q_lora` latent (no extra `wq_a`).
    - The same `hca_mqa_with_sink` dispatcher handles the actual core
      attention — only the topk_idxs construction differs.

Forward shape walk (single block, B, T multiple of `m`):
    H (B, T, d)
    Q latent: H -> q_a_proj -> q_a_layernorm -> q_lora_latent (B, T, q_lora_rank)
    Q full:   q_lora_latent -> q_b_proj -> per-head q-norm -> partial RoPE
    SWA K=V (single shared head): H -> swa_kv_proj -> kv_norm -> partial RoPE
    Compressed K=V: TokenLevelCompressor(H) -> (B, T/m, c)  [overlap=True]
    Concatenate: full_kv = cat([swa_kv, compressed_kv], dim=1)
    Window topk_idxs: causal sliding window of size n_win
    Indexer topk_idxs: LightningIndexer(H, q_lora_latent) -> top_k compressed entries
    topk_idxs = cat([window_idxs, indexer_topk_idxs], dim=-1)
    hca_mqa_with_sink(Q, full_kv, sink, topk_idxs) -> (B, T, n_h, c)
    Output partial RoPE inverse
    GroupedOutputProjection -> (B, T, d)

Stage C4b: standalone block, parity-validated. Cache wiring + the
hybrid CSA/HCA backbone come later.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.cache.hca_attention import hca_mqa_with_sink
from mini_infer.models.blocks.hca import _build_window_topk_idxs
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims
from mini_infer.models.blocks.v4 import (
    AttentionSink,
    GroupedOutputProjection,
    LightningIndexer,
    TokenLevelCompressor,
)


class CSAAttention(nn.Module):
    """Compressed Sparse Attention block (V4 §2.3, with Lightning Indexer)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_head_dim: int,
        rope_head_dim: int,
        num_groups: int,
        o_lora_rank: int,
        window_size: int,
        compression_ratio: int,
        index_num_heads: int,
        index_head_dim: int,
        index_top_k: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if rope_head_dim < 0 or rope_head_dim > kv_head_dim:
            raise ValueError(
                f"rope_head_dim={rope_head_dim} must be in [0, kv_head_dim={kv_head_dim}]"
            )
        if rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim must be even, got {rope_head_dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_head_dim = kv_head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.compression_ratio = compression_ratio
        self.rms_norm_eps = rms_norm_eps

        # --- Q low-rank path (shared latent reused by the indexer below) ---
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * kv_head_dim, bias=False)

        # --- SWA K=V (single shared head) ---
        self.swa_kv_proj = nn.Linear(hidden_size, kv_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_head_dim, eps=rms_norm_eps)

        # --- Main compressor (overlap mode, m=4 in V4) ---
        self.compressor = TokenLevelCompressor(
            hidden_size=hidden_size,
            kv_head_dim=kv_head_dim,
            rope_head_dim=rope_head_dim,
            compression_ratio=compression_ratio,
            rms_norm_eps=rms_norm_eps,
            overlap_mode=True,
        )

        # --- Lightning Indexer ---
        self.indexer = LightningIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            num_heads=index_num_heads,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compression_ratio=compression_ratio,
            top_k=index_top_k,
            rms_norm_eps=rms_norm_eps,
        )

        # --- Per-head sink + grouped output projection ---
        self.sink = AttentionSink(num_heads=num_heads)
        self.grouped_output = GroupedOutputProjection(
            num_heads=num_heads,
            kv_head_dim=kv_head_dim,
            num_groups=num_groups,
            o_lora_rank=o_lora_rank,
            hidden_size=hidden_size,
        )

        self.softmax_scale = kv_head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run one CSA block on a packed prefill batch.

        Same `(token, compressed)` position-embedding contract as
        `HCAAttention`. The indexer reuses the token-level table for
        its Q rotation and the compressed table for its own compressor.
        """
        bsz, seqlen, _ = hidden_states.shape
        n_h = self.num_heads
        c = self.kv_head_dim
        rd = self.rope_head_dim

        # ---- Q low-rank latent (shared with the indexer's `wq_b`) ----
        q_lora_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))

        # ---- Full Q + per-head q-norm (no scale) + partial RoPE ----
        q = self.q_b_proj(q_lora_latent).view(bsz, seqlen, n_h, c)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(
            q.dtype
        )
        cos_t, sin_t = token_position_embeddings
        if rd > 0:
            q = apply_partial_rope_last_n_dims(q, cos_t, sin_t, rd)

        # ---- SWA K=V (single shared head) ----
        swa_kv = self.kv_norm(self.swa_kv_proj(hidden_states))
        if rd > 0:
            swa_kv = apply_partial_rope_last_n_dims(swa_kv, cos_t, sin_t, rd)

        # ---- Main compressed K=V ----
        compressed_kv = self.compressor(hidden_states, compressed_position_embeddings)
        # compressed_kv: (B, T // m, c)
        full_kv = torch.cat([swa_kv, compressed_kv], dim=1)  # (B, T + T/m, c)

        # ---- Window topk indices ----
        win_topk = (
            _build_window_topk_idxs(
                seqlen=seqlen, window_size=self.window_size, device=hidden_states.device
            )
            .unsqueeze(0)
            .expand(bsz, -1, -1)
        )

        # ---- Indexer topk indices over the compressed branch ----
        indexer_topk = self.indexer(
            hidden_states=hidden_states,
            q_lora_latent=q_lora_latent,
            token_position_embeddings=token_position_embeddings,
            compressed_position_embeddings=compressed_position_embeddings,
            compressed_offset=seqlen,  # uncompressed kv occupies [0, seqlen) in `full_kv`
        )

        # ---- Concat: window + indexer top-k ----
        topk_idxs = torch.cat([win_topk, indexer_topk], dim=-1).contiguous()

        # ---- MQA with sink ----
        attn_out = hca_mqa_with_sink(
            q=q,
            kv=full_kv,
            sink_logits=self.sink.sink_logits,
            topk_idxs=topk_idxs,
            softmax_scale=self.softmax_scale,
        )

        # ---- Output partial RoPE inverse ----
        if rd > 0:
            attn_out = apply_partial_rope_last_n_dims(attn_out, cos_t, sin_t, rd, inverse=True)

        # ---- Grouped output projection ----
        out: torch.Tensor = self.grouped_output(attn_out)
        return out
