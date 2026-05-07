"""Heavily Compressed Attention (HCA) — DeepSeek-V4 §2.3.

HCA is one of two attention modes V4 alternates between (the other is
CSA — Compressed Sparse Attention, which adds a Lightning Indexer +
top-k selector). HCA itself has three KV branches that all feed a
single Shared-KV MQA:

  1. **Compressed**: every `m'` consecutive tokens are squashed into
     one KV entry by `TokenLevelCompressor`. At V4-Pro scale `m'=128`,
     that's a 128x sequence-length reduction for this branch.
  2. **Sliding window**: the last `n_win` raw KV entries (no
     compression). Provides fine-grained local context.
  3. **Attention sink**: one learnable scalar logit per head, added
     to the softmax denominator. Stabilizes streaming generation.

Per-query attention reads the union: window slots (causal, last-`n_win`)
plus all compressed entries that fully predate the query.

The reference inference code (`deepseek_v4_reference/model.py::Attention`)
is one unified module with a `compress_ratio` flag — `0` = pure SWA,
`128` = HCA, `4` = CSA. We split HCA into its own block here because
a) CSA needs additional state (the Indexer), b) the per-stream KV
shapes differ (compressed vs raw), c) it's easier to read this way.
The shared primitives (`TokenLevelCompressor`, `AttentionSink`,
`GroupedOutputProjection`) live under `models/blocks/v4/` and will be
imported again by CSA in Stage C4b.

Forward shape walk (single block, batch=B, len=T, multiple of `m`):
    H (B, T, d)
    Q low-rank: H -> q_a_proj -> q_a_layernorm -> q_b_proj
                  -> (B, T, n_h * c) -> (B, T, n_h, c)
    Q-norm (no-scale rsqrt) per head
    Partial RoPE on Q's last `rope_head_dim` dims
    SWA K=V (single shared head): H -> swa_kv_proj -> kv_norm
                                     -> partial RoPE on last rope_head_dim dims
    Compressed K=V: TokenLevelCompressor(H) -> (B, T/m, c)
    Concatenate: full_kv = cat([swa_kv (B, T, c), compressed (B, T/m, c)], dim=1)
    Build topk_idxs per query: causal window slots + causal compressed slots
    hca_mqa_with_sink(Q, full_kv, sink_logits, topk_idxs) -> (B, T, n_h, c)
    Output partial RoPE INVERSE (recovers relative position)
    GroupedOutputProjection -> (B, T, d)

This stage (C4a) is the standalone forward — no PagedKVCache integration.
The cache wiring (per-stream paged compressed branch + per-request state
cache for SWA) lands in C4c when we build the hybrid backbone.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.cache.hca_attention import hca_mqa_with_sink
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims
from mini_infer.models.blocks.v4 import (
    AttentionSink,
    GroupedOutputProjection,
    TokenLevelCompressor,
)


def _build_hca_topk_idxs(
    *,
    seqlen: int,
    window_size: int,
    compression_ratio: int,
    n_compressed: int,
    compressed_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Build per-query gather indices for one HCA prefill.

    Returns `(seqlen, window_slots + n_compressed)` int64 indices into
    the concatenated `[uncompressed_kv ; compressed_kv]` tensor. `-1`
    marks padding (causally-future or out-of-window slots that the
    `hca_mqa_with_sink` dispatcher masks to `-inf` in the softmax).

    Layout matches the reference exactly so any sparsity-correctness
    bug shows up at the same query position with the same index.

    Window section (mirror of `get_window_topk_idxs`):
        For query at position `i`, attend to KV positions
        `[max(i - n_win + 1, 0), i]`. Pad with `-1` if fewer than `n_win`.

    Compressed section (mirror of `get_compress_topk_idxs`):
        For query at position `i`, attend to compressed block `j`
        only when `j < (i + 1) / m` (i.e. block `j` covers tokens
        `[j*m, (j+1)*m - 1]`, all of which must precede `i`). The
        absolute index into the concatenated tensor is `j + offset`
        where `offset = seqlen` (compressed entries follow the
        uncompressed window in the concat layout).
    """
    win_slots = min(seqlen, window_size)
    base = torch.arange(seqlen, device=device).unsqueeze(1)  # (seqlen, 1)
    # Window: indices `start_q .. start_q + win_slots - 1` where `start_q = max(i - win + 1, 0)`.
    win_idxs = (base - window_size + 1).clamp(min=0) + torch.arange(win_slots, device=device)
    win_idxs = torch.where(win_idxs > base, -1, win_idxs)  # (seqlen, win_slots)

    # Compressed: indices `j` for blocks `j < (i+1)/m`; otherwise `-1`.
    cmp_idxs = torch.arange(n_compressed, device=device).repeat(seqlen, 1)  # (seqlen, n_cmp)
    cutoff = (torch.arange(1, seqlen + 1, device=device) // compression_ratio).unsqueeze(1)
    cmp_idxs = torch.where(cmp_idxs >= cutoff, -1, cmp_idxs + compressed_offset)

    return torch.cat([win_idxs, cmp_idxs], dim=-1).to(torch.int64)


class HCAAttention(nn.Module):
    """Heavily Compressed Attention block (V4 §2.3, no Lightning Indexer)."""

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

        # --- Q low-rank path: H -> wq_a -> q_norm -> wq_b -> per-head q-norm (no scale).
        # The first norm has a learnable weight (RMSNorm on q_lora_rank); the second is a
        # plain rsqrt-of-mean-square per-head (no parameter) applied AFTER `wq_b`.
        # The reference does it this way — both norms matter for parity.
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * kv_head_dim, bias=False)

        # --- SWA K=V branch: H -> swa_kv_proj (single shared head) -> kv_norm -> partial RoPE.
        self.swa_kv_proj = nn.Linear(hidden_size, kv_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_head_dim, eps=rms_norm_eps)

        # --- Compressed K=V branch: H -> TokenLevelCompressor -> (B, T/m, c).
        self.compressor = TokenLevelCompressor(
            hidden_size=hidden_size,
            kv_head_dim=kv_head_dim,
            rope_head_dim=rope_head_dim,
            compression_ratio=compression_ratio,
            rms_norm_eps=rms_norm_eps,
        )

        # --- Per-head attention sink.
        self.sink = AttentionSink(num_heads=num_heads)

        # --- Grouped output projection.
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
        """Run one HCA block on a packed prefill batch.

        Args:
            hidden_states: `(B, T, hidden_size)` with `T` a multiple of
                `compression_ratio`.
            token_position_embeddings: `(cos, sin)` for the `T` raw token
                positions; each `(B, T, rope_head_dim)`.
            compressed_position_embeddings: `(cos, sin)` for the `T // m`
                compressed positions (block `i` -> token position `i*m`).
                Each `(B, T // m, rope_head_dim)`.

        Returns:
            `(B, T, hidden_size)` attention output.
        """
        bsz, seqlen, _ = hidden_states.shape
        n_h = self.num_heads
        c = self.kv_head_dim
        rd = self.rope_head_dim

        # ---- Q low-rank + per-head q-norm (no scale) + partial RoPE ----
        q = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q).view(bsz, seqlen, n_h, c)
        # Per-head q-norm: rsqrt(mean(q^2)) without learnable weight. Reference
        # does this AFTER `wq_b` and AFTER reshaping into per-head form.
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(
            q.dtype
        )
        cos_t, sin_t = token_position_embeddings
        if rd > 0:
            q = apply_partial_rope_last_n_dims(q, cos_t, sin_t, rd)

        # ---- SWA K=V (single shared head) ----
        swa_kv = self.swa_kv_proj(hidden_states)  # (B, T, c)
        swa_kv = self.kv_norm(swa_kv)
        if rd > 0:
            swa_kv = apply_partial_rope_last_n_dims(swa_kv, cos_t, sin_t, rd)

        # ---- Compressed K=V ----
        compressed_kv = self.compressor(hidden_states, compressed_position_embeddings)
        # compressed_kv: (B, T // m, c)

        # ---- Concatenate; uncompressed first so its indices are 0..T-1 ----
        full_kv = torch.cat([swa_kv, compressed_kv], dim=1)  # (B, T + T/m, c)
        n_compressed = compressed_kv.shape[1]

        # ---- Per-query gather indices ----
        topk_idxs = _build_hca_topk_idxs(
            seqlen=seqlen,
            window_size=self.window_size,
            compression_ratio=self.compression_ratio,
            n_compressed=n_compressed,
            compressed_offset=seqlen,  # uncompressed kv occupies [0, seqlen)
            device=hidden_states.device,
        )
        topk_idxs = topk_idxs.unsqueeze(0).expand(bsz, -1, -1).contiguous()

        # ---- MQA with sink ----
        attn_out = hca_mqa_with_sink(
            q=q,
            kv=full_kv,
            sink_logits=self.sink.sink_logits,
            topk_idxs=topk_idxs,
            softmax_scale=self.softmax_scale,
        )  # (B, T, n_h, c)

        # ---- Output partial RoPE inverse (relative-position recovery) ----
        if rd > 0:
            attn_out = apply_partial_rope_last_n_dims(attn_out, cos_t, sin_t, rd, inverse=True)

        # ---- Grouped output projection ----
        out: torch.Tensor = self.grouped_output(attn_out)
        return out
