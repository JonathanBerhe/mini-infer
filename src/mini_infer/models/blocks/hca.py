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

Two forward modes:
  - `forward(...)`: standalone packed prefill, no cache. Used by the
    HCA prefill parity test.
  - `forward_decode(..., state_cache, layer_idx)`: single-token decode
    that reads SWA + compressed history from a per-request `StateCache`
    and updates the compressor's in-flight state for that layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from mini_infer.cache.state_cache import StateCache


def _build_window_topk_idxs(
    *,
    seqlen: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-query causal sliding-window gather indices.

    Returns `(seqlen, min(seqlen, window_size))` int64 indices: for query
    at position `i`, slot `j` of the window holds key index
    `max(i - n_win + 1, 0) + j` if that key is causally valid, else `-1`.
    Mirrors the reference's `get_window_topk_idxs` exactly so window-edge
    bugs surface at the same `(query_idx, slot_idx)` pair.

    Shared by HCA and CSA (both branches use the same window construction).
    """
    win_slots = min(seqlen, window_size)
    base = torch.arange(seqlen, device=device).unsqueeze(1)  # (seqlen, 1)
    win_idxs = (base - window_size + 1).clamp(min=0) + torch.arange(win_slots, device=device)
    win_idxs = torch.where(win_idxs > base, -1, win_idxs)  # (seqlen, win_slots)
    return win_idxs.to(torch.int64)


def _build_window_decode_topk_idxs(
    *,
    window_size: int,
    start_pos: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-query causal sliding-window gather indices for ONE decode step.

    Returns a 1-D `(window_size,)` int64 tensor of indices into a circular
    SWA buffer of length `window_size`. Mirrors `get_window_topk_idxs` from
    the reference at `start_pos > 0`.

      - When `start_pos >= window_size - 1` the SWA buffer is full; the
        indices wrap: `[start_pos+1, ..., window_size-1, 0, ..., start_pos]`
        (all modulo `window_size`). All slots are valid.
      - When `0 < start_pos < window_size - 1` only the first
        `start_pos + 1` slots are populated; the tail is `-1`-padded.

    Shared by HCA decode and CSA decode (both use the same circular SWA).
    """
    if start_pos >= window_size - 1:
        position_mod_window = start_pos % window_size
        return torch.cat(
            [
                torch.arange(position_mod_window + 1, window_size, device=device),
                torch.arange(0, position_mod_window + 1, device=device),
            ]
        ).to(torch.int64)
    valid_indices = torch.arange(start_pos + 1, device=device)
    padding = torch.full(
        (window_size - start_pos - 1,), -1, dtype=valid_indices.dtype, device=device
    )
    return torch.cat([valid_indices, padding]).to(torch.int64)


def _build_hca_decode_topk_idxs(
    *,
    window_size: int,
    compression_ratio: int,
    start_pos: int,
    compressed_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-query gather indices for one HCA decode step (single new token).

    Returns a 1-D `(window_size + n_valid_compressed,)` int64 tensor of
    indices into the concatenated `[swa_circular ; compressed_history]`
    cache. Window section comes from `_build_window_decode_topk_idxs`;
    compressed section follows the reference's
    `arange(0, (start_pos + 1) // compression_ratio) + offset`.

    HCA includes ALL causally-valid compressed blocks (no sparse selection).
    CSA replaces the compressed section with a `LightningIndexer` top-k pick.
    """
    window_idxs = _build_window_decode_topk_idxs(
        window_size=window_size, start_pos=start_pos, device=device
    )
    n_valid_compressed = (start_pos + 1) // compression_ratio
    compressed_idxs = torch.arange(0, n_valid_compressed, device=device) + compressed_offset
    return torch.cat([window_idxs, compressed_idxs]).to(torch.int64)


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

    Compressed section (mirror of `get_compress_topk_idxs`): for query
    at position `i`, attend to compressed block `j` only when
    `j < (i + 1) / m` (i.e. block `j` covers tokens `[j*m, (j+1)*m - 1]`,
    all of which must precede `i`). The absolute index into the
    concatenated tensor is `j + compressed_offset`.

    HCA includes ALL causally-valid compressed blocks (no sparse selection).
    CSA replaces this section with a `LightningIndexer` top-k pick.
    """
    win_idxs = _build_window_topk_idxs(seqlen=seqlen, window_size=window_size, device=device)

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

    def forward_decode(
        self,
        hidden_state: torch.Tensor,
        *,
        start_pos: int,
        state_cache: StateCache,
        layer_idx: int,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """One decode step: read SWA + compressed history from cache, append, attend.

        Args:
            hidden_state: `(B, 1, hidden_size)` — hidden state of the new token.
            start_pos: Global token position (0-indexed) of this token. The
                FIRST decode step uses `start_pos = T_prefill`; each subsequent
                step increments by 1.
            state_cache: Per-request `StateCache`. The layer at `layer_idx`
                holds the SWA circular buffer + compressed history + compressor
                in-flight state for this layer.
            layer_idx: Index into `state_cache._layers`.
            token_position_embeddings: `(cos, sin)` for THIS token's position;
                each `(B, 1, rope_head_dim)`.
            block_position_embeddings: `(cos, sin)` for the just-flushed
                compressed block (block `start_pos // m`, position `(start_pos
                // m) * m`); each `(B, 1, rope_head_dim)`. Required iff
                `rope_head_dim > 0` AND this step closes a block.

        Returns:
            `(B, 1, hidden_size)` attention output for the new token.

        Side effects on `state_cache.layer(layer_idx)`:
            - `swa_kv[:, start_pos % n_win]` overwritten with the new SWA entry.
            - `swa_count` incremented (capped at `n_win`).
            - `cmp_kv_state[:, start_pos % m]` and `cmp_score_state[:, ...]`
              updated with the new token's compressor inputs.
            - On block-flush: `compressed_kv[:, n_compressed_blocks]` written,
              `n_compressed_blocks` incremented.

        The caller is responsible for `state_cache.advance_start_pos(1)`
        after the forward returns — this method does NOT advance the global
        counter so a multi-layer stack can call it `n_layers` times for the
        same `start_pos`.
        """
        bsz, seqlen_in, _ = hidden_state.shape
        if seqlen_in != 1:
            raise ValueError(f"forward_decode expects seqlen=1, got {seqlen_in}")
        n_h = self.num_heads
        c = self.kv_head_dim
        rd = self.rope_head_dim
        n_win = self.window_size
        m = self.compression_ratio

        state = state_cache.layer(layer_idx)

        # ---- Q (same low-rank + per-head q-norm + partial RoPE as prefill) ----
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_state))
        q = self.q_b_proj(q_latent).view(bsz, 1, n_h, c)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(
            q.dtype
        )
        cos_t, sin_t = token_position_embeddings
        if rd > 0:
            q = apply_partial_rope_last_n_dims(q, cos_t, sin_t, rd)

        # ---- New SWA KV: project + norm + RoPE; write to circular buffer ----
        new_swa = self.kv_norm(self.swa_kv_proj(hidden_state))  # (B, 1, c)
        if rd > 0:
            new_swa = apply_partial_rope_last_n_dims(new_swa, cos_t, sin_t, rd)
        state.swa_kv[:, start_pos % n_win] = new_swa.squeeze(1).to(state.swa_kv.dtype)
        state.swa_count = min(state.swa_count + 1, n_win)

        # ---- Compressor decode step (may flush a compressed entry) ----
        flushed = self.compressor.forward_decode_step(
            hidden_state,
            start_pos=start_pos,
            cmp_kv_state=state.cmp_kv_state,
            cmp_score_state=state.cmp_score_state,
            block_position_embeddings=block_position_embeddings,
        )
        if flushed is not None:
            if state.n_compressed_blocks >= state.compressed_kv.shape[1]:
                raise RuntimeError(
                    f"layer {layer_idx}: compressed history is full "
                    f"({state.n_compressed_blocks} entries); raise max_n_compressed"
                )
            state.compressed_kv[:, state.n_compressed_blocks] = flushed.squeeze(1).to(
                state.compressed_kv.dtype
            )
            state.n_compressed_blocks += 1

        # ---- Build full_kv: SWA window (full circular) ; compressed history valid prefix ----
        # The topk_idxs `-1` slots make uninitialized SWA tail safe (the dispatcher
        # masks `-1` to `-inf` in the softmax).
        n_valid_cmp = state.n_compressed_blocks
        full_kv = torch.cat([state.swa_kv, state.compressed_kv[:, :n_valid_cmp]], dim=1)
        # full_kv: (B, n_win + n_valid_cmp, c)

        # ---- Per-query gather indices: window (size n_win) + compressed (n_valid_cmp) ----
        topk_1d = _build_hca_decode_topk_idxs(
            window_size=n_win,
            compression_ratio=m,
            start_pos=start_pos,
            compressed_offset=n_win,
            device=hidden_state.device,
        )
        topk_idxs = topk_1d.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1).contiguous()

        # ---- MQA with sink ----
        attn_out = hca_mqa_with_sink(
            q=q,
            kv=full_kv,
            sink_logits=self.sink.sink_logits,
            topk_idxs=topk_idxs,
            softmax_scale=self.softmax_scale,
        )  # (B, 1, n_h, c)

        # ---- Output partial RoPE inverse ----
        if rd > 0:
            attn_out = apply_partial_rope_last_n_dims(attn_out, cos_t, sin_t, rd, inverse=True)

        # ---- Grouped output projection ----
        out: torch.Tensor = self.grouped_output(attn_out)
        return out
