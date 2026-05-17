"""Compressed Sparse Attention (CSA) — DeepSeek-V4 §2.3.

Tensor parallelism
------------------
Same scheme as HCA: replicated `q_a_proj` latent + replicated `swa_kv_proj`
+ replicated main compressor, with column-parallel `q_b_proj` (by head)
and a TP-aware Lightning Indexer (per-head shards) feeding a TP-aware
grouped output (sharded by group, row-parallel `wo_b`).

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

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.cache.hca_attention import hca_mqa_with_sink
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models.blocks.hca import (
    _build_window_decode_topk_idxs,
    _build_window_topk_idxs,
)
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims
from mini_infer.models.blocks.v4 import (
    AttentionSink,
    GroupedOutputProjection,
    LightningIndexer,
    TokenLevelCompressor,
)

if TYPE_CHECKING:
    from mini_infer.cache.state_cache import StateCache


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
        from mini_infer.distributed.group import get_world_size

        world_size = get_world_size()
        if num_heads % world_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by world_size={world_size}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_local = num_heads // world_size
        self.q_lora_rank = q_lora_rank
        self.kv_head_dim = kv_head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.compression_ratio = compression_ratio
        self.rms_norm_eps = rms_norm_eps

        # --- Q low-rank path (shared latent reused by the indexer below) ---
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        # Column-parallel by head: each rank gets `num_heads_local` Q heads.
        self.q_b_proj = ColumnParallelLinear(q_lora_rank, num_heads * kv_head_dim, bias=False)

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
        n_h_local = self.num_heads_local
        c = self.kv_head_dim
        rd = self.rope_head_dim

        # ---- Q low-rank latent (shared with the indexer's `wq_b`) ----
        q_lora_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))

        # ---- Full Q + per-head q-norm (no scale) + partial RoPE (per-rank slice) ----
        q = self.q_b_proj(q_lora_latent).view(bsz, seqlen, n_h_local, c)
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

    def forward_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        *,
        token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state_cache: StateCache,
        layer_idx: int,
    ) -> torch.Tensor:
        """Cache-aware CSA prefill: same output as `forward`, plus state writes.

        Writes the main compressor's output AND the indexer's compressor's
        output into separate sub-caches, populating both in-flight
        accumulators (overlap mode for both). Supports unaligned seqlen.

        Returns `(B, T, hidden_size)` — identical to
        `forward(hidden_states, token_pe, compressed_pe)` for aligned
        seqlen. Caller must `state_cache.advance_start_pos(seqlen)` after.
        """
        batch_size, seqlen, _ = hidden_states.shape
        num_heads_local = self.num_heads_local
        kv_head_dim = self.kv_head_dim
        rope_dim = self.rope_head_dim
        n_win = self.window_size

        layer_state = state_cache.layer(layer_idx)
        if layer_state.indexer is None:
            raise ValueError(
                f"layer {layer_idx}: state cache must have an indexer slot for CSA layers"
            )

        # ---- Q low-rank latent (shared with the indexer) ----
        q_lora_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))

        # ---- Full Q + per-head q-norm + partial RoPE (per-rank head slice) ----
        q = self.q_b_proj(q_lora_latent).view(batch_size, seqlen, num_heads_local, kv_head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(
            q.dtype
        )
        cos_for_tokens, sin_for_tokens = token_position_embeddings
        if rope_dim > 0:
            q = apply_partial_rope_last_n_dims(q, cos_for_tokens, sin_for_tokens, rope_dim)

        # ---- SWA K=V (single shared head) ----
        swa_kv = self.kv_norm(self.swa_kv_proj(hidden_states))
        if rope_dim > 0:
            swa_kv = apply_partial_rope_last_n_dims(
                swa_kv, cos_for_tokens, sin_for_tokens, rope_dim
            )

        # ---- Main compressor with state writes ----
        compressed_kv = self.compressor.forward_prefill_with_cache(
            hidden_states,
            compressed_position_embeddings=compressed_position_embeddings,
            cmp_kv_state=layer_state.cmp_kv_state,
            cmp_score_state=layer_state.cmp_score_state,
        )
        n_emitted_blocks = compressed_kv.shape[1]
        if n_emitted_blocks > layer_state.compressed_kv.shape[1]:
            raise RuntimeError(
                f"layer {layer_idx}: main compressed history capacity "
                f"({layer_state.compressed_kv.shape[1]}) is too small for {n_emitted_blocks} "
                "entries; raise max_n_compressed"
            )
        layer_state.compressed_kv[:, :n_emitted_blocks] = compressed_kv.to(
            layer_state.compressed_kv.dtype
        )
        layer_state.n_compressed_blocks = n_emitted_blocks

        # ---- SWA cache write: last min(seqlen, n_win) tokens (rotated layout) ----
        if seqlen <= n_win:
            layer_state.swa_kv[:, :seqlen] = swa_kv.to(layer_state.swa_kv.dtype)
        else:
            wrap_cutoff = seqlen % n_win
            last_window = swa_kv[:, -n_win:]
            layer_state.swa_kv[:, wrap_cutoff:n_win] = last_window[:, : n_win - wrap_cutoff].to(
                layer_state.swa_kv.dtype
            )
            layer_state.swa_kv[:, :wrap_cutoff] = last_window[:, n_win - wrap_cutoff :].to(
                layer_state.swa_kv.dtype
            )
        layer_state.swa_count = min(seqlen, n_win)

        # ---- Indexer cache-aware prefill (drives its own compressor + writes history) ----
        indexer_topk_idxs = self.indexer.forward_prefill_with_cache(
            hidden_states,
            q_lora_latent,
            token_position_embeddings=token_position_embeddings,
            compressed_position_embeddings=compressed_position_embeddings,
            indexer_state=layer_state.indexer,
            compressed_offset=seqlen,
        )

        # ---- Build full_kv for THIS prefill's attention ----
        full_kv = torch.cat([swa_kv, compressed_kv], dim=1)

        # ---- Window topk (handles unaligned seqlen) + concat with indexer top-k ----
        window_topk_idxs = (
            _build_window_topk_idxs(seqlen=seqlen, window_size=n_win, device=hidden_states.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        topk_idxs = torch.cat([window_topk_idxs, indexer_topk_idxs], dim=-1).contiguous()

        # ---- MQA with sink ----
        attn_out = hca_mqa_with_sink(
            q=q,
            kv=full_kv,
            sink_logits=self.sink.sink_logits,
            topk_idxs=topk_idxs,
            softmax_scale=self.softmax_scale,
        )

        # ---- Output partial RoPE inverse + grouped output ----
        if rope_dim > 0:
            attn_out = apply_partial_rope_last_n_dims(
                attn_out, cos_for_tokens, sin_for_tokens, rope_dim, inverse=True
            )
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
        """One CSA decode step: read SWA + main compressed history + indexer top-k from cache.

        Args:
            hidden_state: `(B, 1, hidden_size)` — hidden state of the new token.
            start_pos: Global token position (0-indexed).
            state_cache: Per-request `StateCache`. The layer at `layer_idx`
                must have been allocated with `overlap_mode=True` and a
                non-`None` `IndexerStateSpec`.
            layer_idx: Index into `state_cache._layers`.
            token_position_embeddings: `(cos, sin)` for THIS token's position;
                each `(B, 1, rope_head_dim)`.
            block_position_embeddings: `(cos, sin)` for the just-flushed block
                (block index `start_pos // compression_ratio`, position
                `(start_pos // compression_ratio) * compression_ratio`); each
                `(B, 1, rope_head_dim)`. Required iff `rope_head_dim > 0` AND
                this step closes a block — both the main compressor AND the
                indexer's compressor share the same compression_ratio in V4
                so they flush on the same step with the same block position.

        Returns:
            `(B, 1, hidden_size)` attention output for the new token.

        Side effects (on `state_cache.layer(layer_idx)`):
            - SWA circular buffer: write at `start_pos % window_size`.
            - Main compressor in-flight state: write to slot
              `compression_ratio + (start_pos % compression_ratio)`; on flush,
              the previous-block slots are slid to make room.
            - On flush: `compressed_kv[:, n_compressed_blocks]` written;
              `n_compressed_blocks` incremented.
            - Indexer's own compressor state + history advanced via
              `LightningIndexer.forward_decode_step`.
        """
        batch_size, seqlen, _ = hidden_state.shape
        if seqlen != 1:
            raise ValueError(f"forward_decode expects seqlen=1, got {seqlen}")
        num_heads_local = self.num_heads_local
        kv_head_dim = self.kv_head_dim
        rope_dim = self.rope_head_dim
        window_size = self.window_size

        layer_state = state_cache.layer(layer_idx)
        if layer_state.indexer is None:
            raise ValueError(
                f"layer {layer_idx}: state_cache must be allocated with an indexer "
                "(IndexerStateSpec) for CSA decode"
            )

        # ---- Q low-rank latent (shared with the indexer's `wq_b`) ----
        q_lora_latent = self.q_a_layernorm(self.q_a_proj(hidden_state))

        # ---- Full Q + per-head q-norm (no scale) + partial RoPE (per-rank slice) ----
        q = self.q_b_proj(q_lora_latent).view(batch_size, 1, num_heads_local, kv_head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(
            q.dtype
        )
        cos_for_token, sin_for_token = token_position_embeddings
        if rope_dim > 0:
            q = apply_partial_rope_last_n_dims(q, cos_for_token, sin_for_token, rope_dim)

        # ---- New SWA KV: project + norm + RoPE; write to circular buffer ----
        new_swa_kv = self.kv_norm(self.swa_kv_proj(hidden_state))
        if rope_dim > 0:
            new_swa_kv = apply_partial_rope_last_n_dims(
                new_swa_kv, cos_for_token, sin_for_token, rope_dim
            )
        layer_state.swa_kv[:, start_pos % window_size] = new_swa_kv.squeeze(1).to(
            layer_state.swa_kv.dtype
        )
        layer_state.swa_count = min(layer_state.swa_count + 1, window_size)

        # ---- Main compressor decode step (overlap mode) ----
        flushed_main = self.compressor.forward_decode_step(
            hidden_state,
            start_pos=start_pos,
            cmp_kv_state=layer_state.cmp_kv_state,
            cmp_score_state=layer_state.cmp_score_state,
            block_position_embeddings=block_position_embeddings,
        )
        if flushed_main is not None:
            if layer_state.n_compressed_blocks >= layer_state.compressed_kv.shape[1]:
                raise RuntimeError(
                    f"layer {layer_idx}: main compressed history is full "
                    f"({layer_state.n_compressed_blocks} entries); raise max_n_compressed"
                )
            layer_state.compressed_kv[:, layer_state.n_compressed_blocks] = flushed_main.squeeze(
                1
            ).to(layer_state.compressed_kv.dtype)
            layer_state.n_compressed_blocks += 1

        # ---- Indexer decode step: drives its own compressor + selects top-k ----
        indexer_topk_idxs = self.indexer.forward_decode_step(
            hidden_state,
            q_lora_latent,
            start_pos=start_pos,
            indexer_state=layer_state.indexer,
            token_position_embeddings=token_position_embeddings,
            block_position_embeddings=block_position_embeddings,
            compressed_offset=window_size,  # main compressed history starts at slot window_size
        )  # (B, 1, actual_top_k)

        # ---- Build full_kv: SWA window (full circular) ; main compressed history ----
        n_valid_main_compressed = layer_state.n_compressed_blocks
        full_kv = torch.cat(
            [layer_state.swa_kv, layer_state.compressed_kv[:, :n_valid_main_compressed]],
            dim=1,
        )

        # ---- Per-query gather indices: window section + indexer's top-k pick ----
        window_idxs_1d = _build_window_decode_topk_idxs(
            window_size=window_size, start_pos=start_pos, device=hidden_state.device
        )
        window_topk_idxs = (
            window_idxs_1d.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1).contiguous()
        )
        topk_idxs = torch.cat([window_topk_idxs, indexer_topk_idxs], dim=-1).contiguous()

        # ---- MQA with sink ----
        attn_out = hca_mqa_with_sink(
            q=q,
            kv=full_kv,
            sink_logits=self.sink.sink_logits,
            topk_idxs=topk_idxs,
            softmax_scale=self.softmax_scale,
        )

        # ---- Output partial RoPE inverse ----
        if rope_dim > 0:
            attn_out = apply_partial_rope_last_n_dims(
                attn_out, cos_for_token, sin_for_token, rope_dim, inverse=True
            )

        # ---- Grouped output projection ----
        out: torch.Tensor = self.grouped_output(attn_out)
        return out
