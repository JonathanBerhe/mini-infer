"""Multi-head Latent Attention (MLA) — DeepSeek-V2 / V3, Kimi-K2.

The defining trick of MLA: cache TWO compressed streams per layer instead
of per-head K and V tensors. Standard MHA stores
`(num_heads, head_dim) * 2` per token; MLA stores
`(1, kv_lora_rank=512) + (1, qk_rope_head_dim=64)` per token —
**~7x smaller cache** at DeepSeek-V2-Lite scale, way more at V3 / Kimi-K2.

What we cache:
  - `kv_latent` — `kv_a_proj_with_mqa`'s `kv_lora_rank` slice (un-normed).
    Decompressed on read via `kv_a_layernorm + kv_b_proj` to recover
    per-head K (no positional) and per-head V.
  - `k_rope` — `kv_a_proj_with_mqa`'s `qk_rope_head_dim` slice with RoPE
    pre-applied. Broadcast to all heads on read and concatenated with
    the decompressed K.

Asymmetric head dims:
  - Q / K head_dim = `qk_nope_head_dim + qk_rope_head_dim` (=192 V2-Lite)
  - V head_dim     = `v_head_dim` (=128 V2-Lite)
This rules out flash-attn 2 / FlashInfer prefill, which both assume
symmetric Q/K/V head_dim. We dispatch through
`mla_packed_attention_forward` (PyTorch SDPA reference); per the Gemma 4
lesson, models force the `"torch"` attention backend via
`required_attention_backend()`.

Forward shape walk (single-token, batch=1):
  hidden_states (1, 1, H)
  → q (1, num_heads, 1, qk_head_dim)        # via q_proj OR low-rank
  → split q_nope (qk_nope_head_dim) + q_pe (qk_rope_head_dim); RoPE on q_pe
  → kv_a_proj_with_mqa → (1, 1, kv_lora_rank + qk_rope_head_dim)
    → split kv_latent (kv_lora_rank) + k_rope (qk_rope_head_dim); RoPE on k_rope
  → write kv_latent + k_rope to cache as 1-head streams
  → read FULL kv_latent + k_rope from cache (concatenated past + present)
  → kv_a_layernorm + kv_b_proj on kv_latent → split to k_nope (per-head) + v (per-head)
  → broadcast k_rope to all heads, cat with k_nope → K
  → SDPA(Q, K, V) where V has different head_dim than Q/K
  → o_proj

Tensor parallelism
------------------
Per-head matrices are column-parallel along the head axis (`q_proj` /
`q_b_proj`, `kv_b_proj` — the latter holds K-nope + V interleaved per
head). The shared low-rank inputs are replicated: `q_a_proj` /
`q_a_layernorm` (latent fed into `q_b_proj`) and `kv_a_proj_with_mqa` /
`kv_a_layernorm` (single shared MQA head, very small). The output
projection `o_proj` is row-parallel and triggers the single all-reduce
per attention block. At `world_size=1` all column/row-parallel layers
reduce to plain `nn.Linear`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.cache.mla_attention import mla_packed_attention_forward
from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_interleaved_rotary_pos_emb

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class MLAAttention(nn.Module):
    """Multi-head Latent Attention block (text-only, varlen-packed)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        from mini_infer.distributed.group import get_world_size

        world_size = get_world_size()
        if num_heads % world_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by world_size={world_size}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # `num_heads_local` is what reshape sites in `forward` use: each
        # rank only owns `num_heads // world_size` heads after the
        # column-parallel projections.
        self.num_heads_local = num_heads // world_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.layer_idx = layer_idx
        # Q path: direct projection (V2-Lite, q_lora_rank=None) or
        # low-rank decomposition (V2 / V3, q_lora_rank=1536).
        # When sharded under TP: `q_a_proj` (input -> q_lora_rank latent) is
        # *replicated* — the latent is small and downstream `q_b_proj` is
        # column-parallel by head, which is what shards the heavy compute.
        if q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                hidden_size, num_heads * self.qk_head_dim, bias=False
            )
            self.q_a_proj: nn.Linear | None = None
            self.q_a_layernorm: RMSNorm | None = None
            self.q_b_proj: ColumnParallelLinear | None = None
        else:
            self.q_proj = None  # type: ignore[assignment]
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=attention_bias)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank, num_heads * self.qk_head_dim, bias=False
            )
        # KV-down: one combined projection emitting [kv_latent | k_rope].
        # Both branches are *single-head* (MQA pattern) — sharding makes no
        # sense, so `kv_a_proj_with_mqa` is replicated. Storage cost is
        # negligible (rank ~512 + ~64 dims).
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=attention_bias
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        # `kv_b_proj` decompresses the shared latent into `num_heads`-many
        # `(K-nope, V)` pairs. Column-parallel by head: each rank produces
        # its head slice of the decompressed K and V.
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )
        # Row-parallel output: input is the per-head-sharded attention output
        # (already split along the head axis). One all-reduce per block.
        self.o_proj = RowParallelLinear(num_heads * v_head_dim, hidden_size, bias=attention_bias)
        # Softmax scale: 1/sqrt(qk_head_dim) per HF source line 335.
        self._softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (1, total_q, hidden_size). Engine packs the batch
        # along dim 1; per-request boundaries live in cu_seqlens_q.
        bsz, total_q, _ = hidden_states.shape
        assert bsz == 1, "MLAAttention expects packed-batch convention (B=1)"

        # --- Q path ---
        if self.q_lora_rank is None:
            assert self.q_proj is not None
            q = self.q_proj(hidden_states)
        else:
            assert self.q_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        # q: (1, total_q, num_heads_local * qk_head_dim)
        # → (1, num_heads_local, total_q, qk_head_dim). Under TP each rank
        # only computed its slice of heads; the reshape uses the *local*
        # head count so the leading "num_heads" axis isn't off by 1/ws.
        q = q.view(1, total_q, self.num_heads_local, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- KV-down: project + split ---
        kv_combined = self.kv_a_proj_with_mqa(hidden_states)
        # kv_combined: (1, total_q, kv_lora_rank + qk_rope_head_dim)
        kv_latent, k_rope = torch.split(
            kv_combined, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # kv_latent: (1, total_q, kv_lora_rank) — shared across heads
        # k_rope:    (1, total_q, qk_rope_head_dim) — shared across heads

        # Apply RoPE: q_pe shape is (1, num_heads, total_q, qk_rope_head_dim);
        # k_rope reshaped to (1, 1, total_q, qk_rope_head_dim) so the same
        # call rotates both. Uses INTERLEAVED RoPE (DeepSeek convention)
        # — pairs (x[2i], x[2i+1]) rotate together. HF's `apply_rotary_emb`
        # uses the same complex-number formulation; we stay in real
        # arithmetic but the math matches bit-for-bit.
        cos, sin = position_embeddings  # both (1, total_q, qk_rope_head_dim)
        k_rope_for_rope = k_rope.view(1, total_q, 1, self.qk_rope_head_dim).transpose(1, 2)
        q_pe, k_rope_rotated = apply_interleaved_rotary_pos_emb(q_pe, k_rope_for_rope, cos, sin)
        # k_rope_rotated: (1, 1, total_q, qk_rope_head_dim) — still shared across heads

        # --- Append per-stream to cache (packed shape: (total_q, num_kv_heads_s, head_dim_s)) ---
        kv_latent_packed = kv_latent.view(total_q, 1, self.kv_lora_rank).contiguous()
        k_rope_packed = (
            k_rope_rotated.transpose(1, 2).reshape(total_q, 1, self.qk_rope_head_dim).contiguous()
        )
        past_key_values.append_stream_packed(
            kv_latent_packed, cu_seqlens_q, self.layer_idx, "kv_latent"
        )
        past_key_values.append_stream_packed(k_rope_packed, cu_seqlens_q, self.layer_idx, "k_rope")

        # --- Read full history per stream and reconstruct K, V ---
        kv_latent_full, cu_seqlens_k, _ = past_key_values.materialize_packed_stream(
            self.layer_idx, "kv_latent"
        )
        k_rope_full, _, _ = past_key_values.materialize_packed_stream(self.layer_idx, "k_rope")
        # kv_latent_full: (total_k, 1, kv_lora_rank). Drop the 1-head dim, then
        # apply layernorm + decompression.
        kv_latent_2d = kv_latent_full.squeeze(1)  # (total_k, kv_lora_rank)
        decompressed = self.kv_b_proj(self.kv_a_layernorm(kv_latent_2d))
        # decompressed: (total_k, num_heads_local * (qk_nope_head_dim + v_head_dim))
        # `kv_b_proj` is column-parallel by head, so the per-rank output
        # already only carries this rank's head slice.
        decompressed = decompressed.view(
            -1, self.num_heads_local, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_full, v_full = torch.split(
            decompressed, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # k_nope_full: (total_k, num_heads_local, qk_nope_head_dim)
        # v_full:      (total_k, num_heads_local, v_head_dim)

        # Broadcast k_rope to this rank's local heads (k_rope is replicated
        # across ranks since `kv_a_proj_with_mqa` itself is replicated).
        k_rope_broadcast = k_rope_full.expand(-1, self.num_heads_local, -1)
        # Concatenate to form full K with head_dim = qk_head_dim (192 V2-Lite).
        k_full = torch.cat([k_nope_full, k_rope_broadcast], dim=-1)
        # k_full: (total_k, num_heads_local, qk_head_dim)

        # Pack Q for the dispatcher (per-rank head slice).
        q_packed = (
            torch.cat([q_nope, q_pe], dim=-1)
            .transpose(1, 2)
            .reshape(total_q, self.num_heads_local, self.qk_head_dim)
            .contiguous()
        )

        # --- Asymmetric SDPA ---
        attn_out = mla_packed_attention_forward(
            q_packed,
            k_full,
            v_full,
            cu_seqlens_q,
            cu_seqlens_k,
            self._softmax_scale,
        )
        # attn_out: (total_q, num_heads_local, v_head_dim)

        attn_out = attn_out.reshape(1, total_q, self.num_heads_local * self.v_head_dim).contiguous()
        # `o_proj` is row-parallel: takes the (col-sharded) attention
        # output and all-reduces to recover the full hidden state.
        out: torch.Tensor = self.o_proj(attn_out)
        return out
