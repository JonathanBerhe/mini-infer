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
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.cache.mla_attention import mla_packed_attention_forward
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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.layer_idx = layer_idx
        # Q path: direct projection (V2-Lite, q_lora_rank=None) or
        # low-rank decomposition (V2 / V3, q_lora_rank=1536).
        if q_lora_rank is None:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.qk_head_dim, bias=False)
            self.q_a_proj: nn.Linear | None = None
            self.q_a_layernorm: RMSNorm | None = None
            self.q_b_proj: nn.Linear | None = None
        else:
            self.q_proj = None  # type: ignore[assignment]
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=attention_bias)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        # KV-down: one combined projection emitting [kv_latent | k_rope].
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=attention_bias
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=attention_bias)
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
        # q: (1, total_q, num_heads * qk_head_dim) → (1, num_heads, total_q, qk_head_dim)
        q = q.view(1, total_q, self.num_heads, self.qk_head_dim).transpose(1, 2)
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
        # decompressed: (total_k, num_heads * (qk_nope_head_dim + v_head_dim))
        decompressed = decompressed.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_full, v_full = torch.split(
            decompressed, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # k_nope_full: (total_k, num_heads, qk_nope_head_dim)
        # v_full:      (total_k, num_heads, v_head_dim)

        # Broadcast k_rope to all heads: (total_k, 1, qk_rope) -> (total_k, num_heads, qk_rope).
        k_rope_broadcast = k_rope_full.expand(-1, self.num_heads, -1)
        # Concatenate to form full K with head_dim = qk_head_dim (192 V2-Lite).
        k_full = torch.cat([k_nope_full, k_rope_broadcast], dim=-1)
        # k_full: (total_k, num_heads, qk_head_dim)

        # Pack Q for the dispatcher.
        q_packed = (
            torch.cat([q_nope, q_pe], dim=-1)
            .transpose(1, 2)
            .reshape(total_q, self.num_heads, self.qk_head_dim)
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
        # attn_out: (total_q, num_heads, v_head_dim)

        attn_out = attn_out.reshape(1, total_q, self.num_heads * self.v_head_dim).contiguous()
        out: torch.Tensor = self.o_proj(attn_out)
        return out
