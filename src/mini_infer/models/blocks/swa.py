"""Pure Sliding-Window Attention (SWA) — DeepSeek-V4's `compress_ratio=0` layers.

V4 alternates three attention modes per layer based on `compress_ratio`:
  - `0`   -> pure SWA (this file): no compressor, no indexer, just sliding
            window + sink + grouped output. V4-Flash uses 2 of these (the
            first two layers).
  - `4`   -> CSA (compressor m=4 + Lightning Indexer + window + sink).
  - `128` -> HCA (compressor m=128 + window + sink, no indexer).

The reference (`deepseek_v4_reference/model.py::Attention`) handles this
via `if self.compress_ratio:` branches. We split it into its own block
here because the constructor signature is cleaner (no compressor /
indexer args), the state cache spec is shorter, and reading "SWA
without any compression machinery" doesn't require carrying the
compressor / indexer plumbing through the file.

Forward shape walk (single block, batch=B, len=T):
    H (B, T, d)
    Q low-rank: H -> q_a_proj -> q_a_layernorm -> q_b_proj
                  -> (B, T, n_h_local, c)
    Q-norm + partial RoPE on the last `rope_head_dim` dims
    SWA K=V (single shared head): H -> swa_kv_proj -> kv_norm -> partial RoPE
                                  -> (B, T, c)
    Window topk_idxs: (T, win_slots)
    hca_mqa_with_sink(Q, swa_kv, sink, topk_idxs) -> (B, T, n_h_local, c)
    Output partial-RoPE inverse + GroupedOutputProjection -> (B, T, d)

Tensor parallelism
------------------
Same wiring as HCA's non-compressor branch: `q_b_proj` column-parallel
by head, `swa_kv_proj` / `kv_norm` replicated (single shared head),
`sink` sharded by head, `grouped_output` sharded by group with
row-parallel finalisation. At `world_size=1` everything reduces to the
plain forms.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.cache.hca_attention import hca_mqa_with_sink
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models.blocks.hca import _build_window_topk_idxs
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims
from mini_infer.models.blocks.v4 import AttentionSink, GroupedOutputProjection


class SWAAttention(nn.Module):
    """Pure-SWA attention (no compressor, no indexer) for V4's `compress_ratio=0`."""

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
            raise ValueError(
                f"num_heads={num_heads} must be divisible by world_size={world_size}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_local = num_heads // world_size
        self.q_lora_rank = q_lora_rank
        self.kv_head_dim = kv_head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.rms_norm_eps = rms_norm_eps

        # Q low-rank: same structure as HCA/CSA. `q_a_proj` and
        # `q_a_layernorm` are replicated; `q_b_proj` is column-parallel.
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank, num_heads * kv_head_dim, bias=False
        )

        # SWA K=V branch: single shared head (MQA-style), replicated.
        self.swa_kv_proj = nn.Linear(hidden_size, kv_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_head_dim, eps=rms_norm_eps)

        # Per-head learnable sink logit.
        self.sink = AttentionSink(num_heads=num_heads)

        # Grouped output projection (sharded by group; one all-reduce).
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
        # compressed_position_embeddings is accepted but ignored — kept for
        # call-site compatibility with HCA/CSA so the decoder layer can
        # invoke any attention type with the same args.
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Standalone packed-prefill forward (no cache).

        Args:
            hidden_states: `(B, T, hidden_size)`. `T` does NOT need to be
                a multiple of any compression ratio (there's no compressor).
            token_position_embeddings: `(cos, sin)` for the `T` raw token
                positions; each `(B, T, rope_head_dim)`.
            compressed_position_embeddings: Ignored. Present for signature
                parity with `HCAAttention.forward`.

        Returns:
            `(B, T, hidden_size)` attention output.
        """
        del compressed_position_embeddings  # documented as ignored
        bsz, seqlen, _ = hidden_states.shape
        n_h_local = self.num_heads_local
        c = self.kv_head_dim
        rope_dim = self.rope_head_dim

        # ---- Q low-rank + per-head q-norm + partial RoPE ----
        q = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q).view(bsz, seqlen, n_h_local, c)
        # Per-head q-norm: rsqrt(mean(q^2)) without learnable weight.
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(
            q.dtype
        )
        cos_t, sin_t = token_position_embeddings
        if rope_dim > 0:
            q = apply_partial_rope_last_n_dims(q, cos_t, sin_t, rope_dim)

        # ---- SWA K=V (single shared head) ----
        swa_kv = self.swa_kv_proj(hidden_states)  # (B, T, c)
        swa_kv = self.kv_norm(swa_kv)
        if rope_dim > 0:
            swa_kv = apply_partial_rope_last_n_dims(swa_kv, cos_t, sin_t, rope_dim)

        # ---- Window-only gather indices (no compressed branch) ----
        topk_idxs = _build_window_topk_idxs(
            seqlen=seqlen,
            window_size=self.window_size,
            device=hidden_states.device,
        )
        topk_idxs = topk_idxs.unsqueeze(0).expand(bsz, -1, -1).contiguous()

        # ---- MQA with sink ----
        attn_out = hca_mqa_with_sink(
            q=q,
            kv=swa_kv,
            sink_logits=self.sink.sink_logits,
            topk_idxs=topk_idxs,
            softmax_scale=self.softmax_scale,
        )  # (B, T, n_h_local, c)

        # ---- Output partial RoPE inverse (relative-position recovery) ----
        if rope_dim > 0:
            attn_out = apply_partial_rope_last_n_dims(
                attn_out, cos_t, sin_t, rope_dim, inverse=True
            )

        # ---- Grouped output projection ----
        out: torch.Tensor = self.grouped_output(attn_out)
        return out
