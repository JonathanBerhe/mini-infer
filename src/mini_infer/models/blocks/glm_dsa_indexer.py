"""GLM-MoE-DSA Lightning Indexer (DeepSeek Sparse Attention top-k selector).

Unlike the V4 `LightningIndexer` (which scores *compressed* blocks via a
`TokenLevelCompressor`), GLM-5.2's DSA indexer scores **raw tokens**: for
each query it picks the `index_topk` past tokens the main MLA attention will
actually attend to. The unselected tokens are masked to `-inf` in the
attention softmax (see `mla_packed_attention_forward`'s DSA path).

Math (one query at token `t`, scoring against keys `k_j`), matching HF
`GlmMoeDsaIndexer`:
    q_t,h   = wq_b(q_resid_t)               # per-head, pe-first layout
    rope(q_t,h[:rope_dim])                  # NON-interleaved (NeoX) RoPE
    k_j     = k_norm(wk(h_j))               # single shared key, pe-first
    rope(k_j[:rope_dim])
    s_t,h,j = ReLU( (q_t,h . k_j) * d^-0.5 )
    w_t,h   = weights_proj(h_t) * n_heads^-0.5
    score_t,j = sum_h( s_t,h,j * w_t,h )
    topk(t) = argmax_j(score_t,j, k=index_topk)   # causal-masked

The ReLU before the head-sum makes the indexer a per-head "vote": a key
survives only if some head scores it positively. Scores run in fp32 to match
HF (which keeps `weights_proj` in fp32 and upcasts the scoring matmul).

Tensor parallelism
------------------
`wq_b` and `weights_proj` are column-parallel by head; `wk` / `k_norm` are
the single shared key, replicated. The head-summed `index_score` is computed
on each rank's local heads then all-reduced before top-k so every rank picks
identical indices. At `world_size=1` everything reduces to the plain form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.functional import relu

from mini_infer.distributed.comm import all_reduce_sum
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models.blocks.rope import apply_rotary_pos_emb

if TYPE_CHECKING:
    from collections.abc import Callable

    from mini_infer.cache.paged_kv_cache import PagedKVCache


class _Fp32ColumnParallelLinear(ColumnParallelLinear):
    """A `ColumnParallelLinear` whose parameters stay fp32 through dtype casts.

    Mirrors HF's `_keep_in_fp32_modules = ["indexer.weights_proj"]`: a
    whole-model `.to(dtype=torch.bfloat16)` (the `load_model` path) converts
    every other weight but leaves this one fp32-resident, and the subsequent
    weight load upcasts the checkpoint tensor into it (`load_full_weight`
    casts the source to the existing param dtype). Device moves still apply;
    only the dtype is pinned, with the original fp32 bits preserved.
    """

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> _Fp32ColumnParallelLinear:
        frozen = {name: param.data for name, param in self.named_parameters(recurse=False)}
        super()._apply(fn, recurse)  # type: ignore[no-untyped-call]
        for name, param in self.named_parameters(recurse=False):
            original = frozen[name]
            if not param.is_floating_point() or original.is_meta or param.data.is_meta:
                continue
            # Re-derive from the pre-cast bits so a dtype round-trip
            # (fp32 -> bf16 -> fp32) never erodes the resident value.
            param.data = original.to(device=param.data.device, dtype=torch.float32)
        return self


class GlmDsaIndexer(nn.Module):
    """Raw-token top-k selector for GLM-MoE-DSA (DeepSeek Sparse Attention)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        qk_rope_head_dim: int,
        index_topk: int,
        layernorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        from mini_infer.distributed.group import get_world_size

        world_size = get_world_size()
        if num_heads % world_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by world_size={world_size}")
        if not 0 < qk_rope_head_dim <= head_dim:
            raise ValueError(
                f"qk_rope_head_dim={qk_rope_head_dim} must be in (0, head_dim={head_dim}]"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_local = num_heads // world_size
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_topk = index_topk

        # Per-head query up-projection from the shared q_lora latent (the main
        # attention's `q_a_layernorm(q_a_proj(x))`). Column-parallel by head.
        self.wq_b = ColumnParallelLinear(q_lora_rank, num_heads * head_dim, bias=False)
        # Single shared key projection (replicated under TP) + its LayerNorm.
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim, eps=layernorm_eps)
        # Per-token, per-head weighting scalar. Column-parallel by head.
        # fp32-resident (survives whole-model bf16 casts), mirroring HF's
        # `_keep_in_fp32_modules = ["indexer.weights_proj"]`.
        self.weights_proj = _Fp32ColumnParallelLinear(hidden_size, num_heads, bias=False)
        self.softmax_scale = head_dim**-0.5

    def _project(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RoPE'd per-head queries, the shared key, and head weights.

        Returns `(q, k, weights)` with shapes `(1, T, H_local, head_dim)`,
        `(1, T, head_dim)`, `(1, T, H_local)` respectively, all fp32-ready.
        """
        bsz, total_q, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Queries: pe-first split (rope dims lead), NeoX RoPE on the pe slice.
        q = self.wq_b(q_resid).view(bsz, total_q, self.num_heads_local, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )
        # Key: single shared head, pe-first split, same NeoX RoPE.
        k = self.k_norm(self.wk(hidden_states))  # (1, T, head_dim)
        k_pe, k_nope = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )
        # Rotate q_pe (1, T, H, rope_D) and k_pe (1, T, 1, rope_D) together;
        # unsqueeze_dim=2 broadcasts cos/sin over the head axis.
        q_pe, k_pe_rot = apply_rotary_pos_emb(q_pe, k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe_rot.squeeze(2), k_nope], dim=-1)
        # Head weights: an fp32 matmul over upcast hidden states, matching HF
        # 5.12's `weights_proj(hidden_states.to(weights_proj.weight.dtype))`
        # with the fp32-resident weight. The 1/sqrt(n_heads) factor uses the
        # FULL head count.
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (
            self.num_heads**-0.5
        )
        return q, k, weights

    def _causal_topk(
        self,
        q: torch.Tensor,
        k_full: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Score queries vs keys per request, causal-mask, return per-request top-k.

        `q` is `(1, total_q, H, head_dim)`, `k_full` is `(total_k, head_dim)`,
        `weights` is `(1, total_q, H)`. A query at intra-request position `i`
        (global position `k_len - q_len + i`) scores keys `0..(k_len-q_len+i)`,
        matching `mla_packed_attention_forward`'s causal rule. Indices are
        request-local into `[0, k_len)`.
        """
        topk_per_request: list[torch.Tensor] = []
        for r in range(cu_seqlens_q.shape[0] - 1):
            q_start, q_end = int(cu_seqlens_q[r]), int(cu_seqlens_q[r + 1])
            k_start, k_end = int(cu_seqlens_k[r]), int(cu_seqlens_k[r + 1])
            q_len, k_len = q_end - q_start, k_end - k_start
            if q_len == 0:
                topk_per_request.append(torch.empty(0, 0, dtype=torch.int64, device=q.device))
                continue
            q_r = q[0, q_start:q_end].float()  # (q_len, H_local, head_dim)
            k_r = k_full[k_start:k_end].float()  # (k_len, head_dim)
            w_r = weights[0, q_start:q_end]  # (q_len, H_local)
            # Per-head q.k, scaled then ReLU'd, then weighted head-sum.
            per_head = relu(torch.einsum("qhd,kd->qhk", q_r, k_r) * self.softmax_scale)
            # Complete the cross-rank head sum (no-op at world_size=1).
            score = all_reduce_sum(torch.einsum("qhk,qh->qk", per_head, w_r))  # (q_len, k_len)
            q_pos = torch.arange(q_len, device=score.device) + (k_len - q_len)
            k_pos = torch.arange(k_len, device=score.device)
            masked = score.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            topk = min(self.index_topk, k_len)
            topk_per_request.append(masked.topk(topk, dim=-1).indices.to(torch.int64))
        return topk_per_request

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
    ) -> list[torch.Tensor]:
        """No-cache (prefill) top-k: keys derive from the same packed tokens.

        `k_len == q_len` per request. Returns per-request `(q_len, topk_r)`
        int64 indices into `[0, q_len)`. Used by the block-level parity tests
        and by `MLAAttention` when no precomputed selection is supplied.
        """
        q, k, weights = self._project(hidden_states, q_resid, position_embeddings)
        # Keys are the current tokens: k_len == q_len, cu_seqlens_k == cu_seqlens_q.
        return self._causal_topk(q, k.squeeze(0), weights, cu_seqlens_q, cu_seqlens_q)

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        layer_idx: int,
        stream_name: str = "index_k",
    ) -> list[torch.Tensor]:
        """Cache-aware top-k: append this step's keys, score against full history.

        The decode counterpart of `forward`. Keys are RoPE'd at their own
        positions on write and read back in full from the PagedKVCache stream,
        so a 1-token decode step (q_len=1, k_len=context) scores against every
        past token. The query is RoPE'd at the current positions, so q.k carries
        relative position (same convention as the main attention's k_rope
        stream). Returns per-request top-k indices into `[0, k_len)`.
        """
        q, k, weights = self._project(hidden_states, q_resid, position_embeddings)
        total_q = hidden_states.shape[1]
        k_packed = k.view(total_q, 1, self.head_dim).contiguous()
        past_key_values.append_stream_packed(k_packed, cu_seqlens_q, layer_idx, stream_name)
        k_full_packed, cu_seqlens_k, _ = past_key_values.materialize_packed_stream(
            layer_idx, stream_name
        )
        return self._causal_topk(q, k_full_packed.squeeze(1), weights, cu_seqlens_q, cu_seqlens_k)
