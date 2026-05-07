"""HCA / CSA core attention: shared-KV MQA with attention sink and sparse gather.

The DeepSeek-V4 reference dispatches every attention call (HCA, CSA,
pure-SWA) through one tile-lang `sparse_attn` kernel. The kernel takes:

    q          : (B, T, n_h, c)
    kv         : (B, n_kv, c)          # SHARED across heads — V == K == kv
    attn_sink  : (n_h,)                # per-head learnable softmax-denom logit
    topk_idxs  : (B, T, n_topk)        # int indices into kv; -1 = padding

and returns `o : (B, T, n_h, c)`. For each `(B, T)` query, it gathers
the topk_idxs columns of `kv`, runs softmax with the sink baked into
the denominator, and writes the weighted-V output (V is the same kv
tensor — no separate V projection in V4's "Shared KV MQA").

We mirror that signature in pure PyTorch. The kernel exists because
sparse gather + per-head softmax + sink is awkward to express
efficiently in a generic SDPA call — but for parity testing on
synthetic configs, a vectorized PyTorch implementation is sufficient
and serves as the oracle for any future Triton port.

Variables:
- `n_topk` is the number of source positions per query. For HCA at
  prefill: `min(seqlen, window_size) + (seqlen // m')`. For CSA at
  prefill: `min(seqlen, window_size) + min(top_k, seqlen // m)`.
- A `topk_idx` of `-1` means "this slot is padding" and contributes
  nothing to the softmax (logit forced to `-inf`).

Sink semantics (V4 paper §2.3.3 formula 27, confirmed by the reference
kernel):
    den = sum_k exp(score_{q,k,h}) + exp(z'_h)
    o_{q,h,d} = sum_k softmax_k(score_{q,*,h})[k] * kv[k, d]
where the sink contributes to the denominator but has no value, so the
output magnitude shrinks by `exp(z'_h) / den` worth of mass per query.
The "concat one extra key with value=0 and logit=z'_h" formulation is
mathematically identical and is what we implement (it's easier to read
and to differentiate than the explicit two-term denominator).
"""

from __future__ import annotations

import torch


def hca_mqa_with_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink_logits: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse-gather MQA with per-head sink. Shared K=V tensor.

    Args:
        q:           `(B, T, n_h, c)`
        kv:          `(B, n_kv, c)` — single shared head, used as both K and V.
        sink_logits: `(n_h,)` per-head learnable logit added to the softmax denominator.
        topk_idxs:   `(B, T, n_topk)` int positions into `kv` (`-1` = padding).
        softmax_scale: scalar multiplier on the QK^T scores.

    Returns:
        `(B, T, n_h, c)` attention output.
    """
    bsz, seqlen, n_h, c = q.shape
    if kv.shape[0] != bsz or kv.shape[2] != c:
        raise ValueError(
            f"kv shape {tuple(kv.shape)} does not match q ({bsz=}, c={c}); expected (B, n_kv, c)"
        )
    if sink_logits.shape != (n_h,):
        raise ValueError(f"sink_logits shape {tuple(sink_logits.shape)} != (n_h={n_h},)")
    if topk_idxs.shape[:2] != (bsz, seqlen):
        raise ValueError(
            f"topk_idxs leading shape {tuple(topk_idxs.shape[:2])} != (B={bsz}, T={seqlen})"
        )

    # Gather kv at topk_idxs. `-1` indices map to position 0 (placeholder);
    # the corresponding scores are forced to `-inf` so the softmax ignores them.
    is_padding = topk_idxs < 0  # (B, T, n_topk)
    safe_idxs = topk_idxs.clamp(min=0)  # (B, T, n_topk)
    # Expand kv to (B, T, n_kv, c) broadcasted, then gather along the n_kv axis.
    expanded_idxs = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, c)  # (B, T, n_topk, c)
    kv_expanded = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)  # (B, T, n_kv, c)
    gathered = torch.gather(kv_expanded, dim=2, index=expanded_idxs)  # (B, T, n_topk, c)

    # Scores in fp32 for numerical parity with the reference kernel
    # (which accumulates in fp32 even when q/k are bf16).
    scores = torch.einsum("bthd,btkd->bhtk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(is_padding.unsqueeze(1), float("-inf"))

    # Concat one sink "key" per head: logit = sink_logits[h], value = 0.
    sink_col = sink_logits.float().view(1, n_h, 1, 1).expand(bsz, -1, seqlen, 1)  # (B, n_h, T, 1)
    scores_with_sink = torch.cat([scores, sink_col], dim=-1)  # (B, n_h, T, n_topk + 1)

    weights = scores_with_sink.softmax(dim=-1)  # fp32 softmax
    weights = weights[..., :-1]  # drop the sink column (its value is 0)

    # Output = weights · V (V = gathered, same as K).
    out = torch.einsum("bhtk,btkd->bthd", weights, gathered.float())
    return out.to(q.dtype)
