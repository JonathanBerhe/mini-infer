"""MiniMax-M3 MSA: partial RoPE, the block indexer, and the block mask.

The indexer applies partial RoPE (width `rotary_dim` tables shared with the
main branch; the first dims rotate, the tail passes through), matching HF's
slice-to-cos-width apply under the deployment config's partial_rotary_factor.

Validated against independent loop-based references (so a vectorization bug in
the module doesn't hide behind the same vectorization in the check), plus the
strong invariant that when every block is selectable MSA collapses to plain
causal attention. Authoritative HF-vs-ours parity (numbers, not just mechanism)
is the model-level harness in a later phase; here we pin the mechanism.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from mini_infer.models.blocks.minimax_m3_indexer import MiniMaxM3Indexer
from mini_infer.models.blocks.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb_partial,
)


def _rope_tables(
    rotary_dim: int, seqlen: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Width-`rotary_dim` cos/sin for positions [0, seqlen)."""
    rope = RotaryEmbedding(head_dim=rotary_dim, base=base)
    hidden = torch.zeros(1, seqlen, rotary_dim)
    position_ids = torch.arange(seqlen).unsqueeze(0)
    return rope(hidden, position_ids)


def test_apply_rotary_pos_emb_partial_rotates_first_n() -> None:
    """Rotate the leading `rotary_dim` dims; pass the trailing dims unchanged."""
    torch.manual_seed(0)
    head_dim, rotary_dim, seqlen = 8, 4, 6
    q = torch.randn(1, 2, seqlen, head_dim)
    k = torch.randn(1, 2, seqlen, head_dim)
    cos, sin = _rope_tables(rotary_dim, seqlen)

    q_out, k_out = apply_rotary_pos_emb_partial(q, k, cos, sin)
    # Trailing dims pass through untouched.
    assert torch.equal(q_out[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_out[..., rotary_dim:], k[..., rotary_dim:])
    # Leading dims equal a full rope over just that slice.
    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    expected_q = q[..., :rotary_dim] * cos_u + rotate_half(q[..., :rotary_dim]) * sin_u
    assert torch.allclose(q_out[..., :rotary_dim], expected_q, atol=1e-6)
    # Position 0 is identity (cos=1, sin=0).
    assert torch.allclose(q_out[:, :, 0], q[:, :, 0], atol=1e-6)


def _make_indexer(seed: int = 0) -> MiniMaxM3Indexer:
    torch.manual_seed(seed)
    return MiniMaxM3Indexer(
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        block_size=4,
        topk_blocks=2,
        num_query_heads=4,
        local_blocks=1,
    ).eval()


def _selected_sets(block_indices: torch.Tensor) -> list[list[set[int]]]:
    """Per (index head, query) set of selected real blocks (drop the -1 padding)."""
    out: list[list[set[int]]] = []
    for head in range(block_indices.shape[1]):
        out.append(
            [
                {int(b) for b in block_indices[0, head, i].tolist() if b >= 0}
                for i in range(block_indices.shape[2])
            ]
        )
    return out


def _reference_selection(idxer: MiniMaxM3Indexer, hidden: torch.Tensor, cos, sin, position_ids):
    """Loop-based reference for the indexer's per-(head, query) selected-block SET.

    Scores each (head, query, block) as the max over causally-valid tokens in
    the block of the raw idx_q.idx_k dot (no scale), forces the local block(s),
    takes the top-k per head (transformers 5.14 semantics: one selection per
    index head, no head pooling), and drops padding. Set-valued so it is robust
    to top-k tie-breaks among equal / -inf scores (only the mask-relevant set
    matters).
    """
    bsz, seqlen, _ = hidden.shape
    h, d, bs = idxer.num_heads, idxer.head_dim, idxer.block_size
    idx_q = idxer.q_norm(idxer.q_proj(hidden).view(bsz, seqlen, h, d)).transpose(1, 2)
    idx_k = idxer.k_norm(idxer.k_proj(hidden).view(bsz, seqlen, 1, d)).transpose(1, 2)
    idx_q, idx_k = apply_rotary_pos_emb_partial(idx_q, idx_k, cos, sin)
    idx_q, idx_k = idx_q.detach().float(), idx_k.detach().float()
    num_blocks = -(-seqlen // bs)
    sets: list[list[set[int]]] = []
    for head in range(h):
        head_sets: list[set[int]] = []
        for i in range(seqlen):
            qpos = int(position_ids[0, i])
            bscore = [float("-inf")] * num_blocks
            for blk in range(num_blocks):
                best = float("-inf")
                for j in range(blk * bs, min((blk + 1) * bs, seqlen)):
                    if j <= qpos:
                        best = max(best, float(idx_q[0, head, i] @ idx_k[0, 0, j]))
                bscore[blk] = best
            for local in range(idxer.local_blocks):
                bscore[max(qpos // bs - local, 0)] = float("inf")
            topk = min(idxer.topk_blocks, num_blocks)
            order = sorted(range(num_blocks), key=lambda b: bscore[b], reverse=True)[:topk]
            head_sets.append({b for b in order if bscore[b] != float("-inf")})
        sets.append(head_sets)
    return sets


def test_indexer_selection_matches_loop_reference() -> None:
    idxer = _make_indexer()
    torch.manual_seed(1)
    seqlen = 10
    hidden = torch.randn(1, seqlen, 16)
    position_ids = torch.arange(seqlen).unsqueeze(0)
    cos, sin = _rope_tables(4, seqlen)

    block_indices = idxer(hidden, cos, sin, position_ids)
    assert block_indices.shape == (1, 2, seqlen, 2)
    assert _selected_sets(block_indices) == _reference_selection(
        idxer, hidden, cos, sin, position_ids
    )
    # Every (head, query) includes its own (local) block, and never a future block.
    for head in range(2):
        for i in range(seqlen):
            sel = {int(b) for b in block_indices[0, head, i].tolist() if b >= 0}
            assert i // 4 in sel
            assert all(b <= i // 4 for b in sel)


def test_build_block_mask_matches_reference() -> None:
    idxer = _make_indexer()
    torch.manual_seed(2)
    seqlen = 10
    hidden = torch.randn(1, seqlen, 16)
    position_ids = torch.arange(seqlen).unsqueeze(0)
    cos, sin = _rope_tables(4, seqlen)

    block_indices = idxer(hidden, cos, sin, position_ids)
    mask = idxer.build_block_mask(block_indices, seqlen, position_ids, dtype=torch.float32)
    # One mask row per QUERY head; each index head's selection covers its
    # GQA group (num_query_heads=4, 2 index heads -> group size 2).
    assert mask.shape == (1, 4, seqlen, seqlen)

    min_val = torch.finfo(torch.float32).min
    sel = _selected_sets(block_indices)
    group = idxer.num_query_heads // idxer.num_heads
    for qh in range(4):
        idx_head = qh // group
        for i in range(seqlen):
            for j in range(seqlen):
                keep = (j <= i) and (j // idxer.block_size in sel[idx_head][i])
                assert mask[0, qh, i, j].item() == (0.0 if keep else min_val)


def test_msa_collapses_to_causal_when_all_blocks_selected() -> None:
    """topk_blocks >= num_blocks -> every visible block selectable -> the block
    mask equals the plain causal additive mask, so MSA == standard causal GQA."""
    torch.manual_seed(3)
    seqlen, block_size = 12, 4  # 3 blocks
    idxer = MiniMaxM3Indexer(
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        block_size=block_size,
        topk_blocks=8,
        num_query_heads=4,
        local_blocks=1,
    ).eval()
    hidden = torch.randn(1, seqlen, 16)
    position_ids = torch.arange(seqlen).unsqueeze(0)
    cos, sin = _rope_tables(4, seqlen)

    block_indices = idxer(hidden, cos, sin, position_ids)
    mask = idxer.build_block_mask(block_indices, seqlen, position_ids, dtype=torch.float32)

    min_val = torch.finfo(torch.float32).min
    causal = torch.zeros(1, 4, seqlen, seqlen)
    causal = causal.masked_fill(
        torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool), 1), min_val
    )
    assert torch.equal(mask, causal)

    # And end to end: attention with this mask == SDPA causal attention.
    q = torch.randn(1, 4, seqlen, 8)
    kv = torch.randn(1, 4, seqlen, 8)
    scale = 8**-0.5
    scores = (q @ kv.transpose(-1, -2)) * scale + mask
    ours = torch.softmax(scores, dim=-1, dtype=torch.float32) @ kv
    ref = functional.scaled_dot_product_attention(q, kv, kv, is_causal=True, scale=scale)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_topk_smaller_than_blocks_actually_prunes() -> None:
    """With topk < num_blocks, a late query masks out at least one visible block."""
    idxer = _make_indexer(seed=5)
    torch.manual_seed(6)
    seqlen = 12  # 3 blocks, topk=2
    hidden = torch.randn(1, seqlen, 16)
    position_ids = torch.arange(seqlen).unsqueeze(0)
    cos, sin = _rope_tables(4, seqlen)

    block_indices = idxer(hidden, cos, sin, position_ids)
    mask = idxer.build_block_mask(block_indices, seqlen, position_ids, dtype=torch.float32)
    min_val = torch.finfo(torch.float32).min
    # The last query sees 3 blocks but keeps only 2 -> some causally-visible key is masked.
    last = seqlen - 1
    visible_but_masked = [j for j in range(last + 1) if mask[0, 0, last, j].item() == min_val]
    assert visible_but_masked, "topk<num_blocks should prune at least one visible key"
