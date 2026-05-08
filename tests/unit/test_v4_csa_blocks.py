"""Shape and math tests for CSA-specific primitives.

The HF parity test (`test_v4_csa_parity.py`) is the strong correctness
gate. These tests cover behaviors that can drift silently while iterating
on the parity test:

  - `TokenLevelCompressor(overlap_mode=True)`: doubled projection widths,
    softmax over `2m` slots, block-0 overlap pad.
  - `LightningIndexer`: shape contract, causal masking semantics, top-k
    masking when fewer-than-k valid entries exist.
  - `CSAAttention`: end-to-end shape on a small synthetic config.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.blocks import (
    CSAAttention,
    LightningIndexer,
    TokenLevelCompressor,
)
from mini_infer.models.blocks.rope import RotaryEmbedding

# ---------- Overlap-mode compressor ----------


def test_overlap_compressor_emits_one_entry_per_block() -> None:
    """`(B=2, T=32, d=16)` with `m=4, overlap_mode=True` -> compressed shape `(2, 8, c)`."""
    torch.manual_seed(0)
    comp = TokenLevelCompressor(
        hidden_size=16,
        kv_head_dim=8,
        rope_head_dim=4,
        compression_ratio=4,
        overlap_mode=True,
    )
    rope = RotaryEmbedding(head_dim=4)
    x = torch.randn(2, 32, 16)
    n_blocks = 32 // 4
    block_pos = (torch.arange(n_blocks) * 4).unsqueeze(0)
    cos, sin = rope(x[:, :n_blocks], block_pos)
    out = comp(x, (cos.expand(2, -1, -1), sin.expand(2, -1, -1)))
    assert out.shape == (2, n_blocks, 8)
    assert torch.all(torch.isfinite(out))


def test_overlap_compressor_widens_projection_outputs() -> None:
    """`overlap_mode=True` doubles the kv_proj/weight_proj output width."""
    plain = TokenLevelCompressor(
        hidden_size=8, kv_head_dim=4, rope_head_dim=0, compression_ratio=2, overlap_mode=False
    )
    overlap = TokenLevelCompressor(
        hidden_size=8, kv_head_dim=4, rope_head_dim=0, compression_ratio=2, overlap_mode=True
    )
    assert plain.kv_proj.out_features == 4
    assert overlap.kv_proj.out_features == 8
    assert plain.position_bias.shape == (2, 4)
    assert overlap.position_bias.shape == (2, 8)


def test_overlap_decode_step_writes_to_current_half_slot() -> None:
    """`forward_decode_step` with `overlap_mode=True` writes to slot `m + (start_pos % m)`.

    The "current" half of the in-flight buffer lives in slots `[m, 2m)`;
    slots `[0, m)` are reserved for the previous block's overlap data.
    """
    torch.manual_seed(0)
    compression_ratio = 4
    kv_head_dim = 4
    compressor = TokenLevelCompressor(
        hidden_size=8,
        kv_head_dim=kv_head_dim,
        rope_head_dim=0,
        compression_ratio=compression_ratio,
        overlap_mode=True,
    )
    cmp_kv_state = torch.zeros(1, 2 * compression_ratio, 2 * kv_head_dim)
    cmp_score_state = torch.full((1, 2 * compression_ratio, 2 * kv_head_dim), float("-inf"))
    hidden = torch.randn(1, 1, 8)
    flushed = compressor.forward_decode_step(
        hidden,
        start_pos=2,  # pos_in_block = 2, target slot = m + 2 = 6
        cmp_kv_state=cmp_kv_state,
        cmp_score_state=cmp_score_state,
    )
    # Mid-block: no flush.
    assert flushed is None
    # Slot 6 was written; slots 0..5 and 7 are unchanged.
    assert torch.any(cmp_kv_state[0, 6] != 0)
    assert torch.all(cmp_kv_state[0, :6] == 0)
    assert torch.all(cmp_kv_state[0, 7] == 0)
    # Score state: slot 6 written; others stay at -inf.
    assert torch.all(torch.isfinite(cmp_score_state[0, 6]))
    finite_count = torch.isfinite(cmp_score_state[0]).all(dim=-1).sum().item()
    assert finite_count == 1


def test_overlap_decode_matches_prefill_within_one_run() -> None:
    """Strong test: 2 blocks via 8 decode steps == compressor.forward over 8 tokens.

    Validates:
      - slot indexing `m + start_pos % m`
      - flush trigger on `(start_pos + 1) % m == 0`
      - the slide that turns the just-completed block into the "previous overlap"
      - softmax dimension and position-bias indexing match prefill
    """
    torch.manual_seed(0)
    compression_ratio = 4
    kv_head_dim = 8
    rope_head_dim = 4
    seq_len = 8

    compressor = TokenLevelCompressor(
        hidden_size=16,
        kv_head_dim=kv_head_dim,
        rope_head_dim=rope_head_dim,
        compression_ratio=compression_ratio,
        overlap_mode=True,
    )
    rope = RotaryEmbedding(head_dim=rope_head_dim)
    hidden_states = torch.randn(1, seq_len, 16) * 0.5

    # Prefill path: produce both compressed entries in one go.
    n_blocks = seq_len // compression_ratio
    block_positions = torch.arange(n_blocks).unsqueeze(0) * compression_ratio
    cos_blocks, sin_blocks = rope(hidden_states[:, :n_blocks], block_positions)
    with torch.no_grad():
        prefill_compressed = compressor(hidden_states, (cos_blocks, sin_blocks))
    assert prefill_compressed.shape == (1, n_blocks, kv_head_dim)

    # Decode path: 8 steps, collect each flushed entry.
    cmp_kv_state = torch.zeros(1, 2 * compression_ratio, 2 * kv_head_dim)
    cmp_score_state = torch.full((1, 2 * compression_ratio, 2 * kv_head_dim), float("-inf"))
    decode_compressed = []
    with torch.no_grad():
        for token_position in range(seq_len):
            block_pe = None
            if (token_position + 1) % compression_ratio == 0:
                # Block flush this step. Position is (block_idx) * compression_ratio.
                block_idx = token_position // compression_ratio
                block_pos = torch.tensor([[block_idx * compression_ratio]])
                block_pe = rope(torch.zeros(1, 1), block_pos)
            flushed = compressor.forward_decode_step(
                hidden_states[:, token_position : token_position + 1],
                start_pos=token_position,
                cmp_kv_state=cmp_kv_state,
                cmp_score_state=cmp_score_state,
                block_position_embeddings=block_pe,
            )
            if flushed is not None:
                decode_compressed.append(flushed.squeeze(1))
    assert len(decode_compressed) == n_blocks

    decode_compressed_tensor = torch.stack(decode_compressed, dim=1)
    torch.testing.assert_close(decode_compressed_tensor, prefill_compressed, rtol=1e-5, atol=1e-6)


def test_overlap_block_zero_uses_padded_overlap_slots() -> None:
    """The first compressed block has no predecessor — its overlap slots are masked.

    Constructed so that the "current half" of block 0's softmax is zeroed out
    in `kv` (only its overlap slots survive). With overlap pad set to 0
    (kv) and -inf (score), the softmax for block 0 puts zero mass on overlap
    slots and the compressed output equals zero. Block 1 sees block 0's
    "first half" of `kv` in its overlap slots and produces a non-zero output.
    """
    torch.manual_seed(0)
    m, c = 2, 4
    comp = TokenLevelCompressor(
        hidden_size=4, kv_head_dim=c, rope_head_dim=0, compression_ratio=m, overlap_mode=True
    )
    # Force the "current" half of kv_proj to zero so block 0 has only padded slots
    # contributing non-zero kv (and those padded slots are zeroed too).
    with torch.no_grad():
        # kv_proj.weight: (2c, hidden_size). Rows [c, 2c) feed the "current" half;
        # zero them so the current-half kv is zero.
        comp.kv_proj.weight[c:].zero_()
        # Force uniform softmax: zero score weights + zero position bias, but the
        # overlap_transform fills overlap slots with -inf for block 0 so its softmax
        # weight on those slots is zero (since exp(-inf) = 0).
        comp.weight_proj.weight.zero_()
        comp.norm.weight.fill_(1.0)

    x = torch.randn(1, 4, 4)
    cos = torch.zeros(1, 2, 0)
    sin = torch.zeros(1, 2, 0)
    out = comp(x, (cos, sin))
    # Block 0: overlap slots padded to (0, -inf), current slots have kv=0. All slots
    # contribute zero -> output is zero (then RMSNorm with eps gives 0/eps_correction).
    # RMSNorm of zero input is zero (var=0, output = 0 * rsqrt(eps)).
    assert torch.allclose(out[:, 0], torch.zeros_like(out[:, 0]), atol=1e-6)
    # Block 1: overlap slots take block 0's "first half" of kv (non-zero), current slots
    # are zero. Output is non-zero.
    assert not torch.allclose(out[:, 1], torch.zeros_like(out[:, 1]))


# ---------- LightningIndexer ----------


def test_indexer_returns_topk_indices_per_query() -> None:
    """Output `(B, T, top_k)` int64; values in `[offset, offset + n_compressed)` or `-1`."""
    torch.manual_seed(0)
    bsz, seqlen, hidden = 2, 16, 32
    m, top_k = 4, 3
    indexer = LightningIndexer(
        hidden_size=hidden,
        q_lora_rank=16,
        num_heads=2,
        head_dim=8,
        rope_head_dim=4,
        compression_ratio=m,
        top_k=top_k,
    )
    x = torch.randn(bsz, seqlen, hidden)
    q_lora = torch.randn(bsz, seqlen, 16)
    rope = RotaryEmbedding(head_dim=4)
    cos_t, sin_t = rope(x, torch.arange(seqlen).unsqueeze(0).expand(bsz, -1))
    n_cmp = seqlen // m
    cos_c, sin_c = rope(x[:, :n_cmp], (torch.arange(n_cmp) * m).unsqueeze(0).expand(bsz, -1))
    offset = 64
    idxs = indexer(x, q_lora, (cos_t, sin_t), (cos_c, sin_c), compressed_offset=offset)
    assert idxs.shape == (bsz, seqlen, top_k)
    assert idxs.dtype == torch.int64
    valid = idxs >= 0
    if valid.any():
        assert (idxs[valid] >= offset).all()
        assert (idxs[valid] < offset + n_cmp).all()


def test_indexer_early_queries_see_no_valid_compressed_blocks() -> None:
    """Queries before position `m-1` cannot see any compressed block (cutoff = 0)."""
    torch.manual_seed(0)
    bsz, seqlen, hidden = 1, 8, 16
    m, top_k = 4, 2
    indexer = LightningIndexer(
        hidden_size=hidden,
        q_lora_rank=8,
        num_heads=2,
        head_dim=8,
        rope_head_dim=0,
        compression_ratio=m,
        top_k=top_k,
    )
    x = torch.randn(bsz, seqlen, hidden)
    q_lora = torch.randn(bsz, seqlen, 8)
    cos_t = torch.zeros(bsz, seqlen, 0)
    sin_t = torch.zeros(bsz, seqlen, 0)
    n_cmp = seqlen // m
    cos_c = torch.zeros(bsz, n_cmp, 0)
    sin_c = torch.zeros(bsz, n_cmp, 0)
    idxs = indexer(x, q_lora, (cos_t, sin_t), (cos_c, sin_c), compressed_offset=0)
    # Cutoff(t) = (t+1) // m. For t in [0, m-1), cutoff = 0 -> no valid blocks.
    # At t = m-1, cutoff = 1 -> block 0 is valid; the indexer returns its index, not -1.
    assert (idxs[:, : m - 1] == -1).all()
    # And position m-1 must not be all -1 (block 0 is reachable).
    assert (idxs[:, m - 1] != -1).any()


def test_indexer_late_query_has_all_valid_compressed_blocks() -> None:
    """Last query sees every compressed block; if `top_k >= n_compressed`, no `-1`."""
    torch.manual_seed(0)
    bsz, seqlen, hidden = 1, 16, 16
    m, top_k = 4, 4  # top_k == n_compressed (16/4)
    indexer = LightningIndexer(
        hidden_size=hidden,
        q_lora_rank=8,
        num_heads=2,
        head_dim=8,
        rope_head_dim=0,
        compression_ratio=m,
        top_k=top_k,
    )
    x = torch.randn(bsz, seqlen, hidden)
    q_lora = torch.randn(bsz, seqlen, 8)
    cos_t = torch.zeros(bsz, seqlen, 0)
    sin_t = torch.zeros(bsz, seqlen, 0)
    n_cmp = seqlen // m
    cos_c = torch.zeros(bsz, n_cmp, 0)
    sin_c = torch.zeros(bsz, n_cmp, 0)
    idxs = indexer(x, q_lora, (cos_t, sin_t), (cos_c, sin_c), compressed_offset=100)
    # Last query (position seqlen-1=15): cutoff = (15+1)//4 = 4 = n_cmp. All blocks valid.
    assert (idxs[:, -1] != -1).all()
    # Each picked index must be in [100, 104).
    assert (idxs[:, -1] >= 100).all()
    assert (idxs[:, -1] < 104).all()


def test_indexer_rejects_zero_topk() -> None:
    with pytest.raises(ValueError, match="top_k"):
        LightningIndexer(
            hidden_size=8,
            q_lora_rank=4,
            num_heads=2,
            head_dim=4,
            rope_head_dim=0,
            compression_ratio=2,
            top_k=0,
        )


# ---------- CSAAttention shape ----------


def test_csa_attention_forward_runs_and_returns_correct_shape() -> None:
    torch.manual_seed(0)
    bsz, seqlen, hidden = 1, 32, 64
    block = CSAAttention(
        hidden_size=hidden,
        num_heads=4,
        q_lora_rank=32,
        kv_head_dim=16,
        rope_head_dim=8,
        num_groups=2,
        o_lora_rank=16,
        window_size=8,
        compression_ratio=4,
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=4,
    )
    x = torch.randn(bsz, seqlen, hidden)
    rope = RotaryEmbedding(head_dim=8)
    cos_t, sin_t = rope(x, torch.arange(seqlen).unsqueeze(0))
    n_cmp = seqlen // 4
    cos_c, sin_c = rope(x[:, :n_cmp], (torch.arange(n_cmp) * 4).unsqueeze(0))
    out = block(x, (cos_t, sin_t), (cos_c, sin_c))
    assert out.shape == (bsz, seqlen, hidden)
    assert torch.all(torch.isfinite(out))
