"""HF parity: CSAAttention vs DeepSeek-V4-Pro reference inference code.

Cousin of `test_v4_hca_parity.py` but for the CSA path:
`compress_ratios=(4,)` triggers the reference's overlap compressor +
Lightning Indexer + top-k branch. We sync weights tensor-by-tensor
to our `CSAAttention` and assert cosine-sim > 0.999.

What this catches (in addition to the HCA parity coverage):
    - Overlap compressor: doubled `wkv` / `wgate` widths, `2m` softmax
      slots, block-0 padding, position-bias split between current and
      overlap halves.
    - Lightning Indexer: per-head dot product, ReLU before head sum,
      causal mask on compressed positions, `top_k = min(top_k, n_compressed)`
      capping, `-1` rewrite for masked-future picks.
    - Indexer's compressor running with `rotate=True` (Hadamard
      patched to identity in the reference; same math at fp32).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.functional import cosine_similarity

from mini_infer.models.blocks import CSAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding


def _build_synthetic_args(reference_module: Any) -> Any:
    """Small CSA-only config: m=4 with overlap, top-k indexer, short seq."""
    return reference_module.ModelArgs(
        max_batch_size=2,
        max_seq_len=64,
        dtype="bf16",
        dim=64,
        n_layers=1,
        n_heads=4,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(4,),
        original_seq_len=0,
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
    )


def _init_reference_attention(ref_attn: Any, seed: int) -> None:
    """Seed-fill `nn.Parameter(torch.empty(...))` weights and zero buffers."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for p in ref_attn.parameters():
        with torch.no_grad():
            p.data = torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.02
    for buf in ref_attn.buffers():
        if buf.dtype.is_floating_point:
            buf.zero_()


def _sync_weights(our_block: CSAAttention, ref_attn: Any) -> None:
    """Copy reference parameters into our CSAAttention, name-by-name."""
    with torch.no_grad():
        # --- Q low-rank ---
        our_block.q_a_proj.weight.copy_(ref_attn.wq_a.weight)
        our_block.q_a_layernorm.weight.copy_(ref_attn.q_norm.weight)
        our_block.q_b_proj.weight.copy_(ref_attn.wq_b.weight)
        # --- SWA K=V ---
        our_block.swa_kv_proj.weight.copy_(ref_attn.wkv.weight)
        our_block.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        # --- Main compressor (overlap mode, m=4) ---
        our_block.compressor.kv_proj.weight.copy_(ref_attn.compressor.wkv.weight)
        our_block.compressor.weight_proj.weight.copy_(ref_attn.compressor.wgate.weight)
        our_block.compressor.position_bias.copy_(ref_attn.compressor.ape)
        our_block.compressor.norm.weight.copy_(ref_attn.compressor.norm.weight)
        # --- Sink + grouped output projection ---
        our_block.sink.sink_logits.copy_(ref_attn.attn_sink)
        our_block.grouped_output.wo_a.copy_(ref_attn.wo_a.weight)
        our_block.grouped_output.wo_b.weight.copy_(ref_attn.wo_b.weight)
        # --- Lightning Indexer ---
        our_block.indexer.wq_b.weight.copy_(ref_attn.indexer.wq_b.weight)
        our_block.indexer.weights_proj.weight.copy_(ref_attn.indexer.weights_proj.weight)
        # Indexer's own compressor (overlap=True with rotate=True in reference;
        # we patch rotate to identity so our non-rotate compressor matches).
        our_block.indexer.compressor.kv_proj.weight.copy_(ref_attn.indexer.compressor.wkv.weight)
        our_block.indexer.compressor.weight_proj.weight.copy_(
            ref_attn.indexer.compressor.wgate.weight
        )
        our_block.indexer.compressor.position_bias.copy_(ref_attn.indexer.compressor.ape)
        our_block.indexer.compressor.norm.weight.copy_(ref_attn.indexer.compressor.norm.weight)


def _build_position_embeddings(
    rotary: RotaryEmbedding,
    bsz: int,
    seqlen: int,
    compression_ratio: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Compute (cos, sin) for both raw token positions and compressed positions."""
    token_positions = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    cos_t, sin_t = rotary(torch.zeros(bsz, seqlen, device=device), token_positions)

    n_compressed = seqlen // compression_ratio
    compressed_positions = (
        (torch.arange(n_compressed, device=device) * compression_ratio).unsqueeze(0).expand(bsz, -1)
    )
    cos_c, sin_c = rotary(torch.zeros(bsz, n_compressed, device=device), compressed_positions)
    return (cos_t, sin_t), (cos_c, sin_c)


def test_csa_block_matches_v4_reference(reference_module: Any) -> None:
    """Cosine-sim > 0.999 between our CSAAttention and the reference Attention(m=4)."""
    torch.manual_seed(0)
    args = _build_synthetic_args(reference_module)

    ref_attn = reference_module.Attention(0, args)
    _init_reference_attention(ref_attn, seed=42)

    our_block = CSAAttention(
        hidden_size=args.dim,
        num_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        num_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        window_size=args.window_size,
        compression_ratio=args.compress_ratios[0],
        index_num_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        index_top_k=args.index_topk,
        rms_norm_eps=args.norm_eps,
    )
    _sync_weights(our_block, ref_attn)

    bsz, seqlen = 2, args.max_seq_len
    x = torch.randn(bsz, seqlen, args.dim) * 0.5
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_pe, compressed_pe = _build_position_embeddings(
        rotary, bsz, seqlen, args.compress_ratios[0], device=x.device
    )

    with torch.no_grad():
        ours = our_block(x, token_pe, compressed_pe)
        theirs = ref_attn(x, start_pos=0)

    assert ours.shape == theirs.shape == (bsz, seqlen, args.dim)
    cos_sim = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0)
    max_diff = (ours - theirs).abs().max().item()
    rel_err = max_diff / max(theirs.abs().max().item(), 1e-9)
    assert cos_sim > 0.999, (
        f"cosine_sim={cos_sim:.6f}, max_abs_diff={max_diff:.3e}, "
        f"rel_err={rel_err:.3e} (target > 0.999)"
    )
