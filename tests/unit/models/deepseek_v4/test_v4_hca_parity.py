"""HF parity: HCAAttention vs DeepSeek-V4-Pro reference inference code.

The strong correctness gate for HCA. Builds the reference's unified
`Attention` module on a small synthetic config (compress_ratio in
`(128,)` forces the HCA path with no Indexer), syncs weights tensor-
by-tensor to our `HCAAttention`, runs both forwards on the same input,
and asserts cosine-sim > 0.999.

What this catches:
    - Compressor formula correctness (positional bias, softmax axis,
      RoPE position assignment for compressed entries).
    - Attention-sink semantics (per-head logit added to softmax denom).
    - Partial RoPE on the right dim slice; inverse output RoPE direction.
    - Q-norm + K-norm placement relative to RoPE.
    - Grouped output projection (per-group einsum vs monolithic).

The kernel + Hadamard stubbing and the `reference_module` fixture
live in `_v4_reference_helpers`, shared with the CSA parity test.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.nn.functional import cosine_similarity

from mini_infer.models.blocks.rope import RotaryEmbedding
from mini_infer.models.deepseek_v4.hca import HCAAttention


def _build_synthetic_args(reference_module: Any) -> Any:
    """Small HCA-only config that exercises every code path on CPU in <1s."""
    return reference_module.ModelArgs(
        max_batch_size=2,
        max_seq_len=512,
        dtype="bf16",  # not used (default_dtype patched to fp32)
        dim=128,
        n_layers=1,
        n_heads=4,
        q_lora_rank=64,
        head_dim=64,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=64,
        window_size=32,
        compress_ratios=(128,),
        original_seq_len=0,  # disable YaRN
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
    )


def _init_reference_attention(ref_attn: Any, seed: int) -> None:
    """Replace `torch.empty(...)` parameters with deterministic random values.

    The reference uses `nn.Parameter(torch.empty(...))` everywhere — those
    tensors hold uninitialized memory. For a meaningful parity test, we
    seed-fill them so the same module configuration produces the same
    weights every run, and so we have non-NaN values to copy into our
    HCAAttention.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for p in ref_attn.parameters():
        with torch.no_grad():
            p.data = torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.02
    for buf in ref_attn.buffers():
        if buf.dtype.is_floating_point:
            buf.zero_()


def _sync_weights(our_block: HCAAttention, ref_attn: Any) -> None:
    """Copy reference parameters into our HCAAttention, name-by-name."""
    with torch.no_grad():
        # --- Q low-rank ---
        our_block.q_a_proj.weight.copy_(ref_attn.wq_a.weight)
        our_block.q_a_layernorm.weight.copy_(ref_attn.q_norm.weight)
        our_block.q_b_proj.weight.copy_(ref_attn.wq_b.weight)
        # --- SWA K=V branch ---
        our_block.swa_kv_proj.weight.copy_(ref_attn.wkv.weight)
        our_block.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        # --- Compressor ---
        our_block.compressor.kv_proj.weight.copy_(ref_attn.compressor.wkv.weight)
        our_block.compressor.weight_proj.weight.copy_(ref_attn.compressor.wgate.weight)
        our_block.compressor.position_bias.copy_(ref_attn.compressor.ape)
        our_block.compressor.norm.weight.copy_(ref_attn.compressor.norm.weight)
        # --- Sink ---
        our_block.sink.sink_logits.copy_(ref_attn.attn_sink)
        # --- Grouped output projection ---
        our_block.grouped_output.wo_a.copy_(ref_attn.wo_a.weight)
        our_block.grouped_output.wo_b.weight.copy_(ref_attn.wo_b.weight)


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


def test_hca_block_matches_v4_reference(reference_module: Any) -> None:
    """Cosine-sim > 0.999 between our HCAAttention and the reference Attention."""
    torch.manual_seed(0)
    args = _build_synthetic_args(reference_module)

    ref_attn = reference_module.Attention(0, args)
    _init_reference_attention(ref_attn, seed=42)

    our_block = HCAAttention(
        hidden_size=args.dim,
        num_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        num_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        window_size=args.window_size,
        compression_ratio=args.compress_ratios[0],
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


def test_hca_block_handles_uneven_seqlen_assertion(reference_module: Any) -> None:
    """Our standalone block requires `seqlen % m == 0` and rejects otherwise.

    Different from the reference (which handles tail tokens via state buffers
    during incremental compression). Documents the standalone-block contract.
    """
    args = _build_synthetic_args(reference_module)
    our_block = HCAAttention(
        hidden_size=args.dim,
        num_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        num_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        window_size=args.window_size,
        compression_ratio=args.compress_ratios[0],
        rms_norm_eps=args.norm_eps,
    )
    bsz, seqlen = 1, args.compress_ratios[0] + 1  # not a multiple of m
    x = torch.randn(bsz, seqlen, args.dim)
    cos = torch.zeros(bsz, seqlen, args.rope_head_dim)
    sin = torch.zeros(bsz, seqlen, args.rope_head_dim)
    with pytest.raises(ValueError, match="multiple of compression_ratio"):
        our_block(x, (cos, sin), (cos[:, :1], sin[:, :1]))
