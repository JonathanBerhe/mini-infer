"""HF parity: HCAAttention decode vs DeepSeek-V4-Pro reference.

Builds the reference `Attention(compress_ratio=8)` on a small synthetic
config, runs prefill once to populate its kv_cache, then runs N=8
decode steps. Mirrors each step on our `HCAAttention.forward_decode`
backed by a `StateCache` synced from the reference's kv_cache. Asserts
cosine-sim > 0.999 at EVERY step.

What this catches (beyond the prefill parity):
    - Compressor's incremental state machinery: `kv_state` /
      `score_state` slot-indexing by `start_pos % m`, flush trigger at
      `(start_pos + 1) % m == 0`, RoPE position assignment for the
      newly-flushed compressed entry.
    - SWA circular buffer: write at `start_pos % n_win`, the `topk_idxs`
      that wrap when `start_pos >= n_win - 1`.
    - The decode-time `topk_idxs` schema: window section uses circular
      indices, compressed section is `[offset, offset + (start_pos+1)//m)`.
    - StateCache <-> reference kv_cache layout match: when sync at
      end-of-prefill is done correctly, the per-step decode is bit-stable.

Aligned prefill only (`T % m == 0`) so the compressor's `kv_state` and
`score_state` start at their initial defaults (zeros / -inf) matching
ours. The unaligned case adds a sync of `[:remainder]` slots — left for
a follow-up to keep this test focused on the per-step decode math.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.nn.functional import cosine_similarity

from mini_infer.cache.state_cache import StateCache, StateLayerSpec
from mini_infer.models.blocks import HCAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding


def _build_synthetic_args(reference_module: Any) -> Any:
    """Small HCA-only config: m=8, n_win=8, t_prefill=16, decode 8 steps."""
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
        compress_ratios=(8,),
        original_seq_len=0,
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
    )


def _init_reference_attention(ref_attn: Any, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for p in ref_attn.parameters():
        with torch.no_grad():
            p.data = torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.02
    for buf in ref_attn.buffers():
        if buf.dtype.is_floating_point:
            buf.zero_()


def _build_our_block(args: Any) -> HCAAttention:
    return HCAAttention(
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


def _sync_weights(our_block: HCAAttention, ref_attn: Any) -> None:
    with torch.no_grad():
        our_block.q_a_proj.weight.copy_(ref_attn.wq_a.weight)
        our_block.q_a_layernorm.weight.copy_(ref_attn.q_norm.weight)
        our_block.q_b_proj.weight.copy_(ref_attn.wq_b.weight)
        our_block.swa_kv_proj.weight.copy_(ref_attn.wkv.weight)
        our_block.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        our_block.compressor.kv_proj.weight.copy_(ref_attn.compressor.wkv.weight)
        our_block.compressor.weight_proj.weight.copy_(ref_attn.compressor.wgate.weight)
        our_block.compressor.position_bias.copy_(ref_attn.compressor.ape)
        our_block.compressor.norm.weight.copy_(ref_attn.compressor.norm.weight)
        our_block.sink.sink_logits.copy_(ref_attn.attn_sink)
        our_block.grouped_output.wo_a.copy_(ref_attn.wo_a.weight)
        our_block.grouped_output.wo_b.weight.copy_(ref_attn.wo_b.weight)


def _sync_state_after_prefill(
    ref_attn: Any, state_cache: StateCache, *, t_prefill: int, args: Any
) -> None:
    """After reference prefill, copy its kv_cache + compressor state into ours.

    Reference layout (`Attention.kv_cache` shape `(B, win + max_seq_len // m, c)`):
        slots [0, n_win): SWA circular buffer (left-aligned for `T <= n_win`,
                          rotated for `T > n_win` — both equivalent to
                          `slot = pos % n_win` for the LAST n_win positions).
        slots [n_win, n_win + n_completed): compressed history.
    """
    n_win = args.window_size
    m = args.compress_ratios[0]
    layer = state_cache.layer(0)
    bsz = layer.swa_kv.shape[0]
    layer.swa_kv.copy_(ref_attn.kv_cache[:bsz, :n_win])
    n_completed = t_prefill // m
    layer.compressed_kv[:, :n_completed].copy_(ref_attn.kv_cache[:bsz, n_win : n_win + n_completed])
    layer.n_compressed_blocks = n_completed
    layer.swa_count = min(t_prefill, n_win)
    # Aligned prefill: compressor's kv_state / score_state are at their initial
    # values on the reference side too. We don't have to copy anything.
    if t_prefill % m != 0:
        raise NotImplementedError("test config requires t_prefill % m == 0")
    state_cache.start_pos = t_prefill


def _build_token_pe(
    rotary: RotaryEmbedding, position: int, bsz: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor([[position]], device=device).expand(bsz, -1)
    return rotary(torch.zeros(bsz, 1, device=device), pos)


def test_hca_decode_matches_v4_reference(reference_module: Any) -> None:
    torch.manual_seed(0)
    args = _build_synthetic_args(reference_module)
    t_pre = 16
    n_decode = 8
    bsz = 2

    ref_attn = reference_module.Attention(0, args)
    _init_reference_attention(ref_attn, seed=42)
    our_block = _build_our_block(args)
    _sync_weights(our_block, ref_attn)

    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=args.compress_ratios[0],
                n_win=args.window_size,
                max_n_compressed=args.max_seq_len // args.compress_ratios[0],
            )
        ],
        batch_size=bsz,
    )

    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)

    # ---- Prefill on the reference (populates its kv_cache + compressor state) ----
    x_pre = torch.randn(bsz, t_pre, args.dim) * 0.5
    with torch.no_grad():
        ref_attn(x_pre, start_pos=0)

    # ---- Sync reference -> our StateCache ----
    _sync_state_after_prefill(ref_attn, state_cache, t_prefill=t_pre, args=args)

    # ---- Decode N steps; check parity at each ----
    for i in range(n_decode):
        start_pos = t_pre + i
        x_dec = torch.randn(bsz, 1, args.dim) * 0.5
        token_pe = _build_token_pe(rotary, position=start_pos, bsz=bsz, device=x_dec.device)

        # If this step closes a block, the just-flushed compressed entry uses
        # RoPE for position `(start_pos // m) * m`.
        m = args.compress_ratios[0]
        block_pe = None
        if (start_pos + 1) % m == 0:
            block_pe = _build_token_pe(
                rotary, position=(start_pos // m) * m, bsz=bsz, device=x_dec.device
            )

        with torch.no_grad():
            theirs = ref_attn(x_dec, start_pos=start_pos)
            ours = our_block.forward_decode(
                x_dec,
                start_pos=start_pos,
                state_cache=state_cache,
                layer_idx=0,
                token_position_embeddings=token_pe,
                block_position_embeddings=block_pe,
            )

        assert ours.shape == theirs.shape == (bsz, 1, args.dim)
        cs = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0).item()
        max_diff = (ours - theirs).abs().max().item()
        rel_err = max_diff / max(theirs.abs().max().item(), 1e-9)
        assert cs > 0.999, (
            f"decode step {i} (start_pos={start_pos}): "
            f"cosine_sim={cs:.6f}, max_abs_diff={max_diff:.3e}, rel_err={rel_err:.3e}"
        )


def test_forward_decode_rejects_non_unit_seqlen(reference_module: Any) -> None:
    args = _build_synthetic_args(reference_module)
    our_block = _build_our_block(args)
    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=args.compress_ratios[0],
                n_win=args.window_size,
                max_n_compressed=args.max_seq_len // args.compress_ratios[0],
            )
        ],
        batch_size=1,
    )
    x = torch.randn(1, 4, args.dim)
    cos = torch.zeros(1, 4, args.rope_head_dim)
    sin = torch.zeros(1, 4, args.rope_head_dim)
    with pytest.raises(ValueError, match="seqlen=1"):
        our_block.forward_decode(
            x,
            start_pos=16,
            state_cache=state_cache,
            layer_idx=0,
            token_position_embeddings=(cos, sin),
        )
