"""Cache-aware prefill: equivalence vs standalone + end-to-end (prefill+decode) parity.

Two correctness contracts in one file:

  1. **Equivalence** (`*_matches_standalone_for_aligned_input`): for any
     `seqlen` that's a multiple of every layer's `compression_ratio`,
     `forward_prefill_with_cache(...)` produces the SAME attention output
     as the existing standalone `forward(...)`. Fresh weights, fresh
     state cache, fresh inputs — both should be element-wise close.

  2. **End-to-end parity** (`*_matches_v4_reference_no_external_sync`):
     run the reference's `Attention.forward(prefill, start_pos=0)` and our
     `forward_prefill_with_cache(prefill, state_cache=...)` on the same
     input. Then run reference decode + our `forward_decode` for N steps.
     Compare outputs at EVERY step. With cache-aware prefill there's no
     external state-syncing helper any more — the prefill itself populates
     the cache.

Coverage matrix:

    | path | aligned | unaligned |
    |---|---|---|
    | HCA equivalence | ✓ | n/a (standalone forward rejects unaligned) |
    | HCA end-to-end  | ✓ | ✓ |
    | CSA equivalence | ✓ | n/a |
    | CSA end-to-end  | ✓ | ✓ |
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.functional import cosine_similarity

from mini_infer.cache.state_cache import IndexerStateSpec, StateCache, StateLayerSpec
from mini_infer.models.blocks import CSAAttention, HCAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding
from mini_infer.models.blocks.swa import SWAAttention

# ---------- Shared scaffolding (small synthetic configs, m matches HCA/CSA needs) ----------


def _build_hca_args(reference_module: Any) -> Any:
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
        compress_ratios=(8,),  # HCA
        original_seq_len=0,
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
    )


def _build_csa_args(reference_module: Any) -> Any:
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
        compress_ratios=(4,),  # CSA -> indexer
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


def _seed_fill_reference(ref_attn: Any, seed: int) -> None:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in ref_attn.parameters():
        with torch.no_grad():
            parameter.data = torch.randn(parameter.shape, generator=rng, dtype=torch.float32) * 0.02
    for buffer in ref_attn.buffers():
        if buffer.dtype.is_floating_point:
            buffer.zero_()


def _build_hca_block(args: Any) -> HCAAttention:
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


def _build_csa_block(args: Any) -> CSAAttention:
    return CSAAttention(
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


def _sync_hca_weights(our_block: HCAAttention, ref_attn: Any) -> None:
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


def _sync_csa_weights(our_block: CSAAttention, ref_attn: Any) -> None:
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
        our_block.indexer.wq_b.weight.copy_(ref_attn.indexer.wq_b.weight)
        our_block.indexer.weights_proj.weight.copy_(ref_attn.indexer.weights_proj.weight)
        our_block.indexer.compressor.kv_proj.weight.copy_(ref_attn.indexer.compressor.wkv.weight)
        our_block.indexer.compressor.weight_proj.weight.copy_(
            ref_attn.indexer.compressor.wgate.weight
        )
        our_block.indexer.compressor.position_bias.copy_(ref_attn.indexer.compressor.ape)
        our_block.indexer.compressor.norm.weight.copy_(ref_attn.indexer.compressor.norm.weight)


def _build_swa_args(reference_module: Any) -> Any:
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
        compress_ratios=(0,),  # pure SWA: no compressor, no indexer
        original_seq_len=0,
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
    )


def _build_swa_block(args: Any) -> SWAAttention:
    return SWAAttention(
        hidden_size=args.dim,
        num_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        num_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        window_size=args.window_size,
        rms_norm_eps=args.norm_eps,
    )


def _sync_swa_weights(our_block: SWAAttention, ref_attn: Any) -> None:
    with torch.no_grad():
        our_block.q_a_proj.weight.copy_(ref_attn.wq_a.weight)
        our_block.q_a_layernorm.weight.copy_(ref_attn.q_norm.weight)
        our_block.q_b_proj.weight.copy_(ref_attn.wq_b.weight)
        our_block.swa_kv_proj.weight.copy_(ref_attn.wkv.weight)
        our_block.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        our_block.sink.sink_logits.copy_(ref_attn.attn_sink)
        our_block.grouped_output.wo_a.copy_(ref_attn.wo_a.weight)
        our_block.grouped_output.wo_b.weight.copy_(ref_attn.wo_b.weight)


def _build_position_embeddings(
    rotary: RotaryEmbedding,
    bsz: int,
    seqlen: int,
    compression_ratio: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    token_positions = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    cos_t, sin_t = rotary(torch.zeros(bsz, seqlen, device=device), token_positions)
    n_compressed = seqlen // compression_ratio
    block_positions = (
        (torch.arange(n_compressed, device=device) * compression_ratio).unsqueeze(0).expand(bsz, -1)
    )
    cos_c, sin_c = rotary(torch.zeros(bsz, max(n_compressed, 1), device=device), block_positions)
    return (cos_t, sin_t), (cos_c, sin_c)


def _single_token_pe(
    rotary: RotaryEmbedding, position: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor([[position]], device=device).expand(batch_size, -1)
    return rotary(torch.zeros(batch_size, 1, device=device), pos)


# ---------- Equivalence: cache-aware prefill matches standalone for aligned input ----------


def test_hca_cache_aware_prefill_matches_standalone_for_aligned_input(
    reference_module: Any,
) -> None:
    torch.manual_seed(0)
    args = _build_hca_args(reference_module)
    ref_attn = reference_module.Attention(0, args)
    _seed_fill_reference(ref_attn, seed=42)
    our_block = _build_hca_block(args)
    _sync_hca_weights(our_block, ref_attn)

    batch_size, seqlen = 2, 16  # multiple of compression_ratio=8
    hidden_states = torch.randn(batch_size, seqlen, args.dim) * 0.5
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_pe, compressed_pe = _build_position_embeddings(
        rotary, batch_size, seqlen, args.compress_ratios[0], device=hidden_states.device
    )

    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=args.compress_ratios[0],
                n_win=args.window_size,
                max_n_compressed=args.max_seq_len // args.compress_ratios[0],
            )
        ],
        batch_size=batch_size,
    )

    with torch.no_grad():
        standalone_out = our_block(hidden_states, token_pe, compressed_pe)
        cache_aware_out = our_block.forward_prefill_with_cache(
            hidden_states,
            token_position_embeddings=token_pe,
            compressed_position_embeddings=compressed_pe,
            state_cache=state_cache,
            layer_idx=0,
        )

    torch.testing.assert_close(standalone_out, cache_aware_out, rtol=1e-5, atol=1e-6)


def test_csa_cache_aware_prefill_matches_standalone_for_aligned_input(
    reference_module: Any,
) -> None:
    torch.manual_seed(0)
    args = _build_csa_args(reference_module)
    ref_attn = reference_module.Attention(0, args)
    _seed_fill_reference(ref_attn, seed=42)
    our_block = _build_csa_block(args)
    _sync_csa_weights(our_block, ref_attn)

    batch_size, seqlen = 2, 16  # multiple of compression_ratio=4
    hidden_states = torch.randn(batch_size, seqlen, args.dim) * 0.5
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_pe, compressed_pe = _build_position_embeddings(
        rotary, batch_size, seqlen, args.compress_ratios[0], device=hidden_states.device
    )

    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=args.compress_ratios[0],
                n_win=args.window_size,
                max_n_compressed=args.max_seq_len // args.compress_ratios[0],
                overlap_mode=True,
                indexer=IndexerStateSpec(head_dim=args.index_head_dim),
            )
        ],
        batch_size=batch_size,
    )

    with torch.no_grad():
        standalone_out = our_block(hidden_states, token_pe, compressed_pe)
        cache_aware_out = our_block.forward_prefill_with_cache(
            hidden_states,
            token_position_embeddings=token_pe,
            compressed_position_embeddings=compressed_pe,
            state_cache=state_cache,
            layer_idx=0,
        )

    torch.testing.assert_close(standalone_out, cache_aware_out, rtol=1e-5, atol=1e-6)


def test_swa_cache_aware_prefill_matches_standalone_for_swa_layer(reference_module: Any) -> None:
    torch.manual_seed(0)
    args = _build_swa_args(reference_module)
    ref_attn = reference_module.Attention(0, args)
    _seed_fill_reference(ref_attn, seed=42)
    our_block = _build_swa_block(args)
    _sync_swa_weights(our_block, ref_attn)

    batch_size, seqlen = 2, 16  # > window_size=8, exercises the rotated window write
    hidden_states = torch.randn(batch_size, seqlen, args.dim) * 0.5
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_positions = torch.arange(seqlen).unsqueeze(0).expand(batch_size, -1)
    cos_t, sin_t = rotary(torch.zeros(batch_size, seqlen), token_positions)
    token_pe = (cos_t, sin_t)

    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=0,
                n_win=args.window_size,
                max_n_compressed=1,
            )
        ],
        batch_size=batch_size,
    )
    with torch.no_grad():
        standalone_out = our_block(hidden_states, token_pe)
        cache_aware_out = our_block.forward_prefill_with_cache(
            hidden_states,
            token_position_embeddings=token_pe,
            state_cache=state_cache,
            layer_idx=0,
        )
    torch.testing.assert_close(standalone_out, cache_aware_out, rtol=1e-5, atol=1e-6)


# ---------- End-to-end parity: prefill -> decode without external sync ----------


def _run_end_to_end_hca(
    reference_module: Any, *, t_prefill: int, n_decode_steps: int
) -> tuple[float, float]:
    """Returns (min_cosine_sim, max_abs_diff_overall) across all decode steps."""
    torch.manual_seed(0)
    args = _build_hca_args(reference_module)
    ref_attn = reference_module.Attention(0, args)
    _seed_fill_reference(ref_attn, seed=42)
    our_block = _build_hca_block(args)
    _sync_hca_weights(our_block, ref_attn)

    batch_size = 2
    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=args.compress_ratios[0],
                n_win=args.window_size,
                max_n_compressed=args.max_seq_len // args.compress_ratios[0],
            )
        ],
        batch_size=batch_size,
    )
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_pe, compressed_pe = _build_position_embeddings(
        rotary, batch_size, t_prefill, args.compress_ratios[0], device=torch.device("cpu")
    )

    prefill_input = torch.randn(batch_size, t_prefill, args.dim) * 0.5
    with torch.no_grad():
        ref_attn(prefill_input, start_pos=0)
        # Cache-aware prefill — populates state_cache itself, no external sync needed.
        our_block.forward_prefill_with_cache(
            prefill_input,
            token_position_embeddings=token_pe,
            compressed_position_embeddings=compressed_pe,
            state_cache=state_cache,
            layer_idx=0,
        )
    state_cache.start_pos = t_prefill

    min_cos_sim = 1.0
    max_diff_overall = 0.0
    compression_ratio = args.compress_ratios[0]
    for step_idx in range(n_decode_steps):
        global_position = t_prefill + step_idx
        decode_input = torch.randn(batch_size, 1, args.dim) * 0.5
        token_pe_decode = _single_token_pe(rotary, global_position, batch_size, decode_input.device)
        block_pe = None
        if (global_position + 1) % compression_ratio == 0:
            flushed_block_position = (global_position // compression_ratio) * compression_ratio
            block_pe = _single_token_pe(
                rotary, flushed_block_position, batch_size, decode_input.device
            )

        with torch.no_grad():
            theirs = ref_attn(decode_input, start_pos=global_position)
            ours = our_block.forward_decode(
                decode_input,
                start_pos=global_position,
                state_cache=state_cache,
                layer_idx=0,
                token_position_embeddings=token_pe_decode,
                block_position_embeddings=block_pe,
            )
        cs = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0).item()
        diff = (ours - theirs).abs().max().item()
        min_cos_sim = min(min_cos_sim, cs)
        max_diff_overall = max(max_diff_overall, diff)
    return min_cos_sim, max_diff_overall


def _run_end_to_end_csa(
    reference_module: Any, *, t_prefill: int, n_decode_steps: int
) -> tuple[float, float]:
    torch.manual_seed(0)
    args = _build_csa_args(reference_module)
    ref_attn = reference_module.Attention(0, args)
    _seed_fill_reference(ref_attn, seed=42)
    our_block = _build_csa_block(args)
    _sync_csa_weights(our_block, ref_attn)

    batch_size = 2
    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=args.compress_ratios[0],
                n_win=args.window_size,
                max_n_compressed=args.max_seq_len // args.compress_ratios[0],
                overlap_mode=True,
                indexer=IndexerStateSpec(head_dim=args.index_head_dim),
            )
        ],
        batch_size=batch_size,
    )
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_pe, compressed_pe = _build_position_embeddings(
        rotary, batch_size, t_prefill, args.compress_ratios[0], device=torch.device("cpu")
    )

    prefill_input = torch.randn(batch_size, t_prefill, args.dim) * 0.5
    with torch.no_grad():
        ref_attn(prefill_input, start_pos=0)
        our_block.forward_prefill_with_cache(
            prefill_input,
            token_position_embeddings=token_pe,
            compressed_position_embeddings=compressed_pe,
            state_cache=state_cache,
            layer_idx=0,
        )
    state_cache.start_pos = t_prefill

    min_cos_sim = 1.0
    max_diff_overall = 0.0
    compression_ratio = args.compress_ratios[0]
    for step_idx in range(n_decode_steps):
        global_position = t_prefill + step_idx
        decode_input = torch.randn(batch_size, 1, args.dim) * 0.5
        token_pe_decode = _single_token_pe(rotary, global_position, batch_size, decode_input.device)
        block_pe = None
        if (global_position + 1) % compression_ratio == 0:
            flushed_block_position = (global_position // compression_ratio) * compression_ratio
            block_pe = _single_token_pe(
                rotary, flushed_block_position, batch_size, decode_input.device
            )

        with torch.no_grad():
            theirs = ref_attn(decode_input, start_pos=global_position)
            ours = our_block.forward_decode(
                decode_input,
                start_pos=global_position,
                state_cache=state_cache,
                layer_idx=0,
                token_position_embeddings=token_pe_decode,
                block_position_embeddings=block_pe,
            )
        cs = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0).item()
        diff = (ours - theirs).abs().max().item()
        min_cos_sim = min(min_cos_sim, cs)
        max_diff_overall = max(max_diff_overall, diff)
    return min_cos_sim, max_diff_overall


def test_hca_end_to_end_matches_v4_reference_aligned_prefill(reference_module: Any) -> None:
    """Aligned prefill (T=16, m=8 -> 2 full blocks) + 8 decode steps with one flush."""
    min_cos_sim, max_diff = _run_end_to_end_hca(reference_module, t_prefill=16, n_decode_steps=8)
    assert min_cos_sim > 0.999, (
        f"min cosine_sim={min_cos_sim:.6f}, max_abs_diff={max_diff:.3e} (target > 0.999)"
    )


def test_hca_end_to_end_matches_v4_reference_unaligned_prefill(reference_module: Any) -> None:
    """Unaligned prefill (T=18, m=8 -> 2 full blocks + remainder=2) + 8 decode steps.

    The trailing 2 tokens land in the in-flight accumulator at slots [0, 2);
    decode then continues from slot 2, with the third block flushing at
    start_pos=23 (since (23+1) % 8 == 0).
    """
    min_cos_sim, max_diff = _run_end_to_end_hca(reference_module, t_prefill=18, n_decode_steps=8)
    assert min_cos_sim > 0.999, (
        f"min cosine_sim={min_cos_sim:.6f}, max_abs_diff={max_diff:.3e} (target > 0.999)"
    )


def test_csa_end_to_end_matches_v4_reference_aligned_prefill(reference_module: Any) -> None:
    """Aligned prefill (T=16, m=4 -> 4 full blocks) + 8 decode steps with two flushes."""
    min_cos_sim, max_diff = _run_end_to_end_csa(reference_module, t_prefill=16, n_decode_steps=8)
    assert min_cos_sim > 0.999, (
        f"min cosine_sim={min_cos_sim:.6f}, max_abs_diff={max_diff:.3e} (target > 0.999)"
    )


def test_csa_end_to_end_matches_v4_reference_unaligned_prefill(reference_module: Any) -> None:
    """Unaligned prefill (T=18, m=4 -> 4 full blocks + remainder=2) + 8 decode steps.

    Exercises overlap-mode unaligned stashing: trailing 2 tokens at slots
    [m, m+2) of the current half AND the last m tokens of the aligned
    prefix at slots [0, m) of the previous half (the next decode flush
    needs both for its 2m softmax).
    """
    min_cos_sim, max_diff = _run_end_to_end_csa(reference_module, t_prefill=18, n_decode_steps=8)
    assert min_cos_sim > 0.999, (
        f"min cosine_sim={min_cos_sim:.6f}, max_abs_diff={max_diff:.3e} (target > 0.999)"
    )


def _run_end_to_end_swa(
    reference_module: Any, *, t_prefill: int, n_decode_steps: int
) -> tuple[float, float]:
    torch.manual_seed(0)
    args = _build_swa_args(reference_module)
    ref_attn = reference_module.Attention(0, args)
    _seed_fill_reference(ref_attn, seed=42)
    our_block = _build_swa_block(args)
    _sync_swa_weights(our_block, ref_attn)

    batch_size = 2
    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=0,
                n_win=args.window_size,
                max_n_compressed=1,
            )
        ],
        batch_size=batch_size,
    )
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_positions = torch.arange(t_prefill).unsqueeze(0).expand(batch_size, -1)
    cos_t, sin_t = rotary(torch.zeros(batch_size, t_prefill), token_positions)
    token_pe = (cos_t, sin_t)

    prefill_input = torch.randn(batch_size, t_prefill, args.dim) * 0.5
    with torch.no_grad():
        ref_attn(prefill_input, start_pos=0)
        our_block.forward_prefill_with_cache(
            prefill_input,
            token_position_embeddings=token_pe,
            state_cache=state_cache,
            layer_idx=0,
        )
    state_cache.start_pos = t_prefill

    min_cos_sim = 1.0
    max_diff_overall = 0.0
    for step_idx in range(n_decode_steps):
        global_position = t_prefill + step_idx
        decode_input = torch.randn(batch_size, 1, args.dim) * 0.5
        token_pe_decode = _single_token_pe(rotary, global_position, batch_size, decode_input.device)
        with torch.no_grad():
            theirs = ref_attn(decode_input, start_pos=global_position)
            ours = our_block.forward_decode(
                decode_input,
                start_pos=global_position,
                state_cache=state_cache,
                layer_idx=0,
                token_position_embeddings=token_pe_decode,
            )
        cs = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0).item()
        diff = (ours - theirs).abs().max().item()
        min_cos_sim = min(min_cos_sim, cs)
        max_diff_overall = max(max_diff_overall, diff)
    return min_cos_sim, max_diff_overall


def test_swa_end_to_end_matches_v4_reference_long_prefill(reference_module: Any) -> None:
    """Prefill T=16 (> window 8, exercises the rotated window write) + 8 decode steps."""
    min_cos_sim, max_diff = _run_end_to_end_swa(reference_module, t_prefill=16, n_decode_steps=8)
    assert min_cos_sim > 0.999, (
        f"min cosine_sim={min_cos_sim:.6f}, max_abs_diff={max_diff:.3e} (target > 0.999)"
    )


def test_swa_end_to_end_matches_v4_reference_short_prefill(reference_module: Any) -> None:
    """Prefill T=6 (<= window 8) + 8 decode steps, crossing the window boundary mid-decode."""
    min_cos_sim, max_diff = _run_end_to_end_swa(reference_module, t_prefill=6, n_decode_steps=8)
    assert min_cos_sim > 0.999, (
        f"min cosine_sim={min_cos_sim:.6f}, max_abs_diff={max_diff:.3e} (target > 0.999)"
    )
