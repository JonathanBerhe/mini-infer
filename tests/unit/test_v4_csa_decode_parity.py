"""HF parity: CSAAttention decode vs DeepSeek-V4-Pro reference.

Mirror of `test_v4_hca_decode_parity.py` for the CSA path. Builds the
reference `Attention(compress_ratio=4)` (CSA: triggers the indexer +
overlap-mode compressor), runs prefill, syncs reference state into our
`StateCache` allocated with `overlap_mode=True` and an `IndexerStateSpec`,
then runs N decode steps comparing each output.

What this catches (beyond CSA prefill parity + HCA decode parity):
    - Overlap-mode compressor decode: slot indexing
      `compression_ratio + (start_pos % compression_ratio)`, the slide
      `state[:m] = state[m:]` after flush, the `(B, 2m, c)` softmax
      shape that mixes previous-block first-half-features with
      current-block last-half-features.
    - Indexer's own compressor decode (separate state buffers).
    - Indexer top-k selection at decode (no causal masking — all
      compressed entries are by construction in the past).
    - Sync of the prefill-time end-of-block overlap slot from the
      reference's `kv_state[:, :ratio]` into our state.

Aligned prefill only (`T_prefill % compression_ratio == 0`); the
unaligned case extends the sync to copy the partial-block accumulator
into our state and is left for a follow-up.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.nn.functional import cosine_similarity

from mini_infer.cache.state_cache import IndexerStateSpec, StateCache, StateLayerSpec
from mini_infer.models.blocks import CSAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding


def _build_synthetic_args(reference_module: Any) -> Any:
    """Small CSA-only config: m=4 (forces indexer), n_win=8, T=16, decode 8 steps.

    Both prefill (16 tokens = 4 blocks) and decode (8 tokens = 2 more
    blocks) exercise multiple flushes so the slide mechanic + indexer
    history append are both covered.
    """
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
        compress_ratios=(4,),  # m=4 -> CSA path with indexer
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
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in ref_attn.parameters():
        with torch.no_grad():
            parameter.data = torch.randn(parameter.shape, generator=rng, dtype=torch.float32) * 0.02
    for buffer in ref_attn.buffers():
        if buffer.dtype.is_floating_point:
            buffer.zero_()


def _build_our_block(args: Any) -> CSAAttention:
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


def _sync_weights(our_block: CSAAttention, ref_attn: Any) -> None:
    with torch.no_grad():
        # Q low-rank
        our_block.q_a_proj.weight.copy_(ref_attn.wq_a.weight)
        our_block.q_a_layernorm.weight.copy_(ref_attn.q_norm.weight)
        our_block.q_b_proj.weight.copy_(ref_attn.wq_b.weight)
        # SWA K=V
        our_block.swa_kv_proj.weight.copy_(ref_attn.wkv.weight)
        our_block.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        # Main compressor (overlap)
        our_block.compressor.kv_proj.weight.copy_(ref_attn.compressor.wkv.weight)
        our_block.compressor.weight_proj.weight.copy_(ref_attn.compressor.wgate.weight)
        our_block.compressor.position_bias.copy_(ref_attn.compressor.ape)
        our_block.compressor.norm.weight.copy_(ref_attn.compressor.norm.weight)
        # Sink + grouped output
        our_block.sink.sink_logits.copy_(ref_attn.attn_sink)
        our_block.grouped_output.wo_a.copy_(ref_attn.wo_a.weight)
        our_block.grouped_output.wo_b.weight.copy_(ref_attn.wo_b.weight)
        # Lightning Indexer
        our_block.indexer.wq_b.weight.copy_(ref_attn.indexer.wq_b.weight)
        our_block.indexer.weights_proj.weight.copy_(ref_attn.indexer.weights_proj.weight)
        our_block.indexer.compressor.kv_proj.weight.copy_(ref_attn.indexer.compressor.wkv.weight)
        our_block.indexer.compressor.weight_proj.weight.copy_(
            ref_attn.indexer.compressor.wgate.weight
        )
        our_block.indexer.compressor.position_bias.copy_(ref_attn.indexer.compressor.ape)
        our_block.indexer.compressor.norm.weight.copy_(ref_attn.indexer.compressor.norm.weight)


def _sync_state_after_prefill(
    ref_attn: Any, state_cache: StateCache, *, t_prefill: int, args: Any
) -> None:
    """After reference prefill, copy its kv_cache + compressor states into ours.

    Reference's `Attention.kv_cache`:
        slots [0, n_win):                  SWA circular buffer.
        slots [n_win, n_win + n_completed): main compressed history.

    Reference's `compressor.kv_state` / `score_state` (overlap mode):
        slots [0, m):  populated at end of prefill with the LAST `m`
                       tokens' KV/score (so the next block's flush sees
                       this as its "previous overlap"). Aligned prefill
                       only; unaligned would also populate slots [m, ...).

    Reference's `indexer.kv_cache`: append-only history of the indexer's
        compressor outputs.

    Reference's `indexer.compressor.kv_state` / `score_state`:
        same overlap-mode shape as the main compressor. We sync the
        `[:m]` slots from the reference too.
    """
    n_win = args.window_size
    compression_ratio = args.compress_ratios[0]
    main_layer = state_cache.layer(0)
    batch_size = main_layer.swa_kv.shape[0]

    if t_prefill % compression_ratio != 0:
        raise NotImplementedError("test config requires t_prefill % compression_ratio == 0")

    # SWA + main compressed history
    main_layer.swa_kv.copy_(ref_attn.kv_cache[:batch_size, :n_win])
    n_main_completed = t_prefill // compression_ratio
    main_layer.compressed_kv[:, :n_main_completed].copy_(
        ref_attn.kv_cache[:batch_size, n_win : n_win + n_main_completed]
    )
    main_layer.n_compressed_blocks = n_main_completed
    main_layer.swa_count = min(t_prefill, n_win)

    # Main compressor in-flight state (overlap mode): copy [0, m) slots
    # holding the last `m` prefill tokens' KV/score with their `ape` bias.
    main_layer.cmp_kv_state[:, :compression_ratio].copy_(
        ref_attn.compressor.kv_state[:batch_size, :compression_ratio]
    )
    main_layer.cmp_score_state[:, :compression_ratio].copy_(
        ref_attn.compressor.score_state[:batch_size, :compression_ratio]
    )

    # Indexer state: history + its own compressor's [0, m) slots.
    indexer_state = main_layer.indexer
    if indexer_state is None:
        raise RuntimeError("CSA layer must have indexer state allocated")
    indexer_state.compressed_kv[:, :n_main_completed].copy_(
        ref_attn.indexer.kv_cache[:batch_size, :n_main_completed]
    )
    indexer_state.n_compressed_blocks = n_main_completed
    indexer_state.cmp_kv_state[:, :compression_ratio].copy_(
        ref_attn.indexer.compressor.kv_state[:batch_size, :compression_ratio]
    )
    indexer_state.cmp_score_state[:, :compression_ratio].copy_(
        ref_attn.indexer.compressor.score_state[:batch_size, :compression_ratio]
    )

    state_cache.start_pos = t_prefill


def _build_token_pe(
    rotary: RotaryEmbedding, position: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor([[position]], device=device).expand(batch_size, -1)
    return rotary(torch.zeros(batch_size, 1, device=device), pos)


def test_csa_decode_matches_v4_reference(reference_module: Any) -> None:
    torch.manual_seed(0)
    args = _build_synthetic_args(reference_module)
    t_prefill = 16
    n_decode_steps = 8
    batch_size = 2

    ref_attn = reference_module.Attention(0, args)
    _init_reference_attention(ref_attn, seed=42)
    our_block = _build_our_block(args)
    _sync_weights(our_block, ref_attn)

    compression_ratio = args.compress_ratios[0]
    state_cache = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=compression_ratio,
                n_win=args.window_size,
                max_n_compressed=args.max_seq_len // compression_ratio,
                overlap_mode=True,
                indexer=IndexerStateSpec(head_dim=args.index_head_dim),
            )
        ],
        batch_size=batch_size,
    )

    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)

    # ---- Prefill: populates reference's kv_cache + compressor states ----
    prefill_input = torch.randn(batch_size, t_prefill, args.dim) * 0.5
    with torch.no_grad():
        ref_attn(prefill_input, start_pos=0)

    # ---- Sync reference state into our StateCache ----
    _sync_state_after_prefill(ref_attn, state_cache, t_prefill=t_prefill, args=args)

    # ---- Decode N steps; per-step parity ----
    for step_idx in range(n_decode_steps):
        global_position = t_prefill + step_idx
        decode_input = torch.randn(batch_size, 1, args.dim) * 0.5
        token_pe = _build_token_pe(rotary, global_position, batch_size, decode_input.device)
        block_pe = None
        if (global_position + 1) % compression_ratio == 0:
            flushed_block_position = (global_position // compression_ratio) * compression_ratio
            block_pe = _build_token_pe(
                rotary, flushed_block_position, batch_size, decode_input.device
            )

        with torch.no_grad():
            theirs = ref_attn(decode_input, start_pos=global_position)
            ours = our_block.forward_decode(
                decode_input,
                start_pos=global_position,
                state_cache=state_cache,
                layer_idx=0,
                token_position_embeddings=token_pe,
                block_position_embeddings=block_pe,
            )

        assert ours.shape == theirs.shape == (batch_size, 1, args.dim)
        cs = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0).item()
        max_abs_diff = (ours - theirs).abs().max().item()
        rel_err = max_abs_diff / max(theirs.abs().max().item(), 1e-9)
        flushed_marker = "[flush]" if (global_position + 1) % compression_ratio == 0 else ""
        assert cs > 0.999, (
            f"decode step {step_idx} (start_pos={global_position}) {flushed_marker}: "
            f"cosine_sim={cs:.6f}, max_abs_diff={max_abs_diff:.3e}, rel_err={rel_err:.3e}"
        )


def test_csa_forward_decode_rejects_layer_without_indexer_state(reference_module: Any) -> None:
    """A CSA layer requires `IndexerStateSpec` in its `StateLayerSpec`."""
    args = _build_synthetic_args(reference_module)
    our_block = _build_our_block(args)
    state_cache_without_indexer = StateCache(
        [
            StateLayerSpec(
                kv_head_dim=args.head_dim,
                compression_ratio=args.compress_ratios[0],
                n_win=args.window_size,
                max_n_compressed=8,
                overlap_mode=True,
                # indexer=None  # missing
            )
        ],
        batch_size=1,
    )
    decode_input = torch.randn(1, 1, args.dim)
    cos = torch.zeros(1, 1, args.rope_head_dim)
    sin = torch.zeros(1, 1, args.rope_head_dim)
    with pytest.raises(ValueError, match="indexer"):
        our_block.forward_decode(
            decode_input,
            start_pos=16,
            state_cache=state_cache_without_indexer,
            layer_idx=0,
            token_position_embeddings=(cos, sin),
        )
