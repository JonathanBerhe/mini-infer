"""End-to-end demo of mini-infer's DeepSeek-V4 hybrid attention backbone.

Runs a small CSA/HCA/CSA/HCA model on synthetic input to demonstrate:

  1. Prefill through the full hybrid stack (mixed CSA + HCA layers).
  2. Multi-step incremental decode through a per-request `StateCache`,
     including the cross-layer state coherence (each layer maintains
     its own SWA + compressed-history + compressor in-flight buffers).
  3. The flush bookkeeping: layers with `compression_ratio = 4` (CSA)
     emit a new compressed entry every 4 decode steps; layers with
     `compression_ratio = 8` (HCA) emit one every 8.

This is NOT a real inference workload: the model uses random weights
on a 4-layer shrunken config so the demo runs in seconds on a laptop CPU.
The point is to exercise the attention machinery end-to-end and show
the per-layer state advancing correctly across decode steps. For the
real V4-Flash storage format end-to-end (block-FP8 + NVFP4 dequant,
expert-parallel sharding, meta-device per-rank loading on multi-GPU),
see `src/mini_infer/models/deepseek_v4.py::load_weights` and the
2x B200 smoke at `scripts/modal_v4_flash_smoke.py`.

Run with:
    uv run python scripts/demo_deepseek_v4_hybrid.py
"""

from __future__ import annotations

import argparse

import torch

from mini_infer.cache.state_cache import StateCache
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)


def make_demo_config(
    num_layers: int,
    *,
    use_moe_ffn: bool = False,
    use_hyper_connections: bool = False,
    hc_mult: int = 4,
) -> DeepseekV4Config:
    """A small 4-layer hybrid: CSA / HCA / CSA / HCA.

    Real V4-Pro: dim=4096, num_attention_heads=64, kv_head_dim=512,
    rope_head_dim=64, window_size=128, compress_ratios=(...) varying
    per-layer between 4 and 128, num_routed_experts=64,
    num_activated_experts=6, hc_mult=4. We shrink everything for CPU fit
    and toggle the optional primitives via the args.
    """
    if num_layers % 2 != 0:
        raise ValueError(f"demo expects even num_layers (alternating CSA/HCA), got {num_layers}")
    compress_ratios = tuple(4 if layer_idx % 2 == 0 else 8 for layer_idx in range(num_layers))
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=32,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=compress_ratios,
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        # Optional primitives — defaults match the original demo (off).
        use_moe_ffn=use_moe_ffn,
        moe_intermediate_size=64 if use_moe_ffn else 0,
        num_routed_experts=4 if use_moe_ffn else 0,
        num_activated_experts=2 if use_moe_ffn else 0,
        num_hash_routed_layers=num_layers // 2 if use_moe_ffn else 0,
        moe_score_func="softmax",
        moe_route_scale=1.0,
        n_shared_experts=1 if use_moe_ffn else 0,
        use_hyper_connections=use_hyper_connections,
        hc_mult=hc_mult if use_hyper_connections else 0,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


def run_prefill(model: DeepseekV4ForCausalLM, prefill_token_ids: torch.Tensor) -> torch.Tensor:
    """Single packed-prefill forward through the hybrid stack."""
    with torch.inference_mode():
        return model(prefill_token_ids)


def run_decode_loop(
    model: DeepseekV4ForCausalLM,
    *,
    state_cache: StateCache,
    starting_position: int,
    decode_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Stream `decode_token_ids` through `forward_decode_with_cache` one at a time.

    Returns logits for each step, shape `(B, n_decode_steps, vocab_size)`.
    """
    n_decode_steps = decode_token_ids.shape[1]
    per_step_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for step_idx in range(n_decode_steps):
            global_position = starting_position + step_idx
            input_id = decode_token_ids[:, step_idx : step_idx + 1]
            step_logits = model.forward_decode_with_cache(
                input_id, start_pos=global_position, state_cache=state_cache
            )
            per_step_logits.append(step_logits)
            state_cache.advance_start_pos(1)
    return torch.cat(per_step_logits, dim=1)  # (B, n_decode_steps, vocab_size)


def summarize_per_layer_state(
    state_cache: StateCache, *, label: str, compress_ratios: tuple[int, ...]
) -> None:
    """Print the per-layer compressed-counts so you can watch them advance."""
    print(f"\n[{label}] per-layer state:")
    print(f"  global start_pos = {state_cache.start_pos}")
    for layer_idx in range(state_cache.num_layers):
        layer_state = state_cache.layer(layer_idx)
        compression_ratio = compress_ratios[layer_idx]
        attention_kind = "CSA" if compression_ratio == 4 else "HCA"
        indexer_summary = ""
        if layer_state.indexer is not None:
            indexer_summary = (
                f", indexer.n_compressed_blocks={layer_state.indexer.n_compressed_blocks}"
            )
        print(
            f"  layer {layer_idx} ({attention_kind}, m={compression_ratio}): "
            f"swa_count={layer_state.swa_count}, "
            f"n_compressed_blocks={layer_state.n_compressed_blocks}"
            f"{indexer_summary}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prefill-len",
        type=int,
        default=16,
        help="prefill length (must be a multiple of every compress_ratio in the schedule)",
    )
    parser.add_argument("--decode-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-moe-ffn",
        action="store_true",
        help="Replace SwiGLU FFN with HashRoutedMoEFFN (V4 paper §2.2). "
        "Half the layers use hash routing (per-token-id lookup), the rest score-topk.",
    )
    parser.add_argument(
        "--use-hyper-connections",
        action="store_true",
        help="Enable Hyper-Connections (V4 paper §2.5): hidden state carries `hc_mult` "
        "copies through the layer stack, mediated by Sinkhorn-normalized residuals.",
    )
    parser.add_argument(
        "--hc-mult",
        type=int,
        default=4,
        help=(
            "Number of residual copies for Hyper-Connections "
            "(only used with --use-hyper-connections)."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = make_demo_config(
        num_layers=args.num_layers,
        use_moe_ffn=args.use_moe_ffn,
        use_hyper_connections=args.use_hyper_connections,
        hc_mult=args.hc_mult,
    )
    model = DeepseekV4ForCausalLM(cfg).eval()

    csa_layer_count = sum(1 for ratio in cfg.compress_ratios if ratio == 4)
    hca_layer_count = cfg.num_hidden_layers - csa_layer_count
    ffn_kind = "HashRoutedMoEFFN" if cfg.use_moe_ffn else "SwiGLU"
    residual_kind = (
        f"Hyper-Connections (hc_mult={cfg.hc_mult})"
        if cfg.use_hyper_connections
        else "vanilla pre-norm"
    )
    print("=== mini-infer DeepSeek-V4 hybrid backbone demo ===")
    print(
        f"Config: {cfg.num_hidden_layers} layers "
        f"({csa_layer_count} CSA + {hca_layer_count} HCA), "
        f"hidden_size={cfg.hidden_size}, "
        f"compress_ratios={cfg.compress_ratios}, "
        f"window_size={cfg.window_size}, "
        f"ffn={ffn_kind}, residual={residual_kind}"
    )

    # ---- Prefill: pack a synthetic token sequence through the full stack ----
    prefill_token_ids = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.prefill_len), dtype=torch.long
    )
    prefill_logits = run_prefill(model, prefill_token_ids)
    print(
        f"\nPrefill: {tuple(prefill_token_ids.shape)} -> logits "
        f"{tuple(prefill_logits.shape)}, finite={torch.isfinite(prefill_logits).all().item()}"
    )

    # ---- Initialise StateCache as if we'd just finished prefill ----
    # Real serving would write SWA + compressed entries during prefill via a
    # cache-aware forward; we simulate "post-prefill" by setting the counters
    # so the decode loop attends to them. Buffer contents are zero-initialised
    # (the demo focuses on the bookkeeping; a parity test sets the contents).
    state_cache = StateCache(
        build_state_cache_layer_specs(cfg, max_n_compressed=64),
        batch_size=args.batch_size,
    )
    for layer_idx, compression_ratio in enumerate(cfg.compress_ratios):
        layer_state = state_cache.layer(layer_idx)
        layer_state.n_compressed_blocks = args.prefill_len // compression_ratio
        layer_state.swa_count = min(args.prefill_len, cfg.window_size)
        if layer_state.indexer is not None:
            layer_state.indexer.n_compressed_blocks = args.prefill_len // compression_ratio
    state_cache.start_pos = args.prefill_len
    summarize_per_layer_state(
        state_cache, label="After prefill", compress_ratios=cfg.compress_ratios
    )

    # ---- Decode loop: feed N tokens through `forward_decode_with_cache` ----
    decode_token_ids = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.decode_steps), dtype=torch.long
    )
    decode_logits = run_decode_loop(
        model,
        state_cache=state_cache,
        starting_position=args.prefill_len,
        decode_token_ids=decode_token_ids,
    )
    print(
        f"\nDecode: {args.decode_steps} steps -> logits {tuple(decode_logits.shape)}, "
        f"finite={torch.isfinite(decode_logits).all().item()}"
    )

    summarize_per_layer_state(
        state_cache, label="After decode", compress_ratios=cfg.compress_ratios
    )

    # ---- Sanity: argmax tokens at each decode step ----
    predicted_tokens = decode_logits.argmax(dim=-1).tolist()
    print(f"\nGreedy-argmax tokens (random weights, content meaningless): {predicted_tokens}")

    print(
        "\nDemo complete. The hybrid backbone composed CSA + HCA layers "
        "and threaded per-layer state correctly across decode."
    )


if __name__ == "__main__":
    main()
