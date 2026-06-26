"""Prefix-cache participation of the GLM DSA index_k stream.

GLM's "full" indexer layers add an `index_k` cache stream alongside the MLA
`kv_latent`/`k_rope`. Prefix sharing publishes and reuses whole blocks keyed on
token IDs, stream-agnostically, so index_k must ride along: a repeat prompt's
cached block should hand back the same index_k keys the first request wrote.
This is the GLM-specific gate on top of the generic prefix-cache wiring in
`test_prefix_cache_integration.py`.
"""

from __future__ import annotations

import torch

from mini_infer.cache.block_pool import BlockPool, StreamSpec
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

BLOCK_SIZE = 4
INDEX_HEAD_DIM = 8
KV_LORA_RANK = 16
QK_ROPE = 4
NUM_LAYERS = 2


def _make_pool() -> BlockPool:
    # Both layers "full": index_k FIRST (the layer-0 allocation trigger),
    # then the MLA kv_latent / k_rope streams.
    layer = [
        StreamSpec("index_k", 1, INDEX_HEAD_DIM),
        StreamSpec("kv_latent", 1, KV_LORA_RANK),
        StreamSpec("k_rope", 1, QK_ROPE),
    ]
    return BlockPool(
        num_blocks=16,
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK,
        dtype=torch.float32,
        device="cpu",
        layer_streams=[list(layer) for _ in range(NUM_LAYERS)],
        prefix_cache=PrefixCache(block_size=BLOCK_SIZE),
    )


def _index_k_data(num_tokens: int) -> torch.Tensor:
    """index_k where row t is filled with the value t, so reused positions are
    identifiable after a cache hit."""
    return (
        torch.arange(num_tokens, dtype=torch.float32)
        .view(num_tokens, 1, 1)
        .expand(num_tokens, 1, INDEX_HEAD_DIM)
        .contiguous()
    )


def _write_all_streams(cache: PagedKVCache, num_tokens: int) -> None:
    """Append num_tokens to slot 0 across both layers, in stream order
    (index_k first so layer 0 triggers allocation; k_rope last triggers publish)."""
    cu = torch.tensor([0, num_tokens], dtype=torch.int32)
    index_k = _index_k_data(num_tokens)
    kv_latent = torch.full((num_tokens, 1, KV_LORA_RANK), 0.5, dtype=torch.float32)
    k_rope = torch.full((num_tokens, 1, QK_ROPE), 0.7, dtype=torch.float32)
    for layer_idx in range(NUM_LAYERS):
        cache.append_stream_packed(index_k, cu, layer_idx, "index_k")
        cache.append_stream_packed(kv_latent, cu, layer_idx, "kv_latent")
        cache.append_stream_packed(k_rope, cu, layer_idx, "k_rope")


def test_index_k_publishes_and_reuses_on_prefix_hit() -> None:
    pool = _make_pool()
    cache = PagedKVCache(pool)
    prompt = [10, 11, 12, 13, 14, 15, 16, 17]  # 2 full blocks

    cache.add_request_slot(prompt_token_ids=prompt)
    _write_all_streams(cache, num_tokens=len(prompt))
    assert pool.prefix_cache is not None
    assert pool.prefix_cache.num_cached == 2  # both full blocks published
    cache.remove_request(0)

    # Repeat prompt: prefix hit. The last-token rule leaves one block to
    # prefill, so the slot starts pre-populated with one cached block.
    new_idx = cache.add_request_slot(prompt_token_ids=prompt)
    assert cache.seq_lens_list()[new_idx] == BLOCK_SIZE

    # The cached block's index_k reads back as the original positions [0..3]
    # on every full layer (proof the index_k stream rode the shared block).
    for layer_idx in range(NUM_LAYERS):
        index_k_packed, _, _ = cache.materialize_packed_stream(layer_idx, "index_k")
        assert index_k_packed.shape == (BLOCK_SIZE, 1, INDEX_HEAD_DIM)
        expected = _index_k_data(BLOCK_SIZE)
        assert torch.allclose(index_k_packed, expected), (
            f"layer {layer_idx}: cached index_k {index_k_packed[:, 0, 0].tolist()} "
            f"!= expected {expected[:, 0, 0].tolist()}"
        )


def _make_model() -> GlmMoeDsaForCausalLM:
    cfg = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        kv_lora_rank=32,
        q_lora_rank=24,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        index_topk=4,  # < prompt: DSA selection is active
        index_head_dim=16,
        index_n_heads=2,
        mlp_layer_types=("dense", "dense", "dense", "sparse"),
        indexer_types=("full", "shared", "full", "shared"),
    )
    return GlmMoeDsaForCausalLM(cfg).to(torch.float32).eval()


def test_prefix_hit_matches_full_prefill() -> None:
    """End-to-end: a prefix-hit request (cached prefix + suffix prefill) yields
    the same final-token logits as a fresh full-compute prefill of the prompt.

    Same model, same prefix-cache pool: request A prefills the whole prompt
    (no hit, publishes blocks); request B re-runs the same prompt, hits the
    cached prefix, and prefills only the suffix. B must reconstruct A's
    last-position logits, which requires the cached prefix's kv_latent/k_rope
    AND index_k to be correctly reused for the suffix's DSA selection.
    """
    torch.manual_seed(0)
    model = _make_model()
    pool = BlockPool(
        num_blocks=32,
        block_size=BLOCK_SIZE,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=1,
        head_dim=model.cfg.kv_lora_rank,
        dtype=torch.float32,
        device="cpu",
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
        prefix_cache=PrefixCache(block_size=BLOCK_SIZE),
    )
    cache = PagedKVCache(pool)
    prompt = [10, 11, 12, 13, 14, 15, 16, 17]  # 2 full blocks
    plen = len(prompt)

    # Request A: full prefill (cache empty → no hit), publishes blocks.
    a_idx = cache.add_request_slot(prompt_token_ids=prompt)
    assert cache.seq_lens_list()[a_idx] == 0
    with torch.inference_mode():
        logits_a = model(
            input_ids=torch.tensor([prompt], dtype=torch.long),
            position_ids=torch.arange(plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], dtype=torch.int32),
        )
    last_a = logits_a[0, plen - 1]
    cache.remove_request(a_idx)

    # Request B: same prompt → prefix hit. Prefill only the uncached suffix.
    b_idx = cache.add_request_slot(prompt_token_ids=prompt)
    cached = cache.seq_lens_list()[b_idx]
    assert cached > 0, "expected a prefix hit to pre-populate cached tokens"
    suffix = prompt[cached:]
    with torch.inference_mode():
        logits_b = model(
            input_ids=torch.tensor([suffix], dtype=torch.long),
            position_ids=torch.arange(cached, plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, len(suffix)], dtype=torch.int32),
        )
    last_b = logits_b[0, -1]

    assert torch.allclose(last_a, last_b, atol=1e-4), (
        f"prefix-hit last-token logits diverged: max_abs_diff="
        f"{(last_a - last_b).abs().max().item():.6f}"
    )
