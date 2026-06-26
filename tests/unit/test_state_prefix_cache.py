"""Cross-request prefix sharing: generate_ids_prefix_cached == generate_ids.

Reusing a cached prefix (restore + replay the suffix) must produce the SAME
tokens as a fresh full prefill, since sharing changes only the work done, not the
math. Covers miss (full prefill), exact-length hit (stored logits), and prefix
hit (restore + suffix replay). Full hybrid config (SWA + CSA + HCA) so the CSA
indexer sub-state is snapshotted and restored too.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.state_prefix_cache import StatePrefixCache
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM


def _make_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=32,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 4, 8, 4),
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def _make_generator() -> StateCacheGenerator:
    torch.manual_seed(0)
    return StateCacheGenerator(DeepseekV4ForCausalLM(_make_config()).eval())


def test_prefix_cached_miss_matches_generate_ids() -> None:
    gen = _make_generator()
    prompt = list(range(8))
    cache = StatePrefixCache()
    cached = gen.generate_ids_prefix_cached(prompt, max_new_tokens=6, prefix_cache=cache)
    assert cached == gen.generate_ids(prompt, max_new_tokens=6)
    assert len(cache) == 1  # the full prompt was cached for reuse


def test_prefix_cached_exact_hit_matches_generate_ids() -> None:
    gen = _make_generator()
    prompt = list(range(8))
    cache = StatePrefixCache()
    gen.generate_ids_prefix_cached(prompt, max_new_tokens=6, prefix_cache=cache)  # warm the cache
    assert cache.match(prompt)[0] == len(prompt)  # exact hit available
    cached = gen.generate_ids_prefix_cached(prompt, max_new_tokens=6, prefix_cache=cache)
    assert cached == gen.generate_ids(prompt, max_new_tokens=6)


def test_prefix_cached_prefix_hit_matches_generate_ids() -> None:
    gen = _make_generator()
    base = list(range(8))
    extended = [*base, 3, 1, 4]  # shares the first 8 tokens with `base`
    cache = StatePrefixCache()
    gen.generate_ids_prefix_cached(base, max_new_tokens=4, prefix_cache=cache)  # caches `base`
    assert cache.match(extended)[0] == len(base)  # prefix hit: replay only [8:11]
    cached = gen.generate_ids_prefix_cached(extended, max_new_tokens=6, prefix_cache=cache)
    assert cached == gen.generate_ids(extended, max_new_tokens=6)


def test_prefix_cached_validates_inputs() -> None:
    gen = _make_generator()
    cache = StatePrefixCache()
    with pytest.raises(ValueError, match="non-empty"):
        gen.generate_ids_prefix_cached([], max_new_tokens=4, prefix_cache=cache)
    with pytest.raises(ValueError, match="max_new_tokens"):
        gen.generate_ids_prefix_cached([1, 2], max_new_tokens=0, prefix_cache=cache)


def test_prefix_cache_fifo_eviction() -> None:
    gen = _make_generator()
    cache = StatePrefixCache(max_entries=2)
    for prompt in (list(range(8)), list(range(8, 16)), list(range(16, 24))):
        gen.generate_ids_prefix_cached(prompt, max_new_tokens=2, prefix_cache=cache)
    assert len(cache) == 2  # oldest evicted past the cap
    assert cache.match(list(range(8)))[0] == 0  # the first prompt was evicted
