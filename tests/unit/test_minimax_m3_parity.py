"""MiniMax-M3 owned model: full-model bit-parity vs the HF reference.

The 428B checkpoint is out of reach for CPU CI, so this uses a tiny-random config
exercising the distinctive bits: dense (0-2) vs MSA+MoE (3-4) layers, per-head
Gemma QK-norm, first-N partial RoPE, the block indexer + additive mask (sized so
top-k actually drops a block), and the swigluoai MoE. Identical weights are loaded
into both models (via load_weights), so any divergence is a math bug, not a
loading one. Skips cleanly when transformers < 5.12 (no minimax_m3_vl).
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models import REGISTRY
from mini_infer.models.minimax_m3 import MiniMaxM3Config, MiniMaxM3ForCausalLM


def _make_cache(model: MiniMaxM3ForCausalLM, *, num_blocks: int = 32) -> PagedKVCache:
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def test_registry_has_minimax_m3() -> None:
    assert REGISTRY.lookup("MiniMaxM3SparseForConditionalGeneration") is MiniMaxM3ForCausalLM


def _tiny_hf_config():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )

    # 3 dense + 2 sparse layers. index_block_size=4 with total_q=12 -> 3 blocks;
    # topk_blocks=2 + local -> at least one block is dropped (MSA is not a no-op).
    return MiniMaxM3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,  # MoE per-expert
        dense_intermediate_size=128,
        shared_intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=8,
        num_experts_per_tok=4,
        routed_scaling_factor=2.0,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0},
        rotary_dim=8,
        hidden_act="swigluoai",
        tie_word_embeddings=False,
        index_n_heads=2,
        index_head_dim=16,
        index_block_size=4,
        index_topk_blocks=2,
        index_local_blocks=1,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 2,
        layer_types=["full_attention"] * 3 + ["minimax_m3_sparse"] * 2,
    )


def test_full_model_parity_vs_hf() -> None:
    """Load the HF state_dict through load_weights, then full-model logit parity."""
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLForCausalLM as HFModel,
    )

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_config()
    hf_cfg._attn_implementation = "eager"
    hf_model = HFModel(hf_cfg).to(torch.float32).eval()

    my_cfg = MiniMaxM3Config.from_hf(hf_cfg)
    my_model = MiniMaxM3ForCausalLM(my_cfg).to(torch.float32).eval()
    MiniMaxM3ForCausalLM.load_weights(my_model, hf_model.state_dict())

    total_q = 12
    input_ids = torch.randint(0, my_cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)

    with torch.inference_mode():
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        cache = _make_cache(my_model)
        my_logits = my_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )

    assert hf_logits.shape == my_logits.shape, f"{hf_logits.shape} vs {my_logits.shape}"
    cs = float(
        torch.nn.functional.cosine_similarity(hf_logits.flatten(), my_logits.flatten(), dim=0)
    )
    assert cs > 0.999, f"full-model logit parity failed: cos_sim={cs:.6f}"
    assert torch.equal(hf_logits.argmax(dim=-1), my_logits.argmax(dim=-1))
    assert torch.allclose(hf_logits, my_logits, atol=1e-3), (
        f"max_abs_diff={(hf_logits - my_logits).abs().max().item():.6f}"
    )
