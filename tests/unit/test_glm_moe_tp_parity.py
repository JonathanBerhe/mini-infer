"""Full GLM-MoE-DSA model tensor-parallel parity (CPU gloo, world_size=2).

Validates the expert-parallel weight load added to `GlmMoeDsaForCausalLM.load_weights`:
under TP each rank materializes only its slice of routed experts (global->local
remap, off-rank dropped). The whole model (MLA + DSA indexer + noaux_tc MoE +
shared expert + lm_head) must then produce logits identical across ranks and
matching the world_size=1 reference. This is the $0 gate before the NCCL GPU run.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process

_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]


def _make_hf_cfg():  # type: ignore[no-untyped-def]
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,  # 2 per rank at ws=2
        num_key_value_heads=4,
        kv_lora_rank=32,
        q_lora_rank=24,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,  # 2 experts per rank at ws=2
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        index_topk=4,
        index_n_heads=2,
        index_head_dim=16,
        mlp_layer_types=["dense", "dense", "dense", "sparse"],  # exercises MoE layer
        indexer_types=["full", "shared", "full", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        hidden_act="silu",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _build_mini():  # type: ignore[no-untyped-def]
    from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    return GlmMoeDsaForCausalLM(GlmMoeDsaConfig.from_hf(_make_hf_cfg())).to(torch.float32).eval()


def _prefill_logits_list(mini) -> list:  # type: ignore[no-untyped-def]
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    pool = BlockPool(
        num_blocks=16,
        block_size=4,
        num_layers=mini.cfg.num_hidden_layers,
        num_kv_heads=1,
        head_dim=mini.cfg.kv_lora_rank,
        dtype=torch.float32,
        device="cpu",
        layer_streams=mini.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    plen = len(_PROMPT)
    with torch.inference_mode():
        logits = mini(
            input_ids=torch.tensor([_PROMPT], dtype=torch.long),
            position_ids=torch.arange(plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], dtype=torch.int32),
        )
    return logits[0].detach().tolist()


def _moe_tp_worker(rank: int, world_size: int, hf_state_dict: dict) -> list:
    """Build the sharded model at this rank, EP-load the full state_dict, prefill."""
    from mini_infer.models.glm_moe_dsa import GlmMoeDsaForCausalLM

    mini = _build_mini()  # self-shards heads + experts at this rank
    GlmMoeDsaForCausalLM.load_weights(mini, hf_state_dict)
    return _prefill_logits_list(mini)


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_glm_moe_model_world_size_2_matches_reference() -> None:
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM as HFModel,
    )

    from mini_infer.models.glm_moe_dsa import GlmMoeDsaForCausalLM

    torch.manual_seed(0)
    hf_model = HFModel(_make_hf_cfg()).to("cpu", torch.float32).eval()
    state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}

    # world_size=1 reference (EP load is identity at ws=1).
    ref_mini = _build_mini()
    GlmMoeDsaForCausalLM.load_weights(ref_mini, state_dict)
    ref = torch.tensor(_prefill_logits_list(ref_mini))

    per_rank = run_multi_process(2, _moe_tp_worker, state_dict)
    assert len(per_rank) == 2
    rank0 = torch.tensor(per_rank[0])
    rank1 = torch.tensor(per_rank[1])

    # TP consistency: every rank produces the same logits (all-reduces agree).
    assert torch.allclose(rank0, rank1, atol=1e-4), (
        f"ranks diverged: max abs diff {(rank0 - rank1).abs().max().item():.2e}"
    )
    # Correctness: matches the world_size=1 reference.
    cs = float(torch.nn.functional.cosine_similarity(rank0.flatten(), ref.flatten(), dim=0))
    assert cs > 0.999, f"rank0 vs ws=1 ref cosine {cs:.6f}"
    assert torch.equal(rank0.argmax(-1), ref.argmax(-1))
