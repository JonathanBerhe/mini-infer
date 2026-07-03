"""Full MiniMax-M3 model tensor-parallel parity (CPU gloo, world_size=2).

Validates the sharded load in `MiniMaxM3ForCausalLM.load_weights`: under TP each
rank materializes its head slice of the GQA projections (column/row-parallel)
and only its contiguous slice of routed experts (expert-parallel, global->local
remap); the indexer stays replicated (block selection is global per query). The
whole model (GQA + qk-norm + MSA indexer + swigluoai MoE + shared expert +
lm_head) must produce logits identical across ranks and matching the
world_size=1 reference. This is the $0 gate before any real GPU TP run.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process

_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]  # 12 tokens -> 3 index blocks of 4


def _make_hf_cfg():  # type: ignore[no-untyped-def]
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )

    cfg = MiniMaxM3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,  # MoE per-expert
        dense_intermediate_size=128,
        shared_intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=8,  # 4 per rank at ws=2
        num_key_value_heads=2,  # 1 per rank at ws=2
        head_dim=16,
        num_local_experts=8,  # 4 experts per rank at ws=2
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
        index_topk_blocks=2,  # < 3 blocks at the last queries -> selection bites
        index_local_blocks=1,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 2,
        layer_types=["full_attention"] * 3 + ["minimax_m3_sparse"] * 2,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _build_mini():  # type: ignore[no-untyped-def]
    from mini_infer.models.minimax_m3 import MiniMaxM3Config, MiniMaxM3ForCausalLM

    return MiniMaxM3ForCausalLM(MiniMaxM3Config.from_hf(_make_hf_cfg())).to(torch.float32).eval()


def _prefill_logits_list(mini) -> list:  # type: ignore[no-untyped-def]
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    pool = BlockPool(
        num_blocks=16,
        block_size=4,
        num_layers=mini.cfg.num_hidden_layers,
        num_kv_heads=mini.cfg.num_key_value_heads,
        head_dim=mini.cfg.head_dim,
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


def _m3_tp_worker(rank: int, world_size: int, hf_state_dict: dict) -> list:
    """Build the sharded model at this rank, TP/EP-load the full state_dict, prefill."""
    from mini_infer.models.minimax_m3 import MiniMaxM3ForCausalLM

    mini = _build_mini()  # self-shards heads + experts at this rank
    MiniMaxM3ForCausalLM.load_weights(mini, hf_state_dict)
    return _prefill_logits_list(mini)


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_minimax_m3_model_world_size_2_matches_reference() -> None:
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLForCausalLM as HFModel,
    )

    from mini_infer.models.minimax_m3 import MiniMaxM3ForCausalLM

    torch.manual_seed(0)
    hf_model = HFModel(_make_hf_cfg()).to("cpu", torch.float32).eval()
    state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}

    # world_size=1 reference (TP/EP load is identity at ws=1).
    ref_mini = _build_mini()
    MiniMaxM3ForCausalLM.load_weights(ref_mini, state_dict)
    ref = torch.tensor(_prefill_logits_list(ref_mini))

    per_rank = run_multi_process(2, _m3_tp_worker, state_dict)
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
