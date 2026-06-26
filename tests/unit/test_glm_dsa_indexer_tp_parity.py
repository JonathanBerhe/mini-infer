"""Tensor-parallel parity for the GLM DSA indexer.

The indexer shards its heads (wq_b, weights_proj are column-parallel) and
all-reduces the head-summed score before top-k. The correctness property TP
must preserve: every rank picks the SAME keys (otherwise the per-rank attention
masks would diverge and the sharded heads would attend to different tokens).
So the gate is rank-consistency plus a match to the world_size=1 reference.

Pattern mirrors `test_attention_tp_parity.py`: build the world_size=1 reference
in the parent, spawn 2 gloo ranks that self-shard and load the same full
weights, compare.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models.blocks.glm_dsa_indexer import GlmDsaIndexer
from mini_infer.models.blocks.rope import RotaryEmbedding
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process

HIDDEN = 32
Q_LORA_RANK = 24
INDEX_N_HEADS = 2  # 2 / 2 = 1 head per rank at world_size=2
INDEX_HEAD_DIM = 16
QK_ROPE = 8
INDEX_TOPK = 3  # < SEQ_LEN: selection actually discriminates
SEQ_LEN = 6
ROPE_THETA = 10000.0


def _make_indexer() -> GlmDsaIndexer:
    return GlmDsaIndexer(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA_RANK,
        num_heads=INDEX_N_HEADS,
        head_dim=INDEX_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE,
        index_topk=INDEX_TOPK,
    )


def test_indexer_world_size_1_construction() -> None:
    """At world_size=1 the head-parallel projections are TP-aware but full;
    the shared key projection stays a plain replicated nn.Linear."""
    block = _make_indexer()
    assert isinstance(block.wq_b, ColumnParallelLinear)
    assert isinstance(block.weights_proj, ColumnParallelLinear)
    assert isinstance(block.wk, nn.Linear)
    assert not isinstance(block.wk, ColumnParallelLinear)
    assert block.num_heads_local == INDEX_N_HEADS
    assert block.wq_b.weight.shape == (INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK)
    assert block.weights_proj.weight.shape == (INDEX_N_HEADS, HIDDEN)


def _indexer_topk_worker(
    rank: int,
    world_size: int,
    full_wq_b: torch.Tensor,
    full_weights_proj: torch.Tensor,
    full_wk: torch.Tensor,
    full_k_norm_w: torch.Tensor,
    full_k_norm_b: torch.Tensor,
    hidden: torch.Tensor,
    q_resid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Build the indexer at this rank (self-shards heads), load the full
    weights, and return the selected top-k indices for the single request."""
    block = _make_indexer()
    block.wq_b.load_full_weight(full_wq_b)
    block.weights_proj.load_full_weight(full_weights_proj)
    with torch.no_grad():
        block.wk.weight.copy_(full_wk)  # replicated
        block.k_norm.weight.copy_(full_k_norm_w)
        block.k_norm.bias.copy_(full_k_norm_b)
    cu_seqlens_q = torch.tensor([0, hidden.shape[1]], dtype=torch.int32)
    topk = block(hidden, q_resid, (cos, sin), cu_seqlens_q)
    return topk[0].detach().cpu()


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_indexer_topk_world_size_2_matches_reference() -> None:
    """world_size=2: head-sharded + all-reduced selection is identical on
    every rank and equals the world_size=1 reference (per causally-valid key)."""
    torch.manual_seed(0)
    full_wq_b = torch.randn(INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK)
    full_weights_proj = torch.randn(INDEX_N_HEADS, HIDDEN)
    full_wk = torch.randn(INDEX_HEAD_DIM, HIDDEN)
    full_k_norm_w = torch.randn(INDEX_HEAD_DIM)
    full_k_norm_b = torch.randn(INDEX_HEAD_DIM)
    hidden = torch.randn(1, SEQ_LEN, HIDDEN)
    q_resid = torch.randn(1, SEQ_LEN, Q_LORA_RANK)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    cos, sin = RotaryEmbedding(head_dim=QK_ROPE, base=ROPE_THETA)(hidden, position_ids)

    # world_size=1 reference (parent process, no PG → all_reduce is identity).
    ref = _make_indexer()
    ref.wq_b.load_full_weight(full_wq_b)
    ref.weights_proj.load_full_weight(full_weights_proj)
    with torch.no_grad():
        ref.wk.weight.copy_(full_wk)
        ref.k_norm.weight.copy_(full_k_norm_w)
        ref.k_norm.bias.copy_(full_k_norm_b)
    ref_topk = ref(hidden, q_resid, (cos, sin), torch.tensor([0, SEQ_LEN], dtype=torch.int32))[0]

    per_rank = run_multi_process(
        2,
        _indexer_topk_worker,
        full_wq_b,
        full_weights_proj,
        full_wk,
        full_k_norm_w,
        full_k_norm_b,
        hidden,
        q_resid,
        cos,
        sin,
    )
    assert len(per_rank) == 2

    # Both ranks pick identical keys (consistency), matching the reference per
    # query restricted to causally-valid keys (j <= i).
    for i in range(SEQ_LEN):
        valid = set(range(i + 1))
        ref_sel = set(ref_topk[i].tolist()) & valid
        for rank, topk in enumerate(per_rank):
            rank_sel = set(topk[i].tolist()) & valid
            assert rank_sel == ref_sel, (
                f"rank {rank} query {i}: {sorted(rank_sel)} vs ref {sorted(ref_sel)}"
            )
