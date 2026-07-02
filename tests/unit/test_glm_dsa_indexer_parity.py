"""GlmDsaIndexer top-k parity vs HF `GlmMoeDsaIndexer`.

The indexer's job is selection: for each query, which past tokens does the
sparse attention keep? So the correctness gate is that our selected key set
matches HF's, on a *selective* config (`index_topk < seq_len`) where the
choice actually bites. We compare per query restricted to causally-valid keys
(`j <= i`); the padding region a too-small valid count pulls in is `-inf`
either way and the attention path re-masks it, so it carries no signal.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.blocks.glm_dsa_indexer import GlmDsaIndexer
from mini_infer.models.blocks.rope import RotaryEmbedding

HIDDEN = 64
Q_LORA_RANK = 24
INDEX_N_HEADS = 2
INDEX_HEAD_DIM = 16
QK_ROPE = 8
SEQ_LEN = 6
ROPE_THETA = 10000.0


def _make_cfg(index_topk: int):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )

    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=Q_LORA_RANK,
        qk_rope_head_dim=QK_ROPE,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=index_topk,
        indexer_types=["full"],
        rope_parameters={"rope_theta": ROPE_THETA, "rope_type": "default"},
    )


def _make_ours(index_topk: int) -> GlmDsaIndexer:
    return GlmDsaIndexer(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA_RANK,
        num_heads=INDEX_N_HEADS,
        head_dim=INDEX_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE,
        index_topk=index_topk,
    ).eval()


def _sync(ours: GlmDsaIndexer, hf_indexer: torch.nn.Module) -> None:
    with torch.no_grad():
        ours.wq_b.weight.copy_(hf_indexer.wq_b.weight)
        ours.wk.weight.copy_(hf_indexer.wk.weight)
        ours.k_norm.weight.copy_(hf_indexer.k_norm.weight)
        ours.k_norm.bias.copy_(hf_indexer.k_norm.bias)
        ours.weights_proj.weight.copy_(hf_indexer.weights_proj.weight)


def _run_parity(index_topk: int) -> None:
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer

    torch.manual_seed(0)
    cfg = _make_cfg(index_topk)
    hf_indexer = GlmMoeDsaIndexer(cfg, layer_idx=0).eval()
    ours = _make_ours(index_topk)
    _sync(ours, hf_indexer)

    hidden_states = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)
    q_resid = torch.randn(1, SEQ_LEN, Q_LORA_RANK, dtype=torch.float32)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    cos, sin = RotaryEmbedding(head_dim=QK_ROPE, base=ROPE_THETA)(hidden_states, position_ids)

    # HF wants a 3D additive causal mask [B, S, T].
    causal3d = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), float("-inf"), dtype=torch.float32), diagonal=1
    ).unsqueeze(0)
    with torch.no_grad():
        hf_topk = hf_indexer(hidden_states, q_resid, (cos, sin), causal3d, position_ids)
    assert hf_topk.shape == (1, SEQ_LEN, min(index_topk, SEQ_LEN))

    cu_seqlens_q = torch.tensor([0, SEQ_LEN], dtype=torch.int32)
    with torch.no_grad():
        our_topk = ours(hidden_states, q_resid, (cos, sin), cu_seqlens_q)
    assert len(our_topk) == 1
    assert our_topk[0].shape == (SEQ_LEN, min(index_topk, SEQ_LEN))

    # Compare causally-valid selections per query.
    for i in range(SEQ_LEN):
        valid = set(range(i + 1))
        hf_sel = set(hf_topk[0, i].tolist()) & valid
        our_sel = set(our_topk[0][i].tolist()) & valid
        assert hf_sel == our_sel, f"query {i}: HF selected {sorted(hf_sel)}, ours {sorted(our_sel)}"


def test_glm_dsa_indexer_topk_selective() -> None:
    """index_topk < seq_len: the selection actually discriminates."""
    _run_parity(index_topk=3)


def test_glm_dsa_indexer_topk_allpass() -> None:
    """index_topk >= seq_len: every causally-valid key is selected by both."""
    _run_parity(index_topk=SEQ_LEN + 2)
