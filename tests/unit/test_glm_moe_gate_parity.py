"""GLM-MoE-DSA MoE parity vs HF `GlmMoeDsaMoE`.

Two gates: the `noaux_tc` router (selection + weighting) and the full sparse
FFN (router + routed experts + shared expert). The router test runs both
`n_group=1` (the real config, grouping is a no-op) and `n_group=2` so the
grouped-top-k path is actually exercised, and uses a nonzero
`e_score_correction_bias` so the selection-vs-weighting split (biased choice,
unbiased weights) is tested rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN, GlmNoAuxTcGate

HIDDEN = 32
N_ROUTED = 8
TOP_K = 2
MOE_INTER = 16
N_SHARED = 1
SCALING = 2.5
N_TOKENS = 5


def _make_cfg(n_group: int, topk_group: int):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )

    return GlmMoeDsaConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        n_routed_experts=N_ROUTED,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTER,
        n_shared_experts=N_SHARED,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=SCALING,
        hidden_act="silu",
    )


def _dense(idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Scatter (T, top_k) weights into a dense (T, n_routed) vector (order-free)."""
    return torch.zeros(idx.shape[0], N_ROUTED, dtype=w.dtype).scatter(1, idx, w)


def _gate_parity(n_group: int, topk_group: int) -> None:
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMoE

    torch.manual_seed(0)
    cfg = _make_cfg(n_group, topk_group)
    hf_moe = GlmMoeDsaMoE(cfg).eval()
    with torch.no_grad():
        torch.nn.init.normal_(hf_moe.gate.weight, std=0.5)
        hf_moe.gate.e_score_correction_bias.copy_(torch.randn(N_ROUTED))

    ours = GlmNoAuxTcGate(
        hidden_size=HIDDEN,
        n_routed_experts=N_ROUTED,
        top_k=TOP_K,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=SCALING,
    )
    with torch.no_grad():
        ours.weight.copy_(hf_moe.gate.weight)
        ours.e_score_correction_bias.copy_(hf_moe.gate.e_score_correction_bias)

    x = torch.randn(N_TOKENS, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        hf_idx, hf_w = hf_moe.route_tokens_to_experts(hf_moe.gate(x))
        our_idx, our_w = ours(x)

    for t in range(N_TOKENS):
        assert set(hf_idx[t].tolist()) == set(our_idx[t].tolist()), (
            f"token {t}: HF {sorted(hf_idx[t].tolist())} vs ours {sorted(our_idx[t].tolist())}"
        )
    assert torch.allclose(_dense(hf_idx, hf_w), _dense(our_idx, our_w), atol=1e-5)


def test_glm_moe_gate_no_grouping() -> None:
    """n_group=1: grouping degenerates to plain top-k (the real config)."""
    _gate_parity(n_group=1, topk_group=1)


def test_glm_moe_gate_grouped() -> None:
    """n_group=2: the grouped top-k path actually masks groups."""
    _gate_parity(n_group=2, topk_group=1)


def test_glm_moe_ffn_parity() -> None:
    """Full sparse FFN (router + routed experts + shared) matches HF."""
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMoE

    torch.manual_seed(0)
    cfg = _make_cfg(n_group=1, topk_group=1)
    hf_moe = GlmMoeDsaMoE(cfg).eval()
    with torch.no_grad():
        torch.nn.init.normal_(hf_moe.gate.weight, std=0.5)
        hf_moe.gate.e_score_correction_bias.copy_(torch.randn(N_ROUTED))
        torch.nn.init.normal_(hf_moe.experts.gate_up_proj, std=0.1)
        torch.nn.init.normal_(hf_moe.experts.down_proj, std=0.1)

    ours = GlmMoeFFN(
        hidden_size=HIDDEN,
        moe_intermediate_size=MOE_INTER,
        n_routed_experts=N_ROUTED,
        top_k=TOP_K,
        n_shared_experts=N_SHARED,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=SCALING,
    ).eval()

    with torch.no_grad():
        ours.gate.weight.copy_(hf_moe.gate.weight)
        ours.gate.e_score_correction_bias.copy_(hf_moe.gate.e_score_correction_bias)
        for j in range(N_ROUTED):
            gate_up = hf_moe.experts.gate_up_proj[j]  # (2*inter, hidden)
            ours.experts[j].w1.weight.copy_(gate_up[:MOE_INTER])  # gate
            ours.experts[j].w3.weight.copy_(gate_up[MOE_INTER:])  # up
            ours.experts[j].w2.weight.copy_(hf_moe.experts.down_proj[j])  # down
        assert ours.shared_experts is not None
        ours.shared_experts.w1.weight.copy_(hf_moe.shared_experts.gate_proj.weight)
        ours.shared_experts.w3.weight.copy_(hf_moe.shared_experts.up_proj.weight)
        ours.shared_experts.w2.weight.copy_(hf_moe.shared_experts.down_proj.weight)

    x = torch.randn(1, N_TOKENS, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        hf_out = hf_moe(x)
        our_out = ours(x)

    assert hf_out.shape == our_out.shape
    cs = float(
        torch.nn.functional.cosine_similarity(hf_out.flatten(), our_out.flatten(), dim=0).item()
    )
    assert cs > 0.999, f"GLM MoE FFN parity failed: cos_sim={cs:.6f}"
    assert torch.allclose(hf_out, our_out, atol=1e-4), (
        f"max_abs_diff={(hf_out - our_out).abs().max().item():.6f}"
    )
