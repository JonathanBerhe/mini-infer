"""swigluoai activation + its use in the M3 dense MLP and MoE experts.

`swigluoai` is the only genuinely-new math in the non-MSA half of MiniMax-M3; the
rest is the GLM-MoE-DSA gate + dispatch reused unchanged. These tests check:
  - the clamped formula against an independent reference, including the clamps;
  - that the default `swiglu` combiner leaves existing blocks bit-identical;
  - that `GlmMoeFFN` with swigluoai experts reproduces M3's MoE math (sigmoid +
    selection-bias routing, unbiased normalized weights, routed scaling, shared
    expert) against a from-scratch reference forward.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from mini_infer.models.blocks.activations import swigluoai
from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN
from mini_infer.models.blocks.mixtral_moe import MixtralExpert
from mini_infer.models.blocks.swiglu import SwiGLU


def test_swigluoai_matches_reference_formula() -> None:
    """(clamp(up,-L,L)+1) * g * sigmoid(alpha*g), g=clamp(gate,max=L), incl. clamps."""
    torch.manual_seed(0)
    # Spread values well past +/-7 so both clamps are exercised on real entries.
    gate = torch.randn(4, 32) * 6.0
    up = torch.randn(4, 32) * 6.0
    alpha, limit = 1.702, 7.0

    g = gate.clamp(max=limit)
    u = up.clamp(min=-limit, max=limit)
    expected = (u + 1.0) * g * torch.sigmoid(alpha * g)

    assert torch.equal(swigluoai(gate, up), expected)
    # Clamps actually fired (some entries exceeded the bounds).
    assert (gate > limit).any() and (up.abs() > limit).any()


def test_swigluoai_gate_clamp_is_max_only() -> None:
    """Gate has no lower clamp; a very negative gate passes through to the SiLU."""
    gate = torch.tensor([[-100.0]])
    up = torch.tensor([[0.0]])
    # up+1=1, g=-100 (unclamped below), g*sigmoid(1.702*g) ~ 0 -> output ~ 0, finite.
    out = swigluoai(gate, up)
    assert torch.isfinite(out).all()
    assert out.abs().item() < 1e-3  # silu(-100) ~ 0


def test_swiglu_default_leaves_blocks_identical() -> None:
    """Default `swiglu` combiner == the old silu(gate)*up path (regression guard)."""
    torch.manual_seed(1)
    x = torch.randn(2, 5, 16)

    mlp = SwiGLU(16, 32).eval()  # default activation
    ref = mlp.down_proj(functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert torch.equal(mlp(x), ref)

    expert = MixtralExpert(16, 32).eval()  # default activation
    ref_e = expert.w2(functional.silu(expert.w1(x)) * expert.w3(x))
    assert torch.equal(expert(x), ref_e)


def test_swiglu_uses_the_given_combiner() -> None:
    """SwiGLU(activation=swigluoai) applies the clamped combiner, not silu."""
    torch.manual_seed(2)
    x = torch.randn(2, 4, 16)
    mlp = SwiGLU(16, 32, activation=swigluoai).eval()
    expected = mlp.down_proj(swigluoai(mlp.gate_proj(x), mlp.up_proj(x)))
    assert torch.equal(mlp(x), expected)
    # And it differs from the silu default (sanity that the swap took effect).
    assert not torch.allclose(mlp(x), SwiGLU(16, 32).eval()(x))


def _m3_moe_reference(
    moe: GlmMoeFFN, x: torch.Tensor, *, top_k: int, scaling: float
) -> torch.Tensor:
    """From-scratch M3 MoE forward (sigmoid+bias selection, unbiased normalized
    weights, routed scaling, swigluoai experts + shared), read from `moe`'s own
    weights. n_group=topk_group=1 so grouping is a no-op (plain top-k)."""
    flat = x.view(-1, x.shape[-1])
    w = functional.linear(flat.float(), moe.gate.weight.float()).sigmoid()  # [T,E]
    choice = w + moe.gate.e_score_correction_bias
    top_idx = torch.topk(choice, top_k, dim=-1).indices
    top_w = w.gather(1, top_idx)
    top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-20)
    top_w = (top_w * scaling).to(x.dtype)

    def expert_fwd(e: MixtralExpert, t: torch.Tensor) -> torch.Tensor:
        return e.w2(swigluoai(e.w1(t), e.w3(t)))

    out = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(top_k):
            e = int(top_idx[token, slot])
            out[token] += (
                top_w[token, slot] * expert_fwd(moe.experts[e], flat[token : token + 1])[0]
            )
    assert moe.shared_experts is not None
    out = out + expert_fwd(moe.shared_experts, flat)
    return out.view(x.shape)


def test_glm_moe_with_swigluoai_matches_m3_reference() -> None:
    """GlmMoeFFN(activation=swigluoai) reproduces M3's MoE math end to end."""
    torch.manual_seed(3)
    hidden, moe_inter, n_exp, top_k, scaling = 16, 8, 8, 2, 2.0
    moe = GlmMoeFFN(
        hidden_size=hidden,
        moe_intermediate_size=moe_inter,
        n_routed_experts=n_exp,
        top_k=top_k,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=scaling,
        activation=swigluoai,
    ).eval()
    # Non-zero selection bias so the biased-selection / unbiased-weight split is
    # actually exercised (default bias is all zeros).
    moe.gate.e_score_correction_bias.copy_(torch.randn(n_exp))

    x = torch.randn(1, 6, hidden)
    expected = _m3_moe_reference(moe, x, top_k=top_k, scaling=scaling)
    assert torch.allclose(moe(x), expected, atol=1e-6, rtol=1e-5)
