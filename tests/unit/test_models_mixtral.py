"""Mixtral MoE block parity vs HF reference.

Block-level: our `MoEFFN` (top-k router + per-expert SwiGLU MLP +
weighted sum) reproduces HF Mixtral's `MixtralSparseMoeBlock` output
on shared random input + matching weights. CPU-only, no model load.
"""

import torch
from transformers import MixtralConfig as HFMixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from mini_infer.models.blocks import MixtralExpert, MoEFFN


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().to(torch.float32), b.flatten().to(torch.float32), dim=0
        ).item()
    )


def _sync_weights_from_ours_to_hf(ours: MoEFFN, theirs: MixtralSparseMoeBlock) -> None:
    """Copy our parameter values into HF's fused-tensor expert layout.

    HF stores experts as 3D tensors (`gate_up_proj`, `down_proj`); we store
    them per-expert (`experts[j].w1/w2/w3`). Map ours -> theirs so the two
    blocks compute the same output on identical input.
    """
    with torch.no_grad():
        theirs.gate.weight.copy_(ours.gate.weight)
        # `gate_up_proj[j] = concat([w1, w3], dim=0)`: HF fuses along the
        # intermediate-dim axis so the linear's `chunk(2, dim=-1)` recovers
        # gate (w1) and up (w3) in that order.
        for j, expert in enumerate(ours.experts):
            theirs.experts.gate_up_proj[j, : expert.w1.weight.shape[0]].copy_(expert.w1.weight)
            theirs.experts.gate_up_proj[j, expert.w1.weight.shape[0] :].copy_(expert.w3.weight)
            theirs.experts.down_proj[j].copy_(expert.w2.weight)


def test_moe_ffn_matches_hf_mixtral() -> None:
    torch.manual_seed(0)
    hidden_size = 32
    intermediate_size = 64
    num_experts = 4
    top_k = 2
    seq_len = 12

    ours = MoEFFN(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    hf_cfg = HFMixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_act="silu",
        router_jitter_noise=0.0,
    )
    theirs = MixtralSparseMoeBlock(hf_cfg)
    _sync_weights_from_ours_to_hf(ours, theirs)
    theirs.eval()
    ours.eval()

    x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)

    with torch.no_grad():
        out_ours = ours(x)
        out_theirs = theirs(x)

    assert _cos_sim(out_ours, out_theirs) > 0.999
    assert torch.allclose(out_ours, out_theirs, atol=1e-5)


def test_mixtral_expert_matches_hand_rolled_swiglu() -> None:
    """`MixtralExpert(w1, w2, w3)` is the standard SwiGLU computation.

    Verifies the w1/w2/w3 -> gate/down/up mapping by comparing against
    a direct silu-gated FFN on the same weights.
    """
    torch.manual_seed(0)
    hidden_size = 16
    intermediate_size = 48
    expert = MixtralExpert(hidden_size, intermediate_size)

    x = torch.randn(4, hidden_size, dtype=torch.float32)
    out = expert(x)

    # Reference: silu(x @ w1.T) * (x @ w3.T) @ w2.T
    gate = torch.nn.functional.silu(x @ expert.w1.weight.T)
    up = x @ expert.w3.weight.T
    expected = (gate * up) @ expert.w2.weight.T

    assert torch.allclose(out, expected, atol=1e-6)
