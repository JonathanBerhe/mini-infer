"""`_KimiMoe` expert dispatch against the per-expert loop it replaced.

The old forward ran a `torch.where` over the routing mask of every expert,
one host sync per expert on a GPU. `_per_expert_loop` keeps that code as the
reference. Both configurations are covered: Kimi Linear (biased expert
weights, SwiGLU, full-width experts) and Kimi K3 (unbiased weights, `-inf`
group mask, SiTU, latent MoE with its RMSNorm, two shared experts).

`functional.grouped_mm` also runs on the CPU, so the grouped path is tested
here in fp32 by patching `supports_grouped_experts` in the Kimi module, as
`tests/unit/models/blocks/moe/test_grouped_experts.py` does for the other
families. The no-sync test of the grouped path needs a CUDA GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.nn import functional

from mini_infer.models.blocks.moe.grouped_experts import supports_grouped_experts
from mini_infer.models.kimi_linear import kimi_linear
from mini_infer.models.kimi_linear.kimi_linear import KimiLinearConfig, _KimiMoe

NUM_EXPERTS = 16


def _config(kind: str) -> KimiLinearConfig:
    """One MoE layer of each convention, with four expert groups."""
    fields: dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=4,
        num_shared_experts=1,
        routed_scaling_factor=2.446,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_expert_group=4,
        topk_group=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    if kind == "k3":
        fields.update(
            hidden_act="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            num_shared_experts=2,
            routed_expert_hidden_size=32,
            latent_moe_use_norm=True,
        )
    cfg = KimiLinearConfig.from_hf(SimpleNamespace(**fields))
    assert cfg.moe_weights_from_biased_scores is (kind == "kimi_linear")
    return cfg


def _moe(kind: str, dtype: torch.dtype = torch.float32) -> _KimiMoe:
    torch.manual_seed(0)
    moe = _KimiMoe(_config(kind))
    with torch.no_grad():
        # A nonzero correction bias makes the biased and unbiased weights differ.
        moe.gate.e_score_correction_bias.uniform_(-0.2, 0.2)
        if hasattr(moe, "routed_expert_norm"):
            moe.routed_expert_norm.weight.uniform_(0.5, 1.5)
    return moe.to(dtype).eval()


def _per_expert_loop(moe: _KimiMoe, x: torch.Tensor) -> torch.Tensor:
    """The `_KimiMoe.forward` before the grouped dispatch, verbatim."""
    input_shape = x.shape
    flat = x.view(-1, x.shape[-1])
    topk_indices, topk_weights = moe.gate(flat)
    routed_in = moe.routed_expert_down_proj(flat) if moe.latent_size is not None else flat

    routed = torch.zeros(routed_in.shape, dtype=torch.float32, device=flat.device)
    expert_mask = functional.one_hot(topk_indices, num_classes=moe.num_experts).permute(2, 1, 0)
    for expert_idx in range(moe.num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        weights = topk_weights[token_idx, top_k_pos, None]
        routed.index_add_(
            0, token_idx, moe.experts[expert_idx](routed_in[token_idx]).float() * weights
        )
    out = routed.to(x.dtype)
    if moe.latent_size is not None:
        if hasattr(moe, "routed_expert_norm"):
            out = moe.routed_expert_norm(out)
        out = moe.routed_expert_up_proj(out)
    if moe.shared_experts is not None:
        out = out + moe.shared_experts(flat)
    return out.view(input_shape)


def _hidden(tokens: int, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    torch.manual_seed(tokens)
    return torch.randn(1, tokens, 48, dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 5, 64])
@pytest.mark.parametrize("kind", ["kimi_linear", "k3"])
def test_loop_has_the_bits_of_the_per_expert_loop(
    kind: str, tokens: int, dtype: torch.dtype
) -> None:
    moe = _moe(kind, dtype)
    x = _hidden(tokens, dtype)
    with torch.inference_mode():
        got = moe(x)
        want = _per_expert_loop(moe, x)
    assert moe._expert_pack is None
    assert torch.equal(got, want)


def test_loop_skips_experts_without_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """One token picks 4 of 16 experts: only those 4 run, and nothing reads `nonzero`."""
    moe = _moe("kimi_linear")
    ran: list[int] = []
    for idx, expert in enumerate(moe.experts):
        expert.register_forward_hook(lambda module, args, out, idx=idx: ran.append(idx))
    x = _hidden(1, torch.float32)
    with torch.inference_mode(), torch.profiler.profile(acc_events=True) as prof:
        moe(x)
    chosen, _ = moe.gate(x.view(1, -1))
    assert ran == sorted(chosen[0].tolist())
    assert not [e for e in prof.events() if e.name == "aten::nonzero"]


def _force_grouped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kimi_linear, "supports_grouped_experts", lambda device, dtype=None: True)


@pytest.mark.skipif(
    not hasattr(functional, "grouped_mm"), reason="needs torch.nn.functional.grouped_mm"
)
@pytest.mark.parametrize("tokens", [1, 5, 64])
@pytest.mark.parametrize("kind", ["kimi_linear", "k3"])
def test_grouped_path_matches_the_per_expert_loop(
    monkeypatch: pytest.MonkeyPatch, kind: str, tokens: int
) -> None:
    moe = _moe(kind)
    keys = list(moe.state_dict())
    x = _hidden(tokens, torch.float32)
    with torch.inference_mode():
        want = _per_expert_loop(moe, x)
        _force_grouped(monkeypatch)
        got = moe(x)
        again = moe(x)
    assert moe._expert_pack is not None
    assert list(moe.state_dict()) == keys
    # Same products; only the order of the fp32 sum over a token's experts differs.
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert torch.equal(again, got)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("kind", ["kimi_linear", "k3"])
def test_grouped_path_has_no_host_sync(kind: str) -> None:
    if not supports_grouped_experts("cuda", torch.bfloat16):
        pytest.skip("the grouped path needs a Hopper or newer GPU")
    moe = _moe(kind).to(device="cuda", dtype=torch.bfloat16)
    x = _hidden(8, torch.bfloat16, device="cuda")
    with torch.inference_mode():
        want = _per_expert_loop(moe, x)
        moe(x)  # packs the experts
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            got = moe(x)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)
