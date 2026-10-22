"""Grouped-matmul MoE experts against the per-expert loop.

`functional.grouped_mm` also runs on the CPU (a per-group fallback inside
PyTorch), so the routing, sorting and un-sorting logic is tested here in
fp32 without a GPU. The blocks only take the grouped path on Hopper and
newer GPUs; the tests force it by patching `supports_grouped_experts` in the
block's module.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from mini_infer.models.blocks import glm_moe_gate, grouped_experts, mixtral_moe
from mini_infer.models.blocks.fp8_expert import Fp8Expert, PackedFp8Experts
from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN
from mini_infer.models.blocks.grouped_experts import grouped_expert_mlp, supports_grouped_experts
from mini_infer.models.blocks.mixtral_moe import MixtralExpert, MoEFFN, PackedSwigluExperts

pytestmark = pytest.mark.skipif(
    not hasattr(functional, "grouped_mm"), reason="needs torch.nn.functional.grouped_mm"
)

HIDDEN = 32
INTER = 16


def _relu_squared(x: torch.Tensor) -> torch.Tensor:
    return functional.relu(x).square()


def _reference(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
    expert_start: int,
) -> torch.Tensor:
    """One token and one expert at a time; off-rank experts add nothing."""
    out = torch.zeros(x.shape[0], down.shape[1], dtype=torch.float64)
    for token in range(x.shape[0]):
        for slot in range(indices.shape[1]):
            local = int(indices[token, slot]) - expert_start
            if not 0 <= local < gate_up.shape[0]:
                continue
            hidden = activation(x[token : token + 1] @ gate_up[local].T)
            out[token] += (weights[token, slot] * (hidden @ down[local].T)[0]).double()
    return out.to(x.dtype)


def _routing(tokens: int, num_experts: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn(tokens, num_experts)
    weights, indices = torch.topk(scores.softmax(dim=-1), top_k, dim=-1)
    return indices, weights


@pytest.mark.parametrize("tokens", [1, 3, 40])
def test_plain_mlp_matches_reference(tokens: int) -> None:
    torch.manual_seed(0)
    num_experts, top_k = 8, 2
    x = torch.randn(tokens, HIDDEN)
    gate_up = torch.randn(num_experts, INTER, HIDDEN) * 0.1
    down = torch.randn(num_experts, HIDDEN, INTER) * 0.1
    indices, weights = _routing(tokens, num_experts, top_k)

    got = grouped_expert_mlp(
        x, indices, weights, gate_up, down, _relu_squared, num_experts=num_experts
    )
    want = _reference(x, indices, weights, gate_up, down, _relu_squared, expert_start=0)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_gated_mlp_matches_reference() -> None:
    torch.manual_seed(1)
    num_experts, top_k, tokens = 8, 3, 17
    x = torch.randn(tokens, HIDDEN)
    gate_up = torch.randn(num_experts, 2 * INTER, HIDDEN) * 0.1
    down = torch.randn(num_experts, HIDDEN, INTER) * 0.1
    indices, weights = _routing(tokens, num_experts, top_k)

    def swiglu(h: torch.Tensor) -> torch.Tensor:
        return functional.silu(h[:, :INTER]) * h[:, INTER:]

    got = grouped_expert_mlp(x, indices, weights, gate_up, down, swiglu, num_experts=num_experts)
    want = _reference(x, indices, weights, gate_up, down, swiglu, expert_start=0)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("expert_start", [0, 4, 12])
def test_off_rank_experts_contribute_zero(expert_start: int) -> None:
    """Expert parallelism: this rank holds 4 of 16 experts."""
    torch.manual_seed(2)
    num_experts, num_local, top_k, tokens = 16, 4, 4, 25
    x = torch.randn(tokens, HIDDEN)
    gate_up = torch.randn(num_local, INTER, HIDDEN) * 0.1
    down = torch.randn(num_local, HIDDEN, INTER) * 0.1
    indices, weights = _routing(tokens, num_experts, top_k)

    got = grouped_expert_mlp(
        x,
        indices,
        weights,
        gate_up,
        down,
        _relu_squared,
        num_experts=num_experts,
        expert_start=expert_start,
    )
    want = _reference(x, indices, weights, gate_up, down, _relu_squared, expert_start)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_fp8_kernel_gets_the_expert_count_of_all_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FP8 kernel picks its tile from the rows over all 16 experts, not the 4 local ones."""
    seen: list[int] = []

    def kernel(
        rows: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, **kwargs: int
    ) -> torch.Tensor:
        seen.append(kwargs["num_experts"])
        return rows.new_zeros(rows.shape[0], weight.shape[1])

    monkeypatch.setattr(grouped_experts, "block_fp8_grouped_mm", kernel)
    num_local, tokens = 4, 25
    gate_up = grouped_experts.BlockFp8Weight(
        torch.zeros(num_local, INTER, HIDDEN), torch.ones(num_local, 1, 1), 128
    )
    down = grouped_experts.BlockFp8Weight(
        torch.zeros(num_local, HIDDEN, INTER), torch.ones(num_local, 1, 1), 128
    )
    indices, weights = _routing(tokens, 16, 4)
    grouped_expert_mlp(
        torch.randn(tokens, HIDDEN),
        indices,
        weights,
        gate_up,
        down,
        _relu_squared,
        num_experts=16,
        expert_start=4,
    )
    assert seen == [16, 16]


def test_unused_experts_and_fp32_accumulation() -> None:
    """Every token on two experts out of eight; bf16 rows summed in fp32."""
    torch.manual_seed(3)
    num_experts, tokens = 8, 9
    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16)
    gate_up = (torch.randn(num_experts, INTER, HIDDEN) * 0.1).bfloat16()
    down = (torch.randn(num_experts, HIDDEN, INTER) * 0.1).bfloat16()
    indices = torch.tensor([[5, 2]] * tokens)
    weights = torch.rand(tokens, 2)

    got = grouped_expert_mlp(
        x,
        indices,
        weights,
        gate_up,
        down,
        _relu_squared,
        num_experts=num_experts,
        accumulate_dtype=torch.float32,
    )
    assert got.dtype == torch.bfloat16
    want = _reference(x.float(), indices, weights, gate_up.float(), down.float(), _relu_squared, 0)
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)


def test_supports_grouped_experts_needs_a_hopper_gpu() -> None:
    assert not supports_grouped_experts("cpu")
    assert not supports_grouped_experts(torch.device("cpu"))


def _experts(count: int) -> torch.nn.ModuleList:
    torch.manual_seed(4)
    return torch.nn.ModuleList(MixtralExpert(HIDDEN, INTER) for _ in range(count))


def test_packing_keeps_experts_and_state_dict_keys() -> None:
    experts = _experts(4)
    x = torch.randn(5, HIDDEN)
    before = [expert(x) for expert in experts]
    keys = list(experts.state_dict())

    pack = PackedSwigluExperts(experts)

    assert list(experts.state_dict()) == keys
    for expert, want in zip(experts, before, strict=True):
        torch.testing.assert_close(expert(x), want)
    assert experts[2].w3.weight.data_ptr() == pack.gate_up[2, INTER:].data_ptr()
    assert pack.matches(experts)


def test_pack_follows_load_state_dict_and_repacks_after_move() -> None:
    experts = _experts(4)
    pack = PackedSwigluExperts(experts)
    fresh = _experts(4)
    for module in fresh:
        torch.nn.init.normal_(module.w2.weight)
    # load_state_dict copies in place, so the pack sees the new weights.
    experts.load_state_dict(fresh.state_dict())
    assert pack.matches(experts)
    torch.testing.assert_close(pack.down[3], fresh[3].w2.weight)

    # `.to` gives every parameter new storage, which breaks the link.
    experts.to(torch.float64)
    assert not pack.matches(experts)
    repacked = PackedSwigluExperts.current(pack, experts)
    assert repacked is not pack
    assert repacked.gate_up.dtype == torch.float64
    assert PackedSwigluExperts.current(repacked, experts) is repacked


def test_pack_made_in_inference_mode_still_trains() -> None:
    experts = _experts(2)
    with torch.inference_mode():
        PackedSwigluExperts(experts)
    x = torch.randn(3, HIDDEN)
    experts[1](x).sum().backward()
    assert experts[1].w1.weight.grad is not None


def _force_grouped(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    monkeypatch.setattr(module, "supports_grouped_experts", lambda device: True)


@pytest.mark.parametrize("n_shared_experts", [0, 1])
def test_moe_ffn_grouped_matches_loop(
    monkeypatch: pytest.MonkeyPatch, n_shared_experts: int
) -> None:
    torch.manual_seed(5)
    moe = MoEFFN(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_experts=8,
        top_k=2,
        n_shared_experts=n_shared_experts,
    ).eval()
    hidden = torch.randn(1, 13, HIDDEN)
    with torch.no_grad():
        looped = moe(hidden)
        _force_grouped(monkeypatch, mixtral_moe)
        grouped = moe(hidden)
        again = moe(hidden)
    assert moe._expert_pack is not None
    torch.testing.assert_close(grouped, looped, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(again, grouped)


def test_glm_moe_ffn_grouped_matches_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(6)
    moe = GlmMoeFFN(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        n_routed_experts=8,
        top_k=2,
        routed_scaling_factor=2.5,
    ).eval()
    hidden = torch.randn(1, 11, HIDDEN)
    with torch.no_grad():
        looped = moe(hidden)
        _force_grouped(monkeypatch, glm_moe_gate)
        grouped = moe(hidden)
    torch.testing.assert_close(grouped, looped, rtol=1e-5, atol=1e-6)


def test_glm_fp8_experts_off_the_block_width_keep_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FP8 experts stack gate over up only when the width fills whole scale blocks."""
    assert not PackedFp8Experts.can_pack(INTER)
    assert PackedFp8Experts.can_pack(256)
    monkeypatch.setattr(glm_moe_gate, "supports_block_fp8_grouped_mm", lambda device: True)
    moe = GlmMoeFFN(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        n_routed_experts=4,
        top_k=2,
        expert_dtype="fp8",
    ).eval()
    _force_grouped(monkeypatch, glm_moe_gate)
    with torch.no_grad():
        moe(torch.randn(1, 3, HIDDEN))
    assert moe._expert_pack is None


def _fp8_experts(count: int, hidden: int, inter: int, block: int) -> torch.nn.ModuleList:
    """`Fp8Expert`s with random e4m3 weights and positive block scales."""
    torch.manual_seed(8)
    experts = torch.nn.ModuleList(Fp8Expert(hidden, inter, block=block) for _ in range(count))
    with torch.no_grad():
        for expert in experts:
            for name in ("w1", "w2", "w3"):
                weight = getattr(expert, name)
                weight.copy_((torch.randn(weight.shape) * 0.5).to(torch.float8_e4m3fn))
                scale = getattr(expert, f"{name}_scale")
                scale.copy_(torch.rand(scale.shape) * 0.02 + 0.01)
    return experts


def test_fp8_packing_keeps_experts_and_state_dict_keys() -> None:
    experts = _fp8_experts(3, hidden=48, inter=32, block=16)
    x = torch.randn(4, 48)
    before = [expert(x) for expert in experts]
    keys = list(experts.state_dict())

    pack = PackedFp8Experts(experts)

    assert list(experts.state_dict()) == keys
    for expert, want in zip(experts, before, strict=True):
        torch.testing.assert_close(expert(x), want)
    assert pack.gate_up.weight.shape == (3, 64, 48)
    assert pack.gate_up.scale.shape == (3, 4, 3)
    assert pack.down.scale.shape == (3, 3, 2)
    assert experts[1].w3_scale.data_ptr() == pack.gate_up.scale[1, 2:].data_ptr()
    assert pack.matches(experts)


def test_fp8_pack_repacks_after_a_buffer_is_replaced() -> None:
    """The meta-device loader registers new buffers instead of copying in place."""
    experts = _fp8_experts(2, hidden=32, inter=16, block=16)
    pack = PackedFp8Experts(experts)
    experts[0].register_buffer("w1_scale", experts[0].w1_scale.clone() * 2)
    assert not pack.matches(experts)
    repacked = PackedFp8Experts.current(pack, experts)
    assert repacked is not pack
    torch.testing.assert_close(repacked.gate_up.scale[0, :1], experts[0].w1_scale)


def test_bench_toggle_disables_the_grouped_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grouped_experts, "is_hopper_or_newer_device", lambda device: True)
    assert supports_grouped_experts("cuda")
    monkeypatch.setattr(grouped_experts, "_DISABLED_FOR_BENCH", True)
    assert not supports_grouped_experts("cuda")


def test_nemotron_moe_grouped_matches_gathered_and_looped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_infer.models import nemotron_h
    from mini_infer.models.nemotron_h import NemotronHConfig, _NemotronMoE

    cfg = NemotronHConfig.from_hf(
        SimpleNamespace(
            vocab_size=64,
            hidden_size=HIDDEN,
            layers_block_type=["moe"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            mamba_num_heads=4,
            mamba_head_dim=8,
            ssm_state_size=8,
            n_groups=2,
            conv_kernel=4,
            chunk_size=4,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=INTER,
            moe_shared_expert_intermediate_size=24,
            routed_scaling_factor=2.5,
            max_position_embeddings=128,
        )
    )
    torch.manual_seed(7)
    moe = _NemotronMoE(cfg).eval()
    for tokens in (3, 40):
        hidden = torch.randn(1, tokens, HIDDEN)
        flat = hidden.reshape(-1, HIDDEN)
        with torch.no_grad():
            indices, weights = moe.gate(flat)
            gathered = moe._gathered(flat, indices, weights)
            looped = moe._looped(flat, indices, weights)
            eager = moe(hidden)
            monkeypatch.setattr(nemotron_h, "supports_grouped_experts", lambda device: True)
            grouped = moe(hidden)
            monkeypatch.undo()
        torch.testing.assert_close(gathered, looped, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(grouped, eager, rtol=1e-5, atol=1e-6)
