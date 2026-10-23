"""DeepSeek-V4 MoE on the grouped path: packing and dispatch, on the CPU.

`PackedFp4Experts` must keep each `FP4Expert` working and its `state_dict`
keys unchanged. `HashRoutedMoEFFN` with bf16 experts must give the loop's
result on the grouped path (`functional.grouped_mm` also runs on the CPU),
and must keep the loop when the reference's FP8 GEMM-input round trip is on.
The FP4 kernel itself is tested in `test_fp4_grouped_kernel.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from mini_infer.models.blocks import hash_routed_moe_ffn
from mini_infer.models.blocks.fp4_expert import FP4Expert, PackedFp4Experts
from mini_infer.models.blocks.grouped_experts import BlockFp4Weight
from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
from mini_infer.models.blocks.mixtral_moe import PackedSwigluExperts
from mini_infer.models.blocks.v4.act_quant import SimulatedActQuant

pytestmark = pytest.mark.skipif(
    not hasattr(functional, "grouped_mm"), reason="needs torch.nn.functional.grouped_mm"
)

HIDDEN = 64
INTER = 32


def fill_fp4(expert: FP4Expert, generator: torch.Generator) -> None:
    """Random packed e2m1 bytes and positive scales in every buffer."""
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            packed = getattr(expert, f"{name}_packed")
            packed.copy_(
                torch.randint(-128, 128, packed.shape, dtype=torch.int8, generator=generator)
            )
            scale = getattr(expert, f"{name}_scale")
            scale.copy_(torch.rand(scale.shape, generator=generator) * 0.02 + 0.01)


def _fp4_experts(count: int) -> torch.nn.ModuleList:
    generator = torch.Generator().manual_seed(3)
    experts = torch.nn.ModuleList(FP4Expert(HIDDEN, INTER, swiglu_limit=10.0) for _ in range(count))
    for expert in experts:
        fill_fp4(expert, generator)
    return experts


def test_block_fp4_weight_reports_the_dequantized_shape() -> None:
    weight = BlockFp4Weight(torch.zeros(3, 8, 16, dtype=torch.int8), torch.zeros(3, 8, 1), 32)
    assert weight.shape == (3, 8, 32)


def test_fp4_packing_keeps_experts_and_state_dict_keys() -> None:
    experts = _fp4_experts(3)
    x = torch.randn(4, HIDDEN)
    before = [expert(x) for expert in experts]
    keys = list(experts.state_dict())

    pack = PackedFp4Experts(experts)

    assert list(experts.state_dict()) == keys
    for expert, want in zip(experts, before, strict=True):
        torch.testing.assert_close(expert(x), want)
    assert pack.gate_up.packed.shape == (3, 2 * INTER, HIDDEN // 2)
    assert pack.gate_up.scale.shape == (3, 2 * INTER, HIDDEN // 32)
    assert pack.down.packed.shape == (3, HIDDEN, INTER // 2)
    assert experts[1].w3_packed.data_ptr() == pack.gate_up.packed[1, INTER:].data_ptr()
    assert pack.matches(experts)


def test_fp4_pack_repacks_after_a_buffer_is_replaced() -> None:
    """The meta-device loader registers new buffers instead of copying in place."""
    experts = _fp4_experts(2)
    pack = PackedFp4Experts(experts)
    experts[0].register_buffer("w1_scale", experts[0].w1_scale.clone() * 2)
    assert not pack.matches(experts)
    repacked = PackedFp4Experts.current(pack, experts)
    assert repacked is not pack
    torch.testing.assert_close(repacked.gate_up.scale[0, :INTER], experts[0].w1_scale)


def _force_grouped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_grouped_experts", lambda device: True)


@pytest.mark.parametrize("routing_mode", ["score_topk", "hash"])
def test_bf16_experts_grouped_match_the_loop(
    monkeypatch: pytest.MonkeyPatch, routing_mode: str
) -> None:
    torch.manual_seed(4)
    vocab = 50
    moe = HashRoutedMoEFFN(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_routed_experts=8,
        num_activated_experts=3,
        routing_mode=routing_mode,  # type: ignore[arg-type]
        vocab_size=vocab if routing_mode == "hash" else None,
        n_shared_experts=1,
        swiglu_limit=10.0,
    ).eval()
    if routing_mode == "hash":
        with torch.no_grad():
            moe.gate.tid2eid.copy_(torch.randint(0, 8, moe.gate.tid2eid.shape))
    hidden = torch.randn(1, 13, HIDDEN)
    input_ids = torch.randint(0, vocab, (1, 13))
    with torch.no_grad():
        looped = moe(hidden, input_ids)
        _force_grouped(monkeypatch)
        grouped = moe(hidden, input_ids)
    assert isinstance(moe._expert_pack, PackedSwigluExperts)
    torch.testing.assert_close(grouped, looped, rtol=1e-5, atol=1e-6)


def test_fp8_gemm_input_round_trip_keeps_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = HashRoutedMoEFFN(
        hidden_size=128,
        intermediate_size=128,
        num_routed_experts=4,
        num_activated_experts=2,
        routing_mode="score_topk",
        act_quant=SimulatedActQuant(fp8_scale_fmt=None, fp8_gemm_inputs=True),
    ).eval()
    _force_grouped(monkeypatch)
    with torch.no_grad():
        moe(torch.randn(1, 3, 128))
    assert moe._expert_pack is None
