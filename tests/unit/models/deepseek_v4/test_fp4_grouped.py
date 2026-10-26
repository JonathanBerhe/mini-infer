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

from mini_infer.models.blocks.moe.grouped_experts import BlockFp4Weight
from mini_infer.models.blocks.moe.mixtral_moe import PackedSwigluExperts
from mini_infer.models.deepseek_v4 import hash_routed_moe_ffn
from mini_infer.models.deepseek_v4.act_quant import SimulatedActQuant
from mini_infer.models.deepseek_v4.fp4_expert import FP4Expert, PackedFp4Experts
from mini_infer.models.deepseek_v4.hash_routed_moe_ffn import HashRoutedMoEFFN

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


def _power_of_two_scales(experts: torch.nn.ModuleList) -> None:
    """Round every scale to a power of two, as in V4's e8m0 checkpoints."""
    with torch.no_grad():
        for expert in experts:
            for name in ("w1", "w2", "w3"):
                scale = getattr(expert, f"{name}_scale")
                scale.copy_(torch.exp2(torch.round(torch.log2(scale))))


@pytest.mark.parametrize("power_of_two", [False, True])
def test_fp4_pack_takes_fp8_inputs_only_when_the_scales_allow(power_of_two: bool) -> None:
    """`fp8_activations=None` runs W4A8 only with e8m0 scales, and else keeps bf16 inputs."""
    experts = _fp4_experts(2)
    if power_of_two:
        _power_of_two_scales(experts)
    pack = PackedFp4Experts(experts, fp8_activations=None)
    assert pack.fp8_activations is power_of_two
    assert (pack.gate_up.e8m0_scale is not None) is power_of_two
    assert (pack.down.e8m0_scale is not None) is power_of_two
    if pack.down.e8m0_scale is not None:
        torch.testing.assert_close(torch.exp2(pack.down.e8m0_scale.float() - 127), pack.down.scale)
    else:
        with pytest.raises(ValueError, match="power of two"):
            PackedFp4Experts(experts, fp8_activations=True)


@pytest.mark.parametrize("native", [False, True])
def test_auto_fp8_activations_need_a_native_gpu(
    monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    """`fp8_activations=None` takes W4A8 only where `supports_native_w4a8` holds."""
    moe = HashRoutedMoEFFN(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_routed_experts=4,
        num_activated_experts=2,
        routing_mode="score_topk",
        expert_dtype="fp4",
        fp8_activations=None,
    )
    generator = torch.Generator().manual_seed(5)
    for expert in moe.experts:
        fill_fp4(expert, generator)  # type: ignore[arg-type]
    _power_of_two_scales(moe.experts)
    _force_grouped(monkeypatch)
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_block_fp4_grouped_mm", lambda device: True)
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_native_w4a8", lambda device: native)
    rows = torch.zeros(3, HIDDEN, dtype=torch.bfloat16)
    pack = moe._grouped_pack(rows)
    assert isinstance(pack, PackedFp4Experts)
    assert pack.fp8_activations is native
    # The next forward reuses the pack: no new read-back of the scales.
    assert moe._grouped_pack(rows) is pack


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
