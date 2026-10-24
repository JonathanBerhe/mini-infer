"""Packed-FP4 grouped expert kernel (`blocks/fp4_grouped_kernel.py`) vs dequantized math.

The kernel against a row-by-row matmul with `dequantize_nvfp4_to_bf16`, and
the FP4 `HashRoutedMoEFFN` on its grouped path against its per-expert loop.

Runs on CUDA, or on the CPU under Triton's interpreter
(`TRITON_INTERPRET=1`), which is how the kernel is debugged without a GPU.
Skips otherwise.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("triton")

from mini_infer.models.blocks import fp4_grouped_kernel, hash_routed_moe_ffn
from mini_infer.models.blocks.fp4_expert import PackedFp4Experts
from mini_infer.models.blocks.fp4_grouped_kernel import (
    block_fp4_grouped_mm,
    quantize_e4m3_rows,
    to_e8m0,
)
from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16
from tests.unit.test_fp4_grouped import fill_fp4


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _atol(device: str, want: torch.Tensor) -> float:
    """Absolute tolerance as a share of the largest expected output.

    bf16 rounding of the weights and of the output gives errors that scale
    with the output. GPUs round fp32 to bf16 to nearest. Triton's interpreter
    truncates instead, so every weight moves toward zero and the error adds
    up in one direction: allow twice as much on the CPU.
    """
    share = 1e-2 if device == "cuda" else 2e-2
    return share * float(want.abs().max())


@pytest.mark.parametrize(
    ("per_segment", "expert_tiles"),
    [(None, None), (False, False), (True, None), (False, None)],
    ids=["default", "row-tiles", "segments", "expert-tiles"],
)
@pytest.mark.parametrize(
    ("counts", "off_rank", "power_of_two_scales"),
    [
        ([3, 0, 17, 1, 9], 0, False),  # tiles span several experts, a group over 16 rows
        ([3, 0, 17, 1, 9], 4, True),  # rows routed to other ranks; e8m0-style scales
        ([100, 0, 150, 60, 1], 0, False),  # over 256 rows: the large-tile configuration
    ],
)
def test_kernel_matches_dequantized_matmul(
    monkeypatch: pytest.MonkeyPatch,
    counts: list[int],
    off_rank: int,
    power_of_two_scales: bool,
    per_segment: bool | None,
    expert_tiles: bool | None,
) -> None:
    device = _device()
    monkeypatch.setattr(fp4_grouped_kernel, "_PER_SEGMENT_FOR_BENCH", per_segment)
    monkeypatch.setattr(fp4_grouped_kernel, "_EXPERT_TILES_FOR_BENCH", expert_tiles)
    generator = torch.Generator().manual_seed(0)
    num_experts, n_dim, k_dim, block = len(counts), 48, 96, 32
    packed = torch.randint(
        -128, 128, (num_experts, n_dim, k_dim // 2), dtype=torch.int8, generator=generator
    )
    scale = torch.rand(num_experts, n_dim, k_dim // block, generator=generator) * 0.02 + 0.01
    if power_of_two_scales:
        scale = torch.exp2(torch.round(torch.log2(scale)))
    row_experts = torch.cat(
        [
            torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts)),
            torch.full((off_rank,), num_experts),
        ]
    )
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32)
    rows = torch.randn(len(row_experts), k_dim, generator=generator).bfloat16()

    got = block_fp4_grouped_mm(
        rows.to(device),
        packed.to(device),
        scale.to(device),
        block=block,
        row_experts=row_experts.to(device),
        group_ends=group_ends.to(device),
    ).cpu()

    want = torch.zeros(len(row_experts), n_dim)
    for row, expert in enumerate(row_experts.tolist()):
        if expert >= num_experts:
            continue
        deq = dequantize_nvfp4_to_bf16(packed[expert], scale[expert], block_size=block)
        want[row] = rows[row].float() @ deq.float().T
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=_atol(device, want))


def test_e4m3_rows_take_the_smallest_power_of_two_scale_per_block() -> None:
    device = _device()
    generator = torch.Generator().manual_seed(0)
    magnitude = torch.exp2(torch.randint(-12, 6, (5, 8, 1), generator=generator).float())
    x = (torch.randn(5, 8, 32, generator=generator) * magnitude).view(5, 256)
    x[1, :32] = 0.0  # an all-zero block
    x = x.bfloat16()

    q, s = quantize_e4m3_rows(x.to(device))

    blocks = x.float().view(5, 8, 32)
    mantissa, exponent = torch.frexp(blocks.abs().amax(-1) / 448.0)
    # The smallest power of two at least amax / 448, as an e8m0 byte.
    want = (exponent - (mantissa == 0.5).int() + 127).clamp(1, 253)
    want[1, 0] = 1
    assert s.dtype == torch.uint8 and q.dtype == torch.float8_e4m3fn
    assert torch.equal(s.cpu().int(), want.int())
    if device != "cuda":
        # Triton's interpreter casts fp32 to e4m3 wrongly: it drops the carry
        # at a binade edge (31.625 becomes 16) and rounds ties away from zero.
        return
    # The power-of-two scale is exact, so the values are PyTorch's e4m3 cast.
    scaled = blocks * torch.exp2(127 - s.cpu().float())[..., None]
    want_q = scaled.to(torch.float8_e4m3fn).view(5, 256)
    assert torch.equal(q.cpu().view(torch.uint8), want_q.view(torch.uint8))


def test_e8m0_scales_reject_scales_that_are_not_powers_of_two() -> None:
    assert to_e8m0(torch.tensor([0.25, 1.0, 2.0**-20])).tolist() == [125, 127, 107]
    for bad in (0.3, -0.25, 0.0):
        with pytest.raises(ValueError, match="power of two"):
            to_e8m0(torch.tensor([0.25, bad]))


@pytest.mark.parametrize(
    ("per_segment", "expert_tiles"),
    [(None, None), (True, None), (False, None)],
    ids=["default", "segments", "expert-tiles"],
)
@pytest.mark.parametrize(("counts", "off_rank"), [([3, 0, 17, 1, 9], 4), ([100, 0, 150, 60, 1], 0)])
def test_w4a8_kernel_matches_fp32_math_on_its_inputs(
    monkeypatch: pytest.MonkeyPatch,
    counts: list[int],
    off_rank: int,
    per_segment: bool | None,
    expert_tiles: bool | None,
) -> None:
    """W4A8: e4m3 rows times e2m1 weights, both scaled by e8m0, summed in fp32."""
    device = _device()
    monkeypatch.setattr(fp4_grouped_kernel, "_PER_SEGMENT_FOR_BENCH", per_segment)
    monkeypatch.setattr(fp4_grouped_kernel, "_EXPERT_TILES_FOR_BENCH", expert_tiles)
    generator = torch.Generator().manual_seed(0)
    num_experts, n_dim, k_dim, block = len(counts), 48, 96, 32
    packed = torch.randint(
        -128, 128, (num_experts, n_dim, k_dim // 2), dtype=torch.int8, generator=generator
    )
    scale = torch.exp2(torch.randint(-8, -5, (num_experts, n_dim, k_dim // block)).float())
    row_experts = torch.cat(
        [
            torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts)),
            torch.full((off_rank,), num_experts),
        ]
    )
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32)
    rows = torch.randn(len(row_experts), k_dim, generator=generator).bfloat16()
    q, act_scale = quantize_e4m3_rows(rows.to(device))

    got = block_fp4_grouped_mm(
        q,
        packed.to(device),
        to_e8m0(scale).to(device),
        block=block,
        row_experts=row_experts.to(device),
        group_ends=group_ends.to(device),
        act_scale=act_scale,
        out_dtype=torch.bfloat16,
    ).cpu()

    q_rows = q.cpu().float().view(len(row_experts), -1, block)
    q_rows = (q_rows * torch.exp2(act_scale.cpu().float() - 127)[..., None]).view(-1, k_dim)
    want = torch.zeros(len(row_experts), n_dim)
    for row, expert in enumerate(row_experts.tolist()):
        if expert >= num_experts:
            continue
        deq = dequantize_nvfp4_to_bf16(packed[expert], scale[expert], block_size=block)
        want[row] = q_rows[row] @ deq.float().T
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=_atol(device, want))


def test_kernel_with_a_scale_block_wider_than_the_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tile of 64 columns inside a scale block of 128 uses one scale per row."""
    device = _device()
    monkeypatch.setattr(fp4_grouped_kernel, "_TILES_FOR_BENCH", (16, 32, 64, 4, 2))
    generator = torch.Generator().manual_seed(3)
    counts, n_dim, k_dim, block = [5, 0, 11], 32, 256, 128
    packed = torch.randint(-128, 128, (3, n_dim, k_dim // 2), dtype=torch.int8, generator=generator)
    scale = torch.rand(3, n_dim, k_dim // block, generator=generator) * 0.02 + 0.01
    row_experts = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    rows = torch.randn(len(row_experts), k_dim, generator=generator).bfloat16()

    got = block_fp4_grouped_mm(
        rows.to(device),
        packed.to(device),
        scale.to(device),
        block=block,
        row_experts=row_experts.to(device),
        group_ends=torch.tensor(counts).cumsum(0).to(torch.int32).to(device),
    ).cpu()

    want = torch.stack(
        [
            rows[row].float()
            @ dequantize_nvfp4_to_bf16(packed[e], scale[e], block_size=block).float().T
            for row, e in enumerate(row_experts.tolist())
        ]
    )
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=_atol(device, want))


def test_kernel_rejects_a_k_the_scale_block_does_not_divide() -> None:
    device = _device()
    with pytest.raises(ValueError, match="dividing K"):
        block_fp4_grouped_mm(
            torch.zeros(2, 48, dtype=torch.bfloat16, device=device),
            torch.zeros(1, 16, 24, dtype=torch.int8, device=device),
            torch.ones(1, 16, 1, device=device),
            block=32,
            row_experts=torch.zeros(2, dtype=torch.long, device=device),
            group_ends=torch.tensor([2], dtype=torch.int32, device=device),
        )


@pytest.mark.parametrize("routing_mode", ["score_topk", "hash"])
# 100 tokens give 300 (token, expert) rows: the expert tiles.
@pytest.mark.parametrize("tokens", [1, 5, 40, 100])
def test_fp4_moe_grouped_matches_loop(
    monkeypatch: pytest.MonkeyPatch, tokens: int, routing_mode: str
) -> None:
    device = _device()
    torch.manual_seed(1)
    vocab = 64
    moe = HashRoutedMoEFFN(
        hidden_size=128,
        intermediate_size=64,
        num_routed_experts=8,
        num_activated_experts=3,
        routing_mode=routing_mode,  # type: ignore[arg-type]
        vocab_size=vocab if routing_mode == "hash" else None,
        n_shared_experts=1,
        expert_dtype="fp4",
        swiglu_limit=10.0,
    ).eval()
    generator = torch.Generator().manual_seed(2)
    for expert in moe.experts:
        fill_fp4(expert, generator)  # type: ignore[arg-type]
    if routing_mode == "hash":
        with torch.no_grad():
            moe.gate.tid2eid.copy_(torch.randint(0, 8, moe.gate.tid2eid.shape))
    # Only the bf16 parts follow the activation dtype; `.to(dtype)` on the
    # whole block would also convert the fp32 scales of the FP4 experts.
    assert moe.shared_experts is not None
    moe.shared_experts.to(torch.bfloat16)
    moe.to(device)
    hidden = torch.randn(1, tokens, 128, device=device).bfloat16()
    input_ids = torch.randint(0, vocab, (1, tokens), device=device)

    monkeypatch.setattr(hash_routed_moe_ffn, "supports_grouped_experts", lambda device: False)
    with torch.no_grad():
        looped = moe(hidden, input_ids)
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_grouped_experts", lambda device: True)
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_block_fp4_grouped_mm", lambda device: True)
    with torch.no_grad():
        grouped = moe(hidden, input_ids)
    assert isinstance(moe._expert_pack, PackedFp4Experts)
    torch.testing.assert_close(grouped.float(), looped.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 5, 100])
def test_fp4_moe_fp8_activations_stay_near_the_bf16_loop(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    """W4A8 rounds each expert input to e4m3; the rest of the math is the bf16 path's."""
    device = _device()
    torch.manual_seed(1)
    moe = HashRoutedMoEFFN(
        hidden_size=128,
        intermediate_size=64,
        num_routed_experts=8,
        num_activated_experts=3,
        routing_mode="score_topk",
        n_shared_experts=1,
        expert_dtype="fp4",
        swiglu_limit=10.0,
        fp8_activations=True,
    ).eval()
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for expert in moe.experts:
            fill_fp4(expert, generator)  # type: ignore[arg-type]
            for name in ("w1", "w2", "w3"):  # e8m0 holds powers of two only
                scale = getattr(expert, f"{name}_scale")
                scale.copy_(torch.exp2(torch.round(torch.log2(scale))))
    assert moe.shared_experts is not None
    moe.shared_experts.to(torch.bfloat16)
    moe.to(device)
    hidden = torch.randn(1, tokens, 128, device=device).bfloat16()

    monkeypatch.setattr(hash_routed_moe_ffn, "supports_grouped_experts", lambda device: False)
    with torch.no_grad():
        looped = moe(hidden).float()
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_grouped_experts", lambda device: True)
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_block_fp4_grouped_mm", lambda device: True)
    with torch.no_grad():
        grouped = moe(hidden).float()
    pack = moe._expert_pack
    assert isinstance(pack, PackedFp4Experts) and pack.fp8_activations
    assert pack.gate_up.e8m0_scale is not None and pack.down.e8m0_scale is not None
    error = float((grouped - looped).norm() / looped.norm())
    # About 2^-5 per e4m3 rounding. The interpreter's faulty e4m3 cast (see
    # the quantizer test) adds a few values off by up to 2x.
    assert error < (0.05 if device == "cuda" else 0.1), error


def test_fp8_activations_need_fp4_experts() -> None:
    with pytest.raises(ValueError, match="expert_dtype='fp4'"):
        HashRoutedMoEFFN(
            hidden_size=128,
            intermediate_size=64,
            num_routed_experts=8,
            num_activated_experts=3,
            routing_mode="score_topk",
            fp8_activations=True,
        )
