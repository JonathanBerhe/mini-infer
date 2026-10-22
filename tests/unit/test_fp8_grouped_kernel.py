"""Block-FP8 grouped expert kernel (`blocks/fp8_grouped_kernel.py`) vs dequantized math.

The kernel against a row-by-row matmul with `dequantize_block_fp8_to_bf16_partial`,
and the FP8 `GlmMoeFFN` on its grouped path against its per-expert loop.

Runs on CUDA, or on the CPU under Triton's interpreter
(`TRITON_INTERPRET=1`), which is how the kernel is debugged without a GPU.
Skips otherwise.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("triton")

from mini_infer.models.blocks import fp8_grouped_kernel, glm_moe_gate
from mini_infer.models.blocks.fp8_expert import PackedFp8Experts
from mini_infer.models.blocks.fp8_grouped_kernel import block_fp8_grouped_mm
from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN
from mini_infer.quant.nvfp4 import dequantize_block_fp8_to_bf16_partial


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _atol(device: str) -> float:
    """Absolute tolerance against fp32 math on dequantized weights.

    GPUs round fp32 to bf16 to nearest, as PyTorch does. Triton's
    interpreter truncates instead (and ignores `fp_downcast_rounding`), so on
    the CPU the dequantized weights and the output can each be one bf16 step
    further off.
    """
    return 1e-3 if device == "cuda" else 4e-3


def _fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * 0.5).to(torch.float8_e4m3fn)


@pytest.mark.parametrize(
    ("per_segment", "swap_ab", "expert_tiles"),
    [
        (None, None, None),
        (False, False, None),
        (False, True, None),
        (True, False, None),
        (True, True, None),
        (False, False, True),
        (False, True, True),
    ],
    ids=[
        "default",
        "tiles",
        "tiles-swapped",
        "segments",
        "segments-swapped",
        "expert-tiles",
        "expert-tiles-swapped",
    ],
)
@pytest.mark.parametrize(
    ("counts", "off_rank"),
    [
        ([3, 0, 17, 1, 9], 0),  # tiles span several experts, a group over 16 rows
        ([3, 0, 17, 1, 9], 4),  # rows routed to other ranks sort last
        ([100, 0, 150, 60, 1], 0),  # over 256 rows: the large-tile configuration
    ],
)
def test_kernel_matches_dequantized_matmul(
    monkeypatch: pytest.MonkeyPatch,
    counts: list[int],
    off_rank: int,
    per_segment: bool | None,
    swap_ab: bool | None,
    expert_tiles: bool | None,
) -> None:
    device = _device()
    monkeypatch.setattr(fp8_grouped_kernel, "_PER_SEGMENT_FOR_BENCH", per_segment)
    monkeypatch.setattr(fp8_grouped_kernel, "_SWAP_AB_FOR_BENCH", swap_ab)
    monkeypatch.setattr(fp8_grouped_kernel, "_EXPERT_TILES_FOR_BENCH", expert_tiles)
    torch.manual_seed(0)
    num_experts, n_dim, k_dim, block = len(counts), 48, 96, 16
    weight = _fp8((num_experts, n_dim, k_dim))
    scale = torch.rand(num_experts, n_dim // block, k_dim // block) * 0.02 + 0.01
    row_experts = torch.cat(
        [
            torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts)),
            torch.full((off_rank,), num_experts),
        ]
    )
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32)
    rows = torch.randn(len(row_experts), k_dim).bfloat16()

    got = block_fp8_grouped_mm(
        rows.to(device),
        weight.to(device),
        scale.to(device),
        block=block,
        row_experts=row_experts.to(device),
        group_ends=group_ends.to(device),
    ).cpu()

    want = torch.zeros(len(row_experts), n_dim)
    for row, expert in enumerate(row_experts.tolist()):
        if expert >= num_experts:
            continue
        deq = dequantize_block_fp8_to_bf16_partial(
            weight[expert], scale[expert], block_size=(block, block)
        )
        want[row] = rows[row].float() @ deq.float().T
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=_atol(device))


def test_kernel_rejects_a_block_that_is_not_a_power_of_two() -> None:
    device = _device()
    with pytest.raises(ValueError, match="power of two"):
        block_fp8_grouped_mm(
            torch.zeros(2, 48, dtype=torch.bfloat16, device=device),
            _fp8((1, 48, 48)).to(device),
            torch.ones(1, 1, 1, device=device),
            block=48,
            row_experts=torch.zeros(2, dtype=torch.long, device=device),
            group_ends=torch.tensor([2], dtype=torch.int32, device=device),
        )


def _fp8_glm(hidden: int, inter: int, experts: int, top_k: int) -> GlmMoeFFN:
    torch.manual_seed(1)
    moe = GlmMoeFFN(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        n_routed_experts=experts,
        top_k=top_k,
        routed_scaling_factor=2.5,
        expert_dtype="fp8",
    ).eval()
    with torch.no_grad():
        for expert in moe.experts:
            for name in ("w1", "w2", "w3"):
                weight = getattr(expert, name)
                weight.copy_(_fp8(tuple(weight.shape)))
                getattr(expert, f"{name}_scale").uniform_(0.01, 0.03)
    # Only the replicated BF16 parts follow the activation dtype; `.to(dtype)`
    # on the whole block would also convert the e4m3 buffers.
    assert moe.shared_experts is not None
    moe.shared_experts.to(torch.bfloat16)
    return moe


@pytest.mark.parametrize("tokens", [1, 5, 40])
def test_fp8_glm_grouped_matches_loop(monkeypatch: pytest.MonkeyPatch, tokens: int) -> None:
    device = _device()
    moe = _fp8_glm(hidden=128, inter=128, experts=8, top_k=2).to(device)
    hidden = torch.randn(1, tokens, 128, device=device).bfloat16()
    monkeypatch.setattr(glm_moe_gate, "supports_grouped_experts", lambda device: False)
    with torch.no_grad():
        looped = moe(hidden)
    monkeypatch.setattr(glm_moe_gate, "supports_grouped_experts", lambda device: True)
    monkeypatch.setattr(glm_moe_gate, "supports_block_fp8_grouped_mm", lambda device: True)
    with torch.no_grad():
        grouped = moe(hidden)
    assert isinstance(moe._expert_pack, PackedFp8Experts)
    torch.testing.assert_close(grouped.float(), looped.float(), rtol=2e-2, atol=2e-2)
