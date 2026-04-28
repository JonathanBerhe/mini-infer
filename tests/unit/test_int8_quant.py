"""Unit tests for the standalone INT8 quantization primitives.

End-to-end model parity (Qwen2 with quantized linears) lives in
`test_int8_model_integration.py`; these tests cover only the math + the
single-module replacement contract.
"""

import pytest
import torch
from torch import nn

from mini_infer.quant.int8 import (
    Int8Linear,
    dequantize_per_channel,
    quantize_model_to_int8,
    quantize_per_channel,
)


def test_quantize_per_channel_round_trip_within_tolerance() -> None:
    """Quantize then dequantize a random matrix; max abs error bounded by 0.5 * scale per row."""
    torch.manual_seed(0)
    weight = torch.randn(64, 128, dtype=torch.float32) * 0.1
    q_weight, scales = quantize_per_channel(weight)

    assert q_weight.dtype == torch.int8
    assert q_weight.shape == weight.shape
    assert scales.shape == (64,)
    assert scales.dtype == weight.dtype

    dq = dequantize_per_channel(q_weight, scales, weight.dtype)
    # Per-row error bound: a single round() can introduce up to 0.5 * scale.
    per_row_max_err = (dq - weight).abs().amax(dim=1)
    bound = scales * 0.5 + 1e-6
    assert torch.all(per_row_max_err <= bound), (
        f"row errors exceed bound: max ratio = {(per_row_max_err / bound).max().item():.3f}"
    )


def test_quantize_per_channel_uses_full_int8_range() -> None:
    """A row whose max-abs is 1.0 should produce a quantized value of 127 there."""
    weight = torch.tensor([[1.0, -0.5, 0.0, 0.25]], dtype=torch.float32)
    q_weight, scales = quantize_per_channel(weight)
    assert q_weight[0, 0].item() == 127  # max-magnitude element saturates to +127
    assert scales[0].item() == pytest.approx(1.0 / 127.0, rel=1e-5)


def test_quantize_per_channel_handles_zero_row() -> None:
    """An all-zero row must not produce NaN; the quantized row stays zero."""
    weight = torch.zeros(2, 8, dtype=torch.float32)
    weight[1] = 1.0  # non-zero second row to keep things realistic
    q_weight, scales = quantize_per_channel(weight)
    assert torch.all(q_weight[0] == 0)
    assert torch.isfinite(scales).all()


def test_quantize_per_channel_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D weight"):
        quantize_per_channel(torch.randn(4))
    with pytest.raises(ValueError, match="2D weight"):
        quantize_per_channel(torch.randn(2, 3, 4))


def test_dequantize_validates_shapes() -> None:
    q = torch.zeros(4, 8, dtype=torch.int8)
    bad_scales = torch.zeros(8, dtype=torch.float32)  # wrong: should be 4
    with pytest.raises(ValueError, match="scales shape"):
        dequantize_per_channel(q, bad_scales, torch.float32)


def test_int8_linear_matches_fp32_within_cosine() -> None:
    """An Int8Linear built from a random Linear produces output close to the original."""
    torch.manual_seed(1)
    fp_linear = nn.Linear(128, 64, bias=True).float()
    int8_linear = Int8Linear.from_float(fp_linear)

    x = torch.randn(4, 128, dtype=torch.float32)
    out_fp = fp_linear(x)
    out_int8 = int8_linear(x)

    cos = torch.nn.functional.cosine_similarity(out_fp.flatten(), out_int8.flatten(), dim=0)
    assert cos.item() > 0.99, f"cos sim {cos.item():.4f} below 0.99"


def test_int8_linear_preserves_bias_bit_equal() -> None:
    """Bias is float and copied without quantization; it must be bit-equal."""
    fp_linear = nn.Linear(8, 4, bias=True).float()
    int8_linear = Int8Linear.from_float(fp_linear)
    assert int8_linear.bias is not None
    assert torch.equal(int8_linear.bias.detach(), fp_linear.bias.detach())


def test_int8_linear_handles_no_bias() -> None:
    fp_linear = nn.Linear(8, 4, bias=False).float()
    int8_linear = Int8Linear.from_float(fp_linear)
    assert int8_linear.bias is None
    x = torch.randn(2, 8)
    out = int8_linear(x)
    assert out.shape == (2, 4)


def test_int8_linear_propagates_input_dtype() -> None:
    """bf16/fp16 inputs should produce same-dtype outputs (dequant happens in input dtype)."""
    fp_linear = nn.Linear(16, 8, bias=True).float()
    int8_linear = Int8Linear.from_float(fp_linear)

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(3, 16, dtype=dtype)
        out = int8_linear(x)
        assert out.dtype == dtype, f"output dtype {out.dtype} != input dtype {dtype}"


def test_int8_linear_reports_correct_storage() -> None:
    """Int8 weight + fp16 scales must be at most ~55% of an fp16 weight's footprint."""
    fp_linear = nn.Linear(1024, 512, bias=False).float()
    int8_linear = Int8Linear.from_float(fp_linear)
    int8_bytes = (
        int8_linear.weight.numel() * int8_linear.weight.element_size()
        + int8_linear.scales.numel() * int8_linear.scales.element_size()
    )
    # 512*1024 (int8) + 512*2 (fp16 scales) = 525 KiB; an fp16 weight would be 1 MiB.
    fp16_bytes = fp_linear.weight.numel() * 2
    assert int8_bytes <= 0.55 * fp16_bytes


def test_int8_linear_state_dict_round_trip() -> None:
    """State dict save+load preserves quantized weights and scales bit-equal."""
    fp_linear = nn.Linear(32, 16, bias=True).float()
    src = Int8Linear.from_float(fp_linear)

    sd = src.state_dict()
    dst = Int8Linear(in_features=32, out_features=16, bias=True, scale_dtype=torch.float32)
    dst.load_state_dict(sd)

    assert torch.equal(src.weight, dst.weight)
    assert torch.equal(src.scales, dst.scales)
    assert src.bias is not None and dst.bias is not None
    assert torch.equal(src.bias.detach(), dst.bias.detach())


def test_quantize_model_to_int8_replaces_all_linears() -> None:
    """A simple Sequential gets every Linear replaced by an Int8Linear."""
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    n_replaced = quantize_model_to_int8(model, skip_modules=frozenset())
    assert n_replaced == 2
    assert isinstance(model[0], Int8Linear)
    assert isinstance(model[2], Int8Linear)


def test_quantize_model_to_int8_skip_lm_head_default() -> None:
    """Default `skip_modules={'lm_head'}` leaves a module named lm_head untouched."""

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.lm_head = nn.Linear(8, 32)

    model = Toy()
    n_replaced = quantize_model_to_int8(model)
    assert n_replaced == 1
    assert isinstance(model.q_proj, Int8Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, Int8Linear)


def test_quantize_model_to_int8_custom_skip_set() -> None:
    """A user-supplied skip set is honored."""

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)

    model = Toy()
    n_replaced = quantize_model_to_int8(model, skip_modules={"k_proj"})
    assert n_replaced == 2
    assert isinstance(model.q_proj, Int8Linear)
    assert isinstance(model.k_proj, nn.Linear)
    assert isinstance(model.v_proj, Int8Linear)


def test_quantize_model_to_int8_handles_nested_modules() -> None:
    """Linears inside child modules are reachable via `get_submodule`."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = Block()

    model = Outer()
    n_replaced = quantize_model_to_int8(model, skip_modules=frozenset())
    assert n_replaced == 1
    assert isinstance(model.block.proj, Int8Linear)
