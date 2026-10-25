"""Parity tests for the fused W8A16 Triton kernel.

CUDA-only (`@pytest.mark.requires_cuda`); the kernel doesn't run on CPU
or MPS. The naive ``Int8Linear.forward`` is the numerical oracle: fused
output must match it within cosine similarity > 0.999 across decode and
prefill shapes plus the asymmetric K/V projection shape from Qwen2.5
GQA.
"""

import pytest
import torch
from torch import nn

from mini_infer.quant.int8 import Int8Linear


def _make_int8_linear(out_features: int, in_features: int, bias: bool, seed: int) -> Int8Linear:
    """Build a quantized linear from a freshly-randomized fp32 nn.Linear."""
    torch.manual_seed(seed)
    fp_linear = nn.Linear(in_features, out_features, bias=bias).float()
    return Int8Linear.from_float(fp_linear)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
    )


@pytest.mark.requires_cuda
def test_fused_matches_naive_decode_qwen_qproj() -> None:
    """M=1 decode shape with Qwen2.5-0.5B's q_proj dimensions (N=896, K=896)."""
    from mini_infer.quant.int8_kernel import fused_w8a16_linear

    layer = _make_int8_linear(out_features=896, in_features=896, bias=True, seed=0).cuda()
    layer.weight.data = layer.weight.data.contiguous()
    layer.scales.data = layer.scales.data.to(torch.bfloat16)
    if layer.bias is not None:
        layer.bias.data = layer.bias.data.to(torch.bfloat16)

    x = torch.randn(1, 896, dtype=torch.bfloat16, device="cuda")

    naive_out = layer(x)
    fused_out = fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)

    assert fused_out.shape == naive_out.shape
    assert fused_out.dtype == naive_out.dtype
    assert _cosine_sim(fused_out, naive_out) > 0.999


@pytest.mark.requires_cuda
def test_fused_matches_naive_prefill_qwen_qproj() -> None:
    """M=128 prefill shape; same N/K as decode test."""
    from mini_infer.quant.int8_kernel import fused_w8a16_linear

    layer = _make_int8_linear(out_features=896, in_features=896, bias=True, seed=1).cuda()
    layer.scales.data = layer.scales.data.to(torch.bfloat16)
    if layer.bias is not None:
        layer.bias.data = layer.bias.data.to(torch.bfloat16)

    x = torch.randn(128, 896, dtype=torch.bfloat16, device="cuda")

    naive_out = layer(x)
    fused_out = fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)

    assert fused_out.shape == naive_out.shape
    assert _cosine_sim(fused_out, naive_out) > 0.999


@pytest.mark.requires_cuda
def test_fused_matches_naive_kv_proj_shape() -> None:
    """Qwen2.5-0.5B GQA k_proj / v_proj are (N=128, K=896): asymmetric, small N."""
    from mini_infer.quant.int8_kernel import fused_w8a16_linear

    layer = _make_int8_linear(out_features=128, in_features=896, bias=True, seed=2).cuda()
    layer.scales.data = layer.scales.data.to(torch.bfloat16)
    if layer.bias is not None:
        layer.bias.data = layer.bias.data.to(torch.bfloat16)

    x = torch.randn(4, 896, dtype=torch.bfloat16, device="cuda")

    naive_out = layer(x)
    fused_out = fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)

    assert fused_out.shape == naive_out.shape
    assert _cosine_sim(fused_out, naive_out) > 0.999


@pytest.mark.requires_cuda
def test_fused_matches_naive_mlp_down_proj() -> None:
    """Qwen2.5-0.5B MLP down_proj is (N=896, K=4864): the deepest K loop."""
    from mini_infer.quant.int8_kernel import fused_w8a16_linear

    layer = _make_int8_linear(out_features=896, in_features=4864, bias=False, seed=3).cuda()
    layer.scales.data = layer.scales.data.to(torch.bfloat16)

    x = torch.randn(8, 4864, dtype=torch.bfloat16, device="cuda")

    naive_out = layer(x)
    fused_out = fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)

    assert fused_out.shape == naive_out.shape
    assert _cosine_sim(fused_out, naive_out) > 0.999


@pytest.mark.requires_cuda
def test_fused_handles_no_bias() -> None:
    """Bias=None path: kernel must not dereference the bias pointer."""
    from mini_infer.quant.int8_kernel import fused_w8a16_linear

    layer = _make_int8_linear(out_features=64, in_features=128, bias=False, seed=4).cuda()
    layer.scales.data = layer.scales.data.to(torch.bfloat16)
    assert layer.bias is None

    x = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
    fused_out = fused_w8a16_linear(x, layer.weight, layer.scales, None)
    naive_out = layer(x)
    assert _cosine_sim(fused_out, naive_out) > 0.999


@pytest.mark.requires_cuda
def test_fused_input_dtype_propagates() -> None:
    """fp16 in → fp16 out; bf16 in → bf16 out. Output dtype follows x."""
    from mini_infer.quant.int8_kernel import fused_w8a16_linear

    for dtype in (torch.float16, torch.bfloat16):
        layer = _make_int8_linear(out_features=64, in_features=128, bias=True, seed=5).cuda()
        layer.scales.data = layer.scales.data.to(dtype)
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.to(dtype)

        x = torch.randn(4, 128, dtype=dtype, device="cuda")
        fused_out = fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)
        assert fused_out.dtype == dtype, f"dtype mismatch for input {dtype}: got {fused_out.dtype}"


@pytest.mark.requires_cuda
def test_fused_handles_3d_input() -> None:
    """A `(batch, seq, K)` input gets flattened to 2D and reshaped back to 3D."""
    from mini_infer.quant.int8_kernel import fused_w8a16_linear

    layer = _make_int8_linear(out_features=64, in_features=128, bias=True, seed=6).cuda()
    layer.scales.data = layer.scales.data.to(torch.bfloat16)
    if layer.bias is not None:
        layer.bias.data = layer.bias.data.to(torch.bfloat16)

    x = torch.randn(2, 5, 128, dtype=torch.bfloat16, device="cuda")
    fused_out = fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)
    naive_out = layer(x)
    assert fused_out.shape == naive_out.shape == (2, 5, 64)
    assert _cosine_sim(fused_out, naive_out) > 0.999


def test_supports_fused_kernel_returns_false_on_cpu() -> None:
    """Sanity check the dispatch helper: CPU device should never report fused-eligible."""
    from mini_infer.quant.int8_kernel import supports_fused_kernel

    assert supports_fused_kernel("cpu") is False
    assert supports_fused_kernel(torch.device("cpu")) is False
