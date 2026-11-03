"""The split-K decode path of the W8A16 kernel against the naive dequantized matmul.

Runs on CUDA, or on the CPU under the Triton interpreter (`TRITON_INTERPRET=1`
in the `mini-infer-triton-interp` image). Under the interpreter the inputs are
fp32, because its bf16 `tl.dot` gives wrong results.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from mini_infer.quant import int8_kernel
from mini_infer.quant.int8 import Int8Linear
from mini_infer.quant.int8_kernel import DecodeTile, block_m_for, split_k_for

needs_triton = pytest.mark.skipif(not int8_kernel._TRITON_AVAILABLE, reason="needs triton")


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _layer(
    out_features: int, in_features: int, *, bias: bool, dtype: torch.dtype, device: str
) -> Int8Linear:
    torch.manual_seed(out_features + in_features)
    fp_linear = nn.Linear(in_features, out_features, bias=bias).to(dtype)
    return Int8Linear.from_float(fp_linear).to(device)


def _naive(layer: Int8Linear, x: torch.Tensor) -> torch.Tensor:
    w_dq = layer.weight.to(x.dtype) * layer.scales.to(x.dtype).unsqueeze(1)
    bias = layer.bias.to(x.dtype) if layer.bias is not None else None
    return torch.nn.functional.linear(x, w_dq, bias)


def _tolerance(dtype: torch.dtype) -> float:
    return 2e-2 if dtype == torch.bfloat16 else 1e-4


def test_split_k_grows_to_the_program_target_within_the_divisibility() -> None:
    # K=18944 = 2**8 * 74: with 128-wide tiles the split stops at 4, with 64-wide at 8.
    assert split_k_for(18944, 128, 56) == 4
    assert split_k_for(18944, 64, 56) == 8
    # K=3584 with 296 n-tiles needs a split of 2 to pass 512 programs.
    assert split_k_for(3584, 128, 296) == 2
    # Enough tiles already: no split.
    assert split_k_for(3584, 64, 1024) == 1


def test_the_tile_follows_the_architecture_and_the_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(int8_kernel, "_DECODE_TILE", None)
    monkeypatch.setattr(int8_kernel, "is_cuda_device", lambda device: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    assert int8_kernel.decode_tile_for("cuda", 3584) == int8_kernel._HOPPER_TILE
    assert int8_kernel.decode_tile_for("cuda", 18944) == int8_kernel._HOPPER_LONG_K_TILE
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    assert int8_kernel.decode_tile_for("cuda", 3584) == int8_kernel._BLACKWELL_TILE
    assert int8_kernel.decode_tile_for("cuda", 18944) == int8_kernel._BLACKWELL_LONG_K_TILE
    # A tile set by a bench wins everywhere.
    chosen = DecodeTile(block_n=32, block_k=64, split_k=2, num_warps=4, num_stages=2)
    monkeypatch.setattr(int8_kernel, "_DECODE_TILE", chosen)
    assert int8_kernel.decode_tile_for("cuda", 18944) == chosen
    # Off CUDA (the interpreter) the H100 tiles serve.
    monkeypatch.setattr(int8_kernel, "_DECODE_TILE", None)
    monkeypatch.setattr(int8_kernel, "is_cuda_device", lambda device: False)
    assert int8_kernel.decode_tile_for("cpu", 3584) == int8_kernel._HOPPER_TILE
    assert int8_kernel.decode_tile_for("cpu", 18944) == int8_kernel._HOPPER_LONG_K_TILE


def test_the_h100_tiles_take_the_measured_splits_on_the_7b_shapes() -> None:
    """With the split from the shape, the H100 tiles give the splits of the sweep."""
    tile, long_tile = int8_kernel._HOPPER_TILE, int8_kernel._HOPPER_LONG_K_TILE
    splits = {
        name: split_k_for(k_dim, chosen.block_k, -(-n_dim // chosen.block_n))
        for name, n_dim, k_dim, chosen in (
            ("gate/up", 18944, 3584, tile),
            ("qkv", 4608, 3584, tile),
            ("o", 3584, 3584, tile),
            ("down", 3584, 18944, long_tile),
        )
    }
    assert splits == {"gate/up": 2, "qkv": 4, "o": 4, "down": 4}


def test_block_m_pads_the_rows_to_the_next_tile() -> None:
    assert [block_m_for(rows) for rows in (1, 16, 17, 32, 33, 64)] == [16, 16, 32, 32, 64, 64]
    with pytest.raises(ValueError, match="64 rows"):
        block_m_for(65)


# (N, K, bias): the Qwen2.5-7B shapes at the decode batch, and an uneven N.
SHAPES = [(3584, 18944, False), (18944, 3584, False), (4608, 3584, True), (1000, 512, True)]


@needs_triton
@pytest.mark.parametrize(("n_dim", "k_dim", "bias"), SHAPES)
@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
def test_decode_split_k_matches_the_naive_path(
    n_dim: int, k_dim: int, bias: bool, rows: int
) -> None:
    device = _device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    layer = _layer(n_dim, k_dim, bias=bias, dtype=dtype, device=device)
    x = torch.randn(rows, k_dim, dtype=dtype, device=device)
    want = _naive(layer, x)
    got = int8_kernel.fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)
    assert got.shape == want.shape and got.dtype == want.dtype
    torch.testing.assert_close(
        got.float(), want.float(), rtol=_tolerance(dtype), atol=_tolerance(dtype)
    )


@needs_triton
def test_one_launch_per_call_and_the_counters_rest_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last split of every n-tile writes the output, so a call is one launch.

    The arrival counters belong to the weight and read zero after the call,
    ready for the next one without a memset.
    """
    device = _device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    layer = _layer(1000, 1024, bias=False, dtype=dtype, device=device)
    x = torch.randn(3, 1024, dtype=dtype, device=device)
    launches: list[tuple[int, ...]] = []
    kernel = int8_kernel._w8a16_splitk_kernel

    class _Recording:
        def __getitem__(self, grid: tuple[int, ...]) -> object:
            launches.append(grid)
            return kernel[grid]

    monkeypatch.setattr(int8_kernel, "_w8a16_splitk_kernel", _Recording())
    monkeypatch.setattr(
        int8_kernel,
        "_DECODE_TILE",
        DecodeTile(block_n=128, block_k=128, split_k=4, num_warps=4, num_stages=2),
    )
    got = int8_kernel.fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)
    assert launches == [(8, 4)]  # 8 n-tiles of 128 columns, 4 K splits
    torch.testing.assert_close(
        got.float(), _naive(layer, x).float(), rtol=_tolerance(dtype), atol=_tolerance(dtype)
    )
    counters = int8_kernel.tile_counters(layer.weight, 8)
    assert counters.tolist() == [0] * 8
    assert int8_kernel.tile_counters(layer.weight, 8) is counters


@needs_triton
@pytest.mark.parametrize(
    "tile",
    [
        DecodeTile(block_n=32, block_k=64, split_k=8, num_warps=2, num_stages=2),
        DecodeTile(block_n=128, block_k=256, split_k=1, num_warps=8, num_stages=3),
    ],
)
def test_other_tiles_give_the_same_result(
    monkeypatch: pytest.MonkeyPatch, tile: DecodeTile
) -> None:
    device = _device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    layer = _layer(1024, 2048, bias=True, dtype=dtype, device=device)
    x = torch.randn(4, 2048, dtype=dtype, device=device)
    want = _naive(layer, x)
    monkeypatch.setattr(int8_kernel, "_DECODE_TILE", tile)
    got = int8_kernel.fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)
    torch.testing.assert_close(
        got.float(), want.float(), rtol=_tolerance(dtype), atol=_tolerance(dtype)
    )


@needs_triton
@pytest.mark.parametrize(
    ("rows", "k_dim", "why"),
    [
        (65, 512, "more rows than the decode kernel takes"),
        (2, 1088, "a K the tile does not divide"),
    ],
)
def test_other_shapes_dequantize_and_run_the_plain_matmul(
    monkeypatch: pytest.MonkeyPatch, rows: int, k_dim: int, why: str
) -> None:
    """Those shapes never reach the kernels and give the naive result bit for bit."""
    device = _device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def no_kernel(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"the split-K kernel ran for {why}")

    monkeypatch.setattr(int8_kernel, "_decode_split_k", no_kernel)
    layer = _layer(256, k_dim, bias=True, dtype=dtype, device=device)
    x = torch.randn(rows, k_dim, dtype=dtype, device=device)
    got = int8_kernel.fused_w8a16_linear(x, layer.weight, layer.scales, layer.bias)
    assert torch.equal(got, _naive(layer, x))
