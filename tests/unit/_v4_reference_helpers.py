"""Shared fixtures + kernel stub for the DeepSeek-V4 parity tests.

Used by `test_v4_hca_parity.py` and `test_v4_csa_parity.py`. The
reference inference code at `third_party/deepseek_v4_reference/model.py`
imports tilelang kernels (`act_quant`, `sparse_attn`, ...) from a
sibling `kernel` module that we don't ship. We register a `kernel`
stub in `sys.modules` BEFORE the reference imports, replacing those
kernels with PyTorch equivalents (or hard-failure stubs for paths we
don't exercise). We also patch the module-level `rotate_activation`
to identity — Hadamard rotation cancels in `q · k^T` because the
transform is unitary, so dropping it leaves the dot products unchanged.

`install_real_quant_stubs` replaces those identity stubs, for one test,
with real PyTorch mirrors of the FP8 / FP4 round-trip kernels and the
Hadamard transform. The parity tests of the port's `simulate_act_quant`
mode use it to compare against the reference's quantized numerics.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.nn.functional import linear

from mini_infer.models.blocks.v4.lightning_indexer import _topk_lower_index_on_ties
from mini_infer.quant.nvfp4 import dequantize_block_fp8_to_bf16_partial

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "deepseek_v4_reference"


def make_kernel_stub() -> types.ModuleType:
    """Build a stand-in `kernel` module for the reference's `from kernel import ...`.

    Stubbed-as-no-op (used on real paths):
      - `act_quant(..., inplace=True)`: QAT round-trip, identity at fp32.
      - `fp4_act_quant(..., inplace=True)`: same.
      - `sparse_attn`: PyTorch reference (gather + per-head softmax + sink).
        Mirrors our `hca_mqa_with_sink` math but lives in test code.

    Hard-fail (we want loud breakage if a non-CSA/HCA path lights up):
      - `fp8_gemm`, `fp4_gemm` (used only when weights are real fp8/fp4).
      - `hc_split_sinkhorn` (used by the Hyper-Connections backbone, not
        exercised when we instantiate just the `Attention` block).
    """
    kernel = types.ModuleType("kernel")

    def act_quant(
        x: torch.Tensor,
        block_size: int = 128,
        scale_fmt: Any = None,
        scale_dtype: Any = torch.float32,
        inplace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inplace:
            return x
        return x, x.new_ones((*x.shape[:-1], x.shape[-1] // block_size))

    def fp4_act_quant(
        x: torch.Tensor, block_size: int = 32, inplace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inplace:
            return x
        return x, x.new_ones((*x.shape[:-1], x.shape[-1] // block_size))

    def fp8_gemm(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fp8_gemm hit on supposedly fp32 V4 parity path")

    def fp4_gemm(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fp4_gemm hit on supposedly fp32 V4 parity path")

    def sparse_attn(
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        # PyTorch reference for the tilelang kernel. Mirrors our
        # `hca_mqa_with_sink` math (intentionally — the parity test checks
        # that the two implementations agree, so they share math but live
        # in different code, avoiding the "tested a stub against itself" trap).
        b, m, h, d = q.shape
        is_pad = topk_idxs < 0
        safe_idxs = topk_idxs.clamp(min=0).long()
        expanded = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, d)
        kv_expanded = kv.unsqueeze(1).expand(-1, m, -1, -1)
        gathered = torch.gather(kv_expanded, dim=2, index=expanded)
        scores = torch.einsum("bmhd,bmkd->bhmk", q.float(), gathered.float()) * softmax_scale
        scores = scores.masked_fill(is_pad.unsqueeze(1), float("-inf"))
        sink_col = attn_sink.float().view(1, h, 1, 1).expand(b, -1, m, 1)
        scores = torch.cat([scores, sink_col], dim=-1)
        weights = scores.softmax(dim=-1)[..., :-1]
        out = torch.einsum("bhmk,bmkd->bmhd", weights, gathered.float())
        return out.to(q.dtype)

    def hc_split_sinkhorn(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # PyTorch transcription of the kernel — used by the V4 Hyper-Connections
        # parity test. Imports lazily so this stub module doesn't pull in the
        # owned mini_infer block at module-collection time.
        from mini_infer.models.blocks.hyper_connections import (
            hc_split_sinkhorn as owned_hc_split_sinkhorn,
        )

        return owned_hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )

    kernel.act_quant = act_quant  # type: ignore[attr-defined]
    kernel.fp4_act_quant = fp4_act_quant  # type: ignore[attr-defined]
    kernel.fp8_gemm = fp8_gemm  # type: ignore[attr-defined]
    kernel.fp4_gemm = fp4_gemm  # type: ignore[attr-defined]
    kernel.sparse_attn = sparse_attn  # type: ignore[attr-defined]
    kernel.hc_split_sinkhorn = hc_split_sinkhorn  # type: ignore[attr-defined]
    return kernel


def _pow2_ceil_via_frexp(x: torch.Tensor) -> torch.Tensor:
    """`2 ** ceil(log2(x))` for positive FP32 `x` (the kernels' `fast_round_scale`).

    `frexp` writes `x = mantissa * 2**exponent` with mantissa in [0.5, 1).
    `x` is already a power of two iff mantissa == 0.5; otherwise the answer
    is `2**exponent == x / mantissa`, which IEEE division returns exactly.
    Deliberately a different route from the owned bit-twiddling version.
    """
    mantissa, _ = torch.frexp(x)
    return torch.where(mantissa == 0.5, x, x / mantissa)


_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _cast_to_e2m1(y: torch.Tensor) -> torch.Tensor:
    """Nearest FP4 e2m1 value of `y` (already in [-6, 6]), ties to the even code.

    Grid codes 0..7 map to `_E2M1_GRID`; the low code bit is the mantissa bit,
    so round-half-to-even picks the even code on a tie (CUDA's RN cast).
    """
    distance = (y.abs().unsqueeze(-1) - _E2M1_GRID).abs()  # (..., 8)
    is_nearest = distance == distance.amin(dim=-1, keepdim=True)
    codes = torch.arange(8)
    no_code = torch.full_like(codes, 99)
    even_pick = torch.where(is_nearest & (codes % 2 == 0), codes, no_code).amin(dim=-1)
    any_pick = torch.where(is_nearest, codes, no_code).amin(dim=-1)
    code = torch.where(even_pick < 99, even_pick, any_pick)
    return torch.sign(y) * _E2M1_GRID[code]


def real_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Any = None,
    scale_dtype: Any = torch.float32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PyTorch mirror of the tilelang `act_quant` FP8 block quantizer.

    Per block of `block_size` values along the last dim: `amax = max(max|x|,
    1e-4)`, `scale = amax * fp32(1/448)` (rounded up to a power of two when
    `scale_fmt` is set), `codes = e4m3(clamp(x / scale, -448, 448))`. The
    e4m3 cast is PyTorch's native `float8_e4m3fn` conversion.

    `inplace=True` (the QAT round trip) writes `codes * scale` back into `x`
    (a view, at the reference's call sites). `inplace=False` (the input of
    `fp8_gemm` / `fp4_gemm`) returns the FP8 codes and the per-block scales
    stored as `scale_dtype`, like the kernel.
    """
    n = x.size(-1)
    assert n % block_size == 0
    blocks = x.float().reshape(-1, n // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    if scale_fmt is not None:
        scale = _pow2_ceil_via_frexp(scale)
    codes = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if not inplace:
        return codes.reshape(x.shape), scale.reshape(*x.shape[:-1], -1).to(scale_dtype)
    x.copy_((codes.float() * scale).reshape(x.shape).to(x.dtype))
    return x


def real_fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: Any = torch.float32,
) -> torch.Tensor:
    """PyTorch mirror of the tilelang `fp8_gemm`: `C = A @ B^T`, per-128 scales on both.

    `a`: FP8 codes `(..., K)` with `a_s` `(..., K / 128)`; `b`: FP8 codes
    `(N, K)` with `b_s` `(ceil(N / 128), K / 128)`. Like the kernel, each
    128-wide K tile's code product is summed in FP32, multiplied by
    `scale_a * scale_b`, and accumulated in FP32; the output is in the
    default dtype.
    """
    del scale_dtype
    k = a.size(-1)
    m, n = a.numel() // k, b.size(0)
    a_codes = a.float().reshape(m, k)
    a_scale = a_s.float().reshape(m, k // 128)
    b_codes = b.float()
    b_scale_rows = b_s.float().repeat_interleave(128, dim=0)[:n]  # (N, K / 128)
    out = torch.zeros(m, n)
    for tile in range(k // 128):
        cols = slice(tile * 128, (tile + 1) * 128)
        partial = a_codes[:, cols] @ b_codes[:, cols].T  # (M, N)
        out += partial * (a_scale[:, tile, None] * b_scale_rows[None, :, tile])
    return out.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """`float4_e2m1fn_x2` `(N, K / 2)` -> FP32 values `(N, K)`; low nibble first."""
    as_bytes = packed.view(torch.uint8)
    nibbles = torch.stack([as_bytes & 0x0F, as_bytes >> 4], dim=-1).flatten(-2).long()
    magnitude = _E2M1_GRID[nibbles & 0x7]
    return torch.where(nibbles & 0x8 != 0, -magnitude, magnitude)


def real_fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: Any = torch.float32,
) -> torch.Tensor:
    """PyTorch mirror of the tilelang `fp4_gemm`: FP8 activation x FP4 weight.

    `a`: FP8 codes `(..., K)` with per-128 scales `a_s`; `b`: packed e2m1
    `(N, K / 2)` with per-32 E8M0 scales `b_s` `(N, K / 32)`. Like the
    kernel, each 32-wide K tile's code product is summed in FP32, scaled by
    `scale_a` (of its 128 group) then `scale_b`, and accumulated in FP32.
    """
    del scale_dtype
    k = a.size(-1)
    m, n = a.numel() // k, b.size(0)
    a_codes = a.float().reshape(m, k)
    a_scale = a_s.float().reshape(m, k // 128)
    b_values = _unpack_e2m1(b)  # (N, K)
    b_scale = b_s.float()  # (N, K / 32)
    out = torch.zeros(m, n)
    for tile in range(k // 32):
        cols = slice(tile * 32, (tile + 1) * 32)
        partial = a_codes[:, cols] @ b_values[:, cols].T  # (M, N)
        out += partial * a_scale[:, tile // 4, None] * b_scale[None, :, tile]
    return out.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def _quantize_fp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 `(N, K)` -> FP8 codes `(N, K)` + E8M0 scales per 128x128 tile (ue8m0)."""
    n, k = weight.shape
    tiles_n, tiles_k = -(-n // 128), -(-k // 128)
    padded = torch.zeros(tiles_n * 128, tiles_k * 128)
    padded[:n, :k] = weight
    tiles = padded.view(tiles_n, 128, tiles_k, 128)
    amax = tiles.abs().amax(dim=(1, 3)).clamp_min(1e-4)  # (tiles_n, tiles_k)
    scale = _pow2_ceil_via_frexp(amax * torch.tensor(1.0 / 448.0, dtype=torch.float32))
    codes = (tiles / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return codes.view(tiles_n * 128, tiles_k * 128)[:n, :k].contiguous(), scale.to(
        torch.float8_e8m0fnu
    )


def _quantize_fp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 `(N, K)` -> packed e2m1 `(N, K / 2)` + E8M0 scales per 32 along K."""
    n, k = weight.shape
    blocks = weight.reshape(n, k // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0**-126)
    scale = _pow2_ceil_via_frexp(amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))
    values = _cast_to_e2m1((blocks / scale).clamp(-6.0, 6.0)).reshape(n, k)
    magnitude_code = (values.abs().unsqueeze(-1) == _E2M1_GRID).int().argmax(dim=-1)
    nibbles = (magnitude_code + 8 * (values < 0).long()).to(torch.uint8).view(n, k // 2, 2)
    packed = nibbles[..., 0] | (nibbles[..., 1] << 4)
    return packed.view(torch.float4_e2m1fn_x2), scale.reshape(n, k // 32).to(torch.float8_e8m0fnu)


def init_reference_weights(reference: Any, module: Any, seed: int, std: float = 0.02) -> None:
    """Seed-fill a reference module; FP8 / FP4 `Linear`s get quantized random weights.

    Plain floating parameters get `randn * std` in `parameters()` order (FP32,
    as the identity-stub tests always did). An FP8 or FP4 `Linear` gets an
    FP32 draw quantized the checkpoint's way: FP8 e4m3 per 128x128 tile, or
    packed FP4 e2m1 per 32 along K, both with power-of-two E8M0 scales.
    Floating buffers (KV caches, compressor state) are zeroed.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    handled: set[int] = set()
    with torch.no_grad():
        for sub in module.modules():
            if not isinstance(sub, reference.Linear) or sub.scale is None:
                continue
            dense = torch.randn(sub.out_features, sub.in_features, generator=generator) * std
            if sub.weight.dtype == torch.float8_e4m3fn:
                sub.weight.data, sub.scale.data = _quantize_fp8_weight(dense)
            else:
                sub.weight.data, sub.scale.data = _quantize_fp4_weight(dense)
            handled.update((id(sub.weight), id(sub.scale)))
        for parameter in module.parameters():
            if id(parameter) not in handled and parameter.dtype.is_floating_point:
                parameter.data = (
                    torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * std
                )
        for buffer in module.buffers():
            if buffer.dtype.is_floating_point:
                buffer.zero_()


def dense_weight(linear: Any) -> torch.Tensor:
    """FP32 weight of a reference `Linear`, dequantized the loader's way if FP8."""
    if linear.weight.dtype == torch.float8_e4m3fn:
        return dequantize_block_fp8_to_bf16_partial(linear.weight, linear.scale).float()
    weight: torch.Tensor = linear.weight
    return weight


def real_fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False
) -> torch.Tensor:
    """PyTorch mirror of the tilelang `fp4_act_quant(..., inplace=True)` FP4 round trip.

    Per block: `amax = max(max|x|, 6 * 2^-126)`, `scale = 2^ceil(log2(amax *
    fp32(1/6)))`, `x = e2m1(clamp(x / scale, -6, 6)) * scale`, in place.
    """
    if not inplace:
        raise NotImplementedError("fp4_act_quant(inplace=False) is unused on this path")
    n = x.size(-1)
    assert n % block_size == 0
    blocks = x.float().reshape(-1, n // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0**-126)
    scale = _pow2_ceil_via_frexp(amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))
    codes = _cast_to_e2m1((blocks / scale).clamp(-6.0, 6.0))
    x.copy_((codes * scale).reshape(x.shape).to(x.dtype))
    return x


def real_rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """`fast_hadamard_transform.hadamard_transform(x, scale=d ** -0.5)` as a matmul.

    That library documents its result as `F.linear(x, hadamard(d)) * scale`
    with the Sylvester-ordered Hadamard matrix; built here by Kronecker
    products and applied in FP32 (the CUDA kernel's compute dtype).
    """
    d = x.size(-1)
    hadamard = torch.ones(1, 1)
    while hadamard.shape[0] < d:
        hadamard = torch.kron(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), hadamard)
    assert hadamard.shape[0] == d, f"Hadamard needs a power-of-two dim, got {d}"
    return (linear(x.float(), hadamard) * d**-0.5).to(x.dtype)


def install_real_quant_stubs(
    reference: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scale_fmt: str | None = "ue8m0",
    fp8_weights: bool = False,
) -> None:
    """Swap the reference's identity quant stubs for the real round trips (test scope).

    `model.py` binds `act_quant` / `fp4_act_quant` / `rotate_activation` /
    `fp8_gemm` / `fp4_gemm` and the `scale_fmt` / `default_dtype` /
    `scale_dtype` globals at module level, so patching the module attributes
    reaches every call site. `monkeypatch` restores the identity stubs after
    the test, so the module-scoped `reference_module` stays clean.
    `scale_fmt="ue8m0"` is what the reference's `Transformer` sets for every
    released V4 config.

    `fp8_weights=True` makes modules built afterwards use FP8 `Linear`
    weights with E8M0 scales, what `Transformer` sets for a released
    checkpoint (`dtype="fp8"`): their GEMMs then take FP8 inputs. Fill them
    with `init_reference_weights`.
    """
    monkeypatch.setattr(reference, "act_quant", real_act_quant)
    monkeypatch.setattr(reference, "fp4_act_quant", real_fp4_act_quant)
    monkeypatch.setattr(reference, "rotate_activation", real_rotate_activation)
    monkeypatch.setattr(reference, "fp8_gemm", real_fp8_gemm)
    monkeypatch.setattr(reference, "fp4_gemm", real_fp4_gemm)
    monkeypatch.setattr(reference, "scale_fmt", scale_fmt)
    if fp8_weights:
        monkeypatch.setattr(reference, "default_dtype", torch.float8_e4m3fn)
        monkeypatch.setattr(reference, "scale_dtype", torch.float8_e8m0fnu)


def _decode_ties_to_lower_block(forward: Any) -> Any:
    """Wrap the reference `Indexer.forward`: a one-token decode breaks top-k ties by index.

    Its `Tensor.topk` leaves the order of equal scores open, and the order
    changes with the width of the input. Our decode steps give a tie to the
    lower block (`lightning_indexer._topk_lower_index_on_ties`), so the
    reference does the same here. Prefill keeps `Tensor.topk` on both sides.
    """

    def lower_block_topk(
        scores: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert dim in (-1, scores.dim() - 1) and largest
        return _topk_lower_index_on_ties(scores, k)

    def wrapped(self: Any, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int) -> Any:
        if start_pos == 0 or x.shape[1] != 1:
            return forward(self, x, qr, start_pos, offset)
        original = torch.Tensor.topk
        torch.Tensor.topk = lower_block_topk  # type: ignore[assignment]
        try:
            return forward(self, x, qr, start_pos, offset)
        finally:
            torch.Tensor.topk = original  # type: ignore[method-assign]

    return wrapped


@pytest.fixture(scope="module")
def reference_module() -> Any:
    """Import the vendored DeepSeek-V4 reference `model.py` with kernel + Hadamard stubbed.

    Forces fp32 default dtype during import so the reference's `nn.Parameter(
    torch.empty(...))` creates fp32 tensors. Also patches the module-level
    `rotate_activation` to identity (Hadamard is unitary, cancels in `q·k^T`),
    and gives the indexer's one-token decode our tie rule for its top-k.
    """
    if not REFERENCE_DIR.exists():
        pytest.skip(
            f"DeepSeek-V4 reference not vendored at {REFERENCE_DIR}; "
            "run `uv run python scripts/clone_v4_reference.py`"
        )
    sys.modules["kernel"] = make_kernel_stub()
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        if "model" in sys.modules:
            del sys.modules["model"]
        ref = importlib.import_module("model")
        ref.default_dtype = torch.float32
        # Hadamard rotation: identity for parity. Reference's version asserts
        # bf16 input AND imports `fast_hadamard_transform`; both fail in our setup.
        ref.rotate_activation = lambda x: x
        ref.Indexer.forward = _decode_ties_to_lower_block(ref.Indexer.forward)
        yield ref
    finally:
        torch.set_default_dtype(prev_dtype)
