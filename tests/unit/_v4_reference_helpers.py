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
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

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


@pytest.fixture(scope="module")
def reference_module() -> Any:
    """Import the vendored DeepSeek-V4 reference `model.py` with kernel + Hadamard stubbed.

    Forces fp32 default dtype during import so the reference's `nn.Parameter(
    torch.empty(...))` creates fp32 tensors. Also patches the module-level
    `rotate_activation` to identity (Hadamard is unitary, cancels in `q·k^T`).
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
        yield ref
    finally:
        torch.set_default_dtype(prev_dtype)
