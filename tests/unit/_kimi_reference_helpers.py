"""Shared fixture + `fla` stub for the Kimi Linear parity tests.

The vendored reference at `third_party/kimi_linear_reference/modeling_kimi.py`
imports the FLA library's Triton kernels (`chunk_kda`, `fused_recurrent_kda`,
`fused_kda_gate`, `ShortConvolution`, `FusedRMSNormGated`), which need a GPU.
We register a stub `fla` package in `sys.modules` BEFORE the reference
imports, replacing each kernel with the pure-PyTorch semantics FLA itself
documents (the recurrence is a transcription of `fla/ops/kda/naive.py`, MIT,
(c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li). The stubs deliberately do
NOT import `mini_infer.models.blocks.kda` — the parity tests compare our
implementation against these independently-written semantics, so sharing
code would test a stub against itself.

Only the paths the parity tests exercise are implemented; everything else
(varlen `cu_seqlens`, padding masks) raises loudly.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "kimi_linear_reference"


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return x32 * torch.rsqrt(x32.pow(2).sum(dim=-1, keepdim=True) + eps)


def _naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Transcription of `fla/ops/kda/naive.py::naive_recurrent_kda` (no GVA)."""
    dtype = v.dtype
    batch, seq_len, num_heads, dim_k = q.shape
    dim_v = v.shape[-1]
    if scale is None:
        scale = dim_k**-0.5
    q, k, v, g, beta = (t.to(torch.float) for t in (q, k, v, g, beta))
    q = q * scale
    state = k.new_zeros(batch, num_heads, dim_k, dim_v)
    if initial_state is not None:
        state = state + initial_state
    out = torch.zeros_like(v)
    for i in range(seq_len):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        state = state * g_i[..., None].exp()
        state = state + torch.einsum(
            "bhk,bhv->bhkv", b_i[..., None] * k_i, v_i - (k_i[..., None] * state).sum(-2)
        )
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, state)
    return out.to(dtype), (state if output_final_state else None)


def _kda_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Stub for both `chunk_kda` and `fused_recurrent_kda`: the fused wrappers
    L2-normalize q/k in-kernel and default the scale, then run the recurrence."""
    if cu_seqlens is not None:
        raise NotImplementedError("kimi parity stub: varlen cu_seqlens not exercised")
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    return _naive_recurrent_kda(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=output_final_state
    )


def _fused_kda_gate(
    g: torch.Tensor, a_log: torch.Tensor, head_dim: int, g_bias: torch.Tensor | None = None
) -> torch.Tensor:
    """`g = -exp(A_log) * softplus(g + dt_bias)`, reshaped `(..., H*K) -> (..., H, K)`,
    fp32 out (the fla-core 0.4.x signature the reference calls)."""
    num_heads = g.shape[-1] // head_dim
    gate = g.float().view(*g.shape[:-1], num_heads, head_dim)
    if g_bias is not None:
        gate = gate + g_bias.float().view(num_heads, head_dim)
    return -a_log.float().view(num_heads, 1).exp() * functional.softplus(gate)


class _StubShortConvolution(nn.Conv1d):
    """FLA `ShortConvolution`: depthwise causal conv1d + SiLU with a rolling
    `(N, D, W)` cache of the last W raw inputs (newest last)."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )
        assert activation in (None, "silu", "swish")
        self.activation = activation

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            raise NotImplementedError("kimi parity stub: varlen conv not exercised")
        batch, seq_len, channels = x.shape
        kernel = self.kernel_size[0]
        taps = self.weight.squeeze(1).float()  # (D, W)
        x32 = x.float().transpose(1, 2)  # (B, D, T)
        if seq_len == 1:
            # Decode step: roll the cache, write the new raw input, dot with taps.
            if cache is None:
                cache = x32.new_zeros(batch, channels, kernel)
            cache = torch.cat([cache.float()[:, :, 1:], x32], dim=-1)
            y = (cache * taps.unsqueeze(0)).sum(-1, keepdim=True)
        else:
            left = (
                cache.float()[:, :, 1:]
                if cache is not None
                else x32.new_zeros(batch, channels, kernel - 1)
            )
            y = functional.conv1d(
                torch.cat([left, x32], dim=-1), taps.unsqueeze(1), groups=channels
            )
            history = torch.cat(
                [
                    cache.float() if cache is not None else x32.new_zeros(batch, channels, kernel),
                    x32,
                ],
                dim=-1,
            )
            cache = history[:, :, -kernel:]
        if self.activation is not None:
            y = functional.silu(y)
        return y.transpose(1, 2).to(x.dtype), (cache if output_final_state else None)


class _StubFusedRMSNormGated(nn.Module):
    """FLA `FusedRMSNormGated`: fp32 RMSNorm * weight * act(gate), then cast."""

    def __init__(
        self, hidden_size: int, eps: float = 1e-5, activation: str = "swish", **kwargs: Any
    ) -> None:
        super().__init__()
        assert activation in ("swish", "silu", "sigmoid")
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        g32 = g.float()
        y = y * torch.sigmoid(g32) if self.activation == "sigmoid" else y * g32 * torch.sigmoid(g32)
        return y.to(x.dtype)


def _prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=-1, dtype=torch.int32)


def _prepare_cu_seqlens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    lens = _prepare_lens_from_mask(mask)
    return functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))


def shim_transformers_for_reference() -> Callable[[], None]:
    """Bridge the pinned reference (written for transformers ~4.57) to 5.14.

    Two drifts as of 5.14, patched at the old import paths BEFORE the
    reference module binds them (extend here, never in the vendored files,
    if a future pin bump breaks another name):
      - `OutputRecorder` moved from `transformers.utils.generic` to
        `transformers.utils.output_capturing` (re-exported; additive, safe
        to leave in place);
      - `create_causal_mask` renamed `input_embeds` -> `inputs_embeds` and
        dropped `cache_position` (5.x derives it from the cache's
        `get_mask_sizes`, which the reference's `KimiDynamicCache` provides).

    Returns a restore callback the caller MUST invoke right after importing
    the reference: the compat wrapper takes the OLD calling convention, so
    leaving it installed would break every 5.14-native family (the vendored
    module keeps its own binding to the wrapper either way).
    """
    import transformers.masking_utils as hf_masking
    import transformers.utils.generic as hf_generic

    if not hasattr(hf_generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        hf_generic.OutputRecorder = OutputRecorder  # type: ignore[attr-defined]

    import inspect

    if "inputs_embeds" not in str(inspect.signature(hf_masking.create_causal_mask)):
        return lambda: None

    original = hf_masking.create_causal_mask

    def create_causal_mask_compat(
        config: Any = None,
        input_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        past_key_values: Any = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        del cache_position  # 5.x recomputes it from the cache
        return original(
            config=config,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **kwargs,
        )

    hf_masking.create_causal_mask = create_causal_mask_compat  # type: ignore[assignment]

    def restore() -> None:
        hf_masking.create_causal_mask = original  # type: ignore[assignment]

    return restore


def install_fla_stub() -> None:
    """Register the stub `fla` package tree in `sys.modules`."""
    fla = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_modules.FusedRMSNormGated = _StubFusedRMSNormGated  # type: ignore[attr-defined]
    fla_modules.ShortConvolution = _StubShortConvolution  # type: ignore[attr-defined]
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_kda = types.ModuleType("fla.ops.kda")
    fla_ops_kda.chunk_kda = _kda_op  # type: ignore[attr-defined]
    fla_ops_kda.fused_recurrent_kda = _kda_op  # type: ignore[attr-defined]
    fla_ops_kda_gate = types.ModuleType("fla.ops.kda.gate")
    fla_ops_kda_gate.fused_kda_gate = _fused_kda_gate  # type: ignore[attr-defined]
    fla_ops_utils = types.ModuleType("fla.ops.utils")
    fla_ops_utils_index = types.ModuleType("fla.ops.utils.index")
    fla_ops_utils_index.prepare_cu_seqlens_from_mask = (  # type: ignore[attr-defined]
        _prepare_cu_seqlens_from_mask
    )
    fla_ops_utils_index.prepare_lens_from_mask = _prepare_lens_from_mask  # type: ignore[attr-defined]
    fla_utils = types.ModuleType("fla.utils")

    def tensor_cache(fn: Any) -> Any:  # the real one memoizes; identity is enough
        return fn

    fla_utils.tensor_cache = tensor_cache  # type: ignore[attr-defined]

    fla.modules = fla_modules  # type: ignore[attr-defined]
    fla.ops = fla_ops  # type: ignore[attr-defined]
    fla.utils = fla_utils  # type: ignore[attr-defined]
    fla_ops.kda = fla_ops_kda  # type: ignore[attr-defined]
    fla_ops.utils = fla_ops_utils  # type: ignore[attr-defined]
    fla_ops_kda.gate = fla_ops_kda_gate  # type: ignore[attr-defined]
    fla_ops_utils.index = fla_ops_utils_index  # type: ignore[attr-defined]
    for name, module in (
        ("fla", fla),
        ("fla.modules", fla_modules),
        ("fla.ops", fla_ops),
        ("fla.ops.kda", fla_ops_kda),
        ("fla.ops.kda.gate", fla_ops_kda_gate),
        ("fla.ops.utils", fla_ops_utils),
        ("fla.ops.utils.index", fla_ops_utils_index),
        ("fla.utils", fla_utils),
    ):
        sys.modules[name] = module


@pytest.fixture(scope="module")
def kimi_reference() -> Any:
    """Import the vendored Kimi Linear reference with the `fla` stub installed.

    Skips when the reference isn't vendored (run
    `uv run python scripts/clone_kimi_linear_reference.py`).
    """
    if not (REFERENCE_DIR / "modeling_kimi.py").exists():
        pytest.skip(
            f"Kimi Linear reference not vendored at {REFERENCE_DIR}; "
            "run `uv run python scripts/clone_kimi_linear_reference.py`"
        )
    restore = shim_transformers_for_reference()
    install_fla_stub()
    parent = str(REFERENCE_DIR.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        reference = importlib.import_module("kimi_linear_reference.modeling_kimi")
    finally:
        # The reference bound the compat wrapper at import; every other
        # family must keep seeing the 5.14-native function.
        restore()
    _patch_reference_cache_protocol(reference)
    return reference


def _patch_reference_cache_protocol(reference: Any) -> None:
    """Bridge `KimiDynamicCache` (written to the 4.5x Cache mask protocol) to 5.14.

    5.14's mask preprocessing calls `get_query_offset(layer_idx)` (absent in
    the vendored class) and `get_mask_sizes(q_length: int, layer_idx)` (the
    vendored one expects a `cache_position` TENSOR as the first arg). Both
    reduce to the same quantities: queries start after the past tokens, and
    the mask spans past + new. Patched on the class, never in the file.
    """

    def get_query_offset(self: Any, layer_idx: int = 0) -> int:
        return int(self.get_seq_length(layer_idx))

    def get_mask_sizes(self: Any, q_length: int, layer_idx: int) -> tuple[int, int]:
        return q_length + int(self.get_seq_length(layer_idx)), 0

    reference.KimiDynamicCache.get_query_offset = get_query_offset
    reference.KimiDynamicCache.get_mask_sizes = get_mask_sizes
