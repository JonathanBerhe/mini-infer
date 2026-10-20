"""Fused Gated DeltaNet decode step (one token) in two Triton kernels.

The eager decode step of a Gated DeltaNet layer, between its input
projections and `out_proj`, runs about 50 small kernels: the short-conv
update, the q/k/v split and head broadcast, two L2 norms, the gate, the
delta-rule update and readout, and the gated RMSNorm. At batch 1 each of
them moves a few KB, so their count, not their bytes, sets the time. Here
the same math runs in two launches:

1. `_conv_step_kernel`, one program per `(row, channel block)`: shift the
   raw-input tail of the fused q/k/v convolution, write the new input,
   apply the depthwise taps and SiLU. It updates the conv cache in place.
   It is a separate launch because the q and k channels are shared by
   `num_v_heads // num_k_heads` value heads; one program per channel is
   the only owner that can update them without a race.
2. `_recurrent_step_kernel`, one program per `(row, value head)`: L2-norm q
   and k, compute the gate and step size, decay the `(head_k_dim,
   head_v_dim)` fp32 state, apply the delta-rule update, read out, and
   apply the SiLU-gated RMSNorm. It updates the recurrent state in place.

The kernels round to the model dtype at the same points as the eager code
(`blocks/kda.py`, `blocks/gdn.py`), so they differ from it only in fp32
summation order. Both launches are capturable in a CUDA graph.

CUDA only. On other devices `_QwenGatedDeltaNet` keeps the eager path, which
is also the reference in `tests/unit/test_gdn_decode_kernel.py`.
"""

from __future__ import annotations

import torch

from mini_infer.device import is_cuda_device

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # macOS / no-CUDA installs typically lack triton
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


# Runtime toggle for benchmarking. When True, `supports_gdn_decode_kernel`
# reports False even on CUDA, so the eager step runs. The bench uses it to
# A/B the two paths on one model load. Not a public API.
_KERNEL_DISABLED_FOR_BENCH = False

_CONV_BLOCK = 256


def supports_gdn_decode_kernel(device: torch.device | str) -> bool:
    """Whether the fused Gated DeltaNet decode step can run on this device."""
    if _KERNEL_DISABLED_FOR_BENCH or not _TRITON_AVAILABLE:
        return False
    return is_cuda_device(device)


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _conv_step_kernel(  # type: ignore[no-untyped-def]
        mixed_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        conved_ptr,
        channels,
        KERNEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        in_range = offs < channels
        taps = tl.arange(0, KERNEL)

        x = tl.load(mixed_ptr + row * channels + offs, mask=in_range, other=0.0)
        state_rows = conv_state_ptr + (row * channels + offs[:, None]) * KERNEL
        # The tail rolls left by one: slot j takes old slot j + 1, and the
        # newest slot takes this token's raw input.
        shifted = tl.load(
            state_rows + taps[None, :] + 1,
            mask=in_range[:, None] & (taps[None, :] + 1 < KERNEL),
            other=0.0,
        )
        new_state = tl.where(
            taps[None, :] == KERNEL - 1,
            x[:, None].to(conv_state_ptr.dtype.element_ty),
            shifted,
        )
        weight = tl.load(
            conv_weight_ptr + offs[:, None] * KERNEL + taps[None, :],
            mask=in_range[:, None],
            other=0.0,
        )
        y = tl.sum(new_state.to(tl.float32) * weight.to(tl.float32), axis=1)
        y = y * tl.sigmoid(y)
        tl.store(
            conved_ptr + row * channels + offs, y.to(conved_ptr.dtype.element_ty), mask=in_range
        )
        tl.store(state_rows + taps[None, :], new_state, mask=in_range[:, None])

    @triton.jit
    def _log1p(x):  # type: ignore[no-untyped-def]
        # Kahan's form: exact for tiny x, where log(1 + x) would round to 0.
        # The guarded denominator keeps the unused branch free of 0 / 0.
        u = 1.0 + x
        denominator = tl.where(u == 1.0, 1.0, u - 1.0)
        return tl.where(u == 1.0, x, tl.log(u) * x / denominator)

    @triton.jit  # type: ignore[untyped-decorator]
    def _recurrent_step_kernel(  # type: ignore[no-untyped-def]
        conved_ptr,
        a_ptr,
        b_ptr,
        a_log_ptr,
        dt_bias_ptr,
        state_ptr,
        z_ptr,
        norm_weight_ptr,
        out_ptr,
        channels,
        num_v_heads,
        head_ratio,
        key_dim,
        q_scale,
        rms_eps,
        HEAD_K: tl.constexpr,
        HEAD_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        key_head = head // head_ratio
        offs_k = tl.arange(0, HEAD_K)
        offs_v = tl.arange(0, HEAD_V)
        model_dtype = out_ptr.dtype.element_ty

        base = conved_ptr + row * channels
        q = tl.load(base + key_head * HEAD_K + offs_k).to(tl.float32)
        k = tl.load(base + key_dim + key_head * HEAD_K + offs_k).to(tl.float32)
        v = tl.load(base + 2 * key_dim + head * HEAD_V + offs_v).to(tl.float32)
        # L2 norm with eps inside the root, then the 1/sqrt(head_k) query scale.
        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
        q = q * q_scale
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)

        # Per-head decay -exp(A_log) * softplus(a + dt_bias), with torch's
        # softplus threshold of 20. The step size is a sigmoid rounded to the
        # model dtype, as the eager projection output is.
        a = tl.load(a_ptr + row * num_v_heads + head).to(tl.float32)
        a = a + tl.load(dt_bias_ptr + head).to(tl.float32)
        softplus = tl.where(a > 20.0, a, _log1p(tl.exp(a)))
        decay = tl.exp(-tl.exp(tl.load(a_log_ptr + head).to(tl.float32)) * softplus)
        b = tl.load(b_ptr + row * num_v_heads + head).to(tl.float32)
        beta = tl.sigmoid(b).to(b_ptr.dtype.element_ty).to(tl.float32)

        state_tile = (
            state_ptr + ((row * num_v_heads + head) * HEAD_K + offs_k[:, None]) * HEAD_V
        ) + offs_v[None, :]
        state = tl.load(state_tile) * decay
        predicted = tl.sum(k[:, None] * state, axis=0)
        state = state + (beta * k)[:, None] * (v - predicted)[None, :]
        tl.store(state_tile, state)
        o = tl.sum(q[:, None] * state, axis=0)

        # SiLU-gated RMSNorm with the eager rounding points: the readout and
        # the normalized value are rounded to the model dtype before the
        # affine weight, the weighted value again before the gate.
        o = o.to(model_dtype).to(tl.float32)
        normed = o * tl.rsqrt(tl.sum(o * o, axis=0) / HEAD_V + rms_eps)
        weight = tl.load(norm_weight_ptr + offs_v).to(tl.float32)
        normed = (weight * normed.to(model_dtype).to(tl.float32)).to(model_dtype).to(tl.float32)
        z = tl.load(z_ptr + (row * num_v_heads + head) * HEAD_V + offs_v).to(tl.float32)
        gated = normed * (z * tl.sigmoid(z))
        tl.store(out_ptr + (row * num_v_heads + head) * HEAD_V + offs_v, gated.to(model_dtype))


def gdn_decode_step(
    mixed: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    rms_eps: float,
) -> torch.Tensor:
    """One-token Gated DeltaNet step from the projections to the gated norm.

    Args:
        mixed: `(batch, conv_channels)` `in_proj_qkv` output, model dtype.
        conv_state: `(batch, conv_channels, kernel)` raw-input tail, newest
            last; updated in place.
        conv_weight: `(conv_channels, kernel)` depthwise taps.
        a_raw, b_raw: `(batch, num_v_heads)` `in_proj_a` / `in_proj_b` outputs.
        a_log, dt_bias: `(num_v_heads,)` gate parameters.
        recurrent_state: `(batch, num_v_heads, head_k_dim, head_v_dim)` fp32;
            updated in place.
        z: `(batch, num_v_heads * head_v_dim)` `in_proj_z` output.
        norm_weight: `(head_v_dim,)` affine weight of the gated RMSNorm.

    Returns:
        `(batch, num_v_heads * head_v_dim)` in the model dtype, the input of
        `out_proj`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("the fused Gated DeltaNet decode step needs Triton")
    batch, channels = mixed.shape
    num_v_heads = a_raw.shape[-1]
    kernel = conv_state.shape[-1]
    for name, size in (("head_k_dim", head_k_dim), ("head_v_dim", head_v_dim), ("kernel", kernel)):
        if size & (size - 1):
            raise ValueError(f"{name} must be a power of two, got {size}")
    if recurrent_state.dtype != torch.float32:
        raise ValueError(f"recurrent_state must be fp32, got {recurrent_state.dtype}")
    for tensor in (mixed, conv_state, conv_weight, a_raw, b_raw, recurrent_state, z):
        if not tensor.is_contiguous():
            raise ValueError("the fused Gated DeltaNet decode step needs contiguous inputs")

    conved = torch.empty_like(mixed)
    _conv_step_kernel[(batch, triton.cdiv(channels, _CONV_BLOCK))](
        mixed, conv_state, conv_weight, conved, channels, KERNEL=kernel, BLOCK=_CONV_BLOCK
    )
    out = torch.empty(batch, num_v_heads * head_v_dim, dtype=mixed.dtype, device=mixed.device)
    _recurrent_step_kernel[(batch, num_v_heads)](
        conved,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        recurrent_state,
        z,
        norm_weight,
        out,
        channels,
        num_v_heads,
        num_v_heads // num_k_heads,
        num_k_heads * head_k_dim,
        head_k_dim**-0.5,
        rms_eps,
        HEAD_K=head_k_dim,
        HEAD_V=head_v_dim,
        num_warps=8,
    )
    return out
