"""Weight-only INT8 quantization (W8A16) with symmetric per-output-channel scales.

The standard recipe for weight-only INT8: each row of an `(out, in)` weight
matrix gets one fp16/bf16 scale `s = max(|row|) / 127`; quantized values are
`round(W / s).clamp(-127, 127).to(int8)`. Dequant on forward multiplies by `s`
and casts back to the activation's dtype.

Activations are not quantized. There is no calibration step. This is the same
mechanism that HuggingFace `load_in_8bit`, bitsandbytes' `Linear8bitLt`, and
vLLM's W8A16 path use; we implement it from scratch so the integration is
explicit and the math is readable.

The forward path here is naive: `W_dq = W_q.to(dtype) * scales` followed by
`F.linear(x, W_dq, bias)`. The dequant is `O(out * in)` regardless of batch
size, so for large batches it amortizes; for batch=1 decode it's a fixed cost
on top of the matmul. A fused dequant-matmul kernel (Triton) would be a
future fast path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def quantize_per_channel(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-output-channel INT8 quantization.

    Args:
        weight: float tensor of shape `(out_features, in_features)`.

    Returns:
        `(q_weight, scales)` where `q_weight` is int8 with the same shape as
        `weight`, and `scales` is shape `(out_features,)` in `weight`'s dtype.

    The per-row scale is `max(|row|) / 127`. Rows with all zeros get a tiny
    epsilon scale to avoid division by zero; the resulting q_weight stays
    all-zero, which is the correct quantized representation.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"quantize_per_channel expects a 2D weight, got shape {tuple(weight.shape)}"
        )

    abs_max = weight.detach().abs().amax(dim=1)
    # Floor the per-row max with a small eps so all-zero rows don't divide by 0.
    # The eps is well below any real weight scale; q values are still 0.
    eps = torch.finfo(weight.dtype).tiny
    abs_max = abs_max.clamp_min(eps)
    scales = (abs_max / 127.0).to(weight.dtype)

    # Use float32 for the divide+round to avoid bf16/fp16 rounding artifacts in
    # the quantization itself; cast back to int8 at the end.
    scales_f32 = scales.float().unsqueeze(1)
    q_weight = (weight.float() / scales_f32).round().clamp_(-127, 127).to(torch.int8)
    return q_weight, scales


def dequantize_per_channel(q_weight: Tensor, scales: Tensor, dtype: torch.dtype) -> Tensor:
    """Inverse of `quantize_per_channel`. Useful for tests and the naive forward."""
    if q_weight.dim() != 2:
        raise ValueError(
            f"dequantize_per_channel expects a 2D q_weight, got shape {tuple(q_weight.shape)}"
        )
    if scales.dim() != 1 or scales.shape[0] != q_weight.shape[0]:
        raise ValueError(
            f"scales shape {tuple(scales.shape)} mismatches q_weight rows {q_weight.shape[0]}"
        )
    return q_weight.to(dtype) * scales.to(dtype).unsqueeze(1)


class Int8Linear(nn.Module):
    """Drop-in replacement for `nn.Linear` with weight-only INT8 storage.

    The `weight` parameter holds INT8 values, `scales` holds one fp scale per
    output channel, and `bias` (if present) stays in the activation dtype.
    Forward dequantizes the entire weight matrix to the input's dtype and runs
    a standard `F.linear`.

    Buffers (not Parameters) hold the int8 weight and scales: they should not
    be touched by autograd, and they're not optimizer-eligible.
    """

    in_features: int
    out_features: int
    weight: Tensor
    scales: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        scale_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Buffers, not Parameters: int8 has no autograd path and we don't want
        # the optimizer to find them.
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer("scales", torch.empty(out_features, dtype=scale_dtype, device=device))
        if bias:
            # Bias stays in the float dtype matching activations; treat as a
            # Parameter for state_dict naming compatibility with nn.Linear.
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=scale_dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, fp_linear: nn.Linear) -> Int8Linear:
        """Build an Int8Linear from an existing `nn.Linear`, quantizing weights.

        Bias and scales inherit the original Linear's dtype; weight goes to
        int8. The original Linear's device is preserved.
        """
        device = fp_linear.weight.device
        scale_dtype = fp_linear.weight.dtype
        out_features, in_features = fp_linear.weight.shape
        q = cls(
            in_features=in_features,
            out_features=out_features,
            bias=fp_linear.bias is not None,
            scale_dtype=scale_dtype,
            device=device,
        )
        q_weight, scales = quantize_per_channel(fp_linear.weight.detach())
        q.weight.copy_(q_weight)
        q.scales.copy_(scales)
        if fp_linear.bias is not None:
            assert q.bias is not None
            q.bias.detach().copy_(fp_linear.bias.detach())
        return q

    def forward(self, x: Tensor) -> Tensor:
        # Dequant happens in `x.dtype` so downstream matmul stays in that dtype.
        # Multiplying int8 by a float dtype upcasts; the broadcast of (out, in)
        # by (out, 1) yields the dequantized weight matrix in `x.dtype`. The
        # bias may have been stored in a different float dtype than `x` (e.g.
        # an fp32 model called with bf16 input via autocast), so cast it too.
        w_dq = self.weight.to(x.dtype) * self.scales.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w_dq, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, dtype=int8 (W8A16)"
        )


def quantize_model_to_int8(
    model: nn.Module,
    skip_modules: frozenset[str] | set[str] = frozenset({"lm_head"}),
) -> int:
    """Walk `model` and replace each `nn.Linear` (not in `skip_modules`) in place.

    Module names are matched as suffix-of-dotted-path: a module named
    `model.layers.5.self_attn.q_proj` matches `skip_modules = {"q_proj"}`. This
    keeps the API ergonomic without needing the full module path.

    Returns the number of replaced modules.

    The walk is post-order: we first collect the (parent, attribute_name,
    child_module) tuples, then perform the replacements after the iteration to
    avoid mutating during traversal.
    """
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    skip = set(skip_modules)
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
        if leaf_name in skip:
            continue
        parent_path, _, attr_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        targets.append((parent, attr_name, module))

    for parent, attr_name, fp_linear in targets:
        new_linear = Int8Linear.from_float(fp_linear)
        setattr(parent, attr_name, new_linear)

    return len(targets)
