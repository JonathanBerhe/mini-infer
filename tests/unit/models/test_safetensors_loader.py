"""Tests for `load_safetensors_state_dict` dtype handling.

Regression coverage for the V4-Flash load path: packed NVFP4 expert
weights ship as raw int8 bytes (safetensors can't carry the
`float4_e2m1fn_x2` type), at the packed half-width shape
`(out, in // 2)`. The loader must NOT cast these to BF16, because:

  1. The downstream dequant detects packed FP4 by its int8 dtype; a
     premature BF16 cast defeats detection and the tensor passes through
     un-dequantized.
  2. The un-dequantized tensor keeps its half-width `(out, in // 2)`
     shape, which then loads silently and only blows up at the first
     forward matmul (`mat1 and mat2 shapes cannot be multiplied`).

These tests pin the dtype-preservation contract so the bug can't recur.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from mini_infer.models.loader import iter_safetensors_shards, load_safetensors_state_dict


def test_int8_weights_are_not_cast_to_bf16(tmp_path) -> None:
    """Packed-FP4-as-int8 weights survive the load at their stored dtype.

    Casting int8 -> bf16 here is the V4-Flash expert-load bug: it both
    defeats the int8-based dequant detection and freezes the tensor at
    its packed half-width shape.
    """
    out_dim, half_in = 8, 4  # packed shape (out, in // 2)
    packed = torch.randint(0, 256, (out_dim, half_in), dtype=torch.uint8)
    packed_i8 = packed.view(torch.int8)
    # A normal (fp16) weight in the same file must still be cast to bf16,
    # so the int8-preservation rule doesn't accidentally freeze everything.
    plain = torch.randn(8, 16, dtype=torch.float16)

    save_file(
        {"mlp.experts.0.w1.weight": packed_i8, "mlp.gate.weight": plain},
        str(tmp_path / "model.safetensors"),
    )

    state_dict = load_safetensors_state_dict(str(tmp_path), device="cpu", dtype=torch.bfloat16)

    # int8 packed weight preserved exactly (dtype AND shape).
    loaded_packed = state_dict["mlp.experts.0.w1.weight"]
    assert loaded_packed.dtype == torch.int8
    assert loaded_packed.shape == (out_dim, half_in)
    assert torch.equal(loaded_packed, packed_i8)

    # Non-quantized float weight still cast to the requested dtype.
    assert state_dict["mlp.gate.weight"].dtype == torch.bfloat16


def test_uint8_weights_are_not_cast_to_bf16(tmp_path) -> None:
    """uint8 storage path is preserved too (some tools emit uint8, not int8)."""
    packed = torch.randint(0, 256, (8, 4), dtype=torch.uint8)
    save_file({"w.weight": packed}, str(tmp_path / "model.safetensors"))

    state_dict = load_safetensors_state_dict(str(tmp_path), device="cpu", dtype=torch.bfloat16)

    assert state_dict["w.weight"].dtype == torch.uint8
    assert torch.equal(state_dict["w.weight"], packed)


def test_fp32_weights_are_preserved(tmp_path) -> None:
    """FP32 sources (V4 Hyper-Connections params) are NOT downcast to bf16."""
    fp32 = torch.randn(4, 4, dtype=torch.float32)
    save_file({"hc.fn": fp32}, str(tmp_path / "model.safetensors"))

    state_dict = load_safetensors_state_dict(str(tmp_path), device="cpu", dtype=torch.bfloat16)

    assert state_dict["hc.fn"].dtype == torch.float32


def test_bf16_target_casts_float16_source(tmp_path) -> None:
    """A non-target float dtype (fp16) is cast to the requested dtype."""
    fp16 = torch.randn(4, 4, dtype=torch.float16)
    save_file({"w.weight": fp16}, str(tmp_path / "model.safetensors"))

    state_dict = load_safetensors_state_dict(str(tmp_path), device="cpu", dtype=torch.bfloat16)

    assert state_dict["w.weight"].dtype == torch.bfloat16


def test_bounded_batches_split_a_single_file_and_keep_every_tensor(tmp_path) -> None:
    """`max_bytes` yields a large single-file checkpoint in small batches."""
    # fp16 sources, so the bf16 cast applies (fp32 sources are kept as is).
    tensors = {
        f"layer.{i}.weight": torch.full((16, 16), float(i), dtype=torch.float16) for i in range(5)
    }
    tensors["layer.packed"] = torch.randint(0, 256, (8, 4), dtype=torch.uint8).view(torch.int8)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    batches = list(
        iter_safetensors_shards(
            str(tmp_path), device="cpu", dtype=torch.bfloat16, max_bytes=2 * 16 * 16 * 2
        )
    )

    assert len(batches) > 1
    assert all(len(batch) <= 2 for batch in batches)
    merged = {key: value for batch in batches for key, value in batch.items()}
    assert set(merged) == set(tensors)
    assert merged["layer.3.weight"].dtype == torch.bfloat16
    assert torch.equal(merged["layer.3.weight"].float(), tensors["layer.3.weight"])
    assert merged["layer.packed"].dtype == torch.int8
