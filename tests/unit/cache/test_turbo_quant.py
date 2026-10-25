"""Unit tests for the standalone TurboQuant primitives.

End-to-end model parity (Qwen2.5-0.5B with kv_quant="turbo4") lives in
`test_turbo_quant_integration.py`; these tests cover only the math.
"""

import pytest
import torch

from mini_infer.cache.turbo_quant import (
    dequantize_kv_block,
    generate_rotation_matrices,
    inverse_rotate,
    quantize_kv_block,
    rotate,
)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
    )


def test_generate_rotation_matrices_shape_and_dtype() -> None:
    matrices = generate_rotation_matrices(num_layers=4, head_dim=32, dtype=torch.float32)
    assert matrices.shape == (4, 32, 32)
    assert matrices.dtype == torch.float32


def test_generate_rotation_matrices_is_orthogonal() -> None:
    """Each rotation matrix R must satisfy R @ R.T ≈ I."""
    matrices = generate_rotation_matrices(num_layers=3, head_dim=64, dtype=torch.float32)
    eye = torch.eye(64)
    for layer_idx in range(3):
        rotation = matrices[layer_idx]
        product = rotation @ rotation.T
        assert torch.allclose(product, eye, atol=1e-5), (
            f"layer {layer_idx} failed orthogonality: max dev "
            f"{(product - eye).abs().max().item():.2e}"
        )


def test_generate_rotation_matrices_is_deterministic_with_seed() -> None:
    """Same seed -> identical matrices (reproducibility for tests + serving)."""
    a = generate_rotation_matrices(num_layers=2, head_dim=16, dtype=torch.float32, seed=42)
    b = generate_rotation_matrices(num_layers=2, head_dim=16, dtype=torch.float32, seed=42)
    assert torch.equal(a, b)


def test_generate_rotation_matrices_different_seeds_differ() -> None:
    a = generate_rotation_matrices(num_layers=1, head_dim=16, dtype=torch.float32, seed=1)
    b = generate_rotation_matrices(num_layers=1, head_dim=16, dtype=torch.float32, seed=2)
    assert not torch.allclose(a, b)


def test_generate_rotation_matrices_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        generate_rotation_matrices(num_layers=0, head_dim=8)
    with pytest.raises(ValueError, match="must be positive"):
        generate_rotation_matrices(num_layers=2, head_dim=0)


def test_rotate_inverse_rotate_roundtrip() -> None:
    """rotate followed by inverse_rotate returns the original tensor."""
    rotation = generate_rotation_matrices(num_layers=1, head_dim=8, dtype=torch.float32)[0]
    x = torch.randn(4, 2, 8)
    rotated = rotate(x, rotation)
    recovered = inverse_rotate(rotated, rotation)
    assert torch.allclose(x, recovered, atol=1e-5)


def test_rotate_preserves_inner_products() -> None:
    """The defining property: (Q@R) @ (K@R).T == Q @ K.T (rotation is benign)."""
    rotation = generate_rotation_matrices(num_layers=1, head_dim=16, dtype=torch.float32)[0]
    q = torch.randn(3, 16)
    k = torch.randn(5, 16)
    q_rot = rotate(q, rotation)
    k_rot = rotate(k, rotation)
    assert torch.allclose(q @ k.T, q_rot @ k_rot.T, atol=1e-4)


def test_rotate_rejects_dim_mismatch() -> None:
    rotation = torch.eye(8)
    with pytest.raises(ValueError, match="head_dim"):
        rotate(torch.randn(4, 16), rotation)


def test_quantize_dequantize_roundtrip_within_tolerance() -> None:
    """Per-channel 4-bit quant: max abs error per element <= scale / 2."""
    torch.manual_seed(0)
    block_size, num_kv_heads, head_dim = 4, 2, 8
    block = torch.randn(block_size, num_kv_heads, head_dim, dtype=torch.float32)

    packed, low, scale = quantize_kv_block(block)
    recovered = dequantize_kv_block(
        packed, low, scale, block_size, num_kv_heads, head_dim, dtype=torch.float32
    )

    assert recovered.shape == block.shape
    # Per-channel max abs error <= scale / 2 + tiny fp slack.
    err = (recovered - block).abs()
    bound = scale.unsqueeze(0).expand_as(err) / 2.0 + 1e-5
    assert (err <= bound).all(), f"max excess {(err / bound).max().item():.3f}; expected <= 1.0"


def test_quantize_dequantize_high_cosine_similarity() -> None:
    """Cosine similarity of original vs roundtripped block is > 0.99 for typical data."""
    torch.manual_seed(1)
    # 16-token block, GQA shape (2 KV heads, head_dim=64). Roughly Qwen2.5-0.5B.
    block = torch.randn(16, 2, 64, dtype=torch.float32) * 0.5

    packed, low, scale = quantize_kv_block(block)
    recovered = dequantize_kv_block(packed, low, scale, 16, 2, 64, dtype=torch.float32)
    cos = _cosine_sim(block, recovered)
    assert cos > 0.99, f"cos sim {cos:.4f} < 0.99"


def test_quantize_returns_correct_shapes_and_dtypes() -> None:
    block = torch.randn(8, 4, 16, dtype=torch.bfloat16)
    packed, low, scale = quantize_kv_block(block)
    assert packed.shape == (8 * 4 * 16 // 2,)
    assert packed.dtype == torch.int8
    assert low.shape == (4, 16)
    assert low.dtype == torch.bfloat16
    assert scale.shape == (4, 16)
    assert scale.dtype == torch.bfloat16


def test_quantize_handles_constant_block() -> None:
    """All-equal channel: low == high; scale gets a tiny epsilon, dequant returns the constant."""
    block = torch.full((4, 1, 8), 3.5, dtype=torch.float32)
    packed, low, scale = quantize_kv_block(block)
    recovered = dequantize_kv_block(packed, low, scale, 4, 1, 8, dtype=torch.float32)
    # The eps in scale + clamp(0, 15) means q is 0 (since x - low == 0 / eps == 0).
    # Reconstruction: low + 0 * scale == 3.5. Tolerance for fp drift.
    assert torch.allclose(recovered, block, atol=1e-3)


def test_quantize_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match="block_size, num_kv_heads, head_dim"):
        quantize_kv_block(torch.randn(8, 4))


def test_quantize_rejects_odd_total_elements() -> None:
    """4-bit packing requires the total element count to be even."""
    # 1*1*1 = 1, odd — can't pack as 4-bit pairs.
    with pytest.raises(ValueError, match="even"):
        quantize_kv_block(torch.randn(1, 1, 1, dtype=torch.float32))


def test_dequantize_validates_shapes() -> None:
    block = torch.randn(4, 2, 8, dtype=torch.float32)
    packed, low, scale = quantize_kv_block(block)
    with pytest.raises(ValueError, match="packed shape"):
        dequantize_kv_block(packed[:-1], low, scale, 4, 2, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="low/scale shapes"):
        dequantize_kv_block(packed, low[:-1], scale, 4, 2, 8, dtype=torch.float32)


def test_full_pipeline_rotation_plus_quant_within_cosine() -> None:
    """End-to-end: rotate + quantize + dequantize + inverse-rotate ≈ identity."""
    torch.manual_seed(2)
    rotation = generate_rotation_matrices(num_layers=1, head_dim=64, dtype=torch.float32)[0]
    block = torch.randn(16, 2, 64, dtype=torch.float32) * 0.4

    rotated = rotate(block, rotation)
    packed, low, scale = quantize_kv_block(rotated)
    recovered_rotated = dequantize_kv_block(packed, low, scale, 16, 2, 64, dtype=torch.float32)
    recovered = inverse_rotate(recovered_rotated, rotation)

    cos = _cosine_sim(block, recovered)
    assert cos > 0.99, f"end-to-end cos sim {cos:.4f} < 0.99"


# ──────────────────────────────────────────────────────────────────────
# V3 (TurboQuant full): polar + Lloyd-Max + QJL primitives
# ──────────────────────────────────────────────────────────────────────


from mini_infer.cache.turbo_quant import (  # noqa: E402
    lloyd_max_codebook,
    polar_dequantize_block,
    polar_quantize_block,
)


def test_lloyd_max_codebook_4bit_shape_and_symmetry() -> None:
    cb = lloyd_max_codebook(4, dtype=torch.float32, device="cpu")
    assert cb.shape == (16,)
    # Symmetric around 0.
    assert torch.allclose(cb + cb.flip(0), torch.zeros_like(cb), atol=1e-5)
    # Sorted ascending.
    assert torch.all(cb[:-1] < cb[1:])


def test_lloyd_max_codebook_3bit_shape_and_symmetry() -> None:
    cb = lloyd_max_codebook(3, dtype=torch.float32, device="cpu")
    assert cb.shape == (8,)
    assert torch.allclose(cb + cb.flip(0), torch.zeros_like(cb), atol=1e-5)
    assert torch.all(cb[:-1] < cb[1:])


def test_lloyd_max_codebook_rejects_unsupported_bits() -> None:
    with pytest.raises(ValueError, match="3- or 4-bit"):
        lloyd_max_codebook(2, dtype=torch.float32, device="cpu")


def test_polar_quantize_dequantize_uniform_4bit_baseline() -> None:
    """V3a alone (uniform 4-bit on [-1, 1]) is the strawman.

    Unit-vector coords on the rotated sphere have std ~1/sqrt(head_dim), so
    uniform [-1, 1] wastes most of its range — exactly the inefficiency
    V3c (Lloyd-Max codebook) is designed to fix. We only require the
    roundtrip to be sane (> 0.9) here; the > 0.99 threshold is the
    Lloyd-Max test below.
    """
    torch.manual_seed(0)
    block = torch.randn(16, 2, 64, dtype=torch.float32) * 0.5

    packed, radii = polar_quantize_block(block, bits=4, use_lloyd_max=False, use_qjl=False)
    recovered = polar_dequantize_block(
        packed,
        radii,
        16,
        2,
        64,
        dtype=torch.float32,
        bits=4,
        use_lloyd_max=False,
        use_qjl=False,
    )

    cos = _cosine_sim(block, recovered)
    assert cos > 0.9, f"polar + uniform 4-bit cos sim {cos:.4f} < 0.9"


def test_polar_quantize_dequantize_lloyd_max_4bit_high_cosine() -> None:
    """V3a + V3c: polar + Lloyd-Max 4-bit beats uniform on cosine sim."""
    torch.manual_seed(0)
    block = torch.randn(16, 2, 64, dtype=torch.float32) * 0.5

    # Uniform reference.
    packed_u, radii_u = polar_quantize_block(block, bits=4, use_lloyd_max=False, use_qjl=False)
    rec_u = polar_dequantize_block(
        packed_u,
        radii_u,
        16,
        2,
        64,
        dtype=torch.float32,
        bits=4,
        use_lloyd_max=False,
        use_qjl=False,
    )
    cos_u = _cosine_sim(block, rec_u)

    # Lloyd-Max.
    packed_lm, radii_lm = polar_quantize_block(block, bits=4, use_lloyd_max=True, use_qjl=False)
    rec_lm = polar_dequantize_block(
        packed_lm,
        radii_lm,
        16,
        2,
        64,
        dtype=torch.float32,
        bits=4,
        use_lloyd_max=True,
        use_qjl=False,
    )
    cos_lm = _cosine_sim(block, rec_lm)

    assert cos_lm > cos_u, f"Lloyd-Max should beat uniform; got LM={cos_lm:.4f} <= U={cos_u:.4f}"
    assert cos_lm > 0.99, f"Lloyd-Max 4-bit cos sim {cos_lm:.4f} < 0.99"


def test_polar_quantize_dequantize_lloyd_max_3bit_with_qjl() -> None:
    """V3b + V3d: 3-bit Lloyd-Max + QJL residual sign approaches 4-bit fidelity."""
    torch.manual_seed(0)
    block = torch.randn(16, 2, 64, dtype=torch.float32) * 0.5

    # 3-bit alone (8 levels).
    packed_3, radii_3 = polar_quantize_block(block, bits=3, use_lloyd_max=True, use_qjl=False)
    rec_3 = polar_dequantize_block(
        packed_3,
        radii_3,
        16,
        2,
        64,
        dtype=torch.float32,
        bits=3,
        use_lloyd_max=True,
        use_qjl=False,
    )
    cos_3 = _cosine_sim(block, rec_3)

    # 3-bit + QJL (= 4 bits stored, 8 levels with sign-bit refinement).
    packed_3q, radii_3q = polar_quantize_block(block, bits=3, use_lloyd_max=True, use_qjl=True)
    rec_3q = polar_dequantize_block(
        packed_3q,
        radii_3q,
        16,
        2,
        64,
        dtype=torch.float32,
        bits=3,
        use_lloyd_max=True,
        use_qjl=True,
    )
    cos_3q = _cosine_sim(block, rec_3q)

    assert cos_3q > cos_3, (
        f"QJL should improve 3-bit accuracy; got with-QJL={cos_3q:.4f} <= without={cos_3:.4f}"
    )


def test_polar_qjl_only_with_3bit() -> None:
    with pytest.raises(ValueError, match="bits=3"):
        polar_quantize_block(torch.randn(2, 1, 4), bits=4, use_lloyd_max=True, use_qjl=True)


def test_polar_quantize_returns_correct_shapes() -> None:
    block = torch.randn(8, 4, 16, dtype=torch.bfloat16)
    packed, radii = polar_quantize_block(block, bits=4, use_lloyd_max=True)
    assert packed.shape == (8 * 4 * 16 // 2,)
    assert packed.dtype == torch.int8
    assert radii.shape == (8, 4)
    assert radii.dtype == torch.bfloat16


def test_polar_full_pipeline_with_rotation_within_cosine() -> None:
    """End-to-end rotation + polar (Lloyd-Max + QJL): cosine > 0.99."""
    torch.manual_seed(2)
    rotation = generate_rotation_matrices(num_layers=1, head_dim=64, dtype=torch.float32)[0]
    block = torch.randn(16, 2, 64, dtype=torch.float32) * 0.4

    rotated = rotate(block, rotation)
    packed, radii = polar_quantize_block(rotated, bits=3, use_lloyd_max=True, use_qjl=True)
    recovered_rot = polar_dequantize_block(
        packed,
        radii,
        16,
        2,
        64,
        dtype=torch.float32,
        bits=3,
        use_lloyd_max=True,
        use_qjl=True,
    )
    recovered = inverse_rotate(recovered_rot, rotation)

    cos = _cosine_sim(block, recovered)
    # 3-bit with QJL is the most aggressive V3 config; cosine should still
    # comfortably exceed 0.99 because the rotation Gaussianizes the input.
    assert cos > 0.99, f"3-bit+QJL end-to-end cos sim {cos:.4f} < 0.99"
