"""YaRN long-context RoPE: math + bit-parity vs the V4 reference.

YaRN ([Peng et al. 2023](https://arxiv.org/abs/2309.00071)) extends RoPE
past the training context by selectively scaling low-frequency components.
This test file:

  1. **Backward compat**: with `yarn_original_seq_len == 0` (default),
     `RotaryEmbedding` produces the same `inv_freq` as before YaRN landed.
  2. **Correctness**: YaRN-corrected `inv_freq` blends scaled and
     unscaled components — high-frequency components stay unchanged,
     low-frequency components get divided by `scaling_factor`.
  3. **Bit-parity vs reference** on the canonical DeepSeek-V2-Lite YaRN
     config (`original=4096, factor=40, beta_fast=32, beta_slow=1`):
     compare our `inv_freq` and the resulting `(cos, sin)` table to the
     V4 reference's `precompute_freqs_cis(...)` output element-by-element.
"""

from __future__ import annotations

from typing import Any

import torch

from mini_infer.models.blocks.rope import (
    RotaryEmbedding,
    _yarn_correction_dim,
    _yarn_correction_range,
    _yarn_linear_ramp,
    apply_yarn_correction,
)

# ---------- Math primitives ----------


def test_correction_dim_inverts_expected_relation() -> None:
    """`_yarn_correction_dim(num_rotations, ...)` is the inverse of the
    rotation-count formula `max_seq_len * f_i / (2*pi) = num_rotations`."""
    head_dim = 64
    base = 10000.0
    max_seq_len = 4096
    for num_rotations in (1.0, 4.0, 32.0):
        component_idx = _yarn_correction_dim(num_rotations, head_dim, base, max_seq_len)
        # Reconstruct the rotation count from the index: f_i = base^(-component_idx / head_dim).
        # rotations = max_seq_len * f_i / (2*pi) (with the 2x because index uses pairs).
        # We allow some tolerance because of float math.
        import math

        f_i = base ** (-component_idx / (head_dim / 2))
        reconstructed_rotations = max_seq_len * f_i / (2 * math.pi)
        assert abs(reconstructed_rotations - num_rotations) < 1e-3 * num_rotations


def test_correction_range_orders_low_below_high() -> None:
    """`beta_fast=32` (more rotations) sits BELOW `beta_slow=1` in component
    index because higher frequencies have lower indices in the inv_freq table."""
    low, high = _yarn_correction_range(
        beta_fast=32, beta_slow=1, head_dim=64, base=10000.0, max_seq_len=4096
    )
    assert 0 <= low <= high <= 63


def test_linear_ramp_clamps_at_endpoints_and_interpolates_in_between() -> None:
    ramp = _yarn_linear_ramp(min_index=4, max_index=8, num_components=12)
    assert ramp.shape == (12,)
    # Below `min_index`: 0.0
    assert torch.all(ramp[:4] == 0.0)
    # Above `max_index`: 1.0
    assert torch.all(ramp[8:] == 1.0)
    # Between: linear.
    expected_mid = torch.tensor([0.0, 0.25, 0.5, 0.75])
    assert torch.allclose(ramp[4:8], expected_mid, atol=1e-6)


def test_linear_ramp_handles_degenerate_min_equals_max() -> None:
    """`min == max` should not divide by zero — the helper nudges max by 0.001.

    With nudge, `(idx - min) / (max + 0.001 - min) = (idx - min) * 1000`. So
    index < min stays at 0 after clamping, index == min stays at 0 (boundary),
    and index > min jumps straight to 1.
    """
    ramp = _yarn_linear_ramp(min_index=4, max_index=4, num_components=8)
    assert torch.all(ramp[:4] == 0.0)  # below min
    assert ramp[4] == 0.0  # at the boundary (post-nudge division yields 0)
    assert torch.all(ramp[5:] == 1.0)  # above min, clamped to 1


def test_apply_yarn_correction_with_zero_original_seq_len_is_noop() -> None:
    inv_freq = torch.linspace(1.0, 0.01, 32)
    out = apply_yarn_correction(
        inv_freq,
        head_dim=64,
        base=10000.0,
        original_seq_len=0,  # disables YaRN
        scaling_factor=40.0,
    )
    assert torch.equal(out, inv_freq)


def test_apply_yarn_correction_preserves_shape_dtype_device() -> None:
    inv_freq = torch.linspace(1.0, 0.01, 32, dtype=torch.float64)
    out = apply_yarn_correction(
        inv_freq,
        head_dim=64,
        base=10000.0,
        original_seq_len=4096,
        scaling_factor=40.0,
    )
    assert out.shape == inv_freq.shape
    assert out.dtype == inv_freq.dtype
    assert out.device == inv_freq.device


def test_apply_yarn_correction_high_frequency_unchanged_low_frequency_scaled() -> None:
    """For a typical config: highest-frequency components stay near identity;
    lowest-frequency components are scaled by `1/factor`."""
    head_dim = 64
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    factor = 40.0
    out = apply_yarn_correction(
        inv_freq,
        head_dim=head_dim,
        base=base,
        original_seq_len=4096,
        scaling_factor=factor,
    )
    # Highest-frequency component (smallest index) — close to original.
    ratio_high = (out[0] / inv_freq[0]).item()
    assert ratio_high > 0.99, (
        f"highest-freq component should stay ~unchanged, got ratio {ratio_high}"
    )
    # Lowest-frequency component (largest index) — close to 1/factor.
    ratio_low = (out[-1] / inv_freq[-1]).item()
    assert abs(ratio_low - 1.0 / factor) < 0.01, (
        f"lowest-freq component should be ~1/factor={1 / factor:.4f}, got ratio {ratio_low}"
    )


# ---------- Bit-parity against the V4 reference ----------


def _canonical_v2_lite_yarn() -> dict[str, Any]:
    """The actual YaRN config V2-Lite ships in its `rope_scaling` dict."""
    return dict(
        head_dim=64,  # qk_rope_head_dim for V2-Lite
        base=10000.0,
        original_seq_len=4096,
        scaling_factor=40.0,
        beta_fast=32,
        beta_slow=1,
    )


def test_yarn_inv_freq_matches_v4_reference_precompute(reference_module: Any) -> None:
    """Our `apply_yarn_correction` ≡ reference's `precompute_freqs_cis` for
    inv_freq generation on a long-context config that triggers YaRN."""
    cfg = _canonical_v2_lite_yarn()
    head_dim = cfg["head_dim"]
    seq_len = 8192  # past original_seq_len = 4096 so YaRN definitely fires

    # Our YaRN-corrected inv_freq.
    standard_inv_freq = 1.0 / (
        cfg["base"] ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    ours_inv_freq = apply_yarn_correction(standard_inv_freq, **cfg)

    # Reference's full freqs_cis tensor — extract the first row's freqs (the
    # raw inv_freq) by computing position 0 vs position 1 angles.
    freqs_cis = reference_module.precompute_freqs_cis(
        head_dim,
        seq_len,
        cfg["original_seq_len"],
        cfg["base"],
        cfg["scaling_factor"],
        cfg["beta_fast"],
        cfg["beta_slow"],
    )
    # `freqs_cis[1]` = e^(i * inv_freq * 1) = polar(1, inv_freq). The angle is inv_freq.
    reference_inv_freq = freqs_cis[1].angle().to(torch.float32)
    # Reference's freqs_cis is `torch.outer(t, freqs)` with t=arange. So row 1 is freqs itself.

    torch.testing.assert_close(ours_inv_freq, reference_inv_freq, rtol=1e-6, atol=1e-7)


def test_rotary_embedding_with_yarn_matches_reference_cos_sin_table(
    reference_module: Any,
) -> None:
    """Build a `RotaryEmbedding` with YaRN, sample its (cos, sin) at long
    positions, compare to the reference's `precompute_freqs_cis` -> cos/sin."""
    cfg = _canonical_v2_lite_yarn()
    head_dim = cfg["head_dim"]
    seq_len = 8192

    rotary = RotaryEmbedding(
        head_dim=head_dim,
        base=cfg["base"],
        yarn_original_seq_len=cfg["original_seq_len"],
        yarn_scaling_factor=cfg["scaling_factor"],
        yarn_beta_fast=cfg["beta_fast"],
        yarn_beta_slow=cfg["beta_slow"],
    )

    # Sample cos/sin at some positions past original_seq_len where YaRN fires hardest.
    positions = torch.tensor([[0, 1, 100, 4000, 5000, 7000]], dtype=torch.long)
    placeholder_input = torch.zeros(1, positions.shape[1], dtype=torch.float32)
    cos_ours, sin_ours = rotary(placeholder_input, positions)
    # Our cos/sin shape: (1, len, head_dim) — duplicated halves.

    # Reference's freqs_cis: (seqlen, head_dim/2), complex polar form.
    ref_freqs_cis = reference_module.precompute_freqs_cis(
        head_dim,
        seq_len,
        cfg["original_seq_len"],
        cfg["base"],
        cfg["scaling_factor"],
        cfg["beta_fast"],
        cfg["beta_slow"],
    )
    ref_at_positions = ref_freqs_cis[positions[0]]  # (n_positions, head_dim/2)
    ref_cos = ref_at_positions.real.to(torch.float32)  # (n_positions, head_dim/2)
    ref_sin = ref_at_positions.imag.to(torch.float32)

    # Our rotary outputs `cat([freqs, freqs])` so first/second halves are equal;
    # compare the first half against the reference's complex angles.
    half_dim = head_dim // 2
    ours_cos_first_half = cos_ours[0, :, :half_dim]
    ours_sin_first_half = sin_ours[0, :, :half_dim]

    torch.testing.assert_close(ours_cos_first_half, ref_cos, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ours_sin_first_half, ref_sin, rtol=1e-5, atol=1e-6)


# ---------- Backward compatibility ----------


def test_rotary_embedding_default_args_unchanged_by_yarn_introduction() -> None:
    """Constructing without YaRN kwargs must produce the same `inv_freq` as before."""
    head_dim = 64
    base = 10000.0
    rotary_default = RotaryEmbedding(head_dim=head_dim, base=base)
    expected_inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    torch.testing.assert_close(rotary_default.inv_freq, expected_inv_freq, rtol=1e-6, atol=1e-7)


def test_rotary_embedding_with_partial_rotary_and_yarn_combined() -> None:
    """Partial-rotary RoPE (Gemma 4 global layers) + YaRN should work together:
    YaRN modifies the rotated portion only, the nope tail stays zero."""
    head_dim = 128
    rotary = RotaryEmbedding(
        head_dim=head_dim,
        base=10000.0,
        partial_rotary_factor=0.25,  # rotate first head_dim/4 dims
        yarn_original_seq_len=4096,
        yarn_scaling_factor=8.0,
    )
    # head_dim * 0.25 = 32 dims rotated; 32//2 = 16 inv_freq components.
    # head_dim // 2 = 64 total inv_freq slots; 64 - 16 = 48 zeros at the tail.
    rotated_count = int(0.25 * head_dim) // 2  # 16
    nope_count = head_dim // 2 - rotated_count  # 48
    assert torch.all(rotary.inv_freq[rotated_count:] == 0.0), "nope tail must remain zero"
    # The rotated portion is non-zero (and YaRN-modified vs the default).
    assert torch.any(rotary.inv_freq[:rotated_count] != 0.0)
    assert nope_count > 0  # sanity check on the test config
