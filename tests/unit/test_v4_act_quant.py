"""Unit tests for the V4 activation quantize-dequantize round trips (`v4/act_quant.py`).

Expected values are worked out by hand from the kernels' recipe (block
absmax, power-of-two or FP32 scale, round half to even onto the e4m3 /
e2m1 grid) and checked against the formats' known value grids. The
reference-parity side lives in `test_v4_act_quant_parity.py`.
"""

from __future__ import annotations

import math

import pytest
import torch

import mini_infer.models.blocks.v4.act_quant as act_quant_module
from mini_infer.cache.state_cache import StateCache
from mini_infer.models.blocks import CSAAttention, HCAAttention
from mini_infer.models.blocks.activations import swiglu
from mini_infer.models.blocks.fp4_expert import FP4Expert
from mini_infer.models.blocks.mixtral_moe import MixtralExpert
from mini_infer.models.blocks.swa import SWAAttention
from mini_infer.models.blocks.v4.act_quant import (
    SimulatedActQuant,
    fp4_e2m1_roundtrip,
    fp8_e4m3_roundtrip,
    hadamard_rotate,
)
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)

_E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _block(values: list[float], *, width: int, pin: float) -> torch.Tensor:
    """One `width`-wide FP32 block: `values`, then `pin` (sets the block absmax), then zeros."""
    row = torch.zeros(width)
    row[: len(values)] = torch.tensor(values)
    row[len(values)] = pin
    return row


# ---------- FP4 e2m1 ----------


def test_fp4_grid_values_survive_with_unit_scale() -> None:
    # absmax 6 -> scale 2^ceil(log2(6 * fp32(1/6))) = 2^0 = 1: grid values map to themselves.
    grid = _E2M1_GRID + [-v for v in _E2M1_GRID[1:]]
    x = _block(grid, width=32, pin=6.0)
    torch.testing.assert_close(fp4_e2m1_roundtrip(x), x, rtol=0, atol=0)


def test_fp4_rounds_to_nearest_with_ties_to_even() -> None:
    # Scale 1 (pinned by 6.0). Ties go to the even code (mantissa bit 0):
    # 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4.
    inputs = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.3, 2.6, 5.1, -0.75, -5.0]
    expected = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.5, 3.0, 6.0, -1.0, -4.0]
    out = fp4_e2m1_roundtrip(_block(inputs, width=32, pin=6.0))
    assert out[: len(inputs)].tolist() == expected


def test_fp4_scale_is_absmax_over_six_rounded_up_to_a_power_of_two() -> None:
    # absmax 1 -> 1 * fp32(1/6) = 0.1667 -> scale 0.25. Codes x / 0.25:
    # 0.3 -> 1.2 -> 1 (0.25), 0.1 -> 0.4 -> 0.5 (0.125), 0.8 -> 3.2 -> 3 (0.75).
    out = fp4_e2m1_roundtrip(_block([0.3, 0.1, 0.8], width=32, pin=1.0))
    assert out[:4].tolist() == [0.25, 0.125, 0.75, 1.0]
    # absmax exactly 6 * 2^-1 = 3 -> scale 0.5 (no round-up at a power of two):
    # 1.3 -> 2.6 -> 3 (1.5), 3 -> 6 (3).
    out = fp4_e2m1_roundtrip(_block([1.3], width=32, pin=3.0))
    assert out[:2].tolist() == [1.5, 3.0]


def test_fp4_blocks_have_independent_scales() -> None:
    # Block 0: absmax 6 (scale 1), 2.6 -> 3. Block 1: absmax 0.6 -> 0.1 -> scale 0.125,
    # 0.26 -> 2.08 -> 2 (0.25).
    x = torch.cat([_block([2.6], width=32, pin=6.0), _block([0.26], width=32, pin=0.6)])
    out = fp4_e2m1_roundtrip(x)
    assert out[0].item() == 3.0
    assert out[32].item() == 0.25


def test_fp4_all_zero_block_stays_zero() -> None:
    out = fp4_e2m1_roundtrip(torch.zeros(2, 32))
    assert torch.equal(out, torch.zeros(2, 32))


# ---------- FP8 e4m3 ----------


def _all_finite_e4m3_values() -> torch.Tensor:
    codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return codes[torch.isfinite(codes)].unique()  # 253 values: +-[0, 448], one zero


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
def test_fp8_matches_the_e4m3_grid_with_unit_scale(scale_fmt: str | None) -> None:
    """Grid values and their midpoints round exactly like the e4m3fn cast.

    Every block is pinned to absmax 448, which gives scale 1 under both scale
    formats (448 * fp32(1/448) rounds to exactly 1.0), so the round trip is
    the bare e4m3 rounding.
    """
    grid = _all_finite_e4m3_values()
    midpoints = (grid[1:] + grid[:-1]) / 2
    probes = torch.cat([grid, midpoints])
    n_blocks = math.ceil(probes.numel() / 63)
    probes = torch.cat([probes, torch.zeros(n_blocks * 63 - probes.numel())])
    blocks = torch.cat([probes.view(n_blocks, 63), torch.full((n_blocks, 1), 448.0)], dim=1)

    out = fp8_e4m3_roundtrip(blocks, block_size=64, scale_fmt=scale_fmt)  # type: ignore[arg-type]

    expected = blocks.to(torch.float8_e4m3fn).float()
    assert torch.equal(out, expected)
    assert torch.equal(out[:, :63].flatten()[: grid.numel()], grid)  # grid values are fixed points


def test_fp8_ue8m0_scale_hand_computed() -> None:
    # absmax 1 -> 1 * fp32(1/448) = 2^-8.8 -> scale 2^-8. Codes x * 256:
    #   0.3   -> 76.8,  step 8 in [64, 128)    -> 80     -> 0.3125
    #   0.001 -> 0.256, step 1/32 in [1/4, 1/2) -> 0.25   -> 2^-10
    x = _block([0.3, 0.001], width=64, pin=1.0)
    out = fp8_e4m3_roundtrip(x, block_size=64, scale_fmt="ue8m0")
    assert out[:3].tolist() == [0.3125, 2.0**-10, 1.0]


def test_fp8_fp32_scale_hand_computed() -> None:
    # absmax 1 -> scale s = fp32(1/448) (no rounding). 0.3 / s = 134.4, step 16 in
    # [128, 256) -> 128 -> 128 * s. 1 / s = 447.99998 -> 448 -> 448 * s = 1.0 in FP32.
    s = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    out = fp8_e4m3_roundtrip(_block([0.3], width=64, pin=1.0), block_size=64, scale_fmt=None)
    assert out[0].item() == (128 * s).item()
    assert out[1].item() == 1.0


def test_fp8_all_zero_block_stays_zero() -> None:
    out = fp8_e4m3_roundtrip(torch.zeros(3, 128), block_size=64, scale_fmt="ue8m0")
    assert torch.equal(out, torch.zeros(3, 128))


def test_roundtrips_are_exact_in_bf16() -> None:
    """Power-of-two scales: BF16 input gives the FP32 path's values, exactly representable."""
    torch.manual_seed(0)
    x = (torch.randn(4, 5, 128) * torch.logspace(-2, 2, 128)).bfloat16()
    for quantize in (
        lambda t: fp8_e4m3_roundtrip(t, block_size=64, scale_fmt="ue8m0"),
        fp4_e2m1_roundtrip,
    ):
        out_bf16 = quantize(x)
        out_fp32 = quantize(x.float())
        assert out_bf16.dtype == torch.bfloat16
        assert torch.equal(out_bf16.float(), out_fp32)


def test_roundtrips_reject_partial_blocks() -> None:
    with pytest.raises(ValueError, match="multiple of block_size"):
        fp8_e4m3_roundtrip(torch.zeros(24), block_size=64, scale_fmt="ue8m0")
    with pytest.raises(ValueError, match="multiple of block_size"):
        fp4_e2m1_roundtrip(torch.zeros(16))
    with pytest.raises(ValueError, match="scale_fmt"):
        fp8_e4m3_roundtrip(torch.zeros(64), block_size=64, scale_fmt="e5m2")  # type: ignore[arg-type]


# ---------- Hadamard ----------


def test_hadamard_hand_computed_small_dims() -> None:
    # d=2: [3, 1] @ [[1, 1], [1, -1]] / sqrt(2) = [4, 2] / sqrt(2).
    out = hadamard_rotate(torch.tensor([3.0, 1.0]))
    torch.testing.assert_close(out, torch.tensor([4.0, 2.0]) / math.sqrt(2))
    # d=4: Sylvester H4 rows [1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1], / 2.
    out = hadamard_rotate(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert out.tolist() == [5.0, -1.0, -2.0, 0.0]


def test_hadamard_matches_sylvester_matrix_and_is_orthogonal() -> None:
    torch.manual_seed(0)
    d = 128
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < d:
        matrix = torch.kron(torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64), matrix)
    q, k = torch.randn(2, 3, d)
    torch.testing.assert_close(
        hadamard_rotate(q).double(), q.double() @ matrix / math.sqrt(d), rtol=0, atol=1e-5
    )
    # Orthogonal and symmetric: rotating twice is the identity, dot products survive.
    torch.testing.assert_close(hadamard_rotate(hadamard_rotate(q)), q, rtol=0, atol=1e-5)
    torch.testing.assert_close(
        (hadamard_rotate(q) * hadamard_rotate(k)).sum(-1), (q * k).sum(-1), rtol=1e-5, atol=1e-5
    )


def test_hadamard_keeps_dtype_and_rejects_non_power_of_two() -> None:
    assert hadamard_rotate(torch.randn(2, 32).bfloat16()).dtype == torch.bfloat16
    with pytest.raises(ValueError, match="power-of-two"):
        hadamard_rotate(torch.zeros(24))


# ---------- SimulatedActQuant ----------


def test_kv_quantizes_only_the_non_rope_dims() -> None:
    torch.manual_seed(0)
    kv = torch.randn(2, 5, 72)
    out = SimulatedActQuant().kv(kv, rope_head_dim=8)
    assert torch.equal(out[..., 64:], kv[..., 64:])
    expected_nope = fp8_e4m3_roundtrip(kv[..., :64], block_size=64, scale_fmt="ue8m0")
    assert torch.equal(out[..., :64], expected_nope)
    assert not torch.equal(out[..., :64], kv[..., :64])


def test_indexer_is_hadamard_then_fp4() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 64)
    out = SimulatedActQuant().indexer(x)
    assert torch.equal(out, fp4_e2m1_roundtrip(hadamard_rotate(x)))


def test_dimension_checks_fail_early() -> None:
    SimulatedActQuant.check_kv_dims(512, 64)  # V4 shape: 448 = 7 blocks of 64
    SimulatedActQuant.check_indexer_dim(128)
    with pytest.raises(ValueError, match="multiple of 64"):
        SimulatedActQuant.check_kv_dims(32, 8)
    for bad in (16, 96):
        with pytest.raises(ValueError, match="power of two"):
            SimulatedActQuant.check_indexer_dim(bad)
    with pytest.raises(ValueError, match="fp8_scale_fmt"):
        SimulatedActQuant(fp8_scale_fmt="e5m2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="multiple of 64"):
        HCAAttention(
            hidden_size=64,
            num_heads=4,
            q_lora_rank=32,
            kv_head_dim=32,
            rope_head_dim=8,
            num_groups=2,
            o_lora_rank=32,
            window_size=8,
            compression_ratio=8,
            act_quant=SimulatedActQuant(),
        )


# ---------- FP8 GEMM inputs ----------

_GEMM_ACT_QUANT = SimulatedActQuant(fp8_gemm_inputs=True)


def test_gemm_input_is_a_block_128_fp8_round_trip_only_when_enabled() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 256) * torch.logspace(-2, 2, 256)
    expected = fp8_e4m3_roundtrip(x, block_size=128, scale_fmt="ue8m0")
    assert torch.equal(_GEMM_ACT_QUANT.gemm_input(x), expected)
    assert SimulatedActQuant().gemm_input(x) is x


def test_quantize_gemm_inputs_hooks_the_linear_input() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    hooked = torch.nn.Linear(256, 16, bias=False)
    _GEMM_ACT_QUANT.quantize_gemm_inputs(hooked)
    expected = torch.nn.functional.linear(_GEMM_ACT_QUANT.gemm_input(x), hooked.weight)
    assert torch.equal(hooked(x), expected)

    untouched = torch.nn.Linear(256, 16, bias=False)
    SimulatedActQuant().quantize_gemm_inputs(untouched)  # QAT only: no hook
    assert not untouched._forward_pre_hooks
    with pytest.raises(ValueError, match="multiple of 128"):
        _GEMM_ACT_QUANT.quantize_gemm_inputs(torch.nn.Linear(64, 16))


def test_routing_weights_scale_the_activation_before_w2() -> None:
    """Same math as scaling the output; with quantized GEMM inputs, `w2` sees the scaled input."""
    torch.manual_seed(0)
    expert = MixtralExpert(128, 128)
    x = torch.randn(5, 128)
    weights = torch.rand(5, 1)
    torch.testing.assert_close(expert(x, routing_weights=weights), expert(x) * weights)

    _GEMM_ACT_QUANT.quantize_gemm_inputs(expert.w1, expert.w2, expert.w3)
    hidden = swiglu(expert.w1(x), expert.w3(x)) * weights
    expected = torch.nn.functional.linear(_GEMM_ACT_QUANT.gemm_input(hidden), expert.w2.weight)
    assert torch.equal(expert(x, routing_weights=weights), expected)


def test_fp4_expert_quantizes_every_gemm_input() -> None:
    torch.manual_seed(0)
    plain = FP4Expert(128, 256)
    quantized = FP4Expert(128, 256, act_quant=_GEMM_ACT_QUANT)
    generator = torch.Generator().manual_seed(1)
    for name in ("w1", "w2", "w3"):
        packed = getattr(plain, f"{name}_packed")
        packed.copy_(torch.randint(-128, 128, packed.shape, generator=generator, dtype=torch.int8))
        scale = getattr(plain, f"{name}_scale")
        scale.copy_(2.0 ** torch.randint(-6, -2, scale.shape, generator=generator).float())
    quantized.load_state_dict(plain.state_dict())
    x = torch.randn(5, 128)
    weights = torch.rand(5, 1)

    w1, w2, w3 = (plain._dequant(name, x.dtype) for name in ("w1", "w2", "w3"))
    x_q = _GEMM_ACT_QUANT.gemm_input(x)
    hidden = swiglu(x_q @ w1.T, x_q @ w3.T) * weights
    expected = _GEMM_ACT_QUANT.gemm_input(hidden) @ w2.T
    torch.testing.assert_close(
        quantized(x, routing_weights=weights), expected, rtol=1e-6, atol=1e-6
    )
    assert not torch.equal(quantized(x), plain(x))
    with pytest.raises(ValueError, match="multiple of 128"):
        FP4Expert(128, 96, act_quant=_GEMM_ACT_QUANT)


# ---------- Model config wiring ----------


def _config(
    *,
    simulate_act_quant: bool,
    scale_fmt: str | None = "ue8m0",
    gemm_dims: bool = False,
    fp8_gemm_inputs: bool = False,
) -> DeepseekV4Config:
    """SWA, CSA, HCA, CSA; dims wide enough for the quantizers' block sizes.

    `gemm_dims`: a MoE backbone whose GEMM input widths are multiples of 128.
    """
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=128 if gemm_dims else 64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=128 if gemm_dims else 32,
        kv_head_dim=72,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=64 if gemm_dims else 32,
        window_size=8,
        compress_ratios=(0, 4, 8, 4),
        index_num_heads=2,
        index_head_dim=32,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_moe_ffn=gemm_dims,
        moe_intermediate_size=128 if gemm_dims else 0,
        num_routed_experts=4 if gemm_dims else 0,
        num_activated_experts=2 if gemm_dims else 0,
        n_shared_experts=1 if gemm_dims else 0,
        simulate_act_quant=simulate_act_quant,
        act_quant_scale_fmt=scale_fmt,
        simulate_fp8_gemm_inputs=fp8_gemm_inputs,
    )


def test_config_threads_act_quant_into_every_attention_block() -> None:
    model = DeepseekV4ForCausalLM(_config(simulate_act_quant=True, scale_fmt=None))
    expected = SimulatedActQuant(fp8_scale_fmt=None)
    for layer in model.model.layers:
        attention = layer.self_attn
        assert attention.act_quant == expected
        if isinstance(attention, (CSAAttention, HCAAttention)):
            assert attention.compressor.act_quant == expected
            assert not attention.compressor.indexer_keys
        if isinstance(attention, CSAAttention):
            assert attention.indexer.act_quant == expected
            assert attention.indexer.compressor.indexer_keys

    default = DeepseekV4ForCausalLM(_config(simulate_act_quant=False))
    assert all(layer.self_attn.act_quant is None for layer in default.model.layers)
    assert isinstance(default.model.layers[0].self_attn, SWAAttention)


def test_config_validation_and_from_hf() -> None:
    with pytest.raises(ValueError, match="act_quant_scale_fmt"):
        _config(simulate_act_quant=True, scale_fmt="e5m2")
    hf = {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "compress_ratios": [0, 4],
        "head_dim": 72,
        "qk_rope_head_dim": 8,
    }
    assert not DeepseekV4Config.from_hf(hf).simulate_act_quant
    opted_in = DeepseekV4Config.from_hf({**hf, "simulate_act_quant": True})
    assert opted_in.simulate_act_quant
    assert opted_in.act_quant() == SimulatedActQuant(fp8_scale_fmt="ue8m0")
    with_gemm = DeepseekV4Config.from_hf(
        {**hf, "simulate_act_quant": True, "simulate_fp8_gemm_inputs": True}
    )
    assert with_gemm.act_quant() == _GEMM_ACT_QUANT
    with pytest.raises(ValueError, match="requires simulate_act_quant"):
        _config(simulate_act_quant=False, gemm_dims=True, fp8_gemm_inputs=True)


def test_config_hooks_exactly_the_reference_fp8_weight_linears() -> None:
    """Attention wq_a/wq_b/wkv/wo_b, indexer wq_b, every expert's w1/w2/w3: nothing else."""
    model = DeepseekV4ForCausalLM(
        _config(simulate_act_quant=True, gemm_dims=True, fp8_gemm_inputs=True)
    )
    hooked = {name for name, module in model.named_modules() if module._forward_pre_hooks}
    expected: set[str] = set()
    for layer_idx, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_idx}"
        attention = ("q_a_proj", "q_b_proj", "swa_kv_proj", "grouped_output.wo_b")
        expected.update(f"{prefix}.self_attn.{name}" for name in attention)
        if isinstance(layer.self_attn, CSAAttention):
            expected.add(f"{prefix}.self_attn.indexer.wq_b")
        experts = [f"mlp.experts.{index}" for index in range(4)] + ["mlp.shared_experts"]
        expected.update(f"{prefix}.{expert}.{w}" for expert in experts for w in ("w1", "w2", "w3"))
    assert hooked == expected
    assert all(layer.mlp.routing_weights_before_down_proj for layer in model.model.layers)

    qat_only = DeepseekV4ForCausalLM(_config(simulate_act_quant=True, gemm_dims=True))
    assert not any(module._forward_pre_hooks for module in qat_only.modules())
    assert not any(layer.mlp.routing_weights_before_down_proj for layer in qat_only.model.layers)


def _run_every_path(model: DeepseekV4ForCausalLM, cfg: DeepseekV4Config) -> list[torch.Tensor]:
    """Forward, cache-aware prefill, scalar decode and ragged decode; return all logits."""
    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 14))
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        outputs.append(model(ids[:, :8]))
        cache = StateCache(build_state_cache_layer_specs(cfg, max_seq_len=32), batch_size=2)
        outputs.append(model.forward_prefill_with_cache(ids[:, :9], state_cache=cache))
        cache.advance_start_pos(9)
        ragged_cache = StateCache(build_state_cache_layer_specs(cfg, max_seq_len=32), batch_size=2)
        model.forward_prefill_with_cache(ids[:, :9], state_cache=ragged_cache)
        positions = torch.tensor([9, 9])
        for step in range(9, 14):
            token = ids[:, step : step + 1]
            outputs.append(
                model.forward_decode_with_cache(token, start_pos=cache.start_pos, state_cache=cache)
            )
            cache.advance_start_pos(1)
            outputs.append(
                model.forward_decode_with_cache_ragged(
                    token, positions=positions, state_cache=ragged_cache
                )
            )
            positions = positions + 1
    return outputs


def test_flag_off_never_runs_a_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off by default and inert: no quantizer runs, so outputs stay bitwise unchanged."""

    def _fail(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("a QAT round trip ran with simulate_act_quant off")

    for name in ("fp8_e4m3_roundtrip", "fp4_e2m1_roundtrip", "hadamard_rotate"):
        monkeypatch.setattr(act_quant_module, name, _fail)
    cfg = _config(simulate_act_quant=False)
    torch.manual_seed(0)
    _run_every_path(DeepseekV4ForCausalLM(cfg).eval(), cfg)


def test_flag_on_changes_the_numerics_of_every_path() -> None:
    """Same weights, flag on vs off: every path's logits move, and stay finite."""
    cfg_off = _config(simulate_act_quant=False)
    cfg_on = _config(simulate_act_quant=True)
    torch.manual_seed(0)
    model_off = DeepseekV4ForCausalLM(cfg_off).eval()
    model_on = DeepseekV4ForCausalLM(cfg_on).eval()
    model_on.load_state_dict(model_off.state_dict())
    for off, on in zip(
        _run_every_path(model_off, cfg_off), _run_every_path(model_on, cfg_on), strict=True
    ):
        assert torch.isfinite(on).all()
        assert not torch.equal(off, on)


def test_fp8_gemm_inputs_change_the_numerics_of_every_path() -> None:
    """MoE backbone, QAT round trips on, GEMM-input round trips off vs on."""
    cfg_qat = _config(simulate_act_quant=True, gemm_dims=True)
    cfg_gemm = _config(simulate_act_quant=True, gemm_dims=True, fp8_gemm_inputs=True)
    torch.manual_seed(0)
    model_qat = DeepseekV4ForCausalLM(cfg_qat).eval()
    model_gemm = DeepseekV4ForCausalLM(cfg_gemm).eval()
    model_gemm.load_state_dict(model_qat.state_dict())
    for qat, gemm in zip(
        _run_every_path(model_qat, cfg_qat), _run_every_path(model_gemm, cfg_gemm), strict=True
    ):
        assert torch.isfinite(gemm).all()
        assert not torch.equal(qat, gemm)
