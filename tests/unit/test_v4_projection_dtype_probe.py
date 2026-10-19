"""Compressor projection output-dtype policy + the drift probe built on it.

Parity contracts (CPU):
  - fp32 inputs: the two policies are the same arithmetic, so every compressor
    path must be bitwise identical (`torch.equal`, max abs diff == 0).
  - bf16 inputs: the fp32-output policy equals the fp32 GEMM over the bf16
    operands exactly (that IS its definition off-CUDA), and the native policy
    equals that result rounded to bf16, within one bf16 ulp (CPU bf16 GEMMs
    accumulate in fp32 but may sum in a different order).
  - The probe is deterministic: for each control pair (the same policy twice,
    fp32-out against the upcast-GEMM fp32 path, which is the same arithmetic
    off CUDA) every drift it reports is exactly zero, including the
    next-token NLL delta.
"""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from mini_infer.bench.v4_projection_dtype_probe import (
    FP32_OUT,
    FP32_OUT_UPCAST_GEMM,
    NATIVE_OUT,
    IsolatedTaps,
    PairedPrefillReport,
    ProjectionPolicy,
    _summarize_logits,
    applied_policy,
    attention_blocks,
    block_means,
    block_stderr,
    make_synthetic_hybrid_config,
    run_paired_prefill,
    subset_positions,
    tensor_drift,
    topk_set_agreement,
)
from mini_infer.models.blocks.v4 import compressor as compressor_module
from mini_infer.models.blocks.v4.compressor import (
    TokenLevelCompressor,
    configure_projection_out_dtype,
)
from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

_HIDDEN = 64
_KV_HEAD_DIM = 32
_ROPE_DIM = 8
_RATIO = 4
_BF16_ULP_REL = 2.0**-7  # one bf16 unit in the last place, relative to the value


def _compressor_pair(
    *, overlap: bool, dtype: torch.dtype
) -> tuple[TokenLevelCompressor, TokenLevelCompressor]:
    """Two compressors with identical weights: native policy and fp32-output policy."""
    torch.manual_seed(0)
    native = TokenLevelCompressor(
        hidden_size=_HIDDEN,
        kv_head_dim=_KV_HEAD_DIM,
        rope_head_dim=_ROPE_DIM,
        compression_ratio=_RATIO,
        overlap_mode=overlap,
    )
    with torch.no_grad():
        native.position_bias.normal_()
    native = native.to(dtype).eval()
    fp32_out = copy.deepcopy(native)
    fp32_out.projection_out_dtype = torch.float32
    return native, fp32_out


def _rope_tables(batch_size: int, n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.linspace(0.0, 3.0, n * _ROPE_DIM).view(1, n, _ROPE_DIM)  # (1, n, rope)
    angles = angles.expand(batch_size, -1, -1)  # (B, n, rope)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _state_pair(batch_size: int, overlap: bool) -> tuple[torch.Tensor, torch.Tensor]:
    coff = 2 if overlap else 1
    kv_state = torch.zeros(batch_size, coff * _RATIO, coff * _KV_HEAD_DIM)
    score_state = torch.full((batch_size, coff * _RATIO, coff * _KV_HEAD_DIM), float("-inf"))
    return kv_state, score_state


# ---------- policy validation ----------


def test_projection_policy_rejects_unsupported_dtypes() -> None:
    with pytest.raises(ValueError, match="projection_out_dtype"):
        TokenLevelCompressor(
            hidden_size=_HIDDEN,
            kv_head_dim=_KV_HEAD_DIM,
            rope_head_dim=_ROPE_DIM,
            compression_ratio=_RATIO,
            projection_out_dtype=torch.float16,
        )
    compressor = TokenLevelCompressor(
        hidden_size=_HIDDEN, kv_head_dim=_KV_HEAD_DIM, rope_head_dim=_ROPE_DIM, compression_ratio=4
    )
    assert compressor.projection_out_dtype is None
    with pytest.raises(ValueError, match="projection_out_dtype"):
        compressor.projection_out_dtype = torch.bfloat16


# ---------- fp32: bitwise identical ----------


@pytest.mark.parametrize("overlap", [False, True])
def test_fp32_inputs_make_both_policies_bitwise_identical(overlap: bool) -> None:
    native, fp32_out = _compressor_pair(overlap=overlap, dtype=torch.float32)
    batch_size, seqlen = 2, 16
    torch.manual_seed(1)
    hidden = torch.randn(batch_size, seqlen, _HIDDEN)
    cos, sin = _rope_tables(batch_size, seqlen // _RATIO, torch.float32)

    with torch.inference_mode():
        assert torch.equal(native(hidden, (cos, sin)), fp32_out(hidden, (cos, sin)))

        kv_a, score_a = _state_pair(batch_size, overlap)
        kv_b, score_b = _state_pair(batch_size, overlap)
        unaligned = hidden[:, : seqlen - 1]  # trailing remainder exercises the stash paths
        cos_u, sin_u = _rope_tables(batch_size, (seqlen - 1) // _RATIO, torch.float32)
        out_a = native.forward_prefill_with_cache(
            unaligned,
            compressed_position_embeddings=(cos_u, sin_u),
            cmp_kv_state=kv_a,
            cmp_score_state=score_a,
        )
        out_b = fp32_out.forward_prefill_with_cache(
            unaligned,
            compressed_position_embeddings=(cos_u, sin_u),
            cmp_kv_state=kv_b,
            cmp_score_state=score_b,
        )
        assert torch.equal(out_a, out_b)
        assert torch.equal(kv_a, kv_b)
        assert torch.equal(score_a, score_b)

        block_cos, block_sin = _rope_tables(batch_size, 1, torch.float32)
        for start_pos in range(seqlen - 1, seqlen - 1 + _RATIO):
            token = torch.randn(batch_size, 1, _HIDDEN)
            step_a = native.forward_decode_step(
                token,
                start_pos=start_pos,
                cmp_kv_state=kv_a,
                cmp_score_state=score_a,
                block_position_embeddings=(block_cos, block_sin),
            )
            step_b = fp32_out.forward_decode_step(
                token,
                start_pos=start_pos,
                cmp_kv_state=kv_b,
                cmp_score_state=score_b,
                block_position_embeddings=(block_cos, block_sin),
            )
            assert (step_a is None) == (step_b is None)
            if step_a is not None and step_b is not None:
                assert torch.equal(step_a, step_b)
        assert torch.equal(kv_a, kv_b)

        positions = torch.tensor([3, 6])  # one row flushes this step, the other does not
        token = torch.randn(batch_size, 1, _HIDDEN)
        ragged_a, mask_a = native.forward_decode_step_ragged(
            token,
            positions=positions,
            cmp_kv_state=kv_a,
            cmp_score_state=score_a,
            block_position_embeddings=(block_cos, block_sin),
        )
        ragged_b, mask_b = fp32_out.forward_decode_step_ragged(
            token,
            positions=positions,
            cmp_kv_state=kv_b,
            cmp_score_state=score_b,
            block_position_embeddings=(block_cos, block_sin),
        )
        assert torch.equal(mask_a, mask_b)
        assert torch.equal(ragged_a, ragged_b)


# ---------- bf16: fp32-out is the exact fp32 GEMM; native is it rounded ----------


@pytest.mark.parametrize("overlap", [False, True])
def test_bf16_fp32_out_policy_is_the_fp32_gemm_and_native_is_its_rounding(overlap: bool) -> None:
    native, fp32_out = _compressor_pair(overlap=overlap, dtype=torch.bfloat16)
    torch.manual_seed(2)
    hidden = torch.randn(2, 16, _HIDDEN).to(torch.bfloat16)  # (B, T, hidden)

    with torch.inference_mode():
        kv_fp32, score_fp32 = fp32_out._project(hidden)  # each (B, T, coff*c) fp32
        kv_native, score_native = native._project(hidden)  # each (B, T, coff*c) fp32

    assert kv_fp32.dtype == torch.float32 and kv_native.dtype == torch.float32
    kv_ref = torch.nn.functional.linear(hidden.float(), native.kv_proj.weight.float())
    score_ref = torch.nn.functional.linear(hidden.float(), native.weight_proj.weight.float())
    assert torch.equal(kv_fp32, kv_ref)
    assert torch.equal(score_fp32, score_ref)
    # Native output == fp32 output rounded to bf16, up to one bf16 ulp (summation order).
    torch.testing.assert_close(
        kv_native, kv_fp32.to(torch.bfloat16).float(), rtol=_BF16_ULP_REL, atol=1e-6
    )
    torch.testing.assert_close(
        score_native, score_fp32.to(torch.bfloat16).float(), rtol=_BF16_ULP_REL, atol=1e-6
    )


@pytest.mark.requires_cuda
def test_cuda_fp32_out_policy_matches_the_fp32_gemm() -> None:
    """On CUDA the fp32-output policy runs `torch.mm(..., out_dtype=float32)` when available.

    cuBLAS may order the K reduction differently from the CPU reference, so the
    contract is fp32 round-off (K=64 here), not bitwise equality. The native
    policy's output must still be exactly representable in bf16.
    """
    native, fp32_out = _compressor_pair(overlap=True, dtype=torch.bfloat16)
    native, fp32_out = native.cuda(), fp32_out.cuda()
    torch.manual_seed(4)
    hidden = torch.randn(2, 16, _HIDDEN, device="cuda").to(torch.bfloat16)  # (B, T, hidden)

    with torch.inference_mode():
        kv_fp32, score_fp32 = fp32_out._project(hidden)  # each (B, T, 2c) fp32
        kv_native, _ = native._project(hidden)  # (B, T, 2c) fp32, bf16-valued

    kv_ref = torch.nn.functional.linear(hidden.float(), native.kv_proj.weight.float())
    score_ref = torch.nn.functional.linear(hidden.float(), native.weight_proj.weight.float())
    torch.testing.assert_close(kv_fp32, kv_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(score_fp32, score_ref, rtol=1e-5, atol=1e-5)
    assert torch.equal(kv_native, kv_native.to(torch.bfloat16).float())
    assert not torch.equal(kv_fp32, kv_fp32.to(torch.bfloat16).float())


# ---------- model-wide configuration ----------


def _synthetic_model(dtype: torch.dtype) -> DeepseekV4ForCausalLM:
    cfg = make_synthetic_hybrid_config()
    torch.manual_seed(0)
    return DeepseekV4ForCausalLM(cfg).to(dtype).eval()


def test_configure_counts_every_compressor_including_indexer_ones() -> None:
    model = _synthetic_model(torch.float32)
    blocks = attention_blocks(model)
    n_csa = sum(1 for _, attn in blocks if hasattr(attn, "indexer"))
    n_hca = len(blocks) - n_csa
    assert n_csa == 3 and n_hca == 3
    assert configure_projection_out_dtype(model, torch.float32) == 2 * n_csa + n_hca
    assert all(
        c.projection_out_dtype is torch.float32
        for c in model.modules()
        if isinstance(c, TokenLevelCompressor)
    )
    assert configure_projection_out_dtype(model, None) == 2 * n_csa + n_hca


# ---------- metrics ----------


def test_tensor_drift_of_identical_tensors_is_zero() -> None:
    x = torch.randn(3, 5, 7)
    drift = tensor_drift(x, x)
    assert drift.max_abs == 0.0 and drift.rms_rel == 0.0 and drift.numel == 105
    assert drift.cosine == pytest.approx(1.0, abs=1e-6)
    empty = tensor_drift(torch.zeros(1, 0, 4), torch.zeros(1, 0, 4))
    assert empty.numel == 0 and empty.cosine == 1.0


def test_topk_set_agreement_hand_computed() -> None:
    reference = torch.tensor([[[5, 1, 9, 3], [2, 4, -1, -1], [7, 8, 0, 6]]])  # (1, 3, 4)
    candidate = torch.tensor([[[3, 9, 1, 5], [4, 2, -1, -1], [7, 8, 0, 11]]])  # (1, 3, 4)
    agreement = topk_set_agreement(reference, candidate)
    # Query 0: same set in a different order (1.0); query 1: same set with padding
    # (1.0); query 2: one of four slots differs (0.75).
    assert agreement.n_queries == 3
    assert agreement.mean_set_agreement == pytest.approx((1.0 + 1.0 + 0.75) / 3)
    assert agreement.min_set_agreement == pytest.approx(0.75)
    assert agreement.fraction_identical == pytest.approx(2 / 3)
    assert agreement.n_flipped_slots == 1
    assert topk_set_agreement(reference, reference).n_flipped_slots == 0
    empty = torch.empty(1, 4, 0, dtype=torch.long)  # (1, 4, 0): queries that select nothing
    assert topk_set_agreement(empty, empty).mean_set_agreement == 1.0


def test_subset_positions_covers_stride_and_tail() -> None:
    positions = subset_positions(100, stride=32, tail=3)
    assert positions.tolist() == [0, 32, 64, 96, 97, 98, 99]


def test_summarize_logits_scores_the_real_next_token() -> None:
    torch.manual_seed(5)
    logits = torch.randn(1, 7, 11)  # (1, T, V)
    targets = torch.tensor([3, 0, 10, 5, 5, 1])  # (T-1,): tokens 1 .. T-1
    summary = _summarize_logits(logits, torch.tensor([0, 6]), targets, chunk=3)
    expected_nll = -logits[0, :6].log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
    torch.testing.assert_close(summary.target_nll, expected_nll, rtol=0.0, atol=1e-6)
    assert torch.equal(summary.target_top1, logits[0, :6].argmax(dim=-1) == targets)
    assert torch.equal(summary.argmax, logits[0].argmax(dim=-1))
    assert summary.subset_logits.shape == (2, 11)
    with pytest.raises(ValueError, match="targets"):
        _summarize_logits(logits, torch.tensor([0]), targets[:-1])


def test_block_stderr() -> None:
    # N=128, block 8 -> 16 blocks: 8 with mean 1.0, 8 with mean 3.0. Deviations from
    # the grand mean are all 1.0, so the unbiased variance is 16/15 and the standard
    # error sqrt(16/15) / sqrt(16) = sqrt(1/15).
    values = torch.tensor([1.0] * 64 + [3.0] * 64)
    assert block_stderr(values, block_size=8) == pytest.approx(math.sqrt(1 / 15))
    # For short inputs the block shrinks to keep at least eight blocks: N=16 -> block 2.
    short = torch.tensor([1.0] * 8 + [3.0] * 8)
    assert block_stderr(short, block_size=8) == pytest.approx(math.sqrt(8 / 7) / math.sqrt(8))
    assert block_stderr(torch.zeros(64)) == 0.0
    assert block_stderr(torch.zeros(1)) is None


def test_block_means_keeps_a_short_last_block() -> None:
    values = torch.tensor([1.0, 3.0, 5.0, 7.0, 10.0])  # (5,)
    assert block_means(values, block_size=2) == (2.0, 6.0, 10.0)
    assert block_means(torch.zeros(0), block_size=4) == ()
    with pytest.raises(ValueError, match="block_size"):
        block_means(values, block_size=0)


def test_projection_policy_names_and_validation() -> None:
    assert FP32_OUT.name == "float32"
    assert NATIVE_OUT.name == "native"
    assert FP32_OUT_UPCAST_GEMM.name == "float32+upcast_gemm"
    with pytest.raises(ValueError, match="force_upcast_gemm"):
        ProjectionPolicy(None, force_upcast_gemm=True)


def test_applied_policy_patches_and_restores_the_capability_check() -> None:
    model = _synthetic_model(torch.float32)
    original = compressor_module.cuda_mm_fp32_out_supported
    with applied_policy(model, FP32_OUT_UPCAST_GEMM):
        assert compressor_module.cuda_mm_fp32_out_supported is not original
        assert compressor_module.cuda_mm_fp32_out_supported(0) is False
        assert all(
            c.projection_out_dtype is torch.float32
            for c in model.modules()
            if isinstance(c, TokenLevelCompressor)
        )
    assert compressor_module.cuda_mm_fp32_out_supported is original
    with applied_policy(model, NATIVE_OUT):
        assert compressor_module.cuda_mm_fp32_out_supported is original


# ---------- probe end to end (synthetic bf16 model on CPU) ----------


def _probe_inputs(model: DeepseekV4ForCausalLM, seqlen: int = 128) -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randint(0, model.cfg.vocab_size, (1, seqlen), dtype=torch.long)


@pytest.mark.parametrize(
    ("baseline", "alternate", "isolated"),
    [
        (NATIVE_OUT, NATIVE_OUT, True),
        (FP32_OUT, FP32_OUT, False),
        # Off CUDA the upcast-GEMM control runs the very same fp32 path.
        (FP32_OUT, FP32_OUT_UPCAST_GEMM, False),
    ],
)
def test_paired_prefill_controls_report_zero_drift(
    baseline: ProjectionPolicy, alternate: ProjectionPolicy, isolated: bool
) -> None:
    model = _synthetic_model(torch.bfloat16)
    report = run_paired_prefill(
        model,
        _probe_inputs(model),
        baseline=baseline,
        alternate=alternate,
        isolated=isolated,
        max_new_tokens=4,
        position_stride=16,
        position_tail=8,
    )
    assert report.baseline == baseline.name and report.alternate == alternate.name
    # Per CSA layer: 2 compressors + 1 top-k; per HCA layer: 1 compressor.
    assert len(report.isolated) == (3 * 3 + 3 if isolated else 0)
    for rec in report.isolated:
        if rec.tensor is not None:
            assert rec.tensor.max_abs == 0.0
        else:
            assert rec.topk is not None and rec.topk.n_flipped_slots == 0
            assert rec.topk.fraction_identical == 1.0
    assert len(report.attention_outputs) == 6
    assert all(cross.tensor.max_abs == 0.0 for cross in report.attention_outputs)
    lg = report.logits
    assert lg.argmax_agreement == 1.0 and lg.subset_max_abs == 0.0 and lg.subset_mean_kl == 0.0
    assert lg.n_scored_positions == 127
    assert lg.mean_nll_delta == 0.0 and lg.nll_delta_block_stderr == 0.0
    assert lg.baseline_mean_nll == lg.alternate_mean_nll > 0.0
    assert lg.baseline_top1_accuracy == lg.alternate_top1_accuracy
    assert lg.nll_block_size == 256
    # 127 scored positions fit in one block of 256; both runs agree exactly.
    assert len(lg.baseline_nll_by_block) == 1
    assert lg.baseline_nll_by_block == lg.alternate_nll_by_block
    assert lg.baseline_nll_by_block[0] == pytest.approx(lg.baseline_mean_nll, rel=1e-12)
    assert report.tokens.identical and report.tokens.n_generated == 4


def test_paired_prefill_rejects_isolated_taps_across_gemm_kernels() -> None:
    model = _synthetic_model(torch.bfloat16)
    with pytest.raises(ValueError, match="force_upcast_gemm"):
        run_paired_prefill(
            model, _probe_inputs(model, seqlen=32), alternate=FP32_OUT_UPCAST_GEMM, isolated=True
        )


def test_paired_prefill_fp32_vs_native_reports_finite_metrics_and_restores_model() -> None:
    model = _synthetic_model(torch.bfloat16)
    report = run_paired_prefill(
        model,
        _probe_inputs(model),
        baseline=FP32_OUT,
        alternate=NATIVE_OUT,
        max_new_tokens=4,
        position_stride=16,
        position_tail=8,
    )
    assert isinstance(report, PairedPrefillReport)
    assert report.baseline == "float32" and report.alternate == "native"
    roles = sorted({rec.role for rec in report.isolated})
    assert roles == [
        "csa.compressor",
        "csa.indexer.compressor",
        "csa.indexer.topk",
        "hca.compressor",
    ]
    for rec in report.isolated:
        if rec.tensor is not None:
            assert torch.isfinite(torch.tensor([rec.tensor.max_abs, rec.tensor.rms_rel])).all()
            assert -1.0 <= rec.tensor.cosine <= 1.0
        else:
            assert rec.topk is not None
            assert 0.0 <= rec.topk.min_set_agreement <= rec.topk.mean_set_agreement <= 1.0
    assert 0.0 <= report.logits.argmax_agreement <= 1.0
    assert report.logits.subset_mean_kl >= 0.0
    assert report.logits.nll_delta_block_stderr is not None
    assert report.logits.mean_nll_delta == pytest.approx(
        report.logits.alternate_mean_nll - report.logits.baseline_mean_nll, abs=1e-9
    )
    # The model is left on the baseline policy, and the taps are gone.
    assert all(
        c.projection_out_dtype is torch.float32
        for c in model.modules()
        if isinstance(c, TokenLevelCompressor)
    )
    for _, attn in attention_blocks(model):
        assert "forward_prefill_with_cache" not in attn.__dict__
        assert "forward_prefill_with_cache" not in attn.compressor.__dict__
    json.dumps(report.to_dict())


def test_isolated_taps_refuse_a_model_not_on_the_baseline_policy() -> None:
    model = _synthetic_model(torch.bfloat16)
    configure_projection_out_dtype(model, None)
    input_ids = _probe_inputs(model, seqlen=32)
    state_cache = model.build_state_cache(max_seq_len=32, batch_size=1, dtype=torch.bfloat16)
    with (
        IsolatedTaps(model, baseline=torch.float32, alternate=None),
        pytest.raises(RuntimeError, match="baseline policy"),
        torch.inference_mode(),
    ):
        model.forward_prefill_with_cache(input_ids, state_cache=state_cache)
