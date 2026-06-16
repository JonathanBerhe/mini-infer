"""Tests for the uniform benchmark harness logic.

Model-free and fast: every technique here is a stub whose `run` returns canned
`BenchResult`s, so we exercise the harness control flow (applicability gating,
skip recording, failure isolation, parity detection, table formatting) without
loading a model. The real model-driven run is exercised by `scripts/bench_all.py`.
"""

from __future__ import annotations

import pytest

from mini_infer.bench import (
    BenchEnv,
    BenchResult,
    SkippedTechnique,
    Technique,
    Workload,
    format_table,
    parity_violations,
    run_suite,
)

_CPU_ENV = BenchEnv(device="cpu", cuda_available=False, cuda_device_count=0)
_GPU2_ENV = BenchEnv(device="cuda", cuda_available=True, cuda_device_count=2)


def _result(
    technique: str,
    concurrency: int,
    tokens: tuple[int, ...] = (1, 2, 3),
    *,
    lossless: bool = True,
) -> BenchResult:
    return BenchResult(
        technique=technique,
        concurrency=concurrency,
        total_seconds=1.0,
        total_tokens=len(tokens) * concurrency,
        per_request_latencies=[1.0] * concurrency,
        output_signature=tuple(tokens for _ in range(concurrency)),
        lossless=lossless,
    )


def _stub(technique: str, **kwargs: object) -> Technique:
    def _run(workload: Workload, env: BenchEnv) -> list[BenchResult]:
        return [_result(technique, c) for c in workload.concurrency_levels]

    return Technique(name=technique, run=_run, **kwargs)  # type: ignore[arg-type]


def _workload(concurrency: list[int] | None = None) -> Workload:
    return Workload(
        model="stub",
        prompts=["a", "b"],
        concurrency_levels=concurrency or [1, 2],
        max_tokens=4,
    )


def test_prompts_for_replicates_to_fill_slots() -> None:
    workload = _workload()
    assert workload.prompts_for(1) == ["a"]
    assert workload.prompts_for(3) == ["a", "b", "a"]


def test_prompts_for_rejects_bad_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        _workload().prompts_for(0)


def test_bench_result_throughput_and_latency() -> None:
    result = BenchResult(
        technique="x",
        concurrency=2,
        total_seconds=2.0,
        total_tokens=20,
        per_request_latencies=[1.0, 3.0],
        output_signature=((1,), (1,)),
    )
    assert result.tokens_per_sec == 10.0
    assert result.median_latency == 2.0


def test_bench_result_zero_time_is_safe() -> None:
    result = BenchResult("x", 1, 0.0, 5, [], ())
    assert result.tokens_per_sec == 0.0
    assert result.median_latency == 0.0


def test_applicability_pending_is_skipped() -> None:
    can_run, reason = _stub("p", pending="not wired").applicability(_CPU_ENV)
    assert not can_run
    assert "pending: not wired" in reason


def test_applicability_requires_cuda_on_cpu() -> None:
    can_run, reason = _stub("g", requires_cuda=True, note="FlashInfer").applicability(_CPU_ENV)
    assert not can_run
    assert "requires CUDA" in reason and "FlashInfer" in reason


def test_applicability_min_cuda_devices() -> None:
    tech = _stub("tp", min_cuda_devices=2)
    assert tech.applicability(_CPU_ENV)[0] is False
    one_gpu = BenchEnv(device="cuda", cuda_available=True, cuda_device_count=1)
    assert tech.applicability(one_gpu)[0] is False
    assert tech.applicability(_GPU2_ENV)[0] is True


def test_run_suite_runs_applicable_and_skips_rest() -> None:
    techniques = [
        _stub("baseline"),
        _stub("kv_nvfp4", requires_cuda=True),
        _stub("spec", pending="follow-up"),
    ]
    results, skipped = run_suite(_workload([1, 2]), techniques, env=_CPU_ENV)

    assert {r.technique for r in results} == {"baseline"}
    assert len(results) == 2  # two concurrency levels
    skipped_names = {s.technique for s in skipped}
    assert skipped_names == {"kv_nvfp4", "spec"}


def test_run_suite_isolates_a_failing_technique() -> None:
    def _boom(workload: Workload, env: BenchEnv) -> list[BenchResult]:
        raise RuntimeError("kaboom")

    techniques = [_stub("ok"), Technique(name="bad", run=_boom)]
    results, skipped = run_suite(_workload([1]), techniques, env=_CPU_ENV)

    assert [r.technique for r in results] == ["ok"]
    assert skipped == [SkippedTechnique("bad", "failed: kaboom")]


def test_parity_violations_flags_token_divergence() -> None:
    same = [_result("a", 1, (1, 2)), _result("b", 1, (1, 2))]
    assert parity_violations(same) == []

    differ = [_result("a", 1, (1, 2)), _result("b", 1, (9, 9))]
    assert parity_violations(differ) == [("a", "b")]


def test_parity_excludes_lossy_techniques() -> None:
    # A lossy technique (INT8/KV-quant) diverging from baseline is expected, not a bug.
    only_lossy_diverges = [
        _result("baseline", 1, (1, 2)),
        _result("int8", 1, (9, 9), lossless=False),
    ]
    assert parity_violations(only_lossy_diverges) == []

    # A lossless technique diverging is still flagged, even alongside a lossy one.
    lossless_diverges = [
        _result("baseline", 1, (1, 2)),
        _result("prefix_cache", 1, (3, 3)),
        _result("int8", 1, (9, 9), lossless=False),
    ]
    assert parity_violations(lossless_diverges) == [("baseline", "prefix_cache")]


def test_parity_compares_within_each_concurrency_level() -> None:
    # Divergence at concurrency=2 only; concurrency=1 agrees.
    results = [
        _result("a", 1, (1,)),
        _result("b", 1, (1,)),
        _result("a", 2, (1,)),
        _result("b", 2, (2,)),
    ]
    assert parity_violations(results) == [("a", "b")]


def test_format_table_has_rows_and_skip_footnote() -> None:
    results = [_result("baseline", 1), _result("baseline", 2)]
    skipped = [SkippedTechnique("kv_nvfp4", "requires CUDA")]
    table = format_table(results, skipped)

    assert "technique" in table and "tok/s" in table
    assert "baseline" in table
    assert "Skipped:" in table
    assert "kv_nvfp4: requires CUDA" in table


def test_workload_attention_backend_default_and_override() -> None:
    assert _workload().attention_backend == "flash_attn"
    overridden = Workload(
        model="m", prompts=["a"], concurrency_levels=[1], max_tokens=1, attention_backend="torch"
    )
    assert overridden.attention_backend == "torch"


def test_build_registry_lists_expected_techniques() -> None:
    """The registry assembles (no model load) and exposes the full surface."""
    from mini_infer.bench import build_registry

    assert [technique.name for technique in build_registry()] == [
        "baseline",
        "int8_w8a16",
        "prefix_cache",
        "pd_serial",
        "pd_parallel",
        "attn_flashinfer",
        "kv_fp8",
        "kv_nvfp4",
        "spec_decode",
        "tensor_parallel",
    ]
