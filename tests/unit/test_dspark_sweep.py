"""The concurrency-sweep driver (`mini_infer.bench.dspark_sweep`), on CPU.

These run the real engine thread over the micro models, so they test the
driver the GPU benchmark will use, not a copy of it. Exact token equality is
asserted only at concurrency 1: with more requests in flight, the engine
thread's timing decides which requests share a step, and an exact assertion
across different batch compositions would test thread scheduling, not the
driver. Higher concurrency is covered by counts and invariants instead.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path

import pytest
import torch

from mini_infer.bench.dspark_sweep import StepRecord, SweepPoint, run_point, token_agreement
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.dspark import DSparkSpec
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.request_state import Request, RequestHandle, RunningRequest
from tests.unit._dspark_doubles import MicroRunner, micro_setup

_PROMPTS = ["hello world", "qrstu", "xy"]


def test_dspark_point_matches_plain_and_reports_tau() -> None:
    """The fp32 greedy contract through the driver: the speculative arm emits
    exactly the plain arm's tokens, and it reports tau because every request
    really speculated."""
    runner, drafter = micro_setup(seed=6)
    plain = run_point(runner, _PROMPTS, concurrency=1, max_tokens=10)
    spec = run_point(runner, _PROMPTS, concurrency=1, max_tokens=10, dspark=DSparkSpec(drafter))

    agreement = token_agreement(plain.tokens, spec.tokens)
    assert agreement.exact_matches == len(_PROMPTS)
    assert agreement.mean_prefix_fraction == 1.0
    assert plain.arm == "plain" and plain.spec is None
    assert spec.arm == "dspark" and spec.n_not_speculating == 0
    assert spec.spec is not None and spec.spec.tau >= 1.0
    assert spec.generated_tokens == sum(len(t) for t in spec.tokens)
    assert spec.to_json()["tau"] == spec.spec.tau


def test_timed_point_attributes_time_without_changing_output() -> None:
    """The timed scheduler only adds barriers and clocks: same tokens, one
    record per packed forward, and every phase boundary synchronized."""
    runner, drafter = micro_setup(seed=7)
    calls = 0

    def count_sync() -> None:
        nonlocal calls
        calls += 1

    untimed = run_point(runner, _PROMPTS, concurrency=1, max_tokens=8, dspark=DSparkSpec(drafter))
    timed = run_point(
        runner,
        _PROMPTS,
        concurrency=1,
        max_tokens=8,
        dspark=DSparkSpec(drafter),
        timed=True,
        sync=count_sync,
    )

    assert timed.tokens == untimed.tokens
    # Every point records its forwards; only a timed one has phase times.
    assert untimed.phases is None and untimed.steps
    assert timed.phases is not None and timed.steps
    assert timed.phases.forward_s > 0 and timed.phases.draft_s > 0
    # Each step that forwards crosses four barrier-bracketed phases.
    assert calls >= 8 * len(timed.steps)
    for step in timed.steps:
        assert step.batch >= 1 and step.n_verify <= step.batch <= step.total_q
    # At concurrency 1, a verify window is the only way a step feeds more
    # than one token after prefill, so some forward must be wider than 1.
    assert any(s.n_verify == 1 and s.total_q > 1 for s in timed.steps)


def test_point_at_higher_concurrency_completes_with_consistent_counts() -> None:
    runner, drafter = micro_setup(seed=8)
    point = run_point(runner, _PROMPTS, concurrency=3, max_tokens=6, dspark=DSparkSpec(drafter))

    assert point.concurrency == 3 and point.n_requests == len(_PROMPTS)
    assert point.generated_tokens == sum(len(t) for t in point.tokens)
    # Never past max_tokens. Zero is legal: a random target can pick EOS as
    # its very first token, and EOS is never emitted.
    assert all(len(t) <= 6 for t in point.tokens)
    assert point.tokens_per_s > 0


def test_point_counts_requests_that_did_not_speculate() -> None:
    """A prompt over the speculation limit runs the plain path with only a log
    line. The point must count it, or a "DSpark" result could quietly be the
    plain path."""
    runner, drafter = micro_setup(seed=9)
    spec = DSparkSpec(drafter, max_spec_prompt_tokens=4)
    point = run_point(runner, ["hello world", "xyz"], concurrency=1, max_tokens=4, dspark=spec)

    assert point.n_not_speculating == 1  # "hello world" is 11 tokens
    assert point.spec is not None  # pooled from the one request that did


class _CrashingRunner(MicroRunner):
    """A forward that fails with a non-OOM error, which ends the engine thread."""

    def forward_step_packed(  # type: ignore[override]
        self, cache: PagedKVCache, *args: object, **kwargs: object
    ) -> torch.Tensor:
        raise RuntimeError("forced engine-thread crash for test")


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_point_fails_fast_when_the_engine_thread_dies() -> None:
    """A dead engine thread never sends a terminal step. The point must notice
    within seconds, not wait out its whole budget of GPU time."""
    runner, _ = micro_setup(seed=10)
    crashing = _CrashingRunner(runner._model, runner.block_pool)

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="engine thread died"):
        run_point(crashing, ["abc"], concurrency=1, max_tokens=4, timeout_s=600.0)
    assert time.perf_counter() - start < 30.0


def test_wait_with_a_timeout_raises_instead_of_blocking() -> None:
    req = RunningRequest(
        request=Request(prompt="x", sampling_params=SamplingParams(), max_tokens=1),
        output_queue=queue.Queue(),
    )
    with pytest.raises(TimeoutError):
        RequestHandle(req).wait(timeout=0.05)


def test_token_agreement_measures_exact_matches_and_common_prefixes() -> None:
    agreement = token_agreement([[1, 2, 3], [4, 5]], [[1, 2, 9], [4, 5]])

    assert agreement.exact_matches == 1
    assert agreement.common_prefix == [2, 2]
    assert agreement.mean_prefix_fraction == pytest.approx((2 / 3 + 1) / 2)
    with pytest.raises(ValueError, match="reference requests"):
        token_agreement([[1]], [[1], [2]])


@pytest.mark.requires_model
def test_rendered_chat_encodes_to_the_same_ids_as_encode_chat() -> None:
    """The benchmark sends the chat template as TEXT, because `Request` takes
    text and admission runs plain `encode`. That only reproduces the
    templated ids if the tokenizer parses the template's special-token strings
    back into the same ids and adds no BOS. Qwen3 checkpoints share one
    tokenizer, so the small one stands in for the 4B target here; the GPU
    script asserts the same thing on the real tokenizer before it measures."""
    from mini_infer.engine.tokenizer import Tokenizer

    tok = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    for text in ["What is 17 * 23?", "Write a haiku about the sea.\nKeep it short."]:
        rendered = tok.render_chat(text, enable_thinking=False)
        assert tok.encode(rendered) == tok.encode_chat(text, enable_thinking=False)
        assert "<think>" in rendered  # the template pre-closes the think block


def test_sweep_measures_the_grid_and_resumes_without_rerunning(tmp_path: Path) -> None:
    """One file per point, only this grid's points returned, and a rerun over
    the same folder measures nothing that is already there."""
    from mini_infer.bench.dspark_sweep import run_sweep

    runner, drafter = micro_setup(seed=11)
    (tmp_path / "meta.json").write_text("{}")  # a stray file the sweep must ignore
    commits = 0

    def commit() -> None:
        nonlocal commits
        commits += 1

    grid = dict(
        out_dir=tmp_path,
        max_tokens=6,
        concurrencies=[1, 2],
        thresholds=[0.0],
        timed_concurrencies=[1],
        on_checkpoint=commit,
    )
    points = run_sweep(runner, drafter, {"a": ["hello", "xyz"]}, **grid)  # type: ignore[arg-type]

    expected = {
        "a-plain-c1",
        "a-dspark-t0.0-c1",
        "a-plain-c2",
        "a-dspark-t0.0-c2",
        "a-timed-plain-c1",
        "a-timed-dspark-c1",
    }
    assert set(points) == expected
    assert commits == len(expected)
    assert all((tmp_path / f"{k}.json").exists() for k in expected)
    # fp32 greedy at concurrency 1: the dspark arm reproduces the plain arm.
    agreement = points["a-dspark-t0.0-c1"]["agreement_vs_plain"]
    assert agreement["exact_matches"] == agreement["n_requests"] == 2
    assert isinstance(points["a-dspark-t0.0-c1"]["speedup_vs_plain"], float)
    assert points["a-timed-dspark-c1"]["steps"]

    again = run_sweep(runner, drafter, {"a": ["hello", "xyz"]}, **grid)  # type: ignore[arg-type]
    assert commits == len(expected)  # nothing measured twice
    assert set(again) == expected


def test_sweep_refuses_a_dspark_point_that_did_not_speculate(tmp_path: Path) -> None:
    from mini_infer.bench.dspark_sweep import run_sweep

    runner, drafter = micro_setup(seed=12)
    with pytest.raises(AssertionError, match="did not speculate"):
        run_sweep(
            runner,
            drafter,
            {"a": ["hello world"]},
            out_dir=tmp_path,
            max_tokens=4,
            concurrencies=[1],
            thresholds=[0.0],
            max_spec_prompt_tokens=4,
        )


def test_sweep_refuses_a_point_with_cancelled_requests(tmp_path: Path) -> None:
    """A prompt that can never fit the pool is cancelled at admission, and the
    point would then measure fewer requests than the grid claims."""
    from mini_infer.bench.dspark_sweep import run_sweep

    runner, drafter = micro_setup(seed=13)  # 128 blocks of 8 tokens
    too_long = "x" * 1100
    with pytest.raises(AssertionError, match="were cancelled"):
        run_sweep(
            runner,
            drafter,
            {"a": [too_long]},
            out_dir=tmp_path,
            max_tokens=4,
            concurrencies=[1],
            thresholds=[0.0],
        )


def _point_with_steps(steps: list[StepRecord]) -> SweepPoint:
    return SweepPoint(
        arm="plain",
        concurrency=2,
        confidence_threshold=None,
        max_tokens=8,
        n_requests=4,
        wall_s=steps[-1].end_s,
        generated_tokens=steps[-1].emitted,
        tokens=[],
        spec=None,
        n_not_speculating=0,
        finish_reasons={"length": 4},
        phases=None,
        steps=steps,
    )


def test_steady_throughput_ignores_the_drain() -> None:
    """While a prompt waits, every slot is busy; after the last admission the
    batch drains. The steady number covers only the busy part, and the mean
    in-flight count says how far the whole run fell short of the label."""
    steps = [
        StepRecord(batch=2, total_q=2, n_verify=0, waiting=2, forward_s=0.0, end_s=1.0, emitted=10),
        StepRecord(batch=2, total_q=2, n_verify=0, waiting=1, forward_s=0.0, end_s=2.0, emitted=20),
        StepRecord(batch=1, total_q=1, n_verify=0, waiting=0, forward_s=0.0, end_s=4.0, emitted=24),
    ]
    point = _point_with_steps(steps)

    assert point.steady_tokens_per_s == 10.0  # 20 tokens by second 2
    assert point.tokens_per_s == 6.0  # 24 tokens in 4 seconds
    assert point.mean_in_flight == (2 * 1 + 2 * 1 + 1 * 2) / 4
    json_view = point.to_json()
    assert json_view["steady_tokens_per_s"] == 10.0 and json_view["mean_in_flight"] == 1.5


def test_steady_throughput_is_none_without_a_waiting_prompt() -> None:
    steps = [
        StepRecord(batch=2, total_q=2, n_verify=0, waiting=0, forward_s=0.0, end_s=1.0, emitted=4)
    ]
    assert _point_with_steps(steps).steady_tokens_per_s is None


def test_sweep_refuses_points_from_a_different_grid(tmp_path: Path) -> None:
    """A rerun into the same folder with other settings would mix an older
    grid's numbers into the new one, so it raises instead."""
    from mini_infer.bench.dspark_sweep import run_sweep

    runner, drafter = micro_setup(seed=14)
    grid = dict(out_dir=tmp_path, concurrencies=[1], thresholds=[0.0])
    run_sweep(runner, drafter, {"a": ["hello"]}, max_tokens=4, **grid)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="different grid"):
        run_sweep(runner, drafter, {"a": ["hello"]}, max_tokens=5, **grid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different grid"):
        run_sweep(
            runner,
            drafter,
            {"a": ["hello"]},
            max_tokens=4,
            identity="other model",
            **grid,  # type: ignore[arg-type]
        )
