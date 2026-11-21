"""The batch-global verify-length scheduler (ADR-061) and its capacity curve.

The scheduler is a pure function of survival probabilities and a forward-time
curve, so these tests are exact. The property tests check the two facts that
the scheduler's correctness rests on: the decision to verify position k of a
request never looks at that request's later confidences (paper, Appendix A),
and the early stop finds the global optimum when the step time is linear in B.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.dspark.capacity_profile import (
    default_token_counts,
    profile_capacity_curve,
)
from mini_infer.engine.dspark.verify_scheduler import (
    CapacityCurve,
    schedule_verify_lengths,
    survival_probabilities,
)
from tests.unit._dspark_doubles import TAPS, MicroRunner, micro_setup

# ── CapacityCurve ──


def test_curve_interpolates_between_points_and_extends_the_last_segment() -> None:
    curve = CapacityCurve(token_counts=(4, 8, 16), step_seconds=(1.0, 2.0, 2.0))

    assert curve.seconds(1) == 1.0  # below the first point: the first time
    assert curve.seconds(4) == 1.0
    assert curve.seconds(6) == 1.5
    assert curve.seconds(8) == 2.0
    assert curve.seconds(12) == 2.0
    assert curve.seconds(32) == 2.0  # the last segment is flat
    steep = CapacityCurve(token_counts=(1, 2), step_seconds=(1.0, 3.0))
    assert steep.seconds(4) == 7.0  # slope 2 per token past the end
    assert steep.steps_per_second(2) == pytest.approx(1 / 3)


def test_single_point_curve_is_constant() -> None:
    curve = CapacityCurve(token_counts=(8,), step_seconds=(0.5,))
    assert curve.seconds(1) == curve.seconds(8) == curve.seconds(100) == 0.5


@pytest.mark.parametrize(
    ("counts", "seconds", "message"),
    [
        ((), (), "at least one point"),
        ((1, 2), (1.0,), "token counts but"),
        ((0, 2), (1.0, 2.0), "must be positive"),
        ((2, 2), (1.0, 2.0), "must increase"),
        ((1, 2), (1.0, 0.0), "positive and finite"),
        ((1, 2), (1.0, math.inf), "positive and finite"),
        ((1, 2), (2.0, 1.0), "must not decrease"),
    ],
)
def test_curve_rejects_invalid_points(
    counts: tuple[int, ...], seconds: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CapacityCurve(token_counts=counts, step_seconds=seconds)


def test_curve_json_round_trip() -> None:
    curve = CapacityCurve(token_counts=(1, 5, 9), step_seconds=(0.01, 0.012, 0.02))
    assert CapacityCurve.from_json(curve.to_json()) == curve


# ── survival_probabilities ──


def test_survival_is_the_cumulative_product_of_sigmoid_confidences() -> None:
    probs = torch.tensor([0.9, 0.5, 0.8, 0.1])
    logits = torch.logit(probs).unsqueeze(0)

    survival = survival_probabilities(logits, 3)

    assert survival == pytest.approx([0.9, 0.45, 0.36])
    assert survival_probabilities(logits, 0) == []


# ── schedule_verify_lengths ──


def _linear(t0: float, per_token: float, top: int = 256) -> CapacityCurve:
    return CapacityCurve(token_counts=(1, top), step_seconds=(t0 + per_token, t0 + per_token * top))


def _theta(
    survival: list[list[float]],
    lengths: list[int],
    base_tokens: int,
    base_tau: float,
    curve: CapacityCurve,
) -> float:
    tau = base_tau + sum(sum(row[:length]) for row, length in zip(survival, lengths, strict=True))
    return tau / curve.seconds(base_tokens + sum(lengths))


def test_free_tokens_are_all_verified() -> None:
    """With a flat curve every drafted token with a positive survival raises throughput."""
    flat = CapacityCurve(token_counts=(1,), step_seconds=(1.0,))
    survival = [[0.9, 0.5, 0.1], [0.3, 0.0, 0.0], []]

    assert schedule_verify_lengths(survival, base_tokens=3, base_tau=3, curve=flat) == [3, 1, 0]


def test_expensive_tokens_are_all_cut() -> None:
    """When one token doubles the step time, no draft position is worth its cost."""
    steep = _linear(0.0, 1.0)
    survival = [[0.99, 0.98], [0.97]]

    assert schedule_verify_lengths(survival, base_tokens=2, base_tau=2, curve=steep) == [0, 0]


def test_appendix_a_example_stops_at_the_first_non_improving_token() -> None:
    """The paper's Appendix A setup: SPS(1)=1.0, SPS(2)=0.5, SPS(3)=0.45 and a_1=0.8.

    Theta falls from 1.0 to 0.9 at the first token, so the early stop verifies nothing,
    even though verifying both tokens (theta 1.134 when c_2=0.9) would be faster. A search
    past the dip would let c_2, which depends on the drafted token x_1, decide whether
    x_1 is verified. That is the bias that the early stop prevents.
    """
    curve = CapacityCurve(token_counts=(1, 2, 3), step_seconds=(1.0, 2.0, 1 / 0.45))
    for c2 in (0.9, 0.0):
        survival = [[0.8, 0.8 * c2]]
        assert schedule_verify_lengths(survival, base_tokens=1, base_tau=1, curve=curve) == [0]
    assert _theta([[0.8, 0.72]], [2], 1, 1, curve) == pytest.approx(1.134)


def test_highest_survival_candidates_win_across_requests() -> None:
    """The kept tokens are the top candidates by survival, across all requests."""
    # The step takes 1 + 0.1 * B seconds. Every candidate down to 0.6 raises theta,
    # and 0.2 lowers it, so the scheduler keeps five tokens and stops there.
    curve = _linear(1.0, 0.1)
    survival = [[0.95, 0.6, 0.2], [0.9, 0.85, 0.8]]

    lengths = schedule_verify_lengths(survival, base_tokens=2, base_tau=2, curve=curve)

    kept = sorted(
        (a for row, n in zip(survival, lengths, strict=True) for a in row[:n]), reverse=True
    )
    dropped = [a for row, n in zip(survival, lengths, strict=True) for a in row[n:]]
    assert kept and dropped
    assert min(kept) > max(dropped)


def test_drafter_time_makes_longer_verifies_pay_off() -> None:
    """A fixed cost per step (the drafter) lowers the relative cost of one more token."""
    curve = _linear(0.0, 1.0)
    survival = [[0.9]]

    assert schedule_verify_lengths(survival, base_tokens=1, base_tau=1, curve=curve) == [0]
    assert schedule_verify_lengths(
        survival, base_tokens=1, base_tau=1, curve=curve, fixed_seconds=10.0
    ) == [1]


def test_ties_keep_each_request_a_prefix() -> None:
    """A confidence that rounds to 1.0 ties two positions. The earlier one goes first."""
    flat = CapacityCurve(token_counts=(1,), step_seconds=(1.0,))
    assert schedule_verify_lengths([[0.5, 0.5, 0.5]], base_tokens=1, base_tau=1, curve=flat) == [3]


@pytest.mark.parametrize(
    ("survival", "kwargs", "message"),
    [
        ([[0.5, 0.6]], {}, "must not increase"),
        ([[1.5]], {}, r"in \[0, 1\]"),
        ([[-0.1]], {}, r"in \[0, 1\]"),
        ([[0.5], [0.5]], {"base_tokens": 1}, "anchors"),
        ([[0.5]], {"base_tau": -1.0}, "base_tau"),
        ([[0.5]], {"fixed_seconds": -1.0}, "fixed_seconds"),
    ],
)
def test_schedule_rejects_invalid_inputs(
    survival: list[list[float]], kwargs: dict[str, float], message: str
) -> None:
    args: dict[str, object] = {"base_tokens": 4, "base_tau": 2.0, "curve": _linear(0.0, 1.0)}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        schedule_verify_lengths(survival, **args)  # type: ignore[arg-type]


def _random_rows(rng: random.Random, n_requests: int, cap: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for _ in range(n_requests):
        running, row = 1.0, []
        for _ in range(rng.randint(0, cap)):
            running *= rng.random()
            row.append(running)
        rows.append(row)
    return rows


def _random_curve(rng: random.Random, top: int) -> CapacityCurve:
    """A jagged increasing curve, as GPU step times are at tile boundaries."""
    seconds, t = [], rng.uniform(0.5, 2.0)
    for _ in range(top):
        t += rng.choice([0.0, 0.0, rng.uniform(0.0, 0.3)])
        seconds.append(t)
    return CapacityCurve(token_counts=tuple(range(1, top + 1)), step_seconds=tuple(seconds))


def test_decision_for_position_k_ignores_later_confidences() -> None:
    """Non-anticipation (Appendix A): whether request r verifies position k depends only on
    its own c_1..c_k and on the other requests. Later positions are redrawn freely."""
    rng = random.Random(0)
    for _ in range(300):
        rows = _random_rows(rng, rng.randint(1, 4), 5)
        base = len(rows) + rng.randint(0, 3)
        curve = _random_curve(rng, base + 25)
        fixed = rng.choice([0.0, rng.uniform(0.0, 2.0)])
        lengths = schedule_verify_lengths(
            rows, base_tokens=base, base_tau=len(rows), curve=curve, fixed_seconds=fixed
        )
        for r, row in enumerate(rows):
            for k in range(1, len(row) + 1):
                redrawn = [list(x) for x in rows]
                running = row[k - 1]
                for j in range(k, len(row)):
                    running *= rng.random()
                    redrawn[r][j] = running
                again = schedule_verify_lengths(
                    redrawn, base_tokens=base, base_tau=len(rows), curve=curve, fixed_seconds=fixed
                )
                assert (again[r] >= k) == (lengths[r] >= k)


def test_early_stop_is_optimal_when_step_time_is_linear() -> None:
    """With T(B) affine, theta is unimodal along the sorted candidates, so the early stop
    returns the best of every possible choice of verify lengths."""
    rng = random.Random(1)
    for _ in range(200):
        rows = _random_rows(rng, rng.randint(1, 3), 3)
        base = len(rows) + rng.randint(0, 2)
        curve = _linear(rng.uniform(0.0, 2.0), rng.uniform(0.01, 1.0))

        lengths = schedule_verify_lengths(rows, base_tokens=base, base_tau=len(rows), curve=curve)

        best = max(
            _theta(rows, list(choice), base, len(rows), curve)
            for choice in itertools.product(*(range(len(row) + 1) for row in rows))
        )
        assert _theta(rows, lengths, base, len(rows), curve) == pytest.approx(best)


# ── profiling ──


def test_default_token_counts_are_dense_then_every_eighth() -> None:
    assert default_token_counts(2, 3) == list(range(1, 9))
    counts = default_token_counts(16, 7)
    assert counts[:64] == list(range(1, 65))
    assert counts[64:] == list(range(72, 129, 8))
    with pytest.raises(ValueError, match="positive"):
        default_token_counts(0, 7)


class _RecordingRunner(MicroRunner):
    def __init__(self, runner: MicroRunner) -> None:
        super().__init__(runner._model, runner.block_pool)
        self.calls: list[tuple[list[int], list[int], bool]] = []

    def forward_step_packed(  # type: ignore[override]
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
        **kwargs: object,
    ) -> torch.Tensor:
        self.calls.append((list(cu_seqlens_q), list(position_offsets), "tap_layers" in kwargs))
        return super().forward_step_packed(
            cache,
            packed_input_ids,
            cu_seqlens_q,
            position_offsets,
            **kwargs,  # type: ignore[arg-type]
        )


def test_profile_measures_each_width_and_returns_every_block() -> None:
    runner = _RecordingRunner(micro_setup()[0])
    pool = runner.block_pool
    free_before = pool.num_free_blocks

    curve = profile_capacity_curve(
        runner,  # type: ignore[arg-type]
        token_counts=[5, 1, 3, 3],
        max_requests=2,
        context_len=4,
        tap_layers=frozenset(TAPS),
        warmup=1,
        repeats=2,
    )

    assert pool.num_free_blocks == free_before
    assert curve.token_counts == (1, 3, 5)
    assert all(b >= a for a, b in itertools.pairwise(curve.step_seconds))
    # One prefill of both slots, then (warmup + repeats) forwards per width, widest first.
    assert runner.calls[0] == ([0, 4, 8], [0, 0], False)
    timed = runner.calls[1:]
    assert [c[0] for c in timed] == [[0, 3, 5]] * 3 + [[0, 2, 3]] * 3 + [[0, 1]] * 3
    assert all(c[1] == [4] * (len(c[0]) - 1) and c[2] for c in timed)


def test_profile_rejects_invalid_settings() -> None:
    runner = micro_setup()[0]
    with pytest.raises(ValueError, match="token_counts"):
        profile_capacity_curve(runner, token_counts=[0], max_requests=1, context_len=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="repeats"):
        profile_capacity_curve(
            runner,  # type: ignore[arg-type]
            token_counts=[1],
            max_requests=1,
            context_len=1,
            repeats=0,
        )
