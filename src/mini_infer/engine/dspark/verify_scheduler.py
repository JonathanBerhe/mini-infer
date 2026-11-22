"""The batch-global verify-length scheduler of DSpark (Algorithm 1 of the paper).

In each step, every speculating request has a drafted block. The drafter's
confidence head gives `c_j`, the predicted probability that draft token `j`
survives verification when tokens `1..j-1` survived. So token `j` commits with
probability `a_j = c_1 * ... * c_j`, and a request that verifies `l` drafted
tokens commits `1 + a_1 + ... + a_l` tokens on average. The 1 is the bonus token.

Each verified token makes the target forward wider, and a wider forward takes
longer. The scheduler picks the verify length `l_r` of every request so that the
expected tokens per second of the whole step is as high as possible:

    theta = tau / (fixed_seconds + T(B))

`tau` is the expected number of committed tokens in the step. `B` is the number
of query tokens in the forward. `T(B)` is the forward time from a profiled
`CapacityCurve`. `fixed_seconds` is the part of the step time that does not
change with `B`, for example the drafter. With `fixed_seconds = 0`, `theta` is
the paper's `tau * SPS(B)`.

`a_j` never increases with `j`. So when the candidates `(r, j)` are sorted by
`a_j`, the positions of each request come in order, and the next candidate is
the best single token to add. The scheduler adds candidates while `theta`
increases, and stops at the first candidate that does not increase it.

The early stop is necessary for sampling at temperature > 0 (paper, Appendix A).
`c_{j+1}` depends on the drafted token `x_j` through the Markov head. If a search
continues past the first candidate that does not help, `x_j` can decide whether
`x_j` itself is verified, and the output distribution is no longer the target's.
With the early stop, the decision to verify position `j` of a request uses only
`c_1..c_j` of that request and the confidences of the other requests.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class CapacityCurve:
    """Target forward time against forward width: the paper's SPS(B), in seconds per step.

    `token_counts[i]` is a forward width `B` (query tokens over all requests), and
    `step_seconds[i]` is the measured time of one target forward at that width.
    Between two points the time is linear. Below the first point it is the time
    of the first point. Past the last point it follows the slope of the last segment.

    The times must not decrease. A forward does not get faster with more tokens,
    and a decrease gives the scheduler free tokens.
    """

    token_counts: tuple[int, ...]
    step_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.token_counts:
            raise ValueError("a capacity curve needs at least one point")
        if len(self.token_counts) != len(self.step_seconds):
            raise ValueError(
                f"{len(self.token_counts)} token counts but {len(self.step_seconds)} times"
            )
        if self.token_counts[0] < 1:
            raise ValueError(f"token counts must be positive, got {self.token_counts[0]}")
        for lo, hi in zip(self.token_counts, self.token_counts[1:], strict=False):
            if hi <= lo:
                raise ValueError(f"token counts must increase, got {lo} then {hi}")
        for seconds in self.step_seconds:
            if not (math.isfinite(seconds) and seconds > 0.0):
                raise ValueError(f"step times must be positive and finite, got {seconds}")
        for earlier, later in zip(self.step_seconds, self.step_seconds[1:], strict=False):
            if later < earlier:
                raise ValueError(f"step times must not decrease, got {earlier} then {later}")

    def seconds(self, tokens: int) -> float:
        """The forward time at a width of `tokens` query tokens."""
        counts, times = self.token_counts, self.step_seconds
        if tokens <= counts[0]:
            return times[0]
        if len(counts) == 1:
            return times[0]
        # Past the end, the last segment continues.
        i = min(bisect.bisect_left(counts, tokens), len(counts) - 1)
        x0, x1 = counts[i - 1], counts[i]
        y0, y1 = times[i - 1], times[i]
        return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)

    def steps_per_second(self, tokens: int) -> float:
        """SPS(B), as the paper writes it."""
        return 1.0 / self.seconds(tokens)

    def to_json(self) -> dict[str, Any]:
        return {"token_counts": list(self.token_counts), "step_seconds": list(self.step_seconds)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CapacityCurve:
        return cls(
            token_counts=tuple(int(b) for b in data["token_counts"]),
            step_seconds=tuple(float(s) for s in data["step_seconds"]),
        )


def survival_probabilities(confidence_logits: torch.Tensor, limit: int) -> list[float]:
    """`a_1..a_limit` of one proposal: the cumulative product of the sigmoid confidences.

    `confidence_logits` is the `(1, block_size)` raw output of the confidence
    head. The product runs in Python floats. A factor in [0, 1] never makes a
    rounded product larger, so the result does not increase.
    """
    if limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    probs = confidence_logits[0, :limit].float().sigmoid().tolist()
    out: list[float] = []
    running = 1.0
    for c in probs:
        running *= float(c)
        out.append(running)
    return out


def schedule_verify_lengths(
    survival: Sequence[Sequence[float]],
    *,
    base_tokens: int,
    base_tau: float,
    curve: CapacityCurve,
    fixed_seconds: float = 0.0,
) -> list[int]:
    """Algorithm 1 with the early stop: the number of drafted tokens each request verifies.

    `survival[r]` holds `a_{r,1..cap_r}`, the cumulative survival probability of
    each draft position that request `r` can verify. The caller cuts each row to
    its cap first (the block size, the confidence threshold, and the `max_tokens`
    room). The values must be in [0, 1] and must not increase along a row.

    `base_tokens` is the forward width with no drafted token: the prefill chunks,
    one token for each plain decoder, and the anchor of each speculator.
    `base_tau` is the number of tokens that the step commits with no drafted
    token: one for each decoding request.

    Returns `l_r` in `[0, len(survival[r])]` for each request. The kept positions
    of a request are always a prefix of its block.
    """
    if base_tokens < len(survival):
        raise ValueError(
            f"base_tokens={base_tokens} is less than the {len(survival)} speculator anchors"
        )
    if base_tau < 0.0:
        raise ValueError(f"base_tau must be non-negative, got {base_tau}")
    if not (math.isfinite(fixed_seconds) and fixed_seconds >= 0.0):
        raise ValueError(f"fixed_seconds must be non-negative and finite, got {fixed_seconds}")

    # Candidates sort by survival, high first. A tie goes to the earlier position, so
    # each request's positions stay in order even when a confidence rounds to 1.0.
    candidates: list[tuple[float, int, int]] = []
    for r, row in enumerate(survival):
        previous = 1.0
        for j, a in enumerate(row, start=1):
            if not 0.0 <= a <= previous:
                raise ValueError(
                    f"survival of request {r} must be in [0, 1] and must not increase; "
                    f"position {j} has {a} after {previous}"
                )
            previous = a
            if a > 0.0:
                candidates.append((a, j, r))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    lengths = [0] * len(survival)
    tokens = base_tokens
    tau = base_tau
    best = tau / (fixed_seconds + curve.seconds(tokens))
    for a, j, r in candidates:
        tokens += 1
        tau += a
        theta = tau / (fixed_seconds + curve.seconds(tokens))
        # Equal throughput stops too: a token that adds nothing only adds K/V work.
        if theta <= best:
            break
        best = theta
        lengths[r] = j
    return lengths
