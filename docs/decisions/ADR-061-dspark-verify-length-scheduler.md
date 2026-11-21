# ADR-061: Batch-global verify lengths for DSpark

Date: 2026-11-22
Status: Accepted

## Context

[ADR-031](ADR-031-multi-request-dspark.md) runs DSpark for many requests in
one packed forward. Each speculator sends its full drafted block to verify, or
the prefix that a fixed confidence threshold keeps. A wider forward takes
longer, so a draft token that the target rejects costs time for the whole
batch.

Algorithm 1 of the DSpark paper ([arXiv 2607.05147](https://arxiv.org/abs/2607.05147),
section 3.2.2) sets the verify length of each request. Its inputs are the
confidence head and a profiled curve of steps per second, SPS(B). B is the
number of query tokens in the forward. vLLM merged the drafter without this
scheduler.

Terms: `c_j` is the confidence of draft position `j`, the predicted
probability that token `j` survives when tokens `1..j-1` survived.
`a_j = c_1 * ... * c_j` is the probability that token `j` commits.

## Decision

1. `DSparkSpec(capacity=CapacityCurve(...))` turns the scheduler on. It needs
   a drafter with a confidence head. Without a curve, ADR-031 does not change.
2. The scheduler runs after admission, right before the packed forward,
   because B is final only then. B counts the prefill chunks, one token per
   plain decoder, and the anchor and kept drafts of each speculator.
3. The objective is the expected tokens per second of the step,
   `theta = tau / (fixed_seconds + T(B))`. `tau` counts one token per decoding
   request, plus `a_1..a_l` of each speculator that verifies `l` drafts.
   `T(B)` comes from the curve.
4. The candidates (request, position) sort by `a_j`, high first. The scheduler
   adds them while `theta` increases and stops at the first candidate that
   does not increase it. A tie goes to the earlier position, so each request
   keeps a prefix of its block.
5. `fixed_seconds` is the wall time of the drafter phase in the same step.
   The paper uses `tau * SPS(B)`, but its production drafter runs inside the
   target forward. Our drafter is a separate phase, and it took 13 to 25
   percent of the step in the ADR-031 benchmark. With `fixed_seconds = 0`,
   `theta` is the paper's objective.
6. `profile_capacity_curve` times the runner's own forward at each width, with
   the drafter's taps on, before the scheduler starts. Between points the
   curve is linear. Each time is raised to the largest time at a smaller
   width, so timer noise cannot make a token free.
7. A confidence threshold above 0 still cuts first.

## Alternatives Considered

- Search every prefix length and keep the best `theta`. Appendix A of the
  paper shows that this breaks sampling at temperature > 0. `c_{j+1}` depends
  on the drafted token `x_j` through the Markov head, so `x_j` then decides
  whether `x_j` is verified. The paper removes the early stop in production
  and argues that a delay of two steps makes this safe. It gives no proof, so
  we keep the early stop.
- A curve over the request count and B. On the torch backend, a verify
  forward of 8 tokens per request costs 1.10x to 1.25x a decode forward. So
  the cost follows the request count more than B. A curve over B alone
  follows the paper (its footnote 2) and is simple to profile. The profile
  spreads B over `min(B, max_requests)` requests.
- Calibrate the confidences with sequential temperature scaling, as the paper
  does. The released checkpoints ship no calibration. Our measured ECE
  (expected calibration error) is 0.185 to 0.287 on Qwen3-4B, against 3 to 8
  percent in the paper. This is a follow-up.

## Consequences

- Greedy output does not depend on the verify lengths. Unit tests compare
  the tokens with the plain scheduler for flat, steep and linear curves (CPU
  fp32).
- A property test redraws the later confidences of a request and asserts
  that the decision for position `k` does not change. A scheduler without the
  early stop fails this test.
- With an affine `T(B)`, `theta` has one peak along the sorted candidates. A
  brute-force test shows that the early stop then finds the best lengths. On a
  jagged GPU curve the early stop can end at a local peak, as in the Appendix
  A example.
- `SpecSummary.scheduler_cut` counts the drafted tokens that the scheduler
  removed. The sweep has a `dspark-sched` arm, and the Modal script profiles
  the curve once per run and stores it as `capacity.json`.
- No GPU measurement exists yet.
