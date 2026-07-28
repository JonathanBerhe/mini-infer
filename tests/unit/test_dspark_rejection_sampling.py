"""Speculative sampling is distribution-preserving, tested where that is checkable.

At temperature 0 the contract is token equality and provable. Above 0 it is
weaker but more interesting: the emitted tokens are a draw from the TARGET's
distribution, so no individual run matches any particular target-alone run,
yet the distributions agree exactly. That is what makes speculative decoding a
pure latency optimization rather than a quality/speed dial, and it is only
worth anything if it is actually true, so it gets tested three ways:

1. **Algebraically**, on hand-built distributions where the emitted
   probability can be computed in closed form and compared to the target.
2. **Empirically**, by drawing many samples and comparing frequencies.
3. **Adversarially**, by breaking the residual correction on purpose and
   confirming the test notices. A distribution test that passes on a broken
   implementation is worse than no test, and residual-free "accept or take the
   target's argmax" is exactly the plausible-looking shortcut this needs to
   reject.
"""

from __future__ import annotations

import math

import pytest
import torch

from mini_infer.engine.dspark.sampling import (
    accepted_prefix_length,
    logits_to_probs,
    sample_residual,
)


def _emitted_distribution(p_target: torch.Tensor, p_draft: torch.Tensor) -> torch.Tensor:
    """Closed-form distribution of one speculative step over a 1-token proposal.

    Emitting `x` happens two ways: the draft proposed `x` and it was accepted,
    contributing `p_draft(x) * min(1, p_target(x)/p_draft(x)) =
    min(p_draft(x), p_target(x))`; or the draft's proposal was rejected and the
    residual draw produced `x`, contributing
    `P(reject) * normalized_residual(x)`.
    """
    accept_mass = torch.minimum(p_draft, p_target)
    p_reject = 1.0 - accept_mass.sum()
    residual = torch.clamp(p_target - p_draft, min=0.0)
    residual = residual / residual.sum() if residual.sum() > 0 else torch.zeros_like(residual)
    return accept_mass + p_reject * residual


def test_emitted_distribution_equals_target_algebraically() -> None:
    """The closed form collapses to the target for arbitrary mismatched drafts."""
    cases = [
        ([0.5, 0.3, 0.2], [0.5, 0.3, 0.2]),  # perfect draft
        ([0.7, 0.2, 0.1], [0.1, 0.2, 0.7]),  # badly wrong draft
        ([0.9, 0.05, 0.05], [0.34, 0.33, 0.33]),  # flat draft, peaked target
        ([0.34, 0.33, 0.33], [0.9, 0.05, 0.05]),  # peaked draft, flat target
        ([1.0, 0.0, 0.0], [0.2, 0.4, 0.4]),  # degenerate target
    ]
    for target, draft in cases:
        p_t = torch.tensor(target, dtype=torch.float64)
        p_d = torch.tensor(draft, dtype=torch.float64)
        emitted = _emitted_distribution(p_t, p_d)
        assert torch.allclose(emitted, p_t, atol=1e-12), f"target={target} draft={draft}"


def test_sample_residual_draws_only_where_target_exceeds_draft() -> None:
    """The residual is supported exactly where the target wants more mass."""
    p_t = torch.tensor([[0.6, 0.4, 0.0]])
    p_d = torch.tensor([[0.9, 0.1, 0.0]])
    # Residual is [0, 0.3, 0] -> token 1 with probability 1.
    draws = {int(sample_residual(p_t, p_d)[0]) for _ in range(50)}
    assert draws == {1}


def test_sample_residual_falls_back_when_draft_covers_target() -> None:
    """Zero residual mass must fall back to the target, not divide by zero."""
    p_t = torch.tensor([[0.5, 0.5]])
    p_d = torch.tensor([[0.5, 0.5]])
    draws = [int(sample_residual(p_t, p_d)[0]) for _ in range(200)]
    assert set(draws) == {0, 1}, "should sample the target, not error or collapse"
    assert 60 < sum(1 for d in draws if d == 0) < 140, "roughly balanced"


# Case selection matters more than it looks. When the draft UNDER-weights the
# target's argmax, all residual mass lands on that argmax, and "sample the
# residual" and "just emit the argmax" become the same operation, so such a
# case cannot detect a missing residual correction at all. The first three
# cases below are of that easy kind (kept because they still exercise the
# accept path across very different mismatch shapes); the last two have the
# draft OVER-weighting the argmax, which pushes the residual entirely onto
# other tokens and is what actually discriminates. `test_a_broken_residual_
# would_be_caught` pins this down, and caught the first draft of this file
# using only non-discriminating cases.
@pytest.mark.parametrize(
    ("target", "draft"),
    [
        ([0.7, 0.2, 0.1], [0.1, 0.2, 0.7]),
        ([0.5, 0.25, 0.25], [0.34, 0.33, 0.33]),
        ([0.9, 0.05, 0.05], [0.2, 0.4, 0.4]),
        ([0.4, 0.35, 0.25], [0.9, 0.05, 0.05]),
        ([0.34, 0.33, 0.33], [0.9, 0.05, 0.05]),
    ],
)
def test_sampled_frequencies_match_the_target(target: list[float], draft: list[float]) -> None:
    """Monte Carlo: simulate the full accept-or-resample step and compare frequencies.

    Sample count is chosen so the tolerance sits comfortably outside sampling
    noise: at n=20000 the standard error on a frequency is under 0.004, so a
    0.02 band is ~5 sigma and will not flake, while still catching the kind of
    systematic bias a wrong residual introduces (which the adversarial test
    below shows is far larger than 0.02).
    """
    torch.manual_seed(0)
    n = 20000
    p_t = torch.tensor(target)
    p_d = torch.tensor(draft)

    proposals = torch.multinomial(p_d, num_samples=n, replacement=True)
    accept_prob = torch.clamp(p_t[proposals] / p_d[proposals].clamp_min(1e-8), max=1.0)
    accepted = torch.rand(n) < accept_prob

    counts = torch.zeros(len(target))
    for tok in proposals[accepted].tolist():
        counts[tok] += 1
    n_rejected = int((~accepted).sum())
    if n_rejected:
        resampled = sample_residual(p_t.unsqueeze(0), p_d.unsqueeze(0))
        # sample_residual draws one at a time; loop for the rejected mass.
        for _ in range(n_rejected):
            counts[int(sample_residual(p_t.unsqueeze(0), p_d.unsqueeze(0))[0])] += 1
        del resampled

    freq = counts / counts.sum()
    assert torch.allclose(freq, p_t, atol=0.02), f"got {freq.tolist()} want {target}"


def test_a_broken_residual_would_be_caught() -> None:
    """Sanity-check the test's own power.

    The tempting shortcut on rejection is "just emit the target's argmax".
    This asserts the frequency check is sensitive enough to reject it.

    The draft here deliberately OVER-weights the target's argmax (0.9 vs 0.4),
    which is what makes the case discriminating: the argmax is already fully
    covered, so the true residual sits entirely on the other two tokens and
    the shortcut's answer is never the right one. A draft that under-weights
    the argmax would put all residual mass back on the argmax and make the
    shortcut accidentally correct, which is how the first version of this
    file passed while testing nothing.
    """
    torch.manual_seed(0)
    n = 20000
    p_t = torch.tensor([0.4, 0.35, 0.25])
    p_d = torch.tensor([0.9, 0.05, 0.05])

    proposals = torch.multinomial(p_d, num_samples=n, replacement=True)
    accept_prob = torch.clamp(p_t[proposals] / p_d[proposals].clamp_min(1e-8), max=1.0)
    accepted = torch.rand(n) < accept_prob
    counts = torch.zeros(3)
    for tok in proposals[accepted].tolist():
        counts[tok] += 1
    counts[int(p_t.argmax())] += int((~accepted).sum())  # the broken shortcut

    freq = counts / counts.sum()
    assert not torch.allclose(freq, p_t, atol=0.02), (
        f"the broken variant produced {freq.tolist()}, which the tolerance failed to "
        "distinguish from the target; the frequency test is too weak"
    )


# --- the prefix rule -------------------------------------------------------


def test_acceptance_is_a_prefix_not_per_position() -> None:
    """A rejection truncates: later tokens cannot be accepted on their own merits.

    Position 0's draft is hopeless and position 1's is perfect, but position 1
    was conditioned on a token the target rejected, so it must not count.
    """
    vocab = 4
    target = torch.zeros(1, 2, vocab)
    draft = torch.zeros(1, 2, vocab)
    # Position 0: target wants token 3, draft insists on token 0 -> ratio 0.
    target[0, 0, 3] = 1.0
    draft[0, 0, 0] = 1.0
    # Position 1: both agree on token 1 -> would accept in isolation.
    target[0, 1, 1] = 1.0
    draft[0, 1, 1] = 1.0
    proposed = torch.tensor([[0, 1]])
    assert accepted_prefix_length(target, draft, proposed) == 0


def test_all_accepted_when_draft_matches_target() -> None:
    vocab = 4
    target = torch.zeros(1, 3, vocab)
    draft = torch.zeros(1, 3, vocab)
    for i, tok in enumerate([1, 2, 3]):
        target[0, i, tok] = 1.0
        draft[0, i, tok] = 1.0
    proposed = torch.tensor([[1, 2, 3]])
    assert accepted_prefix_length(target, draft, proposed) == 3


def test_acceptance_rate_approaches_one_minus_total_variation() -> None:
    """Expected single-token acceptance is `1 - TV(p_draft, p_target)`.

    This is the identity underneath the whole Stage C tau discussion: the
    confidence head is trained against exactly this quantity, and it is why
    temperature-1.0 acceptance exceeds greedy argmax-agreement whenever the
    target is uncertain.
    """
    torch.manual_seed(0)
    p_t = torch.tensor([0.5, 0.3, 0.2])
    p_d = torch.tensor([0.2, 0.3, 0.5])
    expected = float(torch.minimum(p_t, p_d).sum())  # 1 - TV
    tv = 0.5 * float((p_t - p_d).abs().sum())
    assert math.isclose(expected, 1.0 - tv, abs_tol=1e-6)

    trials, hits = 4000, 0
    for _ in range(trials):
        tok = int(torch.multinomial(p_d, 1))
        target_1 = torch.zeros(1, 1, 3)
        draft_1 = torch.zeros(1, 1, 3)
        target_1[0, 0] = p_t
        draft_1[0, 0] = p_d
        hits += accepted_prefix_length(target_1, draft_1, torch.tensor([[tok]]))
    assert abs(hits / trials - expected) < 0.03, f"{hits / trials:.3f} vs {expected:.3f}"


# --- greedy stays a special case -------------------------------------------


def test_greedy_probs_are_one_hot_so_acceptance_is_argmax_equality() -> None:
    logits = torch.tensor([[[0.1, 5.0, 0.2]]])
    probs = logits_to_probs(logits, 0.0)
    assert probs[0, 0].tolist() == [0.0, 1.0, 0.0]

    target = logits_to_probs(torch.tensor([[[0.1, 5.0, 0.2]]]), 0.0)
    draft_match = logits_to_probs(torch.tensor([[[0.0, 3.0, 0.0]]]), 0.0)
    draft_miss = logits_to_probs(torch.tensor([[[3.0, 0.0, 0.0]]]), 0.0)
    assert accepted_prefix_length(target, draft_match, torch.tensor([[1]])) == 1
    assert accepted_prefix_length(target, draft_miss, torch.tensor([[0]])) == 0
