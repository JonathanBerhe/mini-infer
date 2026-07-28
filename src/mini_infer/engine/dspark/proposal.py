"""Turning one drafter forward into a verifiable proposal.

The drafter always computes a full `block_size`-token block: its backbone is
one parallel pass, so producing 7 positions costs the same as producing 1
(this is what makes DSpark's long blocks affordable in the first place).
What varies is how many of those tokens are worth *verifying*, and that is a
throughput decision, not a quality one: every proposed token widens the
target's verify forward, and a token the target was always going to reject
is pure waste.

`confident_prefix_length` is the batch-1 form of that decision, mirroring
`deepspec/eval/dspark/draft_ops.py::_confident_prefix_length`: keep the
proposal's prefix up to the first position whose predicted survival
probability falls below a threshold, drop the rest. The paper's full
scheduler is the batch-global version, solving for verification lengths
across all in-flight requests against a profiled hardware curve; that needs
multi-request speculative decoding we don't have yet (ADR-027, Stage D).

Truncation cannot change greedy output. It only shortens a round: the tokens
it removes were never committed, and the target still emits its own argmax at
the first position it disagrees with. So this is free to enable under the
temperature-0 parity contract, and the only thing it trades is tokens-per-round
against wasted verify width.
"""

from __future__ import annotations

import torch


def confident_prefix_length(
    confidence_logits: torch.Tensor, *, block_size: int, threshold: float
) -> int:
    """How many leading draft tokens to actually verify.

    `confidence_logits` is `(1, block_size)` RAW logits (the head applies no
    sigmoid; see `confidence_head.py`), ordered by draft position.

    A non-positive `threshold` disables truncation and returns `block_size`,
    matching the reference's convention of treating `0.0` as "off" rather than
    as "drop everything below p=0". Otherwise the result is the index of the
    first position whose `sigmoid(logit)` is below `threshold`, i.e. the length
    of the confident prefix, which may be 0.
    """
    if threshold <= 0.0:
        return int(block_size)
    below = confidence_logits.sigmoid() < threshold
    hits = torch.nonzero(below[0], as_tuple=False)
    if hits.numel() == 0:
        return int(block_size)
    return int(hits[0].item())
