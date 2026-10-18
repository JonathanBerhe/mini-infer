"""The pieces of a DSpark round that batch-1 and the scheduler share.

`DSparkSpeculativeRunner` (speculative.py) owns the batch-1 LOOP; the
multi-request path in `ContinuousScheduler` interleaves the same round with
other requests' work across engine steps. What must not fork between them is
the round itself: how a block is proposed and how a verify window is judged.
Both live here as functions of explicit state, and both callers delegate, so
the fp32-exact batch-1 tests keep gating the math the scheduler runs.

`DSparkSpec` is the scheduler's constructor-facing bundle (drafter plus the
knobs a serving deployment sets once), and `SpecDecodeState` is the
per-request mutable state a `RunningRequest` carries between engine steps.
Neither is used by the batch-1 runner.
"""

from __future__ import annotations

import dataclasses

import torch

from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.engine.dspark.drafter import Qwen3DSparkDrafter
from mini_infer.engine.dspark.proposal import confident_prefix_length
from mini_infer.engine.dspark.sampling import (
    accepted_prefix_length,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
)


@dataclasses.dataclass
class DSparkStats:
    """Per-run counters, named to match `deepspec`'s evaluator metrics.

    `acceptance_lengths[i]` is round i's committed token count INCLUDING the
    bonus (`accepted + 1`), which is what the reference calls acceptance
    length and reports as tau. `proposal_lengths[i]` is how many draft tokens
    that round actually offered for verification, i.e. after confidence
    truncation, so the two together show what truncation traded away.
    `confidence_observations` pairs each offered token's predicted survival
    PROBABILITY (the sigmoid of the confidence head's logit, which the head
    itself does not apply) with whether it actually survived, which is the raw
    material for a calibration curve. Both recorders sigmoid before storing,
    so a consumer that squashes again gets a silently compressed curve.

    In the multi-request scheduler `n_target_forwards` stays 0: a packed
    forward is shared by every in-flight request, so a per-request count of
    target forwards has no meaning there.
    """

    acceptance_lengths: list[int] = dataclasses.field(default_factory=list)
    proposal_lengths: list[int] = dataclasses.field(default_factory=list)
    accepted_draft_lengths: list[int] = dataclasses.field(default_factory=list)
    n_target_forwards: int = 0
    n_drafter_forwards: int = 0
    confidence_observations: list[tuple[float, bool]] = dataclasses.field(default_factory=list)

    @property
    def mean_acceptance_length(self) -> float:
        """Mean committed tokens per verification round (tau), bonus included."""
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    @property
    def mean_proposal_length(self) -> float:
        if not self.proposal_lengths:
            return 0.0
        return sum(self.proposal_lengths) / len(self.proposal_lengths)

    def accept_rates_by_position(self, block_size: int) -> list[float | None]:
        """Cumulative survival per draft position, `deepspec`'s definition.

        Entry `i` is (rounds whose accepted prefix reached past position `i`) /
        (rounds that offered a token at position `i`). Because acceptance is a
        prefix count, this is the probability that positions `0..i` ALL
        survived, not the conditional probability that `i` survived given
        `i-1` did. `None` where no round offered that position (which
        truncation makes common at deep positions).
        """
        offered = [0] * block_size
        survived = [0] * block_size
        for proposal_len, accepted in zip(
            self.proposal_lengths, self.accepted_draft_lengths, strict=True
        ):
            for pos in range(block_size):
                if proposal_len > pos:
                    offered[pos] += 1
                if accepted > pos:
                    survived[pos] += 1
        return [(survived[p] / offered[p]) if offered[p] else None for p in range(block_size)]


def propose_block(
    drafter: Qwen3DSparkDrafter,
    *,
    anchor: int,
    context: torch.Tensor,
    start: int,
    draft_cache: DSparkDraftCache,
    temperature: float,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor]:
    """One drafter round: candidate tokens, confidence logits, and draft probs.

    The probabilities are the Markov-CORRECTED ones, not the backbone's raw
    base logits, because those are the distribution the tokens were actually
    drawn from. Rejection sampling compares the target against the proposal
    distribution, so using the uncorrected logits here would silently break
    losslessness.

    Leaves `draft_cache` at exactly `start` positions: the forward appends
    K/V for the injected context and the block, and the block's share is
    dropped unconditionally before the outcome is known (ADR-027, no
    catch-up step).
    """
    cfg = drafter.cfg
    block_size = cfg.block_size
    device = context.device
    draft_input_ids = torch.full(
        (1, block_size), cfg.mask_token_id, dtype=torch.long, device=device
    )
    draft_input_ids[0, 0] = anchor

    # One contiguous position span covering the injected context and then
    # the block. The drafter's RoPE rotates keys over the whole span and
    # queries over just its tail, so this slice has to cover both segments
    # (ADR-027 point 4); its length is always ctx_len + block_size.
    cache_len = draft_cache.get_seq_length()
    position_ids = torch.arange(
        cache_len, start + block_size, device=device, dtype=torch.long
    ).unsqueeze(0)

    with torch.inference_mode():
        block_hidden = drafter.forward_backbone(
            noise_embedding=drafter.embed_tokens(draft_input_ids),
            target_hidden_states=context,
            position_ids=position_ids,
            past_key_values=draft_cache,
        )
        # Drop this round's block K/V, keep the accumulated context
        # projections. Unconditional, before we know the outcome.
        draft_cache.truncate_to(start)

        hidden = block_hidden[:, :block_size, :]
        base_logits = drafter.compute_logits(hidden)
        sampled, corrected_logits = drafter.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=draft_input_ids[:, 0],
            temperature=temperature,
        )
        draft_probs = logits_to_probs(corrected_logits, temperature)
        confidence = drafter.predict_confidence_step(
            hidden,
            prev_token_ids=torch.cat([draft_input_ids[:, :1], sampled[:, :-1]], dim=1),
        )
    return sampled[0].tolist(), confidence, draft_probs


def verify_block(
    verify_logits: torch.Tensor,
    *,
    offered: list[int],
    draft_probs: torch.Tensor,
    temperature: float,
) -> tuple[int, int]:
    """Accept a prefix of `offered` and pick the token that follows it.

    `verify_logits` is the target's `(1, 1 + len(offered), vocab)` window over
    `[anchor, *offered]`; position i predicts what follows input i.

    Greedy and temperature > 0 are the same algorithm at different
    temperatures (`logits_to_probs` emits one-hots when greedy, which
    collapses the ratio test to argmax equality), but greedy is kept on an
    explicit integer-comparison path: it is the temperature-0 bit-parity
    contract Stage B and C rest on, and routing it through multinomial
    draws would make a provable property depend on an RNG.
    """
    if temperature < 1e-5:
        target_argmax = verify_logits[0].argmax(dim=-1).tolist()
        accepted = 0
        for i, tok in enumerate(offered):
            if target_argmax[i] == tok:
                accepted += 1
            else:
                break
        return accepted, int(target_argmax[accepted])

    target_probs = logits_to_probs(verify_logits, temperature)
    if not offered:
        return 0, int(sample_from_probs(target_probs[:, -1:, :])[0, 0].item())

    proposed = torch.tensor([offered], device=verify_logits.device, dtype=torch.long)
    # verify_logits has one position per input token; position i predicts
    # what follows input i, so the draft positions are all but the last.
    accepted = accepted_prefix_length(
        target_probs[:, :-1, :], draft_probs[:, : len(offered), :], proposed
    )
    if accepted < len(offered):
        # Rejected at `accepted`: emit from the residual so the overall
        # distribution stays exactly the target's.
        bonus_t = sample_residual(target_probs[:, accepted, :], draft_probs[:, accepted, :])
        return accepted, int(bonus_t[0].item())
    # Everything accepted: the bonus is a free draw from the target's
    # distribution at the position after the last draft token.
    return accepted, int(sample_from_probs(target_probs[:, -1:, :])[0, 0].item())


def context_window(
    sink: dict[int, torch.Tensor],
    ordered_taps: list[int],
    lo: int,
    hi: int,
) -> torch.Tensor:
    """One request's slice of a packed tap sink, concatenated in config order.

    `sink[layer]` is `(1, total_q, hidden)` covering EVERY request's q-tokens;
    `[lo, hi)` comes from that request's `cu_seqlens_q` window. Two things are
    load-bearing: the concat follows `ordered_taps` (the drafter's `fc` was
    trained against exactly that ordering, and a sorted or set-ordered concat
    silently feeds it permuted features), and the result is `.clone()`d (an
    un-cloned slice is a view that pins the whole packed activation for the
    request's lifetime).
    """
    rows = torch.cat([sink[i][:, lo:hi, :] for i in ordered_taps], dim=-1).clone()
    assert rows.shape[1] == hi - lo
    return rows


@dataclasses.dataclass
class DSparkSpec:
    """Constructor-facing bundle for speculative decoding in the scheduler.

    `confidence_threshold` of 0 disables truncation (verify the whole block);
    a positive value cuts the proposal at the first draft token whose
    predicted survival probability falls below it.

    `max_spec_prompt_tokens` caps which prompts speculate at all. The injected
    context buffers one row of tapped hidden states per prompt token OUTSIDE
    the block pool's accounting, so its exhaustion is a plain host/GPU OOM that
    `_preempt_on_oom` cannot recover; longer prompts run the plain path.

    Size it from the arithmetic, not the default: a row costs
    `len(target_layer_ids) * hidden_size * dtype_bytes`, which at Qwen3-4B
    shapes (5 taps, hidden 2560, bf16) is 25.6 KB, so 1024 tokens is ~26 MB per
    speculating request and `max_concurrent` of those is what has to fit
    alongside the weights and the pool. The buffer only reaches that peak
    during prefill; after the first verify it shrinks to at most
    `block_size + 1` rows.
    """

    drafter: Qwen3DSparkDrafter
    confidence_threshold: float = 0.0
    max_spec_prompt_tokens: int = 1024

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.confidence_threshold > 0.0 and self.drafter.confidence_head is None:
            raise ValueError(
                "confidence_threshold > 0 requires a drafter with a confidence head; "
                "this checkpoint has enable_confidence_head=False"
            )
        if self.max_spec_prompt_tokens < 1:
            raise ValueError(
                f"max_spec_prompt_tokens must be positive, got {self.max_spec_prompt_tokens}"
            )

    @property
    def block_size(self) -> int:
        return int(self.drafter.cfg.block_size)

    @property
    def tap_layers(self) -> frozenset[int]:
        return frozenset(self.drafter.cfg.target_layer_ids)

    @property
    def ordered_taps(self) -> list[int]:
        return list(self.drafter.cfg.target_layer_ids)

    def new_draft_cache(self) -> DSparkDraftCache:
        return DSparkDraftCache(self.drafter.cfg.num_hidden_layers)

    def propose(
        self,
        *,
        anchor: int,
        context: torch.Tensor,
        start: int,
        draft_cache: DSparkDraftCache,
        temperature: float,
    ) -> tuple[list[int], torch.Tensor | None, torch.Tensor]:
        """Draft one block and truncate it to the confident prefix.

        Returns `(offered, confidence, draft_probs)` where `offered` already
        reflects confidence truncation; the caller still clamps it to the
        request's remaining `max_tokens` room.
        """
        tokens, confidence, draft_probs = propose_block(
            self.drafter,
            anchor=anchor,
            context=context,
            start=start,
            draft_cache=draft_cache,
            temperature=temperature,
        )
        proposal_len = self.block_size
        if confidence is not None:
            proposal_len = confident_prefix_length(
                confidence, block_size=self.block_size, threshold=self.confidence_threshold
            )
        return tokens[:proposal_len], confidence, draft_probs


@dataclasses.dataclass
class SpecDecodeState:
    """Per-request speculative state a `RunningRequest` carries across steps.

    `start` is the target slot length this round was drafted at, and the
    invariant everything hinges on is

        target_slot_seq_len == draft_cache_len + context_rows        (INV)

    because the drafter's RoPE rotates keys against an unsliced cos/sin table
    (`engine/dspark/attention.py`); the scheduler asserts it before every
    draft rather than trusting the arithmetic.

    `offered is not None` marks a round in flight: the verify phase keys its
    membership off that, never off `RequestState`, because a request that
    just finished prefill is DECODING by verify time while having no
    proposal. `drafted_for_start` makes re-drafting idempotent: a retry after
    a failed forward reuses the proposal instead of drafting (and emitting
    state changes) twice, sound because the proposal is a pure function of
    `(anchor, context, start)`.

    `context_parts` accumulates the prompt's tapped hidden states across
    prefill chunks; the first draft concatenates them into `context`, and
    every verify thereafter REPLACES `context` with the freshly committed
    window, so the buffer shrinks to at most `block_size + 1` rows after the
    first round.
    """

    draft_cache: DSparkDraftCache
    stats: DSparkStats
    context: torch.Tensor | None = None
    context_parts: list[torch.Tensor] = dataclasses.field(default_factory=list)
    anchor: int | None = None
    offered: list[int] | None = None
    draft_probs: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    start: int | None = None
    drafted_for_start: int | None = None
