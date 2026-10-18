"""The DSpark round math that the batch-1 runner and the scheduler share.

DSpark is a speculative decoder: a small drafter proposes a block of tokens, and
one target forward verifies them and keeps an accepted prefix. Both
`DSparkSpeculativeRunner` and `ContinuousScheduler` call `propose_block` and
`verify_block`, so the fp32-exact batch-1 tests also cover the scheduler's math.
`DSparkSpec` and `SpecDecodeState` serve only the scheduler (ADR-031).
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
    """Per-run counters, named to match the `deepspec` evaluator metrics.

    `acceptance_lengths[i]` is `accepted + 1` for round i, the bonus included (tau).
    `proposal_lengths[i]` is the draft tokens that round i offered after confidence
    truncation. `confidence_observations` pairs the survival probability of each
    offered token, already through the sigmoid, with whether the token survived.
    The scheduler keeps `n_target_forwards` at 0, because all requests share one
    packed forward.
    """

    acceptance_lengths: list[int] = dataclasses.field(default_factory=list)
    proposal_lengths: list[int] = dataclasses.field(default_factory=list)
    accepted_draft_lengths: list[int] = dataclasses.field(default_factory=list)
    n_target_forwards: int = 0
    n_drafter_forwards: int = 0
    confidence_observations: list[tuple[float, bool]] = dataclasses.field(default_factory=list)
    # Rounds that the scheduler shortened to fit `max_tokens`, and their committed tokens.
    # The batch-1 runner never shortens a round, so both stay 0 there.
    room_clamped_rounds: int = 0
    room_clamped_committed: int = 0

    @property
    def mean_acceptance_length(self) -> float:
        """Mean committed tokens per round (tau), bonus included."""
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    @property
    def mean_proposal_length(self) -> float:
        if not self.proposal_lengths:
            return 0.0
        return sum(self.proposal_lengths) / len(self.proposal_lengths)

    def accept_rates_by_position(self, block_size: int) -> list[float | None]:
        """Cumulative survival rate per draft position, as `deepspec` defines it.

        Entry `i` is (rounds that accepted `i`) / (rounds that offered `i`), or `None`
        if no round offered it. A round that accepted `i` also accepted `0..i-1`.
        """
        offered, survived = self._position_counts(block_size)
        return [(survived[p] / offered[p]) if offered[p] else None for p in range(block_size)]

    def _position_counts(self, block_size: int) -> tuple[list[int], list[int]]:
        """Per draft position: the rounds that offered it and the rounds that accepted it."""
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
        return offered, survived

    def summary(self, block_size: int) -> SpecSummary:
        """Reduce these stats to the fixed-size form that outlives the request."""
        offered, survived = self._position_counts(block_size)
        return SpecSummary(
            rounds=len(self.acceptance_lengths),
            committed=sum(self.acceptance_lengths),
            offered=sum(self.proposal_lengths),
            accepted=sum(self.accepted_draft_lengths),
            drafter_forwards=self.n_drafter_forwards,
            offered_by_position=tuple(offered),
            survived_by_position=tuple(survived),
            room_clamped_rounds=self.room_clamped_rounds,
            room_clamped_committed=self.room_clamped_committed,
        )


@dataclasses.dataclass(frozen=True)
class SpecSummary:
    """One request's speculative statistics, as fixed-size sums that a benchmark can pool.

    The scheduler builds it at finish, because the reap drops `DSparkStats`.

    `committed` counts `accepted + 1` per round, as DeepSpec does. It is not the
    client's token count (`GenerationResult.tokens`): the first anchor is in no
    round, and EOS drops tokens.

    With room for n more tokens, the scheduler offers at most n - 1 drafts, so
    `max_tokens` never drops a committed token. Those last rounds commit less and
    lower `tau`. `tau_unclamped` excludes them: compare it with batch-1 and the paper.
    """

    rounds: int
    committed: int
    offered: int
    accepted: int
    drafter_forwards: int
    offered_by_position: tuple[int, ...]
    survived_by_position: tuple[int, ...]
    room_clamped_rounds: int = 0
    room_clamped_committed: int = 0

    @property
    def tau(self) -> float:
        """Committed tokens per round, bonus included."""
        return self.committed / self.rounds if self.rounds else 0.0

    @property
    def tau_unclamped(self) -> float:
        """`tau` over the rounds whose proposal was not shortened for room."""
        rounds = self.rounds - self.room_clamped_rounds
        committed = self.committed - self.room_clamped_committed
        return committed / rounds if rounds else 0.0

    @classmethod
    def pool(cls, summaries: list[SpecSummary]) -> SpecSummary:
        """Sum the summaries before dividing, as DeepSpec does.

        A mean of per-request `tau` values gives a short request the weight of a long one.
        """
        if not summaries:
            raise ValueError("cannot pool an empty list of summaries")
        width = len(summaries[0].offered_by_position)
        if any(len(s.offered_by_position) != width for s in summaries):
            raise ValueError("summaries come from drafters with different block sizes")
        return cls(
            rounds=sum(s.rounds for s in summaries),
            committed=sum(s.committed for s in summaries),
            offered=sum(s.offered for s in summaries),
            accepted=sum(s.accepted for s in summaries),
            drafter_forwards=sum(s.drafter_forwards for s in summaries),
            offered_by_position=tuple(
                sum(s.offered_by_position[p] for s in summaries) for p in range(width)
            ),
            survived_by_position=tuple(
                sum(s.survived_by_position[p] for s in summaries) for p in range(width)
            ),
            room_clamped_rounds=sum(s.room_clamped_rounds for s in summaries),
            room_clamped_committed=sum(s.room_clamped_committed for s in summaries),
        )


def propose_block(
    drafter: Qwen3DSparkDrafter,
    *,
    anchor: int,
    context: torch.Tensor,
    start: int,
    draft_cache: DSparkDraftCache,
    temperature: float,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor]:
    """One drafter round: draft tokens, confidence logits, and draft probs.

    The drafter samples the tokens from the Markov-corrected logits, and the probs
    come from the same logits. Rejection sampling against the raw base logits
    breaks the output distribution with no error.

    On return, `draft_cache` holds exactly `start` positions: this function always
    drops the block K/V (ADR-027: no catch-up step).
    """
    cfg = drafter.cfg
    block_size = cfg.block_size
    device = context.device
    draft_input_ids = torch.full(
        (1, block_size), cfg.mask_token_id, dtype=torch.long, device=device
    )
    draft_input_ids[0, 0] = anchor

    # One span of positions, ctx_len + block_size long: the injected context, then the
    # block. RoPE rotates keys over all of it and queries over the tail (ADR-027 point 4).
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
    """Accept a prefix of `offered` and pick the bonus token that follows it.

    Returns `(accepted, bonus)`. `verify_logits` is the target's
    `(1, 1 + len(offered), vocab)` window over `[anchor, *offered]`.

    Sampling at temperature 0 gives the same result as greedy. Greedy keeps its own
    integer-compare path anyway: it carries the temperature-0 bit-parity contract,
    and multinomial draws would make that contract depend on an RNG.
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
    # The last row predicts the bonus. The rows before it score the offered tokens.
    accepted = accepted_prefix_length(
        target_probs[:, :-1, :], draft_probs[:, : len(offered), :], proposed
    )
    if accepted < len(offered):
        # Rejected: sample the residual, so the output distribution stays the target's.
        bonus_t = sample_residual(target_probs[:, accepted, :], draft_probs[:, accepted, :])
        return accepted, int(bonus_t[0].item())
    return accepted, int(sample_from_probs(target_probs[:, -1:, :])[0, 0].item())


def context_window(
    sink: dict[int, torch.Tensor],
    ordered_taps: list[int],
    lo: int,
    hi: int,
) -> torch.Tensor:
    """One request's rows of a packed tap sink, concatenated in config order.

    `sink[layer]` is `(1, total_q, hidden)` over all requests, and `[lo, hi)` is this
    request's `cu_seqlens_q` window. The order must follow `ordered_taps`: the drafter's
    `fc` learned that order, and another order feeds it permuted features with no error.
    The clone stops a view from pinning the whole packed activation.
    """
    rows = torch.cat([sink[i][:, lo:hi, :] for i in ordered_taps], dim=-1).clone()
    assert rows.shape[1] == hi - lo
    return rows


@dataclasses.dataclass
class DSparkSpec:
    """The drafter and the speculative-decoding settings that the scheduler takes.

    `confidence_threshold` truncates the proposal at the first draft token whose
    predicted survival probability is below it. 0 turns truncation off.

    `max_spec_prompt_tokens` is the longest prompt that speculates. Longer prompts
    use the plain path. The injected context holds one tap row per prompt token,
    outside the block pool, so `_preempt_on_oom` cannot recover from its OOM. A row
    costs `len(target_layer_ids) * hidden_size * dtype_bytes`, so a 1024-token prompt
    costs ~26 MB per request at Qwen3-4B shapes (5 taps, hidden 2560, bf16). After the
    first verify, the buffer holds at most `block_size + 1` rows.
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

        The caller must still clamp `offered` to the request's `max_tokens` room.
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
    """The speculative state that a `RunningRequest` carries across engine steps.

    `start` is the target slot length at which the scheduler drafted this round.
    The drafter's RoPE uses an unsliced cos/sin table, so the scheduler asserts this
    invariant before every draft:

        target_slot_seq_len == draft_cache_len + context_rows        (INV)

    `offered is not None` marks a round in flight. Verify selects by it, not by
    `RequestState`: a request fresh from prefill is DECODING with no proposal.
    `drafted_for_start` lets a retry after a failed forward reuse the proposal. Its
    inputs `(anchor, context, start)` do not change on a retry. The first draft already
    folded the context into `draft_cache`, so a second draft breaks (INV).

    `context_parts` collects the prompt taps across prefill chunks. The first draft
    joins them into `context`. Each verify replaces `context` with at most
    `block_size + 1` committed rows. See ADR-031.
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
    # True when `max_tokens` room shortened the proposal in flight (see `SpecSummary`).
    room_clamped: bool = False
