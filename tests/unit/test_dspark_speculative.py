"""DSpark speculative decoding: the loop's correctness contract, on CPU micro-configs.

The load-bearing test here is `test_greedy_output_matches_target_alone`: greedy
speculative decoding is only useful if it is *invisible*, producing exactly the
tokens the target would have produced on its own, just in fewer target
forwards. Everything else (truncation boundaries, cache bookkeeping, block-pool
hygiene) exists to keep that property true.

The drafter carries random weights, so it proposes mostly-wrong tokens. That is
deliberate: a drafter that always missed and a drafter that always hit would
both hide accept-reject bugs, whereas a random one exercises partial
acceptance, full rejection, and the bonus path across a run. Two extra tests
pin the endpoints explicitly.

Mechanics: `docs/decisions/ADR-027-dspark-drafter-port.md`.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.dspark import (
    DSparkSpeculativeRunner,
    Qwen3DSparkConfig,
    Qwen3DSparkDrafter,
    confident_prefix_length,
)
from mini_infer.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

_VOCAB = 64
_HIDDEN = 32
_TARGET_LAYERS = 4
_TAPS = [1, 3]
_BLOCK = 3
_EOS = 63


def _target_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=40,
        num_hidden_layers=_TARGET_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def _drafter_config(*, confidence: bool = True) -> Qwen3DSparkConfig:
    return Qwen3DSparkConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        target_layer_ids=list(_TAPS),
        mask_token_id=_VOCAB - 2,
        block_size=_BLOCK,
        markov_rank=5,
        enable_confidence_head=confidence,
        confidence_head_with_markov=confidence,
    )


class _StubTokenizer:
    """`DSparkSpeculativeRunner` only reads `eos_token_id` off the tokenizer."""

    eos_token_id = _EOS
    vocab_size = _VOCAB


class _StubRunner:
    """Minimal `ModelRunner` stand-in: a real Qwen3 plus its own block pool.

    Building a real `ModelRunner` would require a checkpoint on disk; the
    runner under test only uses `.tokenizer`, `.block_pool`, and
    `.forward_step_packed`, so this supplies exactly those against a real
    (randomly initialized) Qwen3 so the forward math and cache writes are
    genuine.
    """

    def __init__(self, model: Qwen3ForCausalLM, pool: BlockPool) -> None:
        self._model = model
        self.block_pool = pool
        self.tokenizer = _StubTokenizer()

    def forward_step_packed(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
        *,
        tap_layers: frozenset[int] | None = None,
        hidden_state_sink: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        position_ids_flat: list[int] = []
        for batch_idx in range(cache.batch_size):
            q_len = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
            offset = position_offsets[batch_idx]
            position_ids_flat.extend(range(offset, offset + q_len))
        extra = {}
        if tap_layers is not None:
            extra = {"tap_layers": tap_layers, "hidden_state_sink": hidden_state_sink}
        with torch.inference_mode():
            logits: torch.Tensor = self._model(
                input_ids=torch.tensor([packed_input_ids], dtype=torch.long),
                position_ids=torch.tensor([position_ids_flat], dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32),
                **extra,
            )
        return logits


def _one_hot(tokens: list[int]) -> torch.Tensor:
    """What `logits_to_probs` yields at temperature 0 for these tokens.

    The greedy path never reads draft probs, but returning a real one-hot
    keeps the fake honest if a caller ever runs it at temperature > 0.
    """
    probs = torch.zeros(1, len(tokens), _VOCAB)
    for i, t in enumerate(tokens):
        probs[0, i, t] = 1.0
    return probs


def _build(seed: int = 0, *, confidence: bool = True, num_blocks: int = 64):
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(_target_config()).eval()
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=8,
        num_layers=_TARGET_LAYERS,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
    )
    drafter = Qwen3DSparkDrafter(_drafter_config(confidence=confidence)).eval()
    return _StubRunner(model, pool), drafter, pool


def _target_alone_greedy(runner: _StubRunner, prompt_ids: list[int], max_tokens: int) -> list[int]:
    """Ordinary one-token-at-a-time greedy decode, the oracle."""
    cache = PagedKVCache(runner.block_pool)
    cache.add_request_slot()
    try:
        logits = runner.forward_step_packed(cache, prompt_ids, [0, len(prompt_ids)], [0])
        token = int(logits[0, -1].argmax())
        out = [token]
        pos = len(prompt_ids)
        while len(out) < max_tokens and token != _EOS:
            logits = runner.forward_step_packed(cache, [token], [0, 1], [pos])
            token = int(logits[0, -1].argmax())
            out.append(token)
            pos += 1
        return out[:max_tokens]
    finally:
        cache.free()


# --- confidence truncation -------------------------------------------------


def test_threshold_zero_disables_truncation() -> None:
    logits = torch.tensor([[-9.0, -9.0, -9.0]])
    assert confident_prefix_length(logits, block_size=3, threshold=0.0) == 3


def test_truncates_at_first_position_below_threshold() -> None:
    # sigmoid: [~0.999, ~0.999, ~0.001] -> the third is the first below 0.5.
    logits = torch.tensor([[7.0, 7.0, -7.0]])
    assert confident_prefix_length(logits, block_size=3, threshold=0.5) == 2


def test_all_below_threshold_yields_empty_proposal() -> None:
    logits = torch.tensor([[-7.0, -7.0, -7.0]])
    assert confident_prefix_length(logits, block_size=3, threshold=0.5) == 0


def test_all_above_threshold_keeps_whole_block() -> None:
    logits = torch.tensor([[7.0, 7.0, 7.0]])
    assert confident_prefix_length(logits, block_size=3, threshold=0.5) == 3


# --- the correctness contract ----------------------------------------------


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.9])
def test_greedy_output_matches_target_alone(threshold: float) -> None:
    """Spec-decode must be invisible: same tokens as plain greedy, at every threshold.

    Truncation changes only how many tokens get verified per round, never
    which tokens are committed, so all three thresholds must agree with the
    oracle and with each other.
    """
    runner, drafter, _ = _build()
    prompt = [5, 9, 13, 21]
    expected = _target_alone_greedy(runner, prompt, 12)

    spec = DSparkSpeculativeRunner(runner, drafter, confidence_threshold=threshold)
    got, stats = spec.run_greedy(prompt, 12)

    assert got == expected, f"threshold={threshold}: {got} != {expected}"
    assert stats.n_target_forwards <= 1 + len(expected), "spec should not exceed one forward/token"


def test_empty_proposals_still_make_progress() -> None:
    """A threshold of 1.0 rejects every draft token; decoding must still terminate.

    sigmoid() is strictly < 1, so no position ever clears the bar and every
    round offers zero draft tokens. The loop degenerates to one target forward
    per token, which must still be correct rather than stalling.
    """
    runner, drafter, _ = _build()
    prompt = [7, 11, 3]
    expected = _target_alone_greedy(runner, prompt, 8)

    spec = DSparkSpeculativeRunner(runner, drafter, confidence_threshold=1.0)
    got, stats = spec.run_greedy(prompt, 8)

    assert got == expected
    assert all(n == 0 for n in stats.proposal_lengths), stats.proposal_lengths
    assert all(a == 1 for a in stats.acceptance_lengths), "only the bonus commits"


def test_self_drafting_accepts_everything() -> None:
    """When the drafter's proposals are the target's own argmax, all get accepted.

    Rather than fake a drafter, this replaces the proposal step with the
    target's greedy continuation, so acceptance is 100% by construction. It
    pins the all-accepted path, which is where V1's loop needed its catch-up
    step and this one must not.
    """
    runner, drafter, _ = _build()
    prompt = [2, 4, 8, 16]
    expected = _target_alone_greedy(runner, prompt, 9)

    spec = DSparkSpeculativeRunner(runner, drafter, confidence_threshold=0.0)

    def _perfect_proposal(*, anchor, context, start, draft_cache):  # type: ignore[no-untyped-def]
        del anchor, context, draft_cache
        # The anchor is `expected[start - len(prompt)]` (start advances by the
        # committed count each round), so the tokens that will be accepted are
        # the ones right after it. Pad past the end with a token the target
        # will not emit, so the tail round still exercises a rejection.
        idx = start - len(prompt)
        nxt = expected[idx + 1 : idx + 1 + _BLOCK]
        toks = [*nxt, *([_EOS] * (_BLOCK - len(nxt)))]
        return toks, None, _one_hot(toks)

    spec._propose = _perfect_proposal  # type: ignore[method-assign]
    got, stats = spec.run_greedy(prompt, 9)

    assert got == expected
    # Every offered token accepted: acceptance length is block+1 each round
    # (except a final short/EOS-truncated one).
    assert stats.accepted_draft_lengths[0] == _BLOCK, stats.accepted_draft_lengths
    assert stats.n_target_forwards < 1 + len(expected), "should beat one forward per token"


def test_eos_stops_generation() -> None:
    """Generation halts at EOS, mid-block if necessary.

    Only the target's own argmax ever commits, so a proposed EOS cannot force
    one. Instead this declares whichever token the target genuinely emits at
    position 3 to be EOS, then asserts the run stops exactly there. Because
    that position generally falls inside a block rather than on a round
    boundary, it also covers truncating a partially-committed block.
    """
    runner, drafter, _ = _build()
    prompt = [1, 2, 3]
    natural = _target_alone_greedy(runner, prompt, 10)
    eos_at = 3
    runner.tokenizer.eos_token_id = natural[eos_at]  # type: ignore[misc]

    spec = DSparkSpeculativeRunner(runner, drafter)
    got, _ = spec.run_greedy(prompt, 20)

    assert got == natural[: eos_at + 1], got
    assert got[-1] == natural[eos_at], "EOS itself is emitted, then nothing after"


# --- bookkeeping -----------------------------------------------------------


def test_block_pool_returns_to_free_after_run() -> None:
    runner, drafter, pool = _build()
    free_before = pool.num_free_blocks
    spec = DSparkSpeculativeRunner(runner, drafter, confidence_threshold=0.5)
    spec.run_greedy([3, 6, 9, 12], 10)
    assert pool.num_free_blocks == free_before, "spec-decode leaked blocks"


def test_stats_are_internally_consistent() -> None:
    runner, drafter, _ = _build()
    spec = DSparkSpeculativeRunner(runner, drafter, confidence_threshold=0.5)
    _, stats = spec.run_greedy([4, 8, 12], 12)

    n = len(stats.acceptance_lengths)
    assert len(stats.proposal_lengths) == n
    assert len(stats.accepted_draft_lengths) == n
    for offered, accepted, committed in zip(
        stats.proposal_lengths, stats.accepted_draft_lengths, stats.acceptance_lengths, strict=True
    ):
        assert 0 <= accepted <= offered <= _BLOCK
        assert committed == accepted + 1, "acceptance length counts the bonus"
    assert stats.mean_acceptance_length >= 1.0
    rates = stats.accept_rates_by_position(_BLOCK)
    assert len(rates) == _BLOCK
    for r in rates:
        assert r is None or 0.0 <= r <= 1.0


def test_confidence_observations_pair_with_acceptance() -> None:
    """Each offered token contributes one (probability, survived) observation."""
    runner, drafter, _ = _build()
    spec = DSparkSpeculativeRunner(runner, drafter, confidence_threshold=0.3)
    _, stats = spec.run_greedy([5, 10, 15], 12)

    assert len(stats.confidence_observations) == sum(stats.proposal_lengths)
    for prob, survived in stats.confidence_observations:
        assert 0.0 <= prob <= 1.0
        assert isinstance(survived, bool)


def test_threshold_requires_a_confidence_head() -> None:
    runner, drafter, _ = _build(confidence=False)
    with pytest.raises(ValueError, match="confidence head"):
        DSparkSpeculativeRunner(runner, drafter, confidence_threshold=0.5)


def test_threshold_out_of_range_rejected() -> None:
    runner, drafter, _ = _build()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DSparkSpeculativeRunner(runner, drafter, confidence_threshold=1.5)


# --- temperature > 0 -------------------------------------------------------


def test_sampling_run_completes_and_reports_stats() -> None:
    """Temperature > 0 takes the rejection-sampling path and stays well-formed.

    Token equality against a baseline is NOT the contract here (both sides
    sample), so this checks structure; the distribution property itself is
    tested in `test_dspark_rejection_sampling.py`.
    """
    runner, drafter, _ = _build()
    spec = DSparkSpeculativeRunner(runner, drafter, temperature=1.0)
    got, stats = spec.run_greedy([5, 9, 13, 21], 12)

    assert 0 < len(got) <= 12
    assert all(0 <= t < _VOCAB for t in got)
    for offered, accepted, committed in zip(
        stats.proposal_lengths, stats.accepted_draft_lengths, stats.acceptance_lengths, strict=True
    ):
        assert 0 <= accepted <= offered <= _BLOCK
        assert committed == accepted + 1


def test_sampling_output_varies_across_seeds() -> None:
    """Sampling must actually sample: different seeds give different sequences.

    Guards against a wiring mistake where the temperature path silently falls
    through to argmax, which would look correct in every structural check while
    quietly making the sampler deterministic.
    """
    runner, drafter, _ = _build()
    spec = DSparkSpeculativeRunner(runner, drafter, temperature=1.5)
    outs = []
    for seed in range(6):
        torch.manual_seed(seed)
        got, _ = spec.run_greedy([5, 9, 13, 21], 10)
        outs.append(tuple(got))
    assert len(set(outs)) > 1, f"identical across seeds, sampler may be stuck: {outs[0]}"


def test_negative_temperature_rejected() -> None:
    runner, drafter, _ = _build()
    with pytest.raises(ValueError, match="non-negative"):
        DSparkSpeculativeRunner(runner, drafter, temperature=-0.5)
