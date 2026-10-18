"""Multi-request DSpark speculative decoding in `ContinuousScheduler`.

Two suites, matching the plan's gate split
(`docs/plans/dspark-multi-request.md`):

- **Bookkeeping, exact**, against a scripted runner and a scripted drafter:
  committed tokens, cache lengths, the RoPE invariant (INV), emission clamps,
  eligibility downgrades, and mixed batches. The scripts make acceptance
  patterns deterministic, so each rule is pinned by construction.
- **Numerics, exact**, on a real micro-config Qwen3 target plus a real
  randomly initialized micro drafter, CPU fp32: greedy speculative output
  must be token-for-token what the PLAIN scheduler produces, alone and with
  two requests in flight. Correctness is drafter-independent (a wrong draft
  costs a round, never a token), which is why random weights are the right
  fixture: they exercise partial acceptance, full rejection, and the bonus
  path without a checkpoint.

The scripted target maps input token t to argmax f(t), so a proposal that
follows f is accepted in full and any deviation is rejected at exactly the
deviating position. Tap sinks carry each row's POSITION as its value, so a
context buffer's provenance (which rows, in which order) is assertable.

The mid-append OOM tests also cover the plain path (no dspark), because they
need this file's real micro model. They inject the failure inside the
production `append_kv_packed` (`_oom_injection.py`), assert the partial state
was really reached, and then assert recovery with exact slot lengths, since
token equality alone cannot see a hole of unwritten K/V in a random model.
"""

from __future__ import annotations

import queue
from collections.abc import Callable

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.engine.dspark import (
    DSparkSpec,
    Qwen3DSparkConfig,
    Qwen3DSparkDrafter,
)
from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.engine.sampler import SamplingParams
from mini_infer.exceptions import OutOfMemoryError
from mini_infer.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from mini_infer.scheduler import ContinuousScheduler, Request
from mini_infer.scheduler.request_state import RequestState, RunningRequest
from tests.unit._oom_injection import MidAppendOom

_VOCAB = 64
_HIDDEN = 32
_TAPS = [1, 3]
_BLOCK = 3
_EOS = 63
_MASK = 62


def _drafter_config(*, confidence: bool = False) -> Qwen3DSparkConfig:
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
        mask_token_id=_MASK,
        block_size=_BLOCK,
        markov_rank=5,
        enable_confidence_head=confidence,
        confidence_head_with_markov=confidence,
    )


class _CharTokenizer:
    """One id per character, ids in [1, 60) so EOS and mask stay reserved."""

    eos_token_id = _EOS
    vocab_size = _VOCAB

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 59) + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


def _enqueue(
    sched: ContinuousScheduler,
    prompt: str,
    max_tokens: int,
    params: SamplingParams | None = None,
) -> RunningRequest:
    """Put a request straight on the waiting queue (no engine thread involved)."""
    running = RunningRequest(
        request=Request(
            prompt=prompt,
            sampling_params=params if params is not None else SamplingParams(temperature=0.0),
            max_tokens=max_tokens,
        ),
        output_queue=queue.Queue(maxsize=256),
    )
    sched._waiting.put(running)
    return running


def _run_to_done(sched: ContinuousScheduler, reqs: list[RunningRequest], limit: int = 200) -> None:
    """Drive `_step` synchronously until every request finishes.

    Synchronous driving keeps interleavings deterministic, which the exact
    gates need; the engine-thread path is covered by the existing scheduler
    suites.
    """
    for _ in range(limit):
        if all(r.state == RequestState.DONE for r in reqs):
            return
        sched._step()
    raise AssertionError(f"requests not done after {limit} steps")


# ─────────────────────────────────────────────────────────────────────
# Bookkeeping, exact: scripted target + scripted drafter
# ─────────────────────────────────────────────────────────────────────


class _ScriptedRunner:
    """Duck-typed ModelRunner whose target argmax is the pure function `f`.

    The returned logits give row r (holding input token t) argmax `f(t)`, so
    greedy accept/reject outcomes are decidable on paper: a proposal that
    follows `f` from the anchor is accepted in full, and the first deviation
    is rejected exactly there. Tap sinks are filled with each row's POSITION
    so context provenance is checkable. The K/V append is the real one, so
    slot lengths, block allocation, and truncation behave exactly as in
    production.
    """

    # This double implements the tap kwargs, so it declares the capability the
    # scheduler refuses dspark without.
    supports_hidden_state_taps = True

    def __init__(
        self, f: Callable[[int], int], *, num_blocks: int = 64, block_size: int = 16
    ) -> None:
        self.block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            attention_backend="torch",
        )
        self.tokenizer = _CharTokenizer()
        self.f = f
        self.calls: list[dict[str, list[int]]] = []

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
        self.calls.append(
            {
                "packed_input_ids": list(packed_input_ids),
                "cu_seqlens_q": list(cu_seqlens_q),
                "position_offsets": list(position_offsets),
                "tapped": [1 if tap_layers is not None else 0],
            }
        )
        total_q = cu_seqlens_q[-1]
        zeros = torch.zeros((total_q, 1, 8), dtype=torch.float32)
        cache.append_kv_packed(
            zeros, zeros.clone(), torch.tensor(cu_seqlens_q, dtype=torch.int32), 0
        )
        if tap_layers is not None:
            assert hidden_state_sink is not None
            positions: list[int] = []
            for batch_idx in range(cache.batch_size):
                q_len = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
                offset = position_offsets[batch_idx]
                positions.extend(range(offset, offset + q_len))
            row_positions = torch.tensor(positions, dtype=torch.float32)
            for layer in tap_layers:
                hidden_state_sink[layer] = (
                    row_positions.view(1, total_q, 1).expand(1, total_q, _HIDDEN).contiguous()
                )
        logits = torch.full((1, total_q, _VOCAB), -1.0)
        for row, token in enumerate(packed_input_ids):
            logits[0, row, self.f(token)] = 1.0
        return logits


class _ScriptedSpec(DSparkSpec):
    """A `DSparkSpec` whose proposals come from a script, not the drafter.

    Each entry of `script` is one round's offered list, already "confidence
    truncated". The draft cache is padded to `start` before returning, which
    is the postcondition the real `propose_block` leaves (context projections
    accumulate, block K/V dropped), so the scheduler's (INV) assertion checks
    the same arithmetic it would in production.
    """

    def __init__(self, script: list[list[int]]) -> None:
        super().__init__(drafter=Qwen3DSparkDrafter(_drafter_config()).eval())
        self.script = list(script)
        self.propose_calls = 0

    def propose(
        self,
        *,
        anchor: int,
        context: torch.Tensor,
        start: int,
        draft_cache: DSparkDraftCache,
        temperature: float,
    ) -> tuple[list[int], torch.Tensor | None, torch.Tensor]:
        self.propose_calls += 1
        offered = self.script.pop(0) if self.script else []
        pad = start - draft_cache.get_seq_length()
        if pad > 0:
            filler = torch.zeros(1, 1, pad, 8)
            draft_cache.update(0, filler, filler.clone())
        draft_probs = torch.zeros(1, self.block_size, _VOCAB)
        for i, tok in enumerate(offered):
            draft_probs[0, i, tok] = 1.0
        return offered, None, draft_probs


def _next_token(t: int) -> int:
    """The scripted target's argmax: deterministic, EOS-free by construction."""
    return (t + 1) % 60 or 1


def _spec_sched(
    f: Callable[[int], int],
    script: list[list[int]],
    *,
    num_blocks: int = 64,
    block_size: int = 16,
    chunk_size: int = 256,
    max_concurrent: int = 4,
) -> tuple[ContinuousScheduler, _ScriptedRunner, _ScriptedSpec]:
    runner = _ScriptedRunner(f, num_blocks=num_blocks, block_size=block_size)
    spec = _ScriptedSpec(script)
    sched = ContinuousScheduler(
        runner,  # type: ignore[arg-type]
        max_concurrent=max_concurrent,
        chunk_size=chunk_size,
        dspark=spec,
    )
    return sched, runner, spec


def test_init_refuses_prefix_cache() -> None:
    """Cached prompt positions never produce taps, so the combination is
    rejected at construction, the loudest available failure."""
    runner = _ScriptedRunner(_next_token)
    runner.block_pool = BlockPool(
        num_blocks=8,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
        prefix_cache=PrefixCache(block_size=16),
    )
    with pytest.raises(ValueError, match="prefix cache"):
        ContinuousScheduler(runner, dspark=_ScriptedSpec([]))  # type: ignore[arg-type]


def test_init_refuses_a_target_without_hidden_state_taps() -> None:
    """The drafter is fed tapped hidden states, which only some model families
    produce. Without this check the mismatch is a TypeError inside the first
    forward carrying a speculator, which kills the engine thread and every
    other in-flight request with it."""
    runner = _ScriptedRunner(_next_token)
    runner.supports_hidden_state_taps = False  # e.g. a Llama or Gemma target
    with pytest.raises(ValueError, match="tap_layers"):
        ContinuousScheduler(runner, dspark=_ScriptedSpec([]))  # type: ignore[arg-type]
    # Without dspark the same target is served exactly as before.
    assert ContinuousScheduler(runner) is not None  # type: ignore[arg-type]


def test_top_k_and_top_p_requests_run_the_plain_path() -> None:
    """The spec path applies temperature only; filtered sampling must not be
    silently changed, so those requests downgrade to the plain path."""
    sched, _, _spec = _spec_sched(_next_token, [])
    top_k = _enqueue(sched, "ab", 2, SamplingParams(temperature=1.0, top_k=5))
    top_p = _enqueue(sched, "ab", 2, SamplingParams(temperature=1.0, top_p=0.9))
    sched._admit_waiting()
    assert top_k.spec is None
    assert top_p.spec is None


def test_long_prompts_run_the_plain_path() -> None:
    """The tap buffer lives outside the block pool's accounting, so prompts
    over the cap are not eligible to speculate."""
    sched, _, spec = _spec_sched(_next_token, [])
    spec.max_spec_prompt_tokens = 3
    over = _enqueue(sched, "abcd", 4)
    under = _enqueue(sched, "abc", 4)
    sched._admit_waiting()
    assert over.spec is None
    assert under.spec is not None


def test_spec_headroom_is_a_small_constant_not_a_multiplier() -> None:
    """Speculation adds exactly ceil((block+1)/block_size)+1 blocks of
    admission headroom, the intra-step transient of one verify window.

    Pinned as an exact bound from both sides: 11 blocks is enough to admit a
    speculating request whose plain cost is 9, and 10 is not. A headroom
    MULTIPLIED by the block length (the design's rejected first draft) would
    need far more than 11 and would fail the admit side.
    """
    # prompt 4 tokens -> 1 block; decode headroom 8; plain total 9.
    # spec adds ceil((3+1)/16)+1 = 2, so 11 exactly.
    sched, _, _ = _spec_sched(_next_token, [], num_blocks=11)
    fits = _enqueue(sched, "abcd", 2)
    sched._admit_waiting()
    assert fits.state == RequestState.PREFILLING
    assert fits.spec is not None  # speculating, headroom granted

    tight, _, _ = _spec_sched(_next_token, [], num_blocks=10)
    downgraded = _enqueue(tight, "abcd", 2)
    tight._admit_waiting()
    assert downgraded.spec is None  # one block short: speculation dropped


def test_pool_too_small_for_headroom_downgrades_instead_of_rejecting() -> None:
    """Enabling dspark must never turn a request the plain path could serve
    into a cancellation.

    When the pool cannot hold the prompt plus verify-window headroom, the
    request loses speculation and is admitted on the plain path; only a
    request that does not fit even plainly is rejected permanently.
    """
    # Plain cost 9 <= 10 blocks, spec cost 11 > 10.
    plain_runner = _ScriptedRunner(_next_token, num_blocks=10)
    plain = ContinuousScheduler(plain_runner)  # type: ignore[arg-type]
    baseline = _enqueue(plain, "abcd", 2)
    plain._admit_waiting()
    assert baseline.state == RequestState.PREFILLING

    sched, _, _ = _spec_sched(_next_token, [], num_blocks=10)
    same = _enqueue(sched, "abcd", 2)
    sched._admit_waiting()
    assert same.state == RequestState.PREFILLING, "dspark turned a servable request away"
    assert same.finish_reason is None
    assert same.spec is None

    # A request that does not fit even on the plain path is still rejected.
    too_big = _enqueue(sched, "abcdefghijklmnopqrstuvwxyz" * 8, 2)
    sched._admit_waiting()
    assert too_big.finish_reason == "cancelled"


def test_accept_all_round_commits_block_plus_bonus() -> None:
    """A proposal that follows the target's argmax chain is accepted in full;
    the round commits offered + bonus and the slot holds exactly the
    committed prefix afterwards."""
    # Prompt "ab" -> ids [(ord('a')%59)+1=39, 40]. Anchor = f(40)=41.
    # Following f from the anchor: 42, 43, 44; bonus f(44)=45.
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    req = _enqueue(sched, "ab", 16)

    sched._step()  # prefill (tapped)
    assert req.state == RequestState.DECODING
    sched._step()  # anchor=41 emitted, draft, verify [41,42,43,44]

    assert req.tokens_generated == [41, 42, 43, 44, 45]
    spec_state = req.spec
    assert spec_state is not None
    assert spec_state.anchor == 45
    assert spec_state.start == 2 + 4  # prompt + anchor + 3 accepted
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [6]
    assert spec_state.offered is None  # round consumed
    assert spec_state.stats.acceptance_lengths == [4]
    # Refreshed context covers the committed window: positions 2..5.
    assert spec_state.context is not None
    assert spec_state.context[0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_full_rejection_still_commits_the_bonus() -> None:
    """A proposal the target disagrees with at position 0 commits exactly the
    bonus, and the slot rolls back to start + 1."""
    sched, _runner, _spec = _spec_sched(_next_token, [[7, 8, 9]])  # f(41)=42 != 7
    req = _enqueue(sched, "ab", 16)

    sched._step()
    sched._step()

    assert req.tokens_generated == [41, 42]  # anchor + bonus only
    spec_state = req.spec
    assert spec_state is not None
    assert spec_state.stats.accepted_draft_lengths == [0]
    assert sched._batched_cache is not None
    # Slot rolled back: prompt(2) + anchor(1) + 0 accepted + bonus committed
    # but not yet fed, so cache holds 3 positions, not 3 + rejected tail.
    assert sched._batched_cache.seq_lens_list() == [2 + 1]
    assert spec_state.context[0, :, 0].tolist() == [2.0]  # anchor row only


def test_partial_acceptance_commits_prefix_and_rolls_back_the_tail() -> None:
    """The interior case: 0 < accepted < offered.

    Every other scripted round is accept-all or reject-all, so the rollback
    length, the refreshed context width, and (INV) are only pinned at their
    endpoints. Here the target agrees at position 0 and disagrees at 1, so
    exactly one draft token plus the bonus commit, the slot rolls back to drop
    the rejected tail's K/V, and the context covers anchor + 1 accepted row.
    """
    # Anchor 41; f(41)=42 accepted, then 7 != f(42)=43 -> rejected at 1.
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 7, 44]])
    req = _enqueue(sched, "ab", 16)

    sched._step()
    sched._step()

    assert req.tokens_generated == [41, 42, 43]  # anchor + 1 accepted + bonus
    spec_state = req.spec
    assert spec_state is not None
    assert spec_state.stats.accepted_draft_lengths == [1]
    assert spec_state.stats.proposal_lengths == [3]
    # The verify forward wrote 4 positions (anchor + 3 offered); the slot
    # keeps only prompt + anchor + 1 accepted, dropping the rejected tail.
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [2 + 2]
    assert spec_state.start == 4
    # Context is the committed window only: the anchor row and the one
    # accepted row, at positions 2 and 3.
    assert spec_state.context is not None
    assert spec_state.context[0, :, 0].tolist() == [2.0, 3.0]


def test_eos_accepted_mid_block_drops_it_and_the_tail() -> None:
    """EOS accepted as a mid-block draft token stops emission before itself
    and discards the committed tokens behind it.

    The bonus case below only proves EOS is not appended when it is last;
    this is the case where the early break actually throws committed work
    away, which is the scheduler's convention and the opposite of
    `run_greedy`'s.
    """

    def f(t: int) -> int:
        if t == 42:  # the token after the first accepted draft token
            return _EOS
        if t == _EOS:
            return 44
        return _next_token(t)

    # Anchor 41; offered [42, EOS, 44] is accepted in full (f(41)=42,
    # f(42)=EOS, f(EOS)=44), so committed = [42, EOS, 44, bonus].
    sched, _runner, _spec = _spec_sched(f, [[42, _EOS, 44]])
    req = _enqueue(sched, "ab", 16)

    sched._step()
    sched._step()

    assert req.finish_reason == "stop"
    # Stops at 42: EOS itself and everything committed behind it are dropped.
    assert req.tokens_generated == [41, 42]
    assert _EOS not in req.tokens_generated


def test_eos_inside_committed_block_is_dropped_and_finishes() -> None:
    """EOS stops emission BEFORE itself (the scheduler's convention, opposite
    of run_greedy) and the request finishes with "stop"."""

    # f maps 44 -> EOS, so the chain from anchor 41 is 42, 43, 44, then EOS
    # as the bonus of a fully accepted [42, 43, 44] proposal.
    def f(t: int) -> int:
        if t == 44:
            return _EOS
        return _next_token(t)

    sched, runner, _spec = _spec_sched(f, [[42, 43, 44]])
    req = _enqueue(sched, "ab", 16)

    sched._step()
    sched._step()

    assert req.finish_reason == "stop"
    assert req.tokens_generated == [41, 42, 43, 44]  # EOS itself dropped
    sched._step()  # reap
    assert req.spec is None  # state dropped with the request
    assert req not in sched._running
    pool = runner.block_pool
    assert pool.num_free_blocks == pool.num_blocks  # blocks freed


def test_max_tokens_clamps_offered_at_draft_time() -> None:
    """With room for 2 more tokens, only 1 draft token is worth verifying
    (the bonus takes the other slot); the request lands exactly on
    max_tokens and never overshoots."""
    sched, runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    req = _enqueue(sched, "ab", 3)  # anchor + 2 more

    sched._step()
    sched._step()

    assert req.finish_reason == "length"
    assert req.tokens_generated == [41, 42, 43]
    assert len(req.tokens_generated) == req.request.max_tokens
    # The verify window carried anchor + 1 clamped draft token, not the
    # full proposal: visible in the packed widths of the second forward.
    assert runner.calls[1]["cu_seqlens_q"] == [0, 2]


def test_room_of_one_verifies_the_anchor_alone() -> None:
    """With one token of room left, the clamp empties the proposal and the
    round degenerates to ordinary decoding: verify the anchor, commit the
    bonus, finish exactly at max_tokens. The same path serves a confidence
    truncation that cuts the whole block."""
    sched, runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    req = _enqueue(sched, "ab", 2)  # anchor + exactly one more

    sched._step()
    sched._step()

    assert req.finish_reason == "length"
    assert req.tokens_generated == [41, 42]
    # The verify window was the anchor alone: width 1, no drafted positions.
    assert runner.calls[1]["cu_seqlens_q"] == [0, 1]


def test_anchor_eos_finishes_without_drafting() -> None:
    """An anchor that samples to EOS finishes the request before any draft;
    the drafter is never consulted."""

    def f(t: int) -> int:
        if t == 40:  # last prompt token of "ab"
            return _EOS
        return _next_token(t)

    sched, _, spec = _spec_sched(f, [[1, 2, 3]])
    req = _enqueue(sched, "ab", 8)

    sched._step()
    sched._step()

    assert req.finish_reason == "stop"
    assert req.tokens_generated == []
    assert spec.propose_calls == 0


def test_retry_reuses_the_pending_proposal() -> None:
    """Re-running the draft phase for the same start (the OOM-retry shape)
    reuses the proposal instead of drafting again."""
    sched, _, spec = _spec_sched(_next_token, [[42, 43, 44], [99, 99, 99]])
    _enqueue(sched, "ab", 16)

    sched._step()  # prefill
    first = sched._draft_speculators()
    assert spec.propose_calls == 1
    again = sched._draft_speculators()
    assert spec.propose_calls == 1  # reused, not re-drafted
    assert first == again


def test_mixed_batch_keeps_plain_request_untouched() -> None:
    """A speculator and a plain top-k request share one packed forward; the
    plain request decodes one token per step as if alone."""
    sched, runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    spec_req = _enqueue(sched, "ab", 6)
    plain_req = _enqueue(sched, "cd", 3, SamplingParams(temperature=0.0, top_k=5))

    _run_to_done(sched, [spec_req, plain_req])

    assert plain_req.spec is None
    # Plain chain: prompt "cd" -> [41, 42]; f(42)=43 onwards, one per step.
    assert plain_req.tokens_generated == [43, 44, 45]
    assert spec_req.tokens_generated[0] == 41
    assert len(spec_req.tokens_generated) == 6
    # At least one forward carried both a verify window and a plain token.
    widths = [
        [c["cu_seqlens_q"][i + 1] - c["cu_seqlens_q"][i] for i in range(len(c["cu_seqlens_q"]) - 1)]
        for c in runner.calls
    ]
    assert any(len(w) == 2 and max(w) > 1 and min(w) == 1 for w in widths)


def test_speculator_behind_another_request_reads_its_own_windows() -> None:
    """A speculator in slot 1 slices ITS OWN rows out of the packed forward,
    for the injected context and the verify window alike.

    This is the one hazard the numerics gate is constitutionally blind to:
    greedy correctness is drafter-independent, so a context built from a
    NEIGHBOUR's rows changes what gets proposed but never what gets
    committed, and token equality still holds. Only the buffer's VALUES
    expose it, which is why the scripted sink carries row positions.
    """
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    plain = _enqueue(sched, "zzzz", 8, SamplingParams(temperature=0.0, top_k=5))
    sched._step()  # plain prefills alone in slot 0 (4 tokens)
    spec_req = _enqueue(sched, "ab", 16)  # slot 1
    sched._step()  # plain decodes at rows [0,1); spec prefills at rows [1,3)

    spec_state = spec_req.spec
    assert spec_state is not None
    assert len(spec_state.context_parts) == 1
    # The prompt context covers the SPECULATOR's positions 0..1, not the
    # plain request's row that sat ahead of it in the packed forward.
    assert spec_state.context_parts[0][0, :, 0].tolist() == [0.0, 1.0]

    sched._step()  # plain decodes + spec anchors, drafts, verifies at lo=1

    # Accept-all round: refreshed context is the anchor + accepted window at
    # the speculator's own positions 2..5, sliced at its packed offset.
    assert spec_state.context is not None
    assert spec_state.context[0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert spec_req.tokens_generated == [41, 42, 43, 44, 45]
    # And the plain request in slot 0 decoded normally throughout: "z"
    # encodes to 5, so its chain is f(5)=6, f(6)=7.
    assert plain.tokens_generated[:2] == [6, 7]


def test_prefill_completion_is_not_verified() -> None:
    """A request that finishes prefill in the same forward as another's
    verify has no proposal; the verify phase must leave it alone."""
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44], [46, 47, 48]])
    first = _enqueue(sched, "ab", 8)
    sched._step()  # first prefills alone
    second = _enqueue(sched, "cd", 8)
    sched._step()  # first's round shares the forward with second's prefill

    assert second.state == RequestState.DECODING
    assert second.last_logits is not None
    assert second.tokens_generated == []  # nothing emitted for it yet
    spec_state = second.spec
    assert spec_state is not None and spec_state.offered is None
    assert first.tokens_generated == [41, 42, 43, 44, 45]


def test_chunked_prefill_accumulates_full_prompt_context() -> None:
    """Round 1's injected context covers every prompt position, in order,
    across chunk boundaries."""
    sched, _runner, _spec = _spec_sched(_next_token, [[1, 2, 3]], chunk_size=2)
    req = _enqueue(sched, "abcde", 8)  # 5 tokens -> 3 chunks

    sched._step()  # chunk 1
    sched._step()  # chunk 2
    assert req.state == RequestState.CHUNKED_PREFILLING
    sched._step()  # chunk 3 completes prefill
    assert req.state == RequestState.DECODING
    spec_state = req.spec
    assert spec_state is not None
    assert len(spec_state.context_parts) == 3
    sched._step()  # anchor + draft + verify

    assert spec_state.context_parts == []
    stats = spec_state.stats
    assert stats.n_drafter_forwards == 1
    # Context at draft time covered positions 0..4; after the verify it was
    # replaced by the committed window, whose rows sit at positions >= 5.
    assert spec_state.context is not None
    assert spec_state.context[0, 0, 0].item() >= 5.0


def _drain_text(req: RunningRequest) -> str:
    """Everything streamed to the client so far, terminal step excluded."""
    parts: list[str] = []
    while True:
        try:
            step = req.output_queue.get_nowait()
        except queue.Empty:
            return "".join(parts)
        if step.finish_reason is None:
            parts.append(step.text)


def test_speculators_survive_a_mid_append_oom() -> None:
    """Two speculators survive an allocation failure inside the verify append.

    The pool runs dry after slot 0 (speculator A) has advanced its token count
    for the new verify window but before any K/V is written, and while slot 1
    (speculator B) is still unallocated. That partial state used to be fatal:
    the retry read A's inflated length, failed the start-vs-slot assertion,
    and the AssertionError killed the engine thread.

    With the rollback the step replays exactly: A and B keep their rounds
    (no second draft, no second emission), the prefiller C absorbs the
    preemption, and both streams come out as if nothing happened.
    """
    # block_size 2 makes every verify window allocate; chunk_size 2 keeps
    # the long prompt C prefilling long enough to be the preemption victim.
    # Script order is A r1, B r1, A r2, B r2: each follows f, so all accept.
    sched, runner, spec = _spec_sched(
        _next_token,
        [[42, 43, 44], [44, 45, 46], [46, 47, 48], [48, 49, 50]],
        block_size=2,
        chunk_size=2,
    )
    inj = MidAppendOom(runner.block_pool, lambda: sched._batched_cache)
    a = _enqueue(sched, "ab", 16)  # slot 0, ids [39, 40], anchor f(40) = 41
    b = _enqueue(sched, "cd", 16)  # slot 1, ids [41, 42], anchor f(42) = 43
    sched._step()  # A and B prefill
    c = _enqueue(sched, "efghijklmnop", 4)  # slot 2, 6 chunks of 2
    sched._step()  # A and B: anchor + round 1; C: chunk 1

    assert a.spec is not None and b.spec is not None
    assert (a.spec.start, b.spec.start) == (6, 6)
    baseline = inj.arm()
    assert baseline == [6, 6, 2]

    with pytest.raises(OutOfMemoryError):
        sched._step()  # A and B draft round 2; the append fails at B

    # Not vacuous: the production append really left A claiming its window
    # (6 + 4 positions) with nothing written, while B and C never advanced.
    assert inj.fired
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [10, 6, 2]
    calls_after_draft = spec.propose_calls
    assert calls_after_draft == 4

    sched._preempt_on_oom()

    assert c.finish_reason == "cancelled"  # the youngest prefiller absorbs it
    assert c not in sched._running
    assert sched._batched_cache.seq_lens_list() == [6, 6]  # A rolled back

    sched._step()  # the retry replays the failed step

    assert spec.propose_calls == calls_after_draft  # proposals reused, not redrafted
    _run_to_done(sched, [a, b])

    # The greedy chain f(t) = t + 1 from each anchor, max_tokens long, with
    # no token lost or doubled across the failed step.
    assert a.tokens_generated == list(range(41, 41 + 16))
    assert b.tokens_generated == list(range(43, 43 + 16))
    tokenizer = runner.tokenizer
    assert _drain_text(a) == tokenizer.decode(a.tokens_generated)
    assert _drain_text(b) == tokenizer.decode(b.tokens_generated)
    sched._step()  # finished requests free their slots at the next reap
    pool = runner.block_pool
    assert pool.num_free_blocks == pool.num_blocks  # nothing leaked


# ─────────────────────────────────────────────────────────────────────
# Numerics, exact: real micro Qwen3 target + real random micro drafter
# ─────────────────────────────────────────────────────────────────────


def _target_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=40,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


class _MicroRunner:
    """Duck-typed ModelRunner over a real (random) Qwen3, CPU fp32."""

    supports_hidden_state_taps = True

    def __init__(self, model: Qwen3ForCausalLM, pool: BlockPool) -> None:
        self._model = model
        self.block_pool = pool
        self.tokenizer = _CharTokenizer()

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


def _micro_setup(
    seed: int = 0, *, confidence: bool = False, block_size: int = 8
) -> tuple[_MicroRunner, Qwen3DSparkDrafter]:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(_target_config()).eval()
    pool = BlockPool(
        num_blocks=128,
        block_size=block_size,
        num_layers=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
    )
    drafter = Qwen3DSparkDrafter(_drafter_config(confidence=confidence)).eval()
    return _MicroRunner(model, pool), drafter


def _generate(sched: ContinuousScheduler, prompts: list[str], max_tokens: int) -> list[list[int]]:
    reqs = [_enqueue(sched, p, max_tokens) for p in prompts]
    _run_to_done(sched, reqs)
    return [list(r.tokens_generated) for r in reqs]


@pytest.mark.parametrize("confidence_threshold", [0.0, 0.5])
def test_spec_scheduler_matches_plain_scheduler_greedy(confidence_threshold: float) -> None:
    """The greedy invisibility contract, through the scheduler: speculative
    output is token-for-token what the plain path produces, for any drafter,
    with and without confidence truncation."""
    runner, drafter = _micro_setup(confidence=confidence_threshold > 0.0)
    prompts = ["the capital of", "abcdefg", "xy"]

    plain = ContinuousScheduler(runner)  # type: ignore[arg-type]
    baseline = _generate(plain, prompts, max_tokens=12)

    spec = ContinuousScheduler(
        runner,  # type: ignore[arg-type]
        dspark=DSparkSpec(drafter=drafter, confidence_threshold=confidence_threshold),
    )
    speculative = _generate(spec, prompts, max_tokens=12)

    assert speculative == baseline


def test_two_concurrent_spec_requests_match_each_alone() -> None:
    """Two speculators in flight produce exactly what each produces alone:
    CPU fp32 is the strict oracle for the batched-vs-serial contract."""
    runner, drafter = _micro_setup(seed=1)

    def fresh() -> ContinuousScheduler:
        return ContinuousScheduler(
            runner,  # type: ignore[arg-type]
            max_concurrent=2,
            dspark=DSparkSpec(drafter=drafter),
        )

    alone_a = _generate(fresh(), ["hello world"], max_tokens=10)[0]
    alone_b = _generate(fresh(), ["qrstu"], max_tokens=10)[0]
    together = _generate(fresh(), ["hello world", "qrstu"], max_tokens=10)

    assert together == [alone_a, alone_b]


def test_spec_scheduler_with_chunked_prefill_matches_plain() -> None:
    """Chunked context accumulation feeds the drafter the same prompt taps a
    single-chunk prefill would, so output is unchanged by chunking."""
    runner, drafter = _micro_setup(seed=2)
    prompt = "a longer prompt that spans chunks"

    plain = ContinuousScheduler(runner)  # type: ignore[arg-type]
    baseline = _generate(plain, [prompt], max_tokens=8)

    spec_chunked = ContinuousScheduler(
        runner,  # type: ignore[arg-type]
        chunk_size=4,
        dspark=DSparkSpec(drafter=drafter),
    )
    speculative = _generate(spec_chunked, [prompt], max_tokens=8)

    assert speculative == baseline


def test_temperature_sampling_smoke() -> None:
    """Rejection sampling through the scheduler completes and respects
    max_tokens; distribution-level guarantees are pinned algebraically in
    test_dspark_rejection_sampling.py."""
    runner, drafter = _micro_setup(seed=3)
    sched = ContinuousScheduler(runner, dspark=DSparkSpec(drafter=drafter))  # type: ignore[arg-type]
    torch.manual_seed(7)
    req = _enqueue(sched, "sample me", 8, SamplingParams(temperature=0.8))
    _run_to_done(sched, [req])

    assert req.finish_reason in ("stop", "length")
    assert 0 < len(req.tokens_generated) <= 8


def _oom_mid_run(
    sched: ContinuousScheduler,
    runner: _MicroRunner,
    survivors: list[RunningRequest],
    victim_prompt: str,
) -> RunningRequest:
    """Drive `sched` into one mid-append OOM and recover, like `_engine_loop`.

    Steps until every survivor decodes, admits a long-prompt victim, then
    arms the injection and runs the step that fails inside the append.
    Asserts the partial state was really reached before recovering, so a
    test built on this cannot pass vacuously.
    """
    inj = MidAppendOom(runner.block_pool, lambda: sched._batched_cache)
    for _ in range(20):
        if all(r.state == RequestState.DECODING for r in survivors):
            break
        sched._step()
    victim = _enqueue(sched, victim_prompt, 4)
    sched._step()  # victim admitted and starts its chunked prefill
    assert all(r.state == RequestState.DECODING for r in survivors), "survivor finished early"
    assert victim.state == RequestState.CHUNKED_PREFILLING

    baseline = inj.arm()
    slot_of = {id(r): r.batch_idx for r in survivors}
    with pytest.raises(OutOfMemoryError):
        sched._step()
    assert inj.fired
    assert sched._batched_cache is not None
    lens = sched._batched_cache.seq_lens_list()
    assert lens[0] > baseline[0], "the append never left slot 0 advanced"

    sched._preempt_on_oom()
    assert victim.finish_reason == "cancelled"
    # The exact check. Token equality alone cannot carry this on the plain
    # path: a micro model with random weights can attend over a hole of
    # unwritten K/V and still pick the same argmax (measured: a missing
    # rollback passed the token assertion). Every survivor must be back at
    # precisely the length it had before the failed forward.
    after = sched._batched_cache.seq_lens_list()
    for r in survivors:
        pre = slot_of[id(r)]
        assert pre is not None and r.batch_idx is not None
        assert after[r.batch_idx] == baseline[pre], (
            f"survivor in slot {r.batch_idx} resumed at length {after[r.batch_idx]}, "
            f"expected {baseline[pre]}"
        )
    return victim


def test_two_speculators_survive_mid_append_oom_token_for_token() -> None:
    """End to end on a real micro model: speculators that live through an
    allocation failure inside the verify append produce exactly what each
    produces alone.

    The token comparison is NOT what catches a missing rollback. The exact
    slot-length assertion in `_oom_mid_run` fails first, and behind it the
    start-vs-slot assertion in `_draft_speculators` fails before attention
    ever runs. The tokens then confirm the second half of the claim: that
    replaying the step with the reused proposals changes no output."""
    runner, drafter = _micro_setup(seed=4, block_size=2)

    def fresh() -> ContinuousScheduler:
        return ContinuousScheduler(
            runner,  # type: ignore[arg-type]
            max_concurrent=3,
            chunk_size=4,
            dspark=DSparkSpec(drafter=drafter),
        )

    alone_a = _generate(fresh(), ["hello world"], max_tokens=12)[0]
    alone_b = _generate(fresh(), ["qrstu"], max_tokens=12)[0]

    sched = fresh()
    a = _enqueue(sched, "hello world", 12)
    b = _enqueue(sched, "qrstu", 12)
    _oom_mid_run(sched, runner, [a, b], "a much longer prompt that keeps prefilling")
    _run_to_done(sched, [a, b])

    assert [a.tokens_generated, b.tokens_generated] == [alone_a, alone_b]


def test_plain_decoders_survive_mid_append_oom_token_for_token() -> None:
    """The same guarantee on the path every model family uses (no dspark).

    Before the rollback existed, a plain decoder in slot 0 resumed at its
    inflated length after the OOM: its next token's K/V landed past a hole of
    never-written positions, silently, with no assertion to catch it.
    """
    runner, _ = _micro_setup(seed=5, block_size=2)

    def fresh() -> ContinuousScheduler:
        return ContinuousScheduler(runner, max_concurrent=3, chunk_size=4)  # type: ignore[arg-type]

    alone_a = _generate(fresh(), ["hello world"], max_tokens=12)[0]
    alone_b = _generate(fresh(), ["qrstu"], max_tokens=12)[0]

    sched = fresh()
    a = _enqueue(sched, "hello world", 12)
    b = _enqueue(sched, "qrstu", 12)
    _oom_mid_run(sched, runner, [a, b], "a much longer prompt that keeps prefilling")
    _run_to_done(sched, [a, b])

    assert [a.tokens_generated, b.tokens_generated] == [alone_a, alone_b]


def test_preemption_after_a_successful_forward_undoes_nothing() -> None:
    """The saved lengths live only while a forward is in flight.

    `_preempt_on_oom` also serves callers that did not just fail an append.
    If the snapshot outlived a forward that landed, a later rollback would
    truncate K/V that really was written. Here a step completes, then a
    preemption runs: the survivor's length must be untouched.
    """
    sched, _runner, _spec = _spec_sched(_next_token, [], block_size=2)
    a = _enqueue(sched, "ab", 8)
    b = _enqueue(sched, "cd", 8)
    sched._step()  # both prefill
    sched._step()  # both decode a round
    assert sched._batched_cache is not None
    before = sched._batched_cache.seq_lens_list()
    assert sched._pre_forward_seq_lens is None

    sched._preempt_on_oom()  # cancels b, the youngest (no prefillers left)

    assert b.finish_reason == "cancelled"
    assert a.batch_idx == 0
    assert sched._batched_cache.seq_lens_list() == before[:1]


def _plain_prefill_pair(
    block_size: int = 2, chunk_size: int = 4
) -> tuple[ContinuousScheduler, _ScriptedRunner, RunningRequest, RunningRequest]:
    """Two plain prefillers mid-prompt: an older survivor in slot 0 and a
    younger victim in slot 1. Each chunk of 4 needs 2 blocks of size 2."""
    runner = _ScriptedRunner(_next_token, block_size=block_size)
    sched = ContinuousScheduler(runner, chunk_size=chunk_size)  # type: ignore[arg-type]
    old = _enqueue(sched, "abcdefghij", 6)  # 10 prompt tokens: chunks 4, 4, 2
    young = _enqueue(sched, "klmnopqrstuvwxyz", 6)  # 16 prompt tokens
    sched._step()  # both prefill chunk 1
    assert old.state == young.state == RequestState.CHUNKED_PREFILLING
    return sched, runner, old, young


def test_surviving_prefiller_is_rolled_back_after_mid_append_oom() -> None:
    """A request still reading its prompt survives a failure in a later slot.

    Its count advanced over the chunk it was given, with nothing written.
    Every earlier OOM test had only the victim prefilling, so a rollback that
    skipped prefilling slots passed them all. Without the rollback, the retry
    stores the chunk past a hole; the prefill-vs-slot assertion now makes
    that loud instead of silent.
    """
    sched, runner, old, young = _plain_prefill_pair()
    inj = MidAppendOom(runner.block_pool, lambda: sched._batched_cache)
    assert inj.arm() == [4, 4]

    with pytest.raises(OutOfMemoryError):
        sched._step()  # old's chunk 2 claims positions 4..7; young's alloc raises

    assert inj.fired
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [8, 4]  # old claims, holds nothing

    sched._preempt_on_oom()

    assert young.finish_reason == "cancelled"  # the youngest prefiller
    assert sched._batched_cache.seq_lens_list() == [4]
    assert old.tokens_prefilled == 4  # counter and slot agree again

    _run_to_done(sched, [old])
    # "j" encodes to (106 % 59) + 1 = 48, so greedy follows f from 49.
    assert old.tokens_generated == list(range(49, 49 + 6))


def test_slot_that_raised_keeps_its_claimed_blocks_without_leaking() -> None:
    """The slot whose allocation fails can keep blocks its length does not need.

    `append_kv_packed` adds each new block to the slot before it advances the
    slot's count. A raise on the slot's SECOND allocation leaves it with one
    extra block and an unchanged length, which `truncate_to` does not touch.
    The retry feeds the same chunk, so it reuses that block, and once the
    request finishes every block is back in the pool.
    """
    sched, runner, old, young = _plain_prefill_pair()
    inj = MidAppendOom(runner.block_pool, lambda: sched._batched_cache)
    inj.arm(slot=0)
    assert sched._batched_cache is not None
    blocks_before = len(sched._batched_cache._block_ids[0])

    with pytest.raises(OutOfMemoryError):
        sched._step()  # old's chunk 2 needs 2 blocks; the second one raises

    assert inj.fired
    assert sched._batched_cache.seq_lens_list() == [4, 4]  # nobody advanced
    assert len(sched._batched_cache._block_ids[0]) == blocks_before + 1  # the kept claim

    sched._preempt_on_oom()
    assert young.finish_reason == "cancelled"

    _run_to_done(sched, [old])
    assert old.tokens_generated == list(range(49, 49 + 6))
    sched._step()  # the reap frees the finished request's slot
    pool = runner.block_pool
    assert pool.num_free_blocks == pool.num_blocks
