"""Multi-request DSpark speculative decoding in `ContinuousScheduler`.

ADR-031 has the design, the terms, and the invariant (INV): the target slot
length equals the draft cache length plus the context rows. The bonus is the
token that the target picks after the last accepted draft token.

Bookkeeping tests are exact against a scripted runner and drafter. The
scripted target maps token t to argmax f(t). A proposal that follows f is
accepted in full, and the first deviation is rejected. Each scripted tap row
holds its position, so a test can assert which rows a context holds.

Numerics tests are exact on a real micro Qwen3 target and drafter with random
weights, CPU fp32. A wrong draft costs a round, never a token. So random
weights are enough, and speculative output must equal plain output.

The mid-append OOM tests also cover the plain path. They inject the failure
inside the real `append_kv_packed` and assert exact slot lengths, because
token equality cannot see a hole of unwritten K/V in a random model.
"""

from __future__ import annotations

import queue
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.engine.dspark import (
    DSparkSpec,
    Qwen3DSparkDrafter,
    SpecSummary,
)
from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.engine.sampler import SamplingParams
from mini_infer.exceptions import OutOfMemoryError
from mini_infer.scheduler import ContinuousScheduler, Request
from mini_infer.scheduler.request_state import (
    GenerationStep,
    RequestHandle,
    RequestState,
    RunningRequest,
)
from tests.unit._dspark_doubles import EOS as _EOS
from tests.unit._dspark_doubles import HIDDEN as _HIDDEN
from tests.unit._dspark_doubles import VOCAB as _VOCAB
from tests.unit._dspark_doubles import CharTokenizer as _CharTokenizer
from tests.unit._dspark_doubles import MicroRunner as _MicroRunner
from tests.unit._dspark_doubles import drafter_config as _drafter_config
from tests.unit._dspark_doubles import micro_setup as _micro_setup
from tests.unit._oom_injection import MidAppendOom


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
    """Drive `_step` with no engine thread, for a deterministic order, until all finish."""
    for _ in range(limit):
        if all(r.state == RequestState.DONE for r in reqs):
            return
        sched._step()
    raise AssertionError(f"requests not done after {limit} steps")


# ── Bookkeeping, exact: scripted target + scripted drafter ──


class _ScriptedRunner:
    """ModelRunner double: the argmax for token t is `f(t)`, and each tap row holds its position.

    The K/V append is real, so slot lengths, blocks, and truncation behave as in production.
    """

    # The scheduler refuses dspark without this flag.
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
    """A `DSparkSpec` whose proposals come from `script`, one offered list per round.

    `propose` leaves the draft cache at `start`, as the real `propose_block` does, so the
    (INV) assertion runs the same arithmetic as in production.
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
    """The argmax of the scripted target: deterministic and never EOS."""
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
    """Cached prompt positions produce no taps, so the constructor refuses a prefix cache."""
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
    """The constructor refuses a target without taps. Otherwise the first spec forward raises a
    TypeError that kills the engine thread and every in-flight request."""
    runner = _ScriptedRunner(_next_token)
    runner.supports_hidden_state_taps = False  # e.g. a Llama or Gemma target
    with pytest.raises(ValueError, match="tap_layers"):
        ContinuousScheduler(runner, dspark=_ScriptedSpec([]))  # type: ignore[arg-type]
    assert ContinuousScheduler(runner) is not None  # type: ignore[arg-type]


def test_top_k_and_top_p_requests_run_the_plain_path() -> None:
    """The spec path applies only temperature, so top-k and top-p requests run plain."""
    sched, _, _spec = _spec_sched(_next_token, [])
    top_k = _enqueue(sched, "ab", 2, SamplingParams(temperature=1.0, top_k=5))
    top_p = _enqueue(sched, "ab", 2, SamplingParams(temperature=1.0, top_p=0.9))
    sched._admit_waiting()
    assert top_k.spec is None
    assert top_p.spec is None


def test_long_prompts_run_the_plain_path() -> None:
    """The tap buffer is outside the block pool, so long prompts run the plain path."""
    sched, _, spec = _spec_sched(_next_token, [])
    spec.max_spec_prompt_tokens = 3
    over = _enqueue(sched, "abcd", 4)
    under = _enqueue(sched, "abc", 4)
    sched._admit_waiting()
    assert over.spec is None
    assert under.spec is not None


def test_spec_headroom_is_a_small_constant_not_a_multiplier() -> None:
    """Speculation adds ceil((B + 1) / block_size) + 1 blocks (B = DSpark block length), room
    for one verify window. The bound is exact: 11 blocks admit, 10 do not. A headroom
    multiplied by B fails the admit side."""
    # Prompt of 4 tokens: 1 block. Decode headroom: 8. Plain total: 9.
    # spec adds ceil((3+1)/16)+1 = 2, so 11 exactly.
    sched, _, _ = _spec_sched(_next_token, [], num_blocks=11)
    fits = _enqueue(sched, "abcd", 2)
    sched._admit_waiting()
    assert fits.state == RequestState.PREFILLING
    assert fits.spec is not None

    tight, _, _ = _spec_sched(_next_token, [], num_blocks=10)
    downgraded = _enqueue(tight, "abcd", 2)
    tight._admit_waiting()
    assert downgraded.spec is None


def test_pool_too_small_for_headroom_downgrades_instead_of_rejecting() -> None:
    """A request without room for the spec headroom runs plain. Only a request that does not
    fit the plain path is rejected."""
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

    too_big = _enqueue(sched, "abcdefghijklmnopqrstuvwxyz" * 8, 2)
    sched._admit_waiting()
    assert too_big.finish_reason == "cancelled"


def test_accept_all_round_commits_block_plus_bonus() -> None:
    """A proposal that follows the target argmax commits in full, plus the bonus. The slot then
    holds exactly the committed prefix."""
    # Prompt "ab" -> ids [(ord('a')%59)+1=39, 40]. Anchor = f(40)=41.
    # Following f from the anchor: 42, 43, 44, then bonus f(44)=45.
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
    assert spec_state.offered is None
    assert spec_state.stats.acceptance_lengths == [4]
    # Refreshed context covers the committed window: positions 2..5.
    assert spec_state.context is not None
    assert spec_state.context[0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_full_rejection_still_commits_the_bonus() -> None:
    """A round rejected at position 0 commits only the bonus. The slot rolls back to start + 1."""
    sched, _runner, _spec = _spec_sched(_next_token, [[7, 8, 9]])  # f(41)=42 != 7
    req = _enqueue(sched, "ab", 16)

    sched._step()
    sched._step()

    assert req.tokens_generated == [41, 42]  # anchor + bonus only
    spec_state = req.spec
    assert spec_state is not None
    assert spec_state.stats.accepted_draft_lengths == [0]
    assert sched._batched_cache is not None
    # Prompt (2) + anchor (1). The bonus is committed but not yet fed.
    assert sched._batched_cache.seq_lens_list() == [2 + 1]
    assert spec_state.context[0, :, 0].tolist() == [2.0]  # anchor row only


def test_partial_acceptance_commits_prefix_and_rolls_back_the_tail() -> None:
    """The interior case, 0 < accepted < offered. One draft token and the bonus commit, the slot
    drops the rejected tail, and the context covers the anchor and accepted rows."""
    # Anchor 41. f(41)=42 is accepted, then 7 != f(42)=43 is rejected at 1.
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 7, 44]])
    req = _enqueue(sched, "ab", 16)

    sched._step()
    sched._step()

    assert req.tokens_generated == [41, 42, 43]  # anchor + 1 accepted + bonus
    spec_state = req.spec
    assert spec_state is not None
    assert spec_state.stats.accepted_draft_lengths == [1]
    assert spec_state.stats.proposal_lengths == [3]
    # The verify wrote anchor + 3 offered. The slot keeps prompt + anchor + 1 accepted.
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [2 + 2]
    assert spec_state.start == 4
    assert spec_state.context is not None
    assert spec_state.context[0, :, 0].tolist() == [2.0, 3.0]


def test_eos_accepted_mid_block_drops_it_and_the_tail() -> None:
    """An accepted EOS mid-block stops emission before EOS and drops the committed tokens after
    it. The scheduler does not emit EOS, but `run_greedy` does."""

    def f(t: int) -> int:
        if t == 42:  # f(42) = EOS, so the target accepts EOS after draft token 42
            return _EOS
        if t == _EOS:
            return 44
        return _next_token(t)

    # Anchor 41. Offered [42, EOS, 44] is accepted in full (f(41)=42,
    # f(42)=EOS, f(EOS)=44), so committed = [42, EOS, 44, bonus].
    sched, _runner, _spec = _spec_sched(f, [[42, _EOS, 44]])
    req = _enqueue(sched, "ab", 16)

    sched._step()
    sched._step()

    assert req.finish_reason == "stop"
    assert req.tokens_generated == [41, 42]
    assert _EOS not in req.tokens_generated


def test_eos_inside_committed_block_is_dropped_and_finishes() -> None:
    """EOS as the bonus is not emitted, and the reap frees the spec state and the blocks."""

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
    assert req.spec is None
    assert req not in sched._running
    pool = runner.block_pool
    assert pool.num_free_blocks == pool.num_blocks


def test_max_tokens_clamps_offered_at_draft_time() -> None:
    """With room for 2 tokens, the round verifies 1 draft token, and the bonus fills the other."""
    sched, runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    req = _enqueue(sched, "ab", 3)  # anchor + 2 more

    sched._step()
    sched._step()

    assert req.finish_reason == "length"
    assert req.tokens_generated == [41, 42, 43]
    assert len(req.tokens_generated) == req.request.max_tokens
    # The verify window is the anchor plus 1 draft token, not the full proposal.
    assert runner.calls[1]["cu_seqlens_q"] == [0, 2]


def test_room_of_one_verifies_the_anchor_alone() -> None:
    """With room for one token, the clamp empties the proposal, so the round is plain decoding.
    A confidence truncation of the whole block takes the same path."""
    sched, runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    req = _enqueue(sched, "ab", 2)  # anchor + exactly one more

    sched._step()
    sched._step()

    assert req.finish_reason == "length"
    assert req.tokens_generated == [41, 42]
    assert runner.calls[1]["cu_seqlens_q"] == [0, 1]


def test_anchor_eos_finishes_without_drafting() -> None:
    """An anchor that samples to EOS finishes the request before the drafter runs."""

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
    """A second draft phase at the same start (an OOM retry) reuses the proposal."""
    sched, _, spec = _spec_sched(_next_token, [[42, 43, 44], [99, 99, 99]])
    _enqueue(sched, "ab", 16)

    sched._step()  # prefill
    first = sched._draft_speculators()
    assert spec.propose_calls == 1
    again = sched._draft_speculators()
    assert spec.propose_calls == 1
    assert first == again


def test_mixed_batch_keeps_plain_request_untouched() -> None:
    """A plain request that shares packed forwards with a speculator decodes as if alone."""
    sched, runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    spec_req = _enqueue(sched, "ab", 6)
    plain_req = _enqueue(sched, "cd", 3, SamplingParams(temperature=0.0, top_k=5))

    _run_to_done(sched, [spec_req, plain_req])

    assert plain_req.spec is None
    # Plain chain: prompt "cd" -> [41, 42], then f(42)=43 onward, one per step.
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
    """A speculator in slot 1 takes its own rows of the packed forward, for context and verify.

    Greedy output does not depend on the drafter, so only the tap values can show a slice at
    the wrong offset."""
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    plain = _enqueue(sched, "zzzz", 8, SamplingParams(temperature=0.0, top_k=5))
    sched._step()  # plain prefills alone in slot 0 (4 tokens)
    spec_req = _enqueue(sched, "ab", 16)  # slot 1
    sched._step()  # plain decodes at rows [0,1), spec prefills at rows [1,3)

    spec_state = spec_req.spec
    assert spec_state is not None
    assert len(spec_state.context_parts) == 1
    # Positions 0..1 of the speculator, not the plain row ahead of it.
    assert spec_state.context_parts[0][0, :, 0].tolist() == [0.0, 1.0]

    sched._step()  # plain decodes + spec anchors, drafts, verifies at lo=1

    # Accept-all round: anchor + accepted window, positions 2..5 of the speculator.
    assert spec_state.context is not None
    assert spec_state.context[0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert spec_req.tokens_generated == [41, 42, 43, 44, 45]
    # "z" encodes to 5, so the plain chain is f(5)=6, f(6)=7.
    assert plain.tokens_generated[:2] == [6, 7]


def test_prefill_completion_is_not_verified() -> None:
    """A request that finishes prefill in a verify forward has no proposal, so verify skips it."""
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44], [46, 47, 48]])
    first = _enqueue(sched, "ab", 8)
    sched._step()  # first prefills alone
    second = _enqueue(sched, "cd", 8)
    sched._step()  # first's round shares the forward with second's prefill

    assert second.state == RequestState.DECODING
    assert second.last_logits is not None
    assert second.tokens_generated == []
    spec_state = second.spec
    assert spec_state is not None and spec_state.offered is None
    assert first.tokens_generated == [41, 42, 43, 44, 45]


def test_chunked_prefill_accumulates_full_prompt_context() -> None:
    """The round-1 context covers every prompt position, in order, across chunk boundaries."""
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
    # After the verify, the committed window (positions >= 5) replaced the prompt context.
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

    The pool runs dry after slot 0 (A) advanced its length, before any K/V is written. Without
    the rollback, the retry fails the start-vs-slot assertion and kills the engine thread.
    With it, A and B keep their rounds and replay exactly, and C is the preemption victim."""
    # block_size 2: every verify window allocates. chunk_size 2: C stays in prefill as victim.
    # Script order: A r1, B r1, A r2, B r2. Each follows f, so all accept.
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
    sched._step()  # A and B: anchor + round 1. C: chunk 1.

    assert a.spec is not None and b.spec is not None
    assert (a.spec.start, b.spec.start) == (6, 6)
    baseline = inj.arm()
    assert baseline == [6, 6, 2]

    with pytest.raises(OutOfMemoryError):
        sched._step()  # A and B draft round 2. The append fails at B.

    # Not vacuous: A claims its window (6 + 4) with nothing written. B and C did not advance.
    assert inj.fired
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [10, 6, 2]
    calls_after_draft = spec.propose_calls
    assert calls_after_draft == 4

    sched._preempt_on_oom()

    assert c.finish_reason == "cancelled"  # the youngest prefiller is the victim
    assert c not in sched._running
    assert sched._batched_cache.seq_lens_list() == [6, 6]  # A rolled back

    sched._step()  # the retry replays the failed step

    assert spec.propose_calls == calls_after_draft  # proposals reused, not redrafted
    _run_to_done(sched, [a, b])

    # Greedy chain f(t) = t + 1 from each anchor: no token lost or doubled.
    assert a.tokens_generated == list(range(41, 41 + 16))
    assert b.tokens_generated == list(range(43, 43 + 16))
    tokenizer = runner.tokenizer
    assert _drain_text(a) == tokenizer.decode(a.tokens_generated)
    assert _drain_text(b) == tokenizer.decode(b.tokens_generated)
    sched._step()  # finished requests free their slots at the next reap
    pool = runner.block_pool
    assert pool.num_free_blocks == pool.num_blocks  # nothing leaked


# ── Numerics, exact: real micro Qwen3 target + real random micro drafter ──


def _generate(sched: ContinuousScheduler, prompts: list[str], max_tokens: int) -> list[list[int]]:
    reqs = [_enqueue(sched, p, max_tokens) for p in prompts]
    _run_to_done(sched, reqs)
    return [list(r.tokens_generated) for r in reqs]


@pytest.mark.parametrize("confidence_threshold", [0.0, 0.5])
def test_spec_scheduler_matches_plain_scheduler_greedy(confidence_threshold: float) -> None:
    """Greedy spec output equals plain output, with and without confidence truncation."""
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
    """Two speculators in flight produce exactly what each produces alone (CPU fp32)."""
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
    """Chunked prefill gives the drafter the same prompt taps, so the output does not change."""
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
    """Rejection sampling completes and respects max_tokens. The distribution tests are in
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
    """Drive `sched` into one mid-append OOM and recover, as `_engine_loop` does.

    It asserts that the partial state occurred, so a test that uses it cannot pass vacuously.
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
    # A random model can attend over unwritten K/V and keep the same argmax. Measured on the
    # plain path: a run with no rollback passed the token assertion. So check pre-forward lengths.
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
    """Speculators that survive a mid-append OOM produce exactly what each produces alone.

    The slot-length check in `_oom_mid_run` catches a missing rollback. The tokens show that a
    replay with the reused proposals changes no output."""
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
    """The same guarantee without dspark. Without the rollback, a plain decoder silently writes
    its next K/V past a hole, which the slot-length check in `_oom_mid_run` catches."""
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
    """An OOM outside the append also runs `_preempt_on_oom`. So saved slot lengths live only
    while a forward is in flight, or a rollback truncates K/V that a completed forward wrote."""
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
    """A prefiller survives a failure in a later slot after it advanced over an unwritten chunk.

    Only this test catches a rollback that skips prefilling slots. With such a rollback, the
    retry fails the prefill-vs-slot assertion."""
    sched, runner, old, young = _plain_prefill_pair()
    inj = MidAppendOom(runner.block_pool, lambda: sched._batched_cache)
    assert inj.arm() == [4, 4]

    with pytest.raises(OutOfMemoryError):
        sched._step()  # old's chunk 2 claims positions 4..7, then young's allocate raises

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
    """A raise on the second allocation of a slot leaves one extra block at the same length.

    `append_kv_packed` adds each block before it advances the length. `truncate_to` keeps the
    extra block, and the retry reuses it, so nothing leaks."""
    sched, runner, old, young = _plain_prefill_pair()
    inj = MidAppendOom(runner.block_pool, lambda: sched._batched_cache)
    inj.arm(slot=0)
    assert sched._batched_cache is not None
    blocks_before = len(sched._batched_cache._block_ids[0])

    with pytest.raises(OutOfMemoryError):
        sched._step()  # old's chunk 2 needs 2 blocks, and the second one raises

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


# ── Benchmark plumbing: token counts, stats that outlive the request, trace ──


def _drain_steps(req: RunningRequest) -> list[GenerationStep]:
    steps: list[GenerationStep] = []
    while True:
        try:
            steps.append(req.output_queue.get_nowait())
        except queue.Empty:
            return steps


def test_every_step_carries_its_token_count() -> None:
    """A verify step streams its committed block as one step, so each step carries its token
    count. On both paths, the counts of a request sum to its generated tokens."""
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44]])
    spec_req = _enqueue(sched, "ab", 8)
    plain_req = _enqueue(sched, "cd", 4, SamplingParams(temperature=0.0, top_k=5))
    _run_to_done(sched, [spec_req, plain_req])

    counts: dict[str, list[int]] = {}
    for name, req in (("spec", spec_req), ("plain", plain_req)):
        steps = _drain_steps(req)
        body, terminal = steps[:-1], steps[-1]
        assert terminal.finish_reason is not None and terminal.num_tokens == 0
        assert sum(s.num_tokens for s in body) == len(req.tokens_generated)
        counts[name] = [s.num_tokens for s in body]

    # The anchor streams alone, then the accept-all round streams 42, 43, 44
    # and the bonus 45 as one step of 4. The plain request is 1 per step.
    assert counts["spec"][:2] == [1, 4]
    assert counts["plain"] == [1, 1, 1, 1]


def test_speculative_stats_outlive_the_request() -> None:
    """`_finish` keeps a fixed-size summary of the stats that the reap drops. A request that did
    not speculate reports None, so a benchmark can see a silent downgrade."""
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44], [46, 7, 48]])
    spec_req = _enqueue(sched, "ab", 16)
    plain_req = _enqueue(sched, "cd", 4, SamplingParams(temperature=0.0, top_k=5))
    _run_to_done(sched, [spec_req, plain_req])
    sched._step()  # the reap: spec state is gone after this

    assert spec_req.spec is None
    result = RequestHandle(spec_req).wait()
    summary = result.spec_summary
    assert summary is not None
    assert result.tokens == spec_req.tokens_generated
    # Round 1 accepts 3 of 3, round 2 accepts 1 of 3 (7 is rejected at
    # position 1), and later rounds offer nothing, so each commits its bonus.
    assert summary.offered_by_position[:3] == (2, 2, 2)
    assert summary.survived_by_position[:3] == (2, 1, 1)
    assert summary.committed == sum([4, 2]) + (summary.rounds - 2)
    assert summary.tau == summary.committed / summary.rounds
    assert summary.drafter_forwards == summary.rounds

    assert RequestHandle(plain_req).wait().spec_summary is None


def test_pooled_tau_sums_before_it_divides() -> None:
    """As in DeepSpec, tau (committed tokens per round) pools all rounds. A mean of per-request
    tau gives the few rounds of a short request too much weight."""
    short = SpecSummary(1, 4, 3, 3, 1, (1, 1, 1), (1, 1, 1))  # tau 4.0
    long = SpecSummary(9, 9, 27, 0, 9, (9, 9, 9), (0, 0, 0))  # tau 1.0
    pooled = SpecSummary.pool([short, long])

    assert pooled.rounds == 10 and pooled.committed == 13
    assert pooled.tau == 1.3
    assert pooled.tau != (short.tau + long.tau) / 2
    assert pooled.offered_by_position == (10, 10, 10)
    assert pooled.survived_by_position == (1, 1, 1)
    with pytest.raises(ValueError, match="block sizes"):
        SpecSummary.pool([short, SpecSummary(1, 1, 0, 0, 1, (0,), (0,))])


def test_trace_records_forward_width_and_verify_count(tmp_path: Path) -> None:
    """Step cost follows query tokens, not requests, so the trace records both and the verify
    count."""
    runner = _ScriptedRunner(_next_token)
    sched = ContinuousScheduler(
        runner,  # type: ignore[arg-type]
        trace_out=str(tmp_path / "trace.json"),
        dspark=_ScriptedSpec([[42, 43, 44]]),
    )
    req = _enqueue(sched, "ab", 8)
    sched._step()  # prefill: 2 query tokens, no verify
    sched._step()  # anchor + 3 offered: 4 query tokens, one verify window

    assert sched._trace_events is not None
    first, second = (e["args"] for e in sched._trace_events[:2])
    assert first == {"B": 1, "phase": "prefill", "total_q": 2, "n_verify": 0}
    assert second == {"B": 1, "phase": "decode", "total_q": 4, "n_verify": 1}
    assert req.tokens_generated == [41, 42, 43, 44, 45]


def test_rounds_shortened_for_room_are_counted_apart() -> None:
    """Rounds shortened to fit max_tokens are counted apart. The batch-1 runner never shortens
    for room, so a tau with these rounds is biased low against it."""
    # max_tokens 6: anchor 41, then room 5 offers [42, 43, 44] in full (a round of 4),
    # then room 1 cuts [46, 47, 48] to nothing (a clamped round of the bonus only).
    sched, _runner, _spec = _spec_sched(_next_token, [[42, 43, 44], [46, 47, 48]])
    req = _enqueue(sched, "ab", 6)
    _run_to_done(sched, [req])

    summary = RequestHandle(req).wait().spec_summary
    assert summary is not None
    assert summary.rounds == 2 and summary.committed == 4 + 1
    assert summary.room_clamped_rounds == 1 and summary.room_clamped_committed == 1
    assert summary.tau == 2.5  # (4 + 1) / 2
    assert summary.tau_unclamped == 4.0  # the clamped round left out
    pooled = SpecSummary.pool([summary, summary])
    assert pooled.room_clamped_rounds == 2 and pooled.tau_unclamped == 4.0
