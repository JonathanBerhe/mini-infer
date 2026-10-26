"""Variable per-request q-length in `ContinuousScheduler._packed_forward`.

The q-length of a request is the number of tokens it feeds into one forward.
The packed inputs and the growth of each cache slot must follow it. Each request
must take `last_logits` from its own last packed row. A neighbor's row gives a
wrong token with no error, so the tests assert the row index, not the text.

Model-free (a fake runner over a real `BlockPool`), so it runs in CI.
"""

from __future__ import annotations

import queue
from typing import Any

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request
from mini_infer.scheduler.request_state import RequestState, RunningRequest

_VOCAB = 8
_HEAD_DIM = 8

# Added to vocab index 0 of every packed row, so greedy argmax is always token 0.
_ARGMAX_BUMP = 1000.0


class _FakeTokenizer:
    """`encode` gives one id per character, and `decode` maps ids to letters."""

    eos_token_id = -1  # never matches the fake forward's argmax (token 0)

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % _VOCAB) or 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


class _RecordingRunner:
    """Duck-typed ModelRunner that records the packed inputs of every call.

    Its logits hold `logits[0, row, 1] == row`, so `_row_of` names the packed row
    that the scheduler sliced. It calls the real `append_kv_packed` with zero K/V,
    so slot lengths and block allocation are real.
    """

    def __init__(self) -> None:
        self.block_pool = BlockPool(
            num_blocks=256,
            block_size=16,
            num_layers=1,
            num_kv_heads=1,
            head_dim=_HEAD_DIM,
            dtype=torch.float32,
            device="cpu",
            attention_backend="torch",
        )
        self.tokenizer = _FakeTokenizer()
        self.calls: list[dict[str, list[int]]] = []

    def forward_step_packed(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
    ) -> torch.Tensor:
        self.calls.append(
            {
                "packed_input_ids": list(packed_input_ids),
                "cu_seqlens_q": list(cu_seqlens_q),
                "position_offsets": list(position_offsets),
            }
        )
        total_q = cu_seqlens_q[-1]
        zeros = torch.zeros((total_q, 1, _HEAD_DIM), dtype=torch.float32)
        cache.append_kv_packed(
            zeros, zeros.clone(), torch.tensor(cu_seqlens_q, dtype=torch.int32), 0
        )
        rows = torch.arange(total_q, dtype=torch.float32).reshape(1, total_q, 1)
        logits = rows.repeat(1, 1, _VOCAB)
        logits[..., 0] += _ARGMAX_BUMP
        return logits


def _enqueue(sched: ContinuousScheduler, prompt: str, max_tokens: int) -> RunningRequest:
    """Put a request straight on the waiting queue (no engine thread involved)."""
    running = RunningRequest(
        request=Request(
            prompt=prompt,
            sampling_params=SamplingParams(temperature=0.0),
            max_tokens=max_tokens,
        ),
        output_queue=queue.Queue(maxsize=64),
    )
    sched._waiting.put(running)
    return running


def _sched() -> tuple[ContinuousScheduler, _RecordingRunner]:
    runner = _RecordingRunner()
    return ContinuousScheduler(runner), runner  # type: ignore[arg-type]


def _row_of(req: RunningRequest) -> int:
    """The packed row `req.last_logits` was sliced from."""
    assert req.last_logits is not None
    return int(req.last_logits[1].item())


def test_prefill_slices_each_request_from_its_own_last_row() -> None:
    """Two prompts of different lengths share a forward, and each reads its own last row."""
    sched, runner = _sched()
    short = _enqueue(sched, "abc", max_tokens=2)
    long = _enqueue(sched, "abcde", max_tokens=2)

    sched._step()

    call = runner.calls[0]
    assert call["cu_seqlens_q"] == [0, 3, 8]
    assert call["position_offsets"] == [0, 0]
    assert short.state == RequestState.DECODING
    assert long.state == RequestState.DECODING
    assert _row_of(short) == 2  # last row of rows 0..2
    assert _row_of(long) == 7  # last row of rows 3..7


def test_plain_decode_contributes_exactly_one_token_per_request() -> None:
    """The plain path feeds one token per decoder per step. The golden tests rely on this."""
    sched, runner = _sched()
    short = _enqueue(sched, "abc", max_tokens=2)
    long = _enqueue(sched, "abcde", max_tokens=2)

    sched._step()  # prefill both
    runner.calls.clear()
    sched._step()  # first decode step

    call = runner.calls[0]
    assert call["cu_seqlens_q"] == [0, 1, 2]
    assert call["position_offsets"] == [3, 5]
    assert call["packed_input_ids"] == [0, 0]
    assert short.tokens_generated == [0]
    assert long.tokens_generated == [0]
    assert _row_of(short) == 0
    assert _row_of(long) == 1


def test_sample_decoders_returns_one_element_lists() -> None:
    """`_sample_decoders` returns a list per decoder, with one token on the plain path."""
    sched, _ = _sched()
    short = _enqueue(sched, "abc", max_tokens=4)
    long = _enqueue(sched, "abcde", max_tokens=4)

    sched._step()  # prefill both
    sampled = sched._sample_decoders()

    assert sampled == {id(short): [0], id(long): [0]}


def test_variable_q_lengths_pack_and_slice_per_request() -> None:
    """A one-token decoder and a three-token decoder share one forward. Packing,
    slot growth, and the logits slice all follow the q-length of each request."""
    sched, runner = _sched()
    one = _enqueue(sched, "abc", max_tokens=8)
    three = _enqueue(sched, "abcde", max_tokens=8)

    sched._step()  # prefill both
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [3, 5]
    runner.calls.clear()

    sched._packed_forward([one, three], {id(one): [1], id(three): [2, 3, 4]})

    call = runner.calls[0]
    assert call["packed_input_ids"] == [1, 2, 3, 4]
    assert call["cu_seqlens_q"] == [0, 1, 4]
    assert call["position_offsets"] == [3, 5]
    assert _row_of(one) == 0
    assert _row_of(three) == 3  # last of rows 1..3
    assert sched._batched_cache.seq_lens_list() == [4, 8]


def test_prefill_chunk_and_multi_token_decoder_in_one_forward() -> None:
    """A multi-token decoder and a prefill chunk pack back to back in one forward."""
    runner = _RecordingRunner()
    sched = ContinuousScheduler(runner, chunk_size=2)  # type: ignore[arg-type]
    decoder = _enqueue(sched, "abc", max_tokens=8)  # 3 tokens, prefills in 2 chunks

    sched._step()
    assert decoder.state == RequestState.CHUNKED_PREFILLING
    sched._step()
    assert decoder.state == RequestState.DECODING

    prefiller = _enqueue(sched, "abcdefg", max_tokens=8)
    sched._admit_waiting()
    runner.calls.clear()

    sched._packed_forward([decoder, prefiller], {id(decoder): [5, 6]})

    call = runner.calls[0]
    assert call["cu_seqlens_q"] == [0, 2, 4]  # 2 decode tokens, then a 2-token chunk
    assert call["position_offsets"] == [3, 0]
    assert _row_of(decoder) == 1
    assert prefiller.state == RequestState.CHUNKED_PREFILLING
    assert prefiller.tokens_prefilled == 2
    assert prefiller.last_logits is None  # no logits until the prompt is done


def test_zero_token_decoder_is_rejected() -> None:
    """An empty token list gives a slot no row, so the slot silently reads the
    logits of the previous request. `_packed_forward` must reject it."""
    sched, _ = _sched()
    first = _enqueue(sched, "abc", max_tokens=8)
    second = _enqueue(sched, "abcde", max_tokens=8)

    sched._step()

    with pytest.raises(AssertionError, match="decoding request contributed no tokens"):
        sched._packed_forward([first, second], {id(first): [1], id(second): []})


def test_zero_token_prefiller_is_rejected() -> None:
    """The prefill branch has the same guard: an empty chunk also reads a neighbor's row."""
    sched, _ = _sched()
    req = _enqueue(sched, "abc", max_tokens=8)

    sched._step()  # the real prefill: counter and slot both reach 3
    # The state machine moves a fully fed prefiller to DECODING, so force only the state
    # back. Counter and slot still agree, so only the zero-chunk guard fires.
    req.state = RequestState.CHUNKED_PREFILLING

    with pytest.raises(AssertionError, match="prefilling request contributed no tokens"):
        sched._packed_forward([req], {})


def test_empty_prompt_finishes_at_admission() -> None:
    """A prompt that tokenizes to nothing finishes at admission, before any forward.

    Otherwise it trips the zero-token guard. `_engine_loop` catches only `OutOfMemoryError`,
    so that `AssertionError` kills the engine thread and fails every request in flight.
    """
    sched, runner = _sched()
    empty = _enqueue(sched, "", max_tokens=4)
    other = _enqueue(sched, "abc", max_tokens=4)

    sched._step()

    assert empty.finish_reason == "stop"
    assert empty.tokens_generated == []
    assert empty.batch_idx is None  # never took a cache slot
    assert empty not in sched._running
    assert empty.output_queue.get_nowait().finish_reason == "stop"
    assert other.state == RequestState.DECODING
    assert other.batch_idx == 0
    assert runner.calls[0]["cu_seqlens_q"] == [0, 3]


def test_a_freed_slot_is_refilled_in_the_same_step() -> None:
    """While a prompt waits, every forward runs a full batch.

    A decoder that reaches `max_tokens` finishes when it samples, before the
    forward. The reap runs before admission, so a waiting prompt takes the
    slot in that same forward instead of one step later.
    """
    runner = _RecordingRunner()
    sched = ContinuousScheduler(runner, max_concurrent=2)  # type: ignore[arg-type]
    reqs = [
        _enqueue(sched, p, max_tokens=n) for p, n in (("ab", 1), ("abc", 3), ("abcd", 2), ("a", 4))
    ]
    forward = runner.forward_step_packed
    seen: list[tuple[int, int]] = []

    def recording_forward(*args: Any) -> torch.Tensor:
        seen.append((len(args[2]) - 1, sched._waiting.qsize()))  # (batch, waiting)
        return forward(*args)

    runner.forward_step_packed = recording_forward  # type: ignore[method-assign]
    for _ in range(50):
        if all(r.state == RequestState.DONE for r in reqs):
            break
        sched._step()
    assert all(r.state == RequestState.DONE for r in reqs)
    assert [len(r.tokens_generated) for r in reqs] == [1, 3, 2, 4]
    assert any(waiting for _, waiting in seen)
    assert all(batch == 2 for batch, waiting in seen if waiting)


def test_requests_ended_at_admission_do_not_block_a_waiter() -> None:
    """A request that admission ends on the spot (cancelled while queued, or
    an empty prompt) takes no seat, so the waiter behind it joins this forward."""
    runner = _RecordingRunner()
    sched = ContinuousScheduler(runner, max_concurrent=2)  # type: ignore[arg-type]
    first = _enqueue(sched, "ab", max_tokens=1)
    second = _enqueue(sched, "abc", max_tokens=3)
    cancelled = _enqueue(sched, "abcd", max_tokens=2)
    cancelled.cancel_event.set()
    empty = _enqueue(sched, "", max_tokens=2)
    waiter = _enqueue(sched, "abcde", max_tokens=2)

    sched._step()  # prefill `first` and `second`
    sched._step()  # `first` finishes when it samples; its seat goes to `waiter`

    assert first.finish_reason == "length"
    assert cancelled.finish_reason == "cancelled"
    assert empty.finish_reason == "stop"
    assert second.state == RequestState.DECODING
    # The second forward: one decode token for `second`, five prompt tokens for `waiter`.
    assert runner.calls[1]["cu_seqlens_q"] == [0, 1, 6]
    assert waiter.state == RequestState.DECODING
    assert all(r.state != RequestState.DONE for r in sched._running)


def test_plain_path_runs_to_max_tokens_unchanged() -> None:
    """Both requests emit `max_tokens` tokens, finish with "length", and free their slots."""
    sched, runner = _sched()
    short = _enqueue(sched, "abc", max_tokens=2)
    long = _enqueue(sched, "abcde", max_tokens=2)

    for _ in range(4):
        sched._step()

    assert short.tokens_generated == [0, 0]
    assert long.tokens_generated == [0, 0]
    assert short.finish_reason == "length"
    assert long.finish_reason == "length"
    assert sched._running == []
    assert sched._batched_cache is None
    assert runner.block_pool.num_free_blocks == runner.block_pool.num_blocks
    # Prefill, then one decode. The last emitted token is never fed back.
    assert [c["cu_seqlens_q"] for c in runner.calls] == [[0, 3, 8], [0, 1, 2]]
