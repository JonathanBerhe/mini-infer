"""Variable per-request q-length in `ContinuousScheduler._packed_forward`.

Decoders hand each step a token LIST rather than a single token, so requests
with different q-lengths ride one packed forward. Two properties everything
built on top of this depends on:

- the packed inputs a step builds (ids, `cu_seqlens_q`, `position_offsets`)
  follow each request's own q-length, and the cache slot advances by that many
  positions;
- a request's `last_logits` is sliced from ITS OWN last row of the packed
  result. Reading a neighbour's row yields a plausible wrong token with no
  error at all, so the offset is asserted on the row index directly rather than
  through decoded text.

The plain sampling path still contributes exactly one token per decoder per
step; that is asserted here too, since it is what keeps the golden suite valid.
The zero-length cases are here as well, on both branches: a slot with no rows of
its own is what makes the slice read a neighbour, and an empty prompt is how a
client can reach that state from outside.

Model-free (a fake runner over a real `BlockPool`), so it runs in CI. The fake
forward performs the real `append_kv_packed`, so slot lengths and block
allocation are the production ones.
"""

from __future__ import annotations

import queue

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request
from mini_infer.scheduler.request_state import RequestState, RunningRequest

_VOCAB = 8
_HEAD_DIM = 8

# Row-index offset baked into the packed logits. Index 0 of every row carries
# `row + _ARGMAX_BUMP` so greedy argmax is token 0 at every position (decode
# stays deterministic), while index 1 carries the bare row index, letting a
# test read back which packed row a request's `last_logits` came from.
_ARGMAX_BUMP = 1000.0


class _FakeTokenizer:
    """`encode` -> one id per character; `decode` maps ids back to letters."""

    eos_token_id = -1  # never matches the fake forward's argmax (token 0)

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % _VOCAB) or 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


class _RecordingRunner:
    """Duck-typed ModelRunner whose packed logits encode their own row index.

    Every call's packed inputs are recorded, and the returned
    `(1, total_q, vocab)` tensor has `logits[0, row, 1] == row`, so
    `req.last_logits[1].item()` names the packed row the scheduler sliced.
    The K/V append is the real one (zeros for values), so per-slot lengths and
    block allocation behave exactly as in production.
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
    """Two prompts of different lengths in one packed forward: each request's
    logits come from the last row of ITS OWN window, not the packed tail."""
    sched, runner = _sched()
    short = _enqueue(sched, "abc", max_tokens=2)  # 3 prompt tokens
    long = _enqueue(sched, "abcde", max_tokens=2)  # 5 prompt tokens

    sched._step()

    call = runner.calls[0]
    assert call["cu_seqlens_q"] == [0, 3, 8]
    assert call["position_offsets"] == [0, 0]
    assert short.state == RequestState.DECODING
    assert long.state == RequestState.DECODING
    assert _row_of(short) == 2  # last row of rows 0..2
    assert _row_of(long) == 7  # last row of rows 3..7


def test_plain_decode_contributes_exactly_one_token_per_request() -> None:
    """The non-speculative path is unchanged: q-length 1 per decoder, and
    `position_offsets` pick up from the lengths the prefill append left."""
    sched, runner = _sched()
    short = _enqueue(sched, "abc", max_tokens=2)
    long = _enqueue(sched, "abcde", max_tokens=2)

    sched._step()  # prefill both
    runner.calls.clear()
    sched._step()  # first decode step

    call = runner.calls[0]
    assert call["cu_seqlens_q"] == [0, 1, 2]
    assert call["position_offsets"] == [3, 5]
    assert call["packed_input_ids"] == [0, 0]  # greedy argmax is token 0
    assert short.tokens_generated == [0]
    assert long.tokens_generated == [0]
    assert _row_of(short) == 0
    assert _row_of(long) == 1


def test_sample_decoders_returns_one_element_lists() -> None:
    """`_sample_decoders` hands out lists, holding exactly one token on the
    plain path."""
    sched, _ = _sched()
    short = _enqueue(sched, "abc", max_tokens=4)
    long = _enqueue(sched, "abcde", max_tokens=4)

    sched._step()  # prefill both, leaving them DECODING with last_logits
    sampled = sched._sample_decoders()

    assert sampled == {id(short): [0], id(long): [0]}


def test_variable_q_lengths_pack_and_slice_per_request() -> None:
    """A decoder feeding three tokens and one feeding a single token share the
    forward: packing, per-slot cache growth, and the logits slice all follow
    each request's own q-length."""
    sched, runner = _sched()
    one = _enqueue(sched, "abc", max_tokens=8)  # 3 prompt tokens
    three = _enqueue(sched, "abcde", max_tokens=8)  # 5 prompt tokens

    sched._step()  # prefill both
    assert sched._batched_cache is not None
    assert sched._batched_cache.seq_lens_list() == [3, 5]
    runner.calls.clear()

    sched._packed_forward([one, three], {id(one): [1], id(three): [2, 3, 4]})

    call = runner.calls[0]
    assert call["packed_input_ids"] == [1, 2, 3, 4]
    assert call["cu_seqlens_q"] == [0, 1, 4]
    assert call["position_offsets"] == [3, 5]
    # Each request read its own last row: 0 for the single-token slot, 3 for
    # the three-token slot (rows 1..3).
    assert _row_of(one) == 0
    assert _row_of(three) == 3
    # And each slot grew by its own q-length, so the next step's offsets are right.
    assert sched._batched_cache.seq_lens_list() == [4, 8]


def test_prefill_chunk_and_multi_token_decoder_in_one_forward() -> None:
    """Mixed batch with a chunked prefiller and a multi-token decoder: the
    prefiller's chunk and the decoder's block are packed back to back."""
    runner = _RecordingRunner()
    sched = ContinuousScheduler(runner, chunk_size=2)  # type: ignore[arg-type]
    decoder = _enqueue(sched, "abc", max_tokens=8)  # 3 tokens, prefills in 2 chunks

    sched._step()  # chunk 1 of the decoder's prompt (2 of 3 tokens)
    assert decoder.state == RequestState.CHUNKED_PREFILLING
    sched._step()  # chunk 2 completes the prompt
    assert decoder.state == RequestState.DECODING

    prefiller = _enqueue(sched, "abcdefg", max_tokens=8)  # 7 tokens, chunked
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
    """An empty token list would give a slot no row of its own, silently
    handing it the previous request's logits. It must not be packable."""
    sched, _ = _sched()
    first = _enqueue(sched, "abc", max_tokens=8)
    second = _enqueue(sched, "abcde", max_tokens=8)

    sched._step()  # prefill both

    with pytest.raises(AssertionError, match="decoding request contributed no tokens"):
        sched._packed_forward([first, second], {id(first): [1], id(second): []})


def test_zero_token_prefiller_is_rejected() -> None:
    """The prefill branch carries the same guard: a chunk of no tokens would
    read a neighbour's logits rather than fail."""
    sched, _ = _sched()
    req = _enqueue(sched, "abc", max_tokens=8)

    sched._admit_waiting()
    # A prefiller with nothing left to feed cannot arise on its own (the state
    # machine moves it to DECODING first), so force it.
    req.tokens_prefilled = len(req.prompt_token_ids)

    with pytest.raises(AssertionError, match="prefilling request contributed no tokens"):
        sched._packed_forward([req], {})


def test_empty_prompt_finishes_at_admission() -> None:
    """A prompt that tokenizes to nothing never reaches the forward.

    It has no q-tokens, so it would put a zero-length window in the packed
    batch and the last-position slice would read index -1 of an empty logits
    tensor. `_engine_loop` only catches `OutOfMemoryError`, so that IndexError
    would kill the engine thread and every other request in flight.
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
    # The request admitted alongside it is unaffected and owns slot 0.
    assert other.state == RequestState.DECODING
    assert other.batch_idx == 0
    assert runner.calls[0]["cu_seqlens_q"] == [0, 3]


def test_plain_path_runs_to_max_tokens_unchanged() -> None:
    """End to end over the synchronous step loop: both requests emit exactly
    `max_tokens` tokens, finish with "length", and free their slots."""
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
    # Every block came back when the slots were reaped.
    assert runner.block_pool.num_free_blocks == runner.block_pool.num_blocks
    # One prefill forward plus one per emitted token beyond the first.
    assert [c["cu_seqlens_q"] for c in runner.calls] == [[0, 3, 8], [0, 1, 2]]
