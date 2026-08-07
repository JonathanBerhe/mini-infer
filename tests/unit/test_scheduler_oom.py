"""ContinuousScheduler mid-step OOM recovery (`_engine_loop` + `_preempt_on_oom`).

The pool watermarks blocks at admission but only physically allocates them
during a forward, so a step can raise `OutOfMemoryError` after admission already
accepted everyone. The engine must preempt the youngest in-flight request and
retry, not crash the thread (which would take down every in-flight and future
request). These tests pin both halves:

- victim selection (`_preempt_on_oom`) directly, hand-building `_running`, and
- end-to-end recovery: a fake runner whose forward OOMs exactly once, driven
  through the real engine thread, must leave the engine alive and serving.

Model-free (a fake runner over a real `BlockPool`), so it runs in CI.
"""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.sampler import SamplingParams
from mini_infer.exceptions import OutOfMemoryError
from mini_infer.scheduler import ContinuousScheduler, Request
from mini_infer.scheduler.request_state import GenerationStep, RequestState, RunningRequest

_VOCAB = 32


class _FakeTokenizer:
    """`encode` -> one id per character; greedy decode is a no-op string."""

    eos_token_id = -1  # never matches the fake forward's argmax (token 0)

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % _VOCAB) or 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


class _FakeRunner:
    """Duck-typed ModelRunner: real BlockPool, fake tokenizer, scriptable forward.

    `oom_on_calls` is the set of 1-based `forward_step_packed` invocation indices
    that raise `OutOfMemoryError`; every other call returns greedy-argmax-0
    logits at every packed position, so decode is deterministic.
    """

    def __init__(self, oom_on_calls: set[int]) -> None:
        self.block_pool = BlockPool(
            num_blocks=256,
            block_size=16,
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            attention_backend="torch",
        )
        self.tokenizer = _FakeTokenizer()
        self.calls = 0
        # Packed q-tokens per successful call, so a test can see what a retry
        # re-fed after a raise.
        self.packed_per_call: list[list[int]] = []

    def _forward_impl(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
        oom_on_calls: set[int],
    ) -> torch.Tensor:
        self.calls += 1
        if self.calls in oom_on_calls:
            raise OutOfMemoryError("forced OOM for test")
        self.packed_per_call.append(list(packed_input_ids))
        # Packed logits over every q position; argmax is token 0 everywhere
        # (greedy-deterministic), matching `forward_step_packed`'s contract.
        logits = torch.full((1, cu_seqlens_q[-1], _VOCAB), -1.0)
        logits[..., 0] = 1.0
        return logits


def _make_runner(oom_on_calls: set[int]) -> _FakeRunner:
    runner = _FakeRunner(oom_on_calls)
    runner.forward_step_packed = (  # type: ignore[attr-defined]
        lambda cache, ids, cu, pos: runner._forward_impl(cache, ids, cu, pos, oom_on_calls)
    )
    return runner


def _running_request(state: RequestState, batch_idx: int) -> RunningRequest:
    req = RunningRequest(
        request=Request(prompt="x", sampling_params=SamplingParams(temperature=0.0), max_tokens=2),
        output_queue=queue.Queue(),
    )
    req.state = state
    req.batch_idx = batch_idx
    return req


def test_preempt_on_oom_prefers_youngest_prefiller() -> None:
    """A decoder is preserved over prefillers, and among prefillers the youngest
    (last-admitted) is the victim."""
    sched = ContinuousScheduler(_make_runner(set()))  # type: ignore[arg-type]
    decoder = _running_request(RequestState.DECODING, 0)
    old_prefiller = _running_request(RequestState.PREFILLING, 1)
    young_prefiller = _running_request(RequestState.CHUNKED_PREFILLING, 2)
    sched._running = [decoder, old_prefiller, young_prefiller]

    sched._preempt_on_oom()

    # Youngest prefiller cancelled and reaped; the decoder and older prefiller live.
    assert young_prefiller.finish_reason == "cancelled"
    assert young_prefiller not in sched._running
    assert sched._running == [decoder, old_prefiller]
    terminal = young_prefiller.output_queue.get_nowait()
    assert isinstance(terminal, GenerationStep)
    assert terminal.finish_reason == "cancelled"


def test_preempt_on_oom_falls_back_to_decoder() -> None:
    """With no prefillers in flight, the youngest decoder is the victim."""
    sched = ContinuousScheduler(_make_runner(set()))  # type: ignore[arg-type]
    decoder0 = _running_request(RequestState.DECODING, 0)
    decoder1 = _running_request(RequestState.DECODING, 1)
    sched._running = [decoder0, decoder1]

    sched._preempt_on_oom()

    assert decoder1.finish_reason == "cancelled"
    assert sched._running == [decoder0]


def test_preempt_on_oom_no_candidates_is_noop() -> None:
    """Nothing in flight: preemption is a safe no-op (the loop just retries)."""
    sched = ContinuousScheduler(_make_runner(set()))  # type: ignore[arg-type]
    sched._running = []
    sched._preempt_on_oom()  # must not raise
    assert sched._running == []


def test_engine_survives_mid_step_oom_and_keeps_serving() -> None:
    """End-to-end: the first forward OOMs; the request is cancelled, the engine
    thread stays alive, and a subsequent request completes normally.

    Each `run` is bounded by a wall-clock timeout so that a regression (the
    engine thread dying on OOM, which leaves `run` blocked on its output queue
    forever) fails the test in seconds instead of hanging CI.
    """
    sched = ContinuousScheduler(_make_runner({1}))  # OOM on the very first forward
    sched.start()
    with ThreadPoolExecutor(max_workers=1) as pool:

        def run(prompt: str):  # type: ignore[no-untyped-def]
            future = pool.submit(
                sched.run,
                Request(
                    prompt=prompt,
                    sampling_params=SamplingParams(temperature=0.0),
                    max_tokens=2,
                ),
            )
            try:
                return future.result(timeout=15.0)
            except FutureTimeout:
                raise AssertionError(
                    "scheduler.run did not return within 15s; the engine thread "
                    "likely died on the OOM instead of recovering"
                ) from None

        try:
            victim = run("hello")
            assert victim.finish_reason == "cancelled"
            assert victim.tokens == []  # never got past prefill
            # The engine thread survived the OOM rather than crashing.
            assert sched._thread is not None and sched._thread.is_alive()

            # Recovery: the next request (forward no longer OOMs) runs to completion.
            survivor = run("world")
            assert survivor.finish_reason == "length"
            assert len(survivor.tokens) == 2
        finally:
            sched.stop()


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


def _step_with_recovery(sched: ContinuousScheduler) -> None:
    """One engine iteration with `_engine_loop`'s OOM handling, driven inline.

    The real loop is a thread, so the interleaving this test needs (a decoder
    mid-flight while a prefiller is the preemption victim) would be racy through
    `submit`. This mirrors the handler at `_engine_loop` exactly.
    """
    try:
        sched._step()
    except OutOfMemoryError:
        sched._preempt_on_oom()


def _drain(req: RunningRequest) -> list[GenerationStep]:
    steps: list[GenerationStep] = []
    while True:
        try:
            steps.append(req.output_queue.get_nowait())
        except queue.Empty:
            return steps


def test_oom_retry_does_not_re_emit_an_already_emitted_token() -> None:
    """A decoder that survives a mid-step OOM must not emit its token twice.

    Sampling happens before the forward and emits as it goes, so a retry that
    re-sampled the same (unchanged) `last_logits` would append and stream a
    duplicate of a token the client already has. The tokens stay pending across
    the failed forward and are re-fed instead.

    The existing end-to-end test above cannot reach this: it OOMs the very first
    forward of a single request that has no `last_logits` yet.
    """
    runner = _make_runner({2})  # OOM on the second forward
    sched = ContinuousScheduler(runner)
    decoder = _enqueue(sched, "abc", max_tokens=4)

    _step_with_recovery(sched)  # forward 1: prefill -> DECODING
    assert decoder.state == RequestState.DECODING

    # A prefiller admitted this step is the preemption victim, so the decoder
    # survives the OOM (`_preempt_on_oom` prefers prefillers).
    prefiller = _enqueue(sched, "defgh", max_tokens=4)
    _step_with_recovery(sched)  # forward 2: decoder samples + emits, then OOM

    assert prefiller.finish_reason == "cancelled"
    assert prefiller not in sched._running
    emitted = list(decoder.tokens_generated)
    assert len(emitted) == 1
    # Held for the retry rather than dropped or re-sampled.
    assert decoder.pending_decode_tokens == emitted

    _step_with_recovery(sched)  # forward 3: re-feeds the pending token
    assert decoder.tokens_generated == emitted, "the retry emitted a second token"
    assert decoder.pending_decode_tokens is None  # consumed by a forward that landed
    assert runner.packed_per_call[-1] == emitted

    _step_with_recovery(sched)  # forward 4: the next token, sampled fresh
    assert len(decoder.tokens_generated) == 2

    # The stream carries exactly one delta per generated token, and the text is
    # the decode of those tokens with nothing repeated.
    deltas = [s.text for s in _drain(decoder) if s.finish_reason is None]
    assert len(deltas) == 2
    assert "".join(deltas) == runner.tokenizer.decode(decoder.tokens_generated)
