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

    `oom_on_calls` is the set of 1-based `forward_step` invocation indices that
    raise `OutOfMemoryError`; every other call returns greedy-argmax-0 logits
    for each request in the batch, so decode is deterministic.
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

    def _forward_impl(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
        oom_on_calls: set[int],
    ) -> list[torch.Tensor]:
        self.calls += 1
        if self.calls in oom_on_calls:
            raise OutOfMemoryError("forced OOM for test")
        # One logits vector per request; argmax is token 0 (greedy-deterministic).
        logits = torch.full((_VOCAB,), -1.0)
        logits[0] = 1.0
        return [logits.clone() for _ in range(cache.batch_size)]


def _make_runner(oom_on_calls: set[int]) -> _FakeRunner:
    runner = _FakeRunner(oom_on_calls)
    runner.forward_step = (  # type: ignore[attr-defined]
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
