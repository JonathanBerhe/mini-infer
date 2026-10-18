"""Engine problems reach the caller as errors, and output is never lost.

Every scheduler runs its engine on a thread that dies on an unexpected
exception, and after that no request in flight ever receives its terminal
step. These tests pin the layers that turn engine problems into errors
(ADR-028, ADR-030) and the event-loop reads of the server (ADR-029):

- `RequestHandle` polls the liveness of the thread that owns its request
  whenever its queue stays empty, hands out every buffered step first, and
  then raises `EngineNotRunningError`. A step marked `failed` raises
  `RequestFailedError`. `get_step_async` waits on the event loop with the
  same rules.
- Each of the five schedulers binds its handles to its engine thread(s),
  rejects `submit()` once that thread is gone, and cancels what is left in a
  clean `stop()`.
- The HTTP endpoint answers a dead engine with a 503, a failed request with
  a 500 (or an SSE error event once a stream has started), never blocks the
  event loop, and keeps every token's text with one executor thread.

All engines here are model-free doubles on CPU. Every blocking call runs
under `_bounded` or `_wait_until`, so a regression fails within seconds
instead of hanging the suite (no pytest-timeout plugin is installed).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mini_infer.api.server import _unhandled_exception_handler, completions
from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.sampler import SamplingParams
from mini_infer.exceptions import EngineNotRunningError, OutOfMemoryError, RequestFailedError
from mini_infer.scheduler import (
    ContinuousScheduler,
    StateCacheContinuousScheduler,
    TensorParallelStateCacheContinuousScheduler,
    TensorParallelStateCacheScheduler,
)
from mini_infer.scheduler.request_state import (
    GenerationStep,
    Request,
    RequestHandle,
    RunningRequest,
    StepQueue,
)
from mini_infer.workers import PDScheduler

T = TypeVar("T")

_VOCAB = 32
# Short enough that a dead engine shows up fast, long enough to be a real slice.
_POLL_S = 0.05
_BOUND_S = 10.0


@pytest.fixture(autouse=True)
def _fast_liveness_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RequestHandle, "LIVENESS_POLL_SECONDS", _POLL_S)


class _InjectedCrashError(Exception):
    """The one exception the doubles raise, so a test can tell the injected
    death apart from a double that broke for some other reason."""


@pytest.fixture
def crashes(monkeypatch: pytest.MonkeyPatch) -> list[type[BaseException]]:
    """Exception types of engine threads that died during the test.

    Replacing the hook also keeps pytest's unhandled-thread-exception warning
    out of the report: the deaths here are the point of the test.
    """
    seen: list[type[BaseException]] = []

    def hook(args: threading.ExceptHookArgs) -> None:
        seen.append(args.exc_type)

    monkeypatch.setattr(threading, "excepthook", hook)
    return seen


def _bounded(fn: Callable[[], T], timeout: float = _BOUND_S) -> T:
    """Run `fn` on a daemon thread; fail the test if it does not return in time."""
    out: dict[str, Any] = {}

    def target() -> None:
        try:
            out["value"] = fn()
        except BaseException as exc:  # re-raised on the test thread below
            out["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        pytest.fail(f"call did not return within {timeout}s (it would hang without a bound)")
    if "error" in out:
        raise out["error"]
    return out["value"]  # type: ignore[no-any-return]


def _wait_until(predicate: Callable[[], object], timeout: float = _BOUND_S) -> None:
    """Poll `predicate` until it is true; fail the test instead of spinning forever."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"condition not reached within {timeout}s")
        time.sleep(0.001)


def _request(max_tokens: int = 4, prompt: str = "abc") -> Request:
    return Request(
        prompt=prompt, sampling_params=SamplingParams(temperature=0.0), max_tokens=max_tokens
    )


def _bare(*steps: GenerationStep) -> RunningRequest:
    req = RunningRequest(request=_request(), output_queue=StepQueue())
    for step in steps:
        req.output_queue.put(step)
    return req


def _token(text: str = "x") -> GenerationStep:
    return GenerationStep(text=text, num_tokens=1)


def _dead() -> bool:
    return False


# ---- RequestHandle ------------------------------------------------------


@pytest.mark.parametrize("call", ["get_step", "steps", "wait"])
def test_handle_on_a_dead_engine_raises_instead_of_blocking(call: str) -> None:
    req = _bare()
    handle = RequestHandle(req, engine_alive=_dead)
    calls: dict[str, Callable[[], object]] = {
        "get_step": handle.get_step,
        "steps": lambda: list(handle.steps()),
        "wait": handle.wait,
    }
    with pytest.raises(EngineNotRunningError):
        _bounded(calls[call])
    # The handle cancels the request, so an engine that still holds it drops it.
    assert req.cancel_event.is_set()


def test_handle_hands_out_buffered_steps_before_the_error() -> None:
    handle = RequestHandle(_bare(_token("a"), _token("b")), engine_alive=_dead)
    assert _bounded(handle.get_step).text == "a"
    assert _bounded(handle.get_step).text == "b"
    with pytest.raises(EngineNotRunningError):
        _bounded(handle.get_step)


def test_handle_steps_yields_buffered_steps_then_raises() -> None:
    handle = RequestHandle(_bare(_token("a"), _token("b")), engine_alive=_dead)
    seen: list[str] = []

    def drain() -> None:
        for step in handle.steps():
            seen.append(step.text)

    with pytest.raises(EngineNotRunningError):
        _bounded(drain)
    assert seen == ["a", "b"]


def test_handle_returns_a_last_step_put_just_before_the_thread_exited() -> None:
    """The thread can put its terminal step after the timed-out read but before
    the liveness read. The handle must look once more instead of raising."""
    req = _bare()

    def exits_after_its_last_put() -> bool:
        req.output_queue.put_nowait(GenerationStep(text="", finish_reason="length"))
        return False

    handle = RequestHandle(req, engine_alive=exits_after_its_last_put)
    assert _bounded(handle.get_step).finish_reason == "length"


def test_handle_wait_returns_a_result_when_the_terminal_step_races_the_exit() -> None:
    req = _bare(_token("a"))

    def exits_after_its_last_put() -> bool:
        req.output_queue.put_nowait(GenerationStep(text="", finish_reason="stop"))
        return False

    result = _bounded(RequestHandle(req, engine_alive=exits_after_its_last_put).wait)
    assert (result.text, result.finish_reason) == ("a", "stop")


def test_handle_keeps_polling_after_a_first_alive_read() -> None:
    """Liveness is read on every empty slice, not once at the start."""
    reads: list[bool] = []

    def alive_once() -> bool:
        reads.append(not reads)
        return reads[-1]

    with pytest.raises(EngineNotRunningError):
        _bounded(RequestHandle(_bare(), engine_alive=alive_once).wait)
    assert reads == [True, False]


def test_handle_waits_through_empty_slices_while_the_engine_lives() -> None:
    """A slow engine is not a dead one: an empty slice alone is no error."""
    req = _bare()

    def produce_late() -> None:
        time.sleep(4 * _POLL_S)
        req.output_queue.put(_token("late"))
        req.output_queue.put(GenerationStep(text="", finish_reason="length"))

    threading.Thread(target=produce_late, daemon=True).start()
    result = _bounded(RequestHandle(req, engine_alive=lambda: True).wait)
    assert (result.text, result.finish_reason) == ("late", "length")


def test_handle_timeout_on_a_live_engine_raises_timeout_error() -> None:
    handle = RequestHandle(_bare(), engine_alive=lambda: True)
    with pytest.raises(TimeoutError):
        _bounded(lambda: handle.wait(timeout=0.1))
    with pytest.raises(TimeoutError):
        _bounded(lambda: handle.get_step(timeout=0.1))


def test_handle_timeout_on_a_dead_engine_reports_the_dead_engine() -> None:
    handle = RequestHandle(_bare(), engine_alive=_dead)
    start = time.monotonic()
    with pytest.raises(EngineNotRunningError):
        _bounded(lambda: handle.wait(timeout=5.0))
    assert time.monotonic() - start < 2.0


def test_handle_reports_a_dead_engine_even_when_the_timeout_is_shorter_than_a_poll() -> None:
    """The server reads in slices shorter than `LIVENESS_POLL_SECONDS` and
    retries on `TimeoutError`. If the deadline won over liveness, that retry
    loop would never see the dead engine."""
    handle = RequestHandle(_bare(), engine_alive=_dead)
    with pytest.raises(EngineNotRunningError):
        _bounded(lambda: handle.get_step(timeout=_POLL_S / 5))


def test_handle_without_a_liveness_check_keeps_blocking() -> None:
    """`engine_alive=None` (a bare handle in a test) keeps the old behavior."""
    req = _bare()
    handle = RequestHandle(req)
    got: list[GenerationStep] = []
    reader = threading.Thread(target=lambda: got.append(handle.get_step()), daemon=True)
    reader.start()
    reader.join(6 * _POLL_S)
    assert reader.is_alive() and not got
    req.output_queue.put(_token("late"))
    reader.join(_BOUND_S)
    assert [s.text for s in got] == ["late"]


def test_build_result_ignores_the_terminal_step_text() -> None:
    """The text of a terminal step is not completion output."""
    req = _bare(_token("a"), GenerationStep(text="not output", finish_reason="cancelled"))
    result = _bounded(RequestHandle(req, engine_alive=lambda: True).wait)
    assert (result.text, result.finish_reason) == ("a", "cancelled")


def _failed_terminal() -> GenerationStep:
    return GenerationStep(text="", finish_reason="cancelled", failed=True)


@pytest.mark.parametrize("call", ["get_step", "steps", "wait", "get_step_async"])
def test_a_failed_request_raises_after_its_buffered_steps(call: str) -> None:
    """A failed request ends with `RequestFailedError`, never with a step that
    looks like a cancellation. Output produced before the failure comes first."""
    handle = RequestHandle(_bare(_token("a"), _failed_terminal()), engine_alive=lambda: True)
    seen: list[str] = []

    def drain() -> None:
        if call == "wait":
            handle.wait()
        elif call == "steps":
            seen.extend(s.text for s in handle.steps())
        else:
            while True:
                if call == "get_step":
                    step = handle.get_step()
                else:
                    step = asyncio.run(handle.get_step_async())
                seen.append(step.text)

    with pytest.raises(RequestFailedError):
        _bounded(drain)
    if call != "wait":
        assert seen == ["a"]


@pytest.mark.parametrize("call", ["get_step", "get_step_async"])
def test_a_failed_terminal_step_is_reported_after_the_engine_exits(call: str) -> None:
    """The last-chance read after a dead engine also checks for failure: the
    failed step lands between the empty read and the liveness read."""
    req = _bare()

    def exits_after_its_failed_put() -> bool:
        req.output_queue.put_nowait(_failed_terminal())
        return False

    handle = RequestHandle(req, engine_alive=exits_after_its_failed_put)
    read = handle.get_step if call == "get_step" else lambda: asyncio.run(handle.get_step_async())
    with pytest.raises(RequestFailedError):
        _bounded(read)


# ---- RequestHandle.get_step_async ------------------------------------------


def _run_async(fn: Callable[[], Any]) -> Any:
    """Run a coroutine function on a fresh event loop, under `_bounded`."""
    return _bounded(lambda: asyncio.run(fn()))


def test_get_step_async_wakes_on_a_put_without_waiting_for_a_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's put wakes the loop at once; the poll slice is only for
    liveness. A slice this long would show up in the latency otherwise."""
    monkeypatch.setattr(RequestHandle, "LIVENESS_POLL_SECONDS", 5.0)
    req = _bare()
    handle = RequestHandle(req, engine_alive=lambda: True)
    threading.Timer(0.05, lambda: req.output_queue.put(_token("now"))).start()
    start = time.monotonic()
    assert _run_async(handle.get_step_async).text == "now"
    assert time.monotonic() - start < 1.0


def test_get_step_async_hands_out_buffered_steps_then_reports_a_dead_engine() -> None:
    req = _bare(_token("a"), _token("b"))
    handle = RequestHandle(req, engine_alive=_dead)
    seen: list[str] = []

    async def drain() -> None:
        while True:
            seen.append((await handle.get_step_async()).text)

    with pytest.raises(EngineNotRunningError):
        _run_async(drain)
    assert seen == ["a", "b"]
    assert req.cancel_event.is_set()
    # The listener does not outlive the wait: a later put must not call into
    # a loop that is gone.
    assert req.output_queue._listener is None  # type: ignore[attr-defined]


def test_get_step_async_returns_a_last_step_put_just_before_the_thread_exited() -> None:
    req = _bare()

    def exits_after_its_last_put() -> bool:
        req.output_queue.put_nowait(GenerationStep(text="", finish_reason="length"))
        return False

    handle = RequestHandle(req, engine_alive=exits_after_its_last_put)
    assert _run_async(handle.get_step_async).finish_reason == "length"


def test_get_step_async_waits_through_empty_slices_while_the_engine_lives() -> None:
    req = _bare()
    threading.Timer(4 * _POLL_S, lambda: req.output_queue.put(_token("late"))).start()
    handle = RequestHandle(req, engine_alive=lambda: True)
    assert _run_async(handle.get_step_async).text == "late"


def test_get_step_async_needs_a_step_queue() -> None:
    req = RunningRequest(request=_request(), output_queue=queue.Queue())
    with pytest.raises(TypeError, match="StepQueue"):
        _run_async(RequestHandle(req, engine_alive=lambda: True).get_step_async)


def test_step_queue_put_survives_a_reader_whose_loop_closed() -> None:
    """The listener runs on the engine thread. A reader's loop that closed
    under it must not turn the engine's put into a crash."""
    loop = asyncio.new_event_loop()
    loop.close()
    out = StepQueue()
    out.set_listener(lambda: loop.call_soon_threadsafe(lambda: None))
    out.put_nowait(_token("kept"))
    assert out.get_nowait().text == "kept"


# ---- model-free engines ---------------------------------------------------


class _CharTokenizer:
    """One id per character; every id decodes to exactly one character."""

    eos_token_id = -1  # never sampled: the doubles favor token 0

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % (_VOCAB - 1)) + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


class _ScriptedRunner:
    """Duck-typed ModelRunner: a real BlockPool and a forward with three knobs.

    `crash_on_call` raises `_InjectedCrashError` on that 1-based forward call,
    which kills a `ContinuousScheduler` engine thread. `gate`, when set,
    makes every forward wait for it, so an engine can be alive but stalled.
    `delay_s` slows each forward down. Logits always favor token 0.
    """

    def __init__(
        self,
        *,
        crash_on_call: int | None = None,
        gate: threading.Event | None = None,
        delay_s: float = 0.0,
    ) -> None:
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
        self.tokenizer = _CharTokenizer()
        self.calls = 0
        self.entered = threading.Event()
        self._crash_on_call = crash_on_call
        self._gate = gate
        self._delay_s = delay_s

    def forward_step_packed(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
    ) -> torch.Tensor:
        self.calls += 1
        self.entered.set()
        if self.calls == self._crash_on_call:
            raise _InjectedCrashError("injected engine-thread crash")
        if self._gate is not None:
            self._gate.wait(_BOUND_S)
        if self._delay_s:
            time.sleep(self._delay_s)
        # Append K/V like a one-layer model, so that the cache really holds
        # blocks from the pool and a test can see them come back.
        kv = torch.zeros(cu_seqlens_q[-1], 1, 8)
        cache.append_kv_packed(kv, kv, torch.tensor(cu_seqlens_q, dtype=torch.int32), 0)
        logits = torch.full((1, cu_seqlens_q[-1], _VOCAB), -1.0)
        logits[..., 0] = 1.0
        return logits


def _emit_then_crash(running: RunningRequest) -> None:
    """What an injected engine step does: emit one token step, then die."""
    running.output_queue.put_nowait(_token("x"))
    raise _InjectedCrashError("injected engine-thread crash")


def _state_cache_generator() -> Any:
    model = SimpleNamespace(build_state_cache=lambda **_: object())
    return SimpleNamespace(
        model=model, device="cpu", dtype=torch.float32, tokenizer=_CharTokenizer()
    )


def _continuous() -> tuple[ContinuousScheduler, int]:
    # Forward 1 is the prefill and forward 2 the first decode, each followed
    # by one emitted token; forward 3 crashes.
    return ContinuousScheduler(_ScriptedRunner(crash_on_call=3)), 2  # type: ignore[arg-type]


def _state_cache() -> tuple[StateCacheContinuousScheduler, int]:
    sched = StateCacheContinuousScheduler(_state_cache_generator(), max_batch_size=2)
    sched._prefill_into_slot = lambda running, _slot: _emit_then_crash(running)  # type: ignore[method-assign]
    return sched, 1


def _tp() -> tuple[TensorParallelStateCacheScheduler, int]:
    server = SimpleNamespace(is_leader=True, tokenizer=_CharTokenizer())
    sched = TensorParallelStateCacheScheduler(server)  # type: ignore[arg-type]
    sched._serve = _emit_then_crash  # type: ignore[method-assign]
    return sched, 1


def _tp_continuous() -> tuple[TensorParallelStateCacheContinuousScheduler, int]:
    server = SimpleNamespace(
        is_leader=True, max_batch_size=2, tokenizer=_CharTokenizer(), shutdown=lambda: None
    )
    sched = TensorParallelStateCacheContinuousScheduler(server)  # type: ignore[arg-type]
    sched._prefill_into_slot = lambda running, _slot: _emit_then_crash(running)  # type: ignore[method-assign]
    return sched, 1


def _crash_on_admit(sched: PDScheduler) -> None:
    """Kill the thread that admits (serial: the only one; parallel: prefill)."""

    def admit() -> None:
        try:
            running = sched._waiting.get_nowait()
        except queue.Empty:
            return
        _emit_then_crash(running)

    sched._admit_waiting = admit  # type: ignore[method-assign]


def _pd_serial() -> tuple[PDScheduler, int]:
    sched = PDScheduler(_ScriptedRunner(), mode="serial")  # type: ignore[arg-type]
    _crash_on_admit(sched)
    return sched, 1


def _pd_parallel_prefill_dies() -> tuple[PDScheduler, int]:
    """Only the prefill thread dies; the decode thread lives on."""
    sched = PDScheduler(_ScriptedRunner(), mode="parallel")  # type: ignore[arg-type]
    _crash_on_admit(sched)
    return sched, 1


_SCHEDULERS: dict[str, Callable[[], tuple[Any, int]]] = {
    "continuous": _continuous,
    "state-cache-continuous": _state_cache,
    "tp-state-cache": _tp,
    "tp-state-cache-continuous": _tp_continuous,
    "pd-serial": _pd_serial,
    "pd-parallel-prefill-dies": _pd_parallel_prefill_dies,
}


@pytest.mark.parametrize("name", list(_SCHEDULERS))
def test_engine_death_fails_the_request_in_flight(
    name: str, crashes: list[type[BaseException]], caplog: pytest.LogCaptureFixture
) -> None:
    sched, n_buffered = _SCHEDULERS[name]()
    sched.start()
    try:
        handle = sched.submit(_request(max_tokens=8))
        texts: list[str] = []
        with pytest.raises(EngineNotRunningError):
            while True:
                step = _bounded(handle.get_step)
                assert step.finish_reason is None
                texts.append(step.text)
        # Steps emitted before the death still reach the caller.
        assert len(texts) == n_buffered
        assert crashes == [_InjectedCrashError]
        # The handle's error points to the log, so the cause must be there.
        assert any(
            r.levelno == logging.ERROR and "crashed" in r.getMessage() and r.exc_info
            for r in caplog.records
        )
        assert not sched.is_running
        with pytest.raises(EngineNotRunningError, match="not running"):
            sched.submit(_request())
    finally:
        sched.stop()


def test_pd_parallel_decode_thread_death_fails_the_request(
    crashes: list[type[BaseException]],
) -> None:
    """The prefill thread lives, but no request can finish without decode."""
    sched = PDScheduler(_ScriptedRunner(), mode="parallel")  # type: ignore[arg-type]
    sched._admit_waiting = lambda: None  # type: ignore[method-assign]
    armed = threading.Event()

    def drain() -> None:
        if armed.is_set():
            raise _InjectedCrashError("injected decode-thread crash")

    sched._drain_new_handoffs = drain  # type: ignore[method-assign]
    sched.start()
    try:
        handle = sched.submit(_request())
        armed.set()
        with pytest.raises(EngineNotRunningError):
            _bounded(handle.wait)
        assert crashes == [_InjectedCrashError]
        assert sched._thread is not None and sched._thread.is_alive()
        assert not sched.is_running
    finally:
        sched.stop()


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_pd_submit_needs_a_running_engine(mode: str) -> None:
    sched = PDScheduler(_ScriptedRunner(), mode=mode)  # type: ignore[arg-type]
    with pytest.raises(EngineNotRunningError, match="not running"):
        sched.submit(_request())
    sched.start()
    sched.stop()
    with pytest.raises(EngineNotRunningError, match="not running"):
        sched.submit(_request())


def _wait_concurrently(handles: list[RequestHandle]) -> tuple[list[Any], list[threading.Thread]]:
    """Start one `wait()` per handle on daemon threads, so they wait during a
    `stop()` instead of after it."""
    results: list[Any] = []

    def wait(handle: RequestHandle) -> None:
        try:
            results.append(handle.wait().finish_reason)
        except EngineNotRunningError as exc:
            results.append(type(exc).__name__)

    waiters = [threading.Thread(target=wait, args=(h,), daemon=True) for h in handles]
    for waiter in waiters:
        waiter.start()
    return results, waiters


def _join_all(waiters: list[threading.Thread]) -> None:
    for waiter in waiters:
        waiter.join(_BOUND_S)
        assert not waiter.is_alive(), "a waiter did not return within the bound"


def test_pd_stop_cancels_a_request_in_flight_instead_of_failing_it() -> None:
    """`stop()` sends 'cancelled' after the engine threads exit; the handle
    must wait for it instead of reporting a dead engine in that window."""
    sched = PDScheduler(_ScriptedRunner(), mode="serial")  # type: ignore[arg-type]
    # Admitted but never prefilled: the request parks in `_prefilling`.
    sched._run_prefill_batch = lambda: None  # type: ignore[method-assign]
    emit_terminal = sched._emit_terminal

    def slow_emit_terminal(r: RunningRequest, reason: str, *, failed: bool = False) -> None:
        time.sleep(4 * _POLL_S)  # several liveness polls see dead threads here
        emit_terminal(r, reason, failed=failed)

    sched._emit_terminal = slow_emit_terminal  # type: ignore[method-assign]
    sched.start()
    try:
        handle = sched.submit(_request())
        _wait_until(lambda: sched._prefilling)
        results, waiters = _wait_concurrently([handle])
        sched.stop()
        _join_all(waiters)
        assert results == ["cancelled"]
    finally:
        sched.stop()


def test_pd_stop_cancels_a_request_that_was_never_admitted() -> None:
    sched = PDScheduler(_ScriptedRunner(), mode="serial")  # type: ignore[arg-type]
    sched._admit_waiting = lambda: None  # type: ignore[method-assign]
    sched.start()
    try:
        handle = sched.submit(_request())
        sched.stop()
        assert _bounded(handle.wait).finish_reason == "cancelled"
    finally:
        sched.stop()


def test_continuous_stop_cancels_requests_in_flight_and_frees_their_blocks() -> None:
    """A clean `stop()` is a boundary: every request left running or waiting
    ends as 'cancelled', the pool gets its blocks back, and a restarted
    engine starts empty."""
    runner = _ScriptedRunner(delay_s=0.002)
    sched = ContinuousScheduler(runner, max_concurrent=1)  # type: ignore[arg-type]
    free_at_start = runner.block_pool.num_free_blocks
    finish = sched._finish

    def slow_finish(req: RunningRequest, reason: Any) -> None:
        time.sleep(4 * _POLL_S)  # the waiters poll a dead thread meanwhile
        finish(req, reason)

    sched.start()
    try:
        running = sched.submit(_request(max_tokens=100_000))
        waiting = sched.submit(_request(max_tokens=4))  # max_concurrent=1: never admitted
        assert _bounded(running.get_step).finish_reason is None
        sched._finish = slow_finish  # type: ignore[method-assign]
        results, waiters = _wait_concurrently([running, waiting])
        sched.stop()
        _join_all(waiters)
        assert results == ["cancelled", "cancelled"]
        assert runner.block_pool.num_free_blocks == free_at_start
        sched._finish = finish  # type: ignore[method-assign]
        sched.start()
        assert _bounded(sched.submit(_request(max_tokens=3)).wait).finish_reason == "length"
    finally:
        sched.stop()


def _occupy_slots(sched: Any) -> None:
    """Make a slot scheduler hold its requests forever, with no model."""

    def occupy(running: RunningRequest, slot_idx: int) -> None:
        sched._slots[slot_idx] = SimpleNamespace(request=running)

    sched._prefill_into_slot = occupy
    sched._decode_step = lambda: time.sleep(0.001)


@pytest.mark.parametrize("name", ["state-cache-continuous", "tp-state-cache-continuous"])
def test_slot_schedulers_stop_cancels_slotted_and_waiting_requests(name: str) -> None:
    sched: Any
    if name == "state-cache-continuous":
        sched = StateCacheContinuousScheduler(_state_cache_generator(), max_batch_size=1)
    else:
        server = SimpleNamespace(
            is_leader=True, max_batch_size=1, tokenizer=_CharTokenizer(), shutdown=lambda: None
        )
        sched = TensorParallelStateCacheContinuousScheduler(server)  # type: ignore[arg-type]
    _occupy_slots(sched)
    finish = sched._finish

    def slow_finish(running: RunningRequest, reason: Any) -> None:
        time.sleep(4 * _POLL_S)  # the waiters poll a dead thread meanwhile
        finish(running, reason)

    sched.start()
    try:
        slotted = sched.submit(_request())
        _wait_until(lambda: any(sched._slots))
        waiting = sched.submit(_request())  # the only slot is taken
        sched._finish = slow_finish
        results, waiters = _wait_concurrently([slotted, waiting])
        sched.stop()
        _join_all(waiters)
        assert results == ["cancelled", "cancelled"]
        assert not any(sched._slots)
    finally:
        sched.stop()


def test_tp_stop_cancels_the_requests_still_waiting() -> None:
    """The real `_serve` finishes the request it serves (the join waits for
    that generation); `stop()` cancels the ones still waiting."""
    serving = threading.Event()
    stop_requested = threading.Event()

    def generate_ids(*_: object, emit: Callable[[int], None], **__: object) -> None:
        emit(0)
        serving.set()
        stop_requested.wait(_BOUND_S)  # a generation that runs until stop()

    server = SimpleNamespace(is_leader=True, tokenizer=_CharTokenizer(), generate_ids=generate_ids)
    sched = TensorParallelStateCacheScheduler(server)  # type: ignore[arg-type]
    sched.start()
    try:
        served = sched.submit(_request(max_tokens=8))
        _wait_until(serving.is_set)
        waiting = sched.submit(_request())
        stopper = threading.Thread(target=sched.stop, daemon=True)
        stopper.start()
        _wait_until(sched._stop_event.is_set)  # the event that ends the loop
        stop_requested.set()
        stopper.join(_BOUND_S)
        assert not stopper.is_alive()
        served_result = _bounded(served.wait)
        assert (served_result.text, served_result.finish_reason) == ("a", "stop")
        assert _bounded(waiting.wait).finish_reason == "cancelled"
    finally:
        stop_requested.set()
        sched.stop()


def _crash_with_requests_tracked(name: str) -> tuple[Any, int, Callable[[], None]]:
    """A scheduler whose engine dies while it still tracks a request.

    Returns the scheduler, how many requests to submit, and a trigger to call
    after the submits. A crash must leave a request where `stop()` looks (a
    slot, the running batch, or a queue); otherwise `stop()` has nothing to
    misreport.
    """
    go = threading.Event()
    sched: Any
    if name == "continuous":
        sched = ContinuousScheduler(_ScriptedRunner(crash_on_call=3))  # type: ignore[arg-type]
        return sched, 1, go.set
    if name in ("state-cache-continuous", "tp-state-cache-continuous"):
        # One slot: the first request takes it, the second one waits.
        if name == "state-cache-continuous":
            sched = StateCacheContinuousScheduler(_state_cache_generator(), max_batch_size=1)
        else:
            server = SimpleNamespace(
                is_leader=True, max_batch_size=1, tokenizer=_CharTokenizer(), shutdown=lambda: None
            )
            sched = TensorParallelStateCacheContinuousScheduler(server)  # type: ignore[arg-type]
        _occupy_slots(sched)

        def decode_step() -> None:  # runs only while the slot is taken
            go.wait(_BOUND_S)  # the second request is waiting by now
            raise _InjectedCrashError("injected engine-thread crash")

        sched._decode_step = decode_step
        return sched, 2, go.set
    if name == "tp-state-cache":
        sched = _tp()[0]

        def serve(running: RunningRequest) -> None:
            go.wait(_BOUND_S)  # the second request is waiting by now
            raise _InjectedCrashError("injected engine-thread crash")

        sched._serve = serve
        return sched, 2, go.set
    if name == "pd-serial":
        sched = PDScheduler(_ScriptedRunner(), mode="serial")  # type: ignore[arg-type]

        def run_prefill_batch() -> None:  # the request sits in `_prefilling`
            if sched._prefilling:
                raise _InjectedCrashError("injected engine-thread crash")

        sched._run_prefill_batch = run_prefill_batch
        return sched, 1, go.set
    sched = PDScheduler(_ScriptedRunner(), mode="parallel")  # type: ignore[arg-type]
    sched._admit_waiting = lambda: None  # the request stays in `_waiting`

    def drain() -> None:
        if go.is_set():
            raise _InjectedCrashError("injected decode-thread crash")

    sched._drain_new_handoffs = drain
    return sched, 1, go.set


@pytest.mark.parametrize(
    "name",
    [
        "continuous",
        "state-cache-continuous",
        "tp-state-cache-continuous",
        "tp-state-cache",
        "pd-serial",
        "pd-parallel",
    ],
)
def test_stop_right_after_a_crash_still_reports_the_crash(
    name: str, crashes: list[type[BaseException]]
) -> None:
    """A `stop()` that lands before the waiters' next liveness poll must not
    turn the crash into 'cancelled': the outcome may not depend on timing."""
    sched, n_requests, trigger = _crash_with_requests_tracked(name)
    sched.start()
    try:
        handles = [sched.submit(_request(max_tokens=8)) for _ in range(n_requests)]
        trigger()
        _wait_until(lambda: not sched.is_running)
        results, waiters = _wait_concurrently(handles)
        sched.stop()
        _join_all(waiters)
        assert results == ["EngineNotRunningError"] * n_requests
        assert crashes == [_InjectedCrashError]
    finally:
        sched.stop()


# What a failing prefill or generation raises. The tests check that this text
# reaches the log and never the caller.
_SECRET = "failed at /srv/models/secret-path, shape (7, 4096)"


def _pd_with_failing_prefill(mode: str) -> PDScheduler:
    sched = PDScheduler(_ScriptedRunner(), mode=mode)  # type: ignore[arg-type]

    def prefill_batch(_requests: list[Request]) -> list[Any]:
        raise RuntimeError(_SECRET)

    sched._prefill_worker.prefill_batch = prefill_batch  # type: ignore[method-assign]
    return sched


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_pd_prefill_failure_fails_the_request_and_keeps_the_engine(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    sched = _pd_with_failing_prefill(mode)
    sched.start()
    try:
        for _ in range(2):  # the engine keeps serving after a failure
            handle = sched.submit(_request())
            with pytest.raises(RequestFailedError) as failure:
                _bounded(handle.wait)
            assert _SECRET not in str(failure.value)
        assert sched.is_running
        assert any(r.exc_info and _SECRET in str(r.exc_info[1]) for r in caplog.records)
    finally:
        sched.stop()


@pytest.mark.parametrize(
    "name",
    [
        "continuous",
        "state-cache-continuous",
        "tp-state-cache-continuous",
        "tp-state-cache",
        "pd-serial",
    ],
)
def test_a_quick_restart_after_a_crash_does_not_revive_its_requests(
    name: str, crashes: list[type[BaseException]]
) -> None:
    """`start()` right after a crash, with no `stop()`: the requests that the
    dead thread left must not move to the new thread (their handles watch the
    dead one, and the state can be half-updated). They end with the crash."""
    sched, n_requests, trigger = _crash_with_requests_tracked(name)
    sched.start()
    try:
        handles = [sched.submit(_request(max_tokens=8)) for _ in range(n_requests)]
        trigger()
        _wait_until(lambda: not sched.is_running)
        sched.start()
        results, waiters = _wait_concurrently(handles)
        _join_all(waiters)
        assert results == ["EngineNotRunningError"] * n_requests
        assert crashes == [_InjectedCrashError]
    finally:
        sched.stop()


def test_a_crash_inside_the_oom_handler_is_a_crash(
    crashes: list[type[BaseException]],
) -> None:
    """The OOM handler runs inside the engine loop too. If it raises, the
    thread dies, and `stop()` must treat it like any other crash."""

    class _OomRunner(_ScriptedRunner):
        def forward_step_packed(self, cache: PagedKVCache, *args: Any) -> torch.Tensor:
            if self.calls >= 2:
                raise OutOfMemoryError("BlockPool: no free blocks")
            return super().forward_step_packed(cache, *args)

    sched = ContinuousScheduler(_OomRunner())  # type: ignore[arg-type]

    def preempt_on_oom() -> None:
        raise _InjectedCrashError("injected crash in the OOM handler")

    sched._preempt_on_oom = preempt_on_oom  # type: ignore[method-assign]
    sched.start()
    try:
        handle = sched.submit(_request(max_tokens=8))
        _wait_until(lambda: not sched.is_running)
        results, waiters = _wait_concurrently([handle])
        sched.stop()
        _join_all(waiters)
        assert results == ["EngineNotRunningError"]
        assert crashes == [_InjectedCrashError]
    finally:
        sched.stop()


def test_pd_stop_with_a_timed_out_join_drains_nothing() -> None:
    """A thread that outlives the join can still touch the state, so `stop()`
    must leave it alone. The handle then fails once the thread exits."""
    sched = PDScheduler(_ScriptedRunner(), mode="serial")  # type: ignore[arg-type]
    entered = threading.Event()
    release = threading.Event()

    def run_prefill_batch() -> None:  # a prefill that outlasts the join
        if sched._prefilling:
            entered.set()
            release.wait(_BOUND_S)

    sched._run_prefill_batch = run_prefill_batch  # type: ignore[method-assign]
    sched.start()
    engine = sched._thread
    assert engine is not None
    try:
        handle = sched.submit(_request())
        assert entered.wait(_BOUND_S)
        sched.stop(timeout=0.05)
        assert engine.is_alive()
        assert len(sched._prefilling) == 1  # not drained under the live thread
        sched.stop(timeout=0.05)  # a second stop must not drain it either
        assert len(sched._prefilling) == 1
        release.set()
        engine.join(_BOUND_S)
        with pytest.raises(EngineNotRunningError):
            _bounded(handle.wait)
        # The next start() drains what the timed-out stop left, so the stale
        # request neither moves to the new thread nor counts toward the
        # concurrency limit.
        sched.start()
        assert sched._prefilling == []
    finally:
        release.set()
        sched.stop()


def test_pd_second_stop_ends_a_timed_out_stops_leftovers_like_start() -> None:
    """After a timed-out stop, the leftover request belongs to a thread that
    its handle watches. A second `stop()` must end it with no terminal step,
    as `start()` does, whether or not the waiter polls during that stop."""
    sched = PDScheduler(_ScriptedRunner(), mode="serial")  # type: ignore[arg-type]
    entered = threading.Event()
    release = threading.Event()

    def run_prefill_batch() -> None:
        if sched._prefilling:
            entered.set()
            release.wait(_BOUND_S)

    sched._run_prefill_batch = run_prefill_batch  # type: ignore[method-assign]
    emit_terminal = sched._emit_terminal

    def slow_emit_terminal(r: RunningRequest, reason: str, *, failed: bool = False) -> None:
        time.sleep(4 * _POLL_S)  # the waiter polls during the second stop
        emit_terminal(r, reason, failed=failed)

    sched._emit_terminal = slow_emit_terminal  # type: ignore[method-assign]
    sched.start()
    engine = sched._thread
    assert engine is not None
    try:
        handle = sched.submit(_request())
        assert entered.wait(_BOUND_S)
        sched.stop(timeout=0.05)
        release.set()
        engine.join(_BOUND_S)
        results, waiters = _wait_concurrently([handle])
        sched.stop()
        _join_all(waiters)
        assert results == ["EngineNotRunningError"]
        assert sched._prefilling == []  # the second stop drained it
    finally:
        release.set()
        sched.stop()


def test_pd_parallel_prefill_thread_hands_back_on_a_clean_stop() -> None:
    """A clean `stop()` that interrupts a blocked handoff still ends the
    request as 'cancelled': the prefill thread hands it back to `stop()`."""
    sched = PDScheduler(_ScriptedRunner(), mode="parallel", max_concurrent=1)  # type: ignore[arg-type]
    prefilled = threading.Semaphore(0)

    def prefill_batch(requests: list[Request]) -> list[Any]:
        for _ in requests:
            prefilled.release()
        return [object() for _ in requests]

    sched._prefill_worker.prefill_batch = prefill_batch  # type: ignore[method-assign]
    # The decode thread takes no handoff; it only waits for the stop.
    sched._decode_thread_loop = lambda: sched._stop_event.wait(_BOUND_S)  # type: ignore[method-assign]
    sched.start()
    try:
        first = sched.submit(_request())
        second = sched.submit(_request())
        for _ in range(2):
            assert prefilled.acquire(timeout=_BOUND_S)
        assert sched._handoff_queue is not None
        _wait_until(sched._handoff_queue.full)  # the second put now blocks
        results, waiters = _wait_concurrently([first, second])
        sched.stop()
        _join_all(waiters)
        assert results == ["cancelled", "cancelled"]
    finally:
        sched.stop()


def test_pd_parallel_prefill_thread_does_not_cancel_after_a_decode_crash(
    crashes: list[type[BaseException]],
) -> None:
    """The prefill thread can be blocked on a full handoff queue when `stop()`
    begins. It hands the request back instead of cancelling it, so `stop()`
    applies the crash rule to it too."""
    sched = PDScheduler(_ScriptedRunner(), mode="parallel", max_concurrent=1)  # type: ignore[arg-type]
    prefilled = threading.Semaphore(0)

    def prefill_batch(requests: list[Request]) -> list[Any]:
        for _ in requests:
            prefilled.release()
        return [object() for _ in requests]

    sched._prefill_worker.prefill_batch = prefill_batch  # type: ignore[method-assign]
    armed = threading.Event()

    def decode_loop() -> None:  # takes no handoff, then crashes when armed
        armed.wait(_BOUND_S)
        raise _InjectedCrashError("injected decode-thread crash")

    sched._decode_thread_loop = decode_loop  # type: ignore[method-assign]
    sched.start()
    try:
        first = sched.submit(_request())
        second = sched.submit(_request())
        for _ in range(2):
            assert prefilled.acquire(timeout=_BOUND_S)
        # The first handoff fills the queue (size 1); the second one blocks.
        assert sched._handoff_queue is not None
        _wait_until(sched._handoff_queue.full)
        armed.set()
        _wait_until(lambda: not sched.is_running)
        results, waiters = _wait_concurrently([first, second])
        sched.stop()
        _join_all(waiters)
        assert results == ["EngineNotRunningError", "EngineNotRunningError"]
        assert crashes == [_InjectedCrashError]
    finally:
        sched.stop()


def test_a_clean_stop_after_a_crash_and_a_restart_cancels(
    crashes: list[type[BaseException]],
) -> None:
    """The crash belongs to the old thread. After `start()`, a clean `stop()`
    ends the new thread's requests as 'cancelled' again."""
    runner = _ScriptedRunner(crash_on_call=3, delay_s=0.002)
    sched = ContinuousScheduler(runner)  # type: ignore[arg-type]
    sched.start()
    try:
        crashed = sched.submit(_request(max_tokens=8))
        with pytest.raises(EngineNotRunningError):
            _bounded(crashed.wait)
        sched.start()  # forward call 3 is behind us; the new thread does not crash
        running = sched.submit(_request(max_tokens=100_000))
        assert _bounded(running.get_step).finish_reason is None
        sched.stop()
        assert _bounded(running.wait).finish_reason == "cancelled"
        assert crashes == [_InjectedCrashError]
    finally:
        sched.stop()


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_pd_prefill_out_of_memory_cancels_the_batch_instead_of_failing_it(mode: str) -> None:
    """A full KV pool is a resource limit, as for `ContinuousScheduler`'s
    preemption, not an engine failure."""
    sched = PDScheduler(_ScriptedRunner(), mode=mode)  # type: ignore[arg-type]

    def prefill_batch(_requests: list[Request]) -> list[Any]:
        raise OutOfMemoryError("BlockPool: no free blocks")

    sched._prefill_worker.prefill_batch = prefill_batch  # type: ignore[method-assign]
    sched.start()
    try:
        assert _bounded(sched.submit(_request()).wait).finish_reason == "cancelled"
        assert sched.is_running
    finally:
        sched.stop()


def test_tp_generation_failure_fails_the_request_and_keeps_the_engine() -> None:
    def generate_ids(*_: object, **__: object) -> None:
        raise RuntimeError(_SECRET)

    server = SimpleNamespace(is_leader=True, tokenizer=_CharTokenizer(), generate_ids=generate_ids)
    sched = TensorParallelStateCacheScheduler(server)  # type: ignore[arg-type]
    sched.start()
    try:
        with pytest.raises(RequestFailedError) as failure:
            _bounded(sched.submit(_request()).wait)
        assert _SECRET not in str(failure.value)
        assert sched.is_running
    finally:
        sched.stop()


def test_pd_terminal_step_lands_even_on_a_full_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dropped terminal step would leave the handle waiting forever on a
    live engine, which no liveness check can see."""
    monkeypatch.setattr(PDScheduler, "TERMINAL_PUT_TIMEOUT_S", 0.01)
    sched = PDScheduler(_ScriptedRunner(), mode="serial")  # type: ignore[arg-type]
    req = RunningRequest(request=_request(), output_queue=StepQueue(maxsize=2))
    req.output_queue.put(_token("a"))
    req.output_queue.put(_token("b"))
    sched._emit_terminal(req, "length")
    result = _bounded(RequestHandle(req, engine_alive=lambda: True).wait)
    # One buffered step made room for the terminal step.
    assert (result.text, result.finish_reason) == ("b", "length")


def test_a_handle_stays_bound_to_the_thread_that_crashed(
    crashes: list[type[BaseException]],
) -> None:
    """After a crash, a quick `start()` brings up a thread that never saw the
    request (the StateCache schedulers keep no trace of it). The handle must
    watch the thread that owned it, or it would wait on the new one forever."""
    sched, _ = _state_cache()  # each prefill emits one step, then crashes
    sched.start()
    try:
        handle = sched.submit(_request())
        _wait_until(lambda: not sched.is_running)
        sched.start()
        assert sched.is_running
        assert _bounded(handle.get_step).text == "x"
        with pytest.raises(EngineNotRunningError):
            _bounded(lambda: handle.wait(timeout=2.0))
        assert crashes == [_InjectedCrashError]
    finally:
        sched.stop()


# ---- HTTP -----------------------------------------------------------------


class _Server:
    """The real `completions` route on a fresh app over one scheduler.

    Every handle the endpoint gets is recorded, so teardown can end it: if a
    regression leaves the endpoint blocked, `TestClient` cannot exit until the
    handle's queue gets its terminal step.
    """

    def __init__(self, sched: Any, *, start: bool = True) -> None:
        self.sched = sched
        self.handles: list[RequestHandle] = []
        self.submitted = threading.Semaphore(0)
        submit = sched.submit

        def spy(request: Request) -> RequestHandle:
            handle: RequestHandle = submit(request)
            self.handles.append(handle)
            self.submitted.release()
            return handle

        sched.submit = spy
        app = FastAPI()
        app.add_api_route("/v1/completions", completions, methods=["POST"], response_model=None)
        app.add_api_route("/ping", lambda: {"ok": True}, methods=["GET"])
        app.add_exception_handler(Exception, _unhandled_exception_handler)
        app.state.scheduler = sched
        self.app = app
        if start:
            sched.start()

    def release(self) -> None:
        """Give every recorded handle a terminal step, making room if needed."""
        for handle in self.handles:
            out = handle._req.output_queue
            with contextlib.suppress(queue.Empty):
                if out.full():
                    out.get_nowait()
            with contextlib.suppress(queue.Full):
                out.put_nowait(GenerationStep(text="", finish_reason="stop"))


@pytest.fixture
def serve() -> Iterator[Callable[..., tuple[_Server, TestClient]]]:
    opened: list[tuple[_Server, TestClient, list[threading.Event]]] = []

    def open_server(
        sched: Any, *, start: bool = True, gates: tuple[threading.Event, ...] = ()
    ) -> tuple[_Server, TestClient]:
        server = _Server(sched, start=start)
        client = TestClient(server.app)
        client.__enter__()
        opened.append((server, client, list(gates)))
        return server, client

    yield open_server
    for server, client, gates in opened:
        for gate in gates:
            gate.set()
        server.release()
        client.__exit__(None, None, None)
        server.sched.stop()


def _post(client: TestClient, *, stream: bool = False, max_tokens: int = 4) -> Any:
    return client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "abc", "max_tokens": max_tokens, "stream": stream},
    )


def _sse_events(client: TestClient, max_tokens: int = 8) -> tuple[int, list[str]]:
    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": "m", "prompt": "abc", "max_tokens": max_tokens, "stream": True},
    ) as response:
        events = [
            line[len("data: ") :] for line in response.iter_lines() if line.startswith("data: ")
        ]
        return response.status_code, events


@pytest.mark.parametrize("stream", [False, True])
def test_http_rejects_a_request_when_the_engine_is_not_running(
    serve: Callable[..., tuple[_Server, TestClient]], stream: bool
) -> None:
    _, client = serve(ContinuousScheduler(_ScriptedRunner()), start=False)  # type: ignore[arg-type]
    response = _bounded(lambda: _post(client, stream=stream))
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "engine_unavailable"


def test_http_non_streaming_request_fails_when_the_engine_dies(
    serve: Callable[..., tuple[_Server, TestClient]], crashes: list[type[BaseException]]
) -> None:
    _, client = serve(ContinuousScheduler(_ScriptedRunner(crash_on_call=1)))  # type: ignore[arg-type]
    response = _bounded(lambda: _post(client))
    assert response.status_code == 503
    assert response.json()["error"] == {
        "type": "engine_unavailable",
        "message": "engine is not running",
    }
    assert crashes == [_InjectedCrashError]


def test_http_stream_reports_a_mid_stream_death_in_band(
    serve: Callable[..., tuple[_Server, TestClient]], crashes: list[type[BaseException]]
) -> None:
    _, client = serve(ContinuousScheduler(_ScriptedRunner(crash_on_call=3)))  # type: ignore[arg-type]
    status, events = _bounded(lambda: _sse_events(client))
    assert status == 200
    *chunks, error, done = events
    assert [json.loads(c)["choices"][0] for c in chunks] == [
        {"text": "a", "index": 0, "finish_reason": None},
        {"text": "a", "index": 0, "finish_reason": None},
    ]
    assert json.loads(error)["error"]["type"] == "engine_unavailable"
    assert done == "[DONE]"
    assert crashes == [_InjectedCrashError]


def test_http_non_streaming_request_failure_is_a_500_without_the_cause(
    serve: Callable[..., tuple[_Server, TestClient]],
) -> None:
    _, client = serve(_pd_with_failing_prefill("serial"))
    response = _bounded(lambda: _post(client))
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert _SECRET not in response.text


def test_http_stream_failure_is_an_error_event_without_the_cause(
    serve: Callable[..., tuple[_Server, TestClient]],
) -> None:
    _, client = serve(_pd_with_failing_prefill("serial"))
    status, events = _bounded(lambda: _sse_events(client))
    assert status == 200
    assert [json.loads(e)["error"]["type"] for e in events[:-1]] == ["server_error"]
    assert events[-1] == "[DONE]"
    assert not any(_SECRET in e for e in events)


def test_http_non_streaming_request_does_not_block_the_event_loop(
    serve: Callable[..., tuple[_Server, TestClient]],
) -> None:
    """While one completion waits on a stalled engine, other routes answer."""
    gate = threading.Event()
    runner = _ScriptedRunner(gate=gate)
    _, client = serve(ContinuousScheduler(runner), gates=(gate,))  # type: ignore[arg-type]
    pending: list[Any] = []
    first = threading.Thread(target=lambda: pending.append(_post(client)), daemon=True)
    first.start()
    assert runner.entered.wait(_BOUND_S)
    # Several liveness slices pass while the engine is stalled: an empty slice
    # on a live engine must not fail the request.
    time.sleep(6 * _POLL_S)
    assert _bounded(lambda: client.get("/ping"), timeout=2.0).status_code == 200
    assert first.is_alive()
    gate.set()
    first.join(_BOUND_S)
    assert not first.is_alive()
    body = pending[0].json()
    assert pending[0].status_code == 200
    assert body["choices"][0] == {"text": "aaaa", "index": 0, "finish_reason": "length"}


def test_http_non_streaming_requests_do_not_starve_each_other_of_threads(
    serve: Callable[..., tuple[_Server, TestClient]],
) -> None:
    """With one executor thread and small output queues, two concurrent
    completions both keep every token. A request that held the thread for its
    whole generation would leave the other's queue unread, and that queue
    drops steps once it is full."""
    gate = threading.Event()
    # 20 ms per forward gives a stalled event loop about 80 ms of slack before
    # a 4-step queue overflows. A reader that holds the only thread for a
    # whole generation still overflows it, whatever the delay.
    sched = ContinuousScheduler(_ScriptedRunner(gate=gate, delay_s=0.02))  # type: ignore[arg-type]
    sched.DEFAULT_OUTPUT_QUEUE_SIZE = 4  # type: ignore[misc]
    server, client = serve(sched, gates=(gate,))
    _one_executor_thread(client)
    _assert_no_text_lost(server, client, gate, n_requests=2, max_tokens=12)


def test_http_requests_waiting_for_admission_do_not_starve_the_running_one(
    serve: Callable[..., tuple[_Server, TestClient]],
) -> None:
    """Two requests wait in the admission queue while one generates, and the
    executor has one thread. A reader that holds a thread while it waits, even
    in short slices, leaves the running request's small queue unread long
    enough to overflow. Reading on the event loop needs no thread at all."""
    gate = threading.Event()
    sched = ContinuousScheduler(  # type: ignore[arg-type]
        _ScriptedRunner(gate=gate, delay_s=0.02), max_concurrent=1
    )
    sched.DEFAULT_OUTPUT_QUEUE_SIZE = 4  # type: ignore[misc]
    server, client = serve(sched, gates=(gate,))
    _one_executor_thread(client)
    _assert_no_text_lost(server, client, gate, n_requests=3, max_tokens=12)


def _one_executor_thread(client: TestClient) -> None:
    """Shrink the app loop's default executor to one thread."""

    async def shrink() -> None:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

    assert client.portal is not None
    client.portal.call(shrink)


def _assert_no_text_lost(
    server: _Server, client: TestClient, gate: threading.Event, *, n_requests: int, max_tokens: int
) -> None:
    """Post `n_requests` concurrent completions while the engine is gated,
    open the gate, and check that each one kept every token's text."""
    responses: list[Any] = []
    posts = [
        threading.Thread(
            target=lambda: responses.append(_post(client, max_tokens=max_tokens)), daemon=True
        )
        for _ in range(n_requests)
    ]
    for post in posts:
        post.start()
    for _ in posts:
        assert server.submitted.acquire(timeout=_BOUND_S)
    gate.set()
    for post in posts:
        post.join(_BOUND_S)
        assert not post.is_alive()
    assert len(responses) == n_requests
    for response in responses:
        assert response.status_code == 200
        assert response.json()["choices"][0]["text"] == "a" * max_tokens
        assert response.json()["usage"]["completion_tokens"] == max_tokens
