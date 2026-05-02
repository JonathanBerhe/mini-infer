"""Scheduler trace export → Chrome Trace Event Format.

The scheduler can record one duration event per engine step (when a forward
runs) and dump them as JSON on `stop()`. The output is loadable in
`chrome://tracing` or Perfetto UI without further processing.
"""

import json
from pathlib import Path

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.requires_model
def test_trace_out_dumps_chrome_trace_format(tmp_path: Path) -> None:
    """Run a request, then verify the trace file is valid Chrome Trace JSON."""
    trace_path = tmp_path / "trace.json"

    runner = ModelRunner.from_pretrained(MODEL_NAME)
    sched = ContinuousScheduler(runner, trace_out=str(trace_path))
    sched.start()
    try:
        result = sched.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=4,
            )
        )
    finally:
        sched.stop()

    assert "Paris" in result.text  # sanity: tracing didn't break decoding
    assert trace_path.exists()

    blob = json.loads(trace_path.read_text())
    events = blob["traceEvents"]
    assert len(events) > 0

    last_ts = -1.0
    phases_seen: set[str] = set()
    for ev in events:
        assert ev["name"] == "step"
        assert ev["ph"] == "X"
        assert isinstance(ev["ts"], (int, float))
        assert isinstance(ev["dur"], (int, float))
        assert ev["dur"] >= 0
        assert ev["ts"] >= last_ts, "ts must be monotonic"
        last_ts = ev["ts"]
        args = ev["args"]
        assert args["B"] >= 1
        assert args["phase"] in {"prefill", "decode", "mixed"}
        phases_seen.add(args["phase"])

    # A full run with max_tokens=4 always covers prefill + at least one decode.
    assert "prefill" in phases_seen
    assert "decode" in phases_seen


def test_classify_phase_labels_correctly() -> None:
    """Phase label reflects the states of the alive set passed to forward."""
    import queue as _queue

    from mini_infer.scheduler.request_state import RequestState, RunningRequest

    def fake(state: RequestState) -> RunningRequest:
        r = RunningRequest(
            request=Request(prompt="p", sampling_params=SamplingParams(), max_tokens=1),
            output_queue=_queue.Queue(),
        )
        r.state = state
        return r

    classify = ContinuousScheduler._classify_phase
    assert classify([fake(RequestState.PREFILLING)]) == "prefill"
    assert classify([fake(RequestState.CHUNKED_PREFILLING)]) == "prefill"
    assert (
        classify([fake(RequestState.PREFILLING), fake(RequestState.CHUNKED_PREFILLING)])
        == "prefill"
    )
    assert classify([fake(RequestState.DECODING)]) == "decode"
    assert classify([fake(RequestState.DECODING), fake(RequestState.DECODING)]) == "decode"
    assert classify([fake(RequestState.PREFILLING), fake(RequestState.DECODING)]) == "mixed"
