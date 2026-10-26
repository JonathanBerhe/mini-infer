import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from mini_infer.api.server import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.mark.requires_model
def test_completions_non_stream_returns_paris(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt": "The capital of France is",
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "Paris" in body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


@pytest.mark.requires_model
def test_completions_stream_yields_chunks_and_done(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        chunks: list[dict[str, object]] = []
        saw_done = False
        for raw_line in response.iter_lines():
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(payload))

    assert saw_done
    assert any(c["choices"][0]["text"] for c in chunks)
    finish_reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert any(fr in {"stop", "length"} for fr in finish_reasons)


def test_unknown_endpoint_returns_404(client: TestClient) -> None:
    assert client.get("/v1/nope").status_code == 404


def test_oversized_prompt_returns_422(client: TestClient) -> None:
    """Pydantic rejects oversized prompts before any engine work happens."""
    from mini_infer.api.schemas import MAX_PROMPT_CHARS

    response = client.post(
        "/v1/completions",
        json={
            "model": "m",
            "prompt": "x" * (MAX_PROMPT_CHARS + 1),
            "max_tokens": 1,
        },
    )
    assert response.status_code == 422


def test_negative_max_tokens_returns_422(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "p", "max_tokens": -1},
    )
    assert response.status_code == 422


def test_request_handle_cancel_marks_event() -> None:
    """`RequestHandle.cancel` is the API-thread side of the cancel contract.

    Verifies the public surface without spinning up the scheduler: the
    engine-side check (`req.cancel_event.is_set()`) is the same flag this
    sets, so a thread-safe `cancel()` call is enough to wire the path.
    """
    import queue

    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler import Request, RequestHandle
    from mini_infer.scheduler.request_state import RunningRequest

    req = RunningRequest(
        request=Request(prompt="p", sampling_params=SamplingParams(), max_tokens=1),
        output_queue=queue.Queue(maxsize=4),
    )
    handle = RequestHandle(req)
    assert not req.cancel_event.is_set()
    handle.cancel()
    assert req.cancel_event.is_set()
    handle.cancel()  # idempotent
    assert req.cancel_event.is_set()
