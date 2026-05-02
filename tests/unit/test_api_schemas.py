import pytest
from pydantic import ValidationError

from mini_infer.api.schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)


def test_completion_request_defaults() -> None:
    req = CompletionRequest(model="m", prompt="p")
    assert req.max_tokens == 16
    assert req.temperature == 0.0
    assert req.top_p == 1.0
    assert req.top_k == 0
    assert req.stream is False


def test_completion_request_requires_model_and_prompt() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="p")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        CompletionRequest(model="m")  # type: ignore[call-arg]


def test_completion_request_accepts_stream_flag() -> None:
    req = CompletionRequest(model="m", prompt="p", stream=True)
    assert req.stream is True


def test_completion_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="")


def test_completion_request_rejects_oversized_prompt() -> None:
    """Prompt length is bounded by `MINI_INFER_MAX_PROMPT_CHARS` (default 131072)."""
    from mini_infer.api.schemas import MAX_PROMPT_CHARS

    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="x" * (MAX_PROMPT_CHARS + 1))


def test_completion_request_rejects_invalid_max_tokens() -> None:
    """`max_tokens` must be in [1, MAX_OUTPUT_TOKENS]."""
    from mini_infer.api.schemas import MAX_OUTPUT_TOKENS

    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="p", max_tokens=0)
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="p", max_tokens=-1)
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="p", max_tokens=MAX_OUTPUT_TOKENS + 1)


def test_completion_request_rejects_invalid_sampling_params() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="p", temperature=-0.1)
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="p", top_p=0.0)
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="p", top_p=1.1)
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="p", top_k=-1)


def test_completion_response_round_trips_through_json() -> None:
    resp = CompletionResponse(
        id="cmpl-abc",
        created=1234,
        model="m",
        choices=[CompletionChoice(text="hello", finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )
    blob = resp.model_dump_json()
    parsed = CompletionResponse.model_validate_json(blob)
    assert parsed == resp
    assert parsed.object == "text_completion"
