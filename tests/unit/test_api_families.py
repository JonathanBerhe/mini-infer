"""Per-family API smoke: each PagedKVCache family serves over the shared
`/v1/completions` interface, through the real server lifespan + routing.

This turns "works by construction" into "tested" for the families that have a
small pinned checkpoint. It drives the REAL `app` (its lifespan loads the model
named by `MINI_INFER_MODEL`, routes it to `ContinuousScheduler`, and serves),
so the whole path is exercised per family, not just the shared HTTP layer.

`requires_model`: each case downloads its pinned small model; these run in the
bit-parity CI job and locally when the models are present.

Coverage note: only four families have a small enough checkpoint to smoke this
way (Qwen2, Qwen3, Llama, Gemma 3). Mistral / Gemma 4 / Mixtral / DeepSeek-V2
have no small public checkpoint (Mixtral and V2 are far too large for CI), but
they ride the identical `ModelRunner` + `ContinuousScheduler` +
`forward_step_packed` path proven here, and each one's forward is covered by its
golden tests.
DeepSeek-V4 (the StateCache path) is smoked separately and synthetically in
`test_api_state_cache.py`, since it has no single-GPU checkpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mini_infer.api.server import app

# One pinned small model per PagedKVCache architecture family (matches
# tests/_pinned_models.toml).
_PINNED_FAMILY_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # Qwen2
    "Qwen/Qwen3-0.6B",  # Qwen3
    "HuggingFaceTB/SmolLM2-135M-Instruct",  # Llama family
    "unsloth/gemma-3-1b-it",  # Gemma 3
]


@pytest.mark.requires_model
@pytest.mark.parametrize("model_name", _PINNED_FAMILY_MODELS)
def test_api_serves_pinned_family_over_http(
    model_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pinned family loads via the real lifespan and serves a completion."""
    monkeypatch.setenv("MINI_INFER_MODEL", model_name)
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": model_name, "prompt": "The capital of France is", "max_tokens": 8},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"]  # served some text
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
