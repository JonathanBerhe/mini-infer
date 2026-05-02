"""Pydantic schemas for the OpenAI-compatible HTTP API.

Field bounds (`max_tokens`, prompt length) protect the engine from
remote-callable resource exhaustion. Defaults are conservative; deployments
that need larger limits override via env vars (`MINI_INFER_MAX_PROMPT_CHARS`,
`MINI_INFER_MAX_OUTPUT_TOKENS`) at process start.
"""

import os
from typing import Literal

from pydantic import BaseModel, Field

# Hard upper bounds on per-request inputs. Validated by Pydantic so a remote
# caller can't force the engine to tokenize a multi-megabyte prompt or
# allocate decode space for an unbounded `max_tokens`. The defaults assume a
# 0.5B-7B-class model on a single GPU; bump via env var for bigger
# deployments that have already sized their block pool accordingly.
MAX_PROMPT_CHARS: int = int(os.environ.get("MINI_INFER_MAX_PROMPT_CHARS", "131072"))
MAX_OUTPUT_TOKENS: int = int(os.environ.get("MINI_INFER_MAX_OUTPUT_TOKENS", "4096"))


class CompletionRequest(BaseModel):
    model: str = Field(min_length=1, max_length=512)
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_tokens: int = Field(default=16, ge=1, le=MAX_OUTPUT_TOKENS)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Literal["stop", "length", "cancelled"] | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion.chunk"] = "text_completion.chunk"
    created: int
    model: str
    choices: list[CompletionChoice]
