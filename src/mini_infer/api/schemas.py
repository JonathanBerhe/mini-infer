from typing import Literal

from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Literal["stop", "length"] | None = None


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
