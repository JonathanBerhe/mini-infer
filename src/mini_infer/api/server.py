import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import StreamingResponse

from mini_infer.api.schemas import (
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_name = os.environ.get("MINI_INFER_MODEL", DEFAULT_MODEL)
    runner = ModelRunner.from_pretrained(model_name)
    scheduler = ContinuousScheduler(runner)
    scheduler.start()
    app.state.scheduler = scheduler
    try:
        yield
    finally:
        scheduler.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/completions", response_model=None)
async def completions(
    req: CompletionRequest,
    fastapi_req: FastAPIRequest,
) -> StreamingResponse | CompletionResponse:
    scheduler: ContinuousScheduler = fastapi_req.app.state.scheduler
    internal = Request(
        prompt=req.prompt,
        sampling_params=SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        ),
        max_tokens=req.max_tokens,
    )
    completion_id = f"cmpl-{uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        # Sync generator so Starlette runs it in a threadpool: blocking queue.get()
        # in the scheduler doesn't stall the event loop when other requests are in flight.
        def event_stream() -> Iterator[str]:
            for step in scheduler.stream(internal):
                chunk = CompletionChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[
                        CompletionChoice(text=step.text, finish_reason=step.finish_reason),
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = scheduler.run(internal)
    return CompletionResponse(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[CompletionChoice(text=result.text, finish_reason=result.finish_reason)],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=len(result.tokens),
            total_tokens=result.prompt_tokens + len(result.tokens),
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mini_infer.api.server:app", host="0.0.0.0", port=8000)
