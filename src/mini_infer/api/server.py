"""OpenAI-compatible HTTP API for mini-infer.

Security posture (the server is meant to run behind a trusted reverse proxy
or on localhost — these guards are belt-and-suspenders):

- Defaults to binding `127.0.0.1`. Set `MINI_INFER_HOST=0.0.0.0` to expose.
- Optional bearer-token auth: set `MINI_INFER_API_KEY=<secret>` and clients
  must send `Authorization: Bearer <secret>`. Unset = open access (intended
  for dev / trusted networks).
- Per-request input bounds (`prompt` length, `max_tokens`) are enforced by
  Pydantic in `schemas.py`; oversized payloads return 422 before any
  engine work happens.
- Streaming endpoint polls for client disconnect between yielded tokens
  and cancels the request when the client drops the connection.
- A global exception handler converts uncaught engine errors into a sanitized
  503 instead of letting FastAPI emit a stack trace.
"""

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mini_infer.api.schemas import (
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)
from mini_infer.cache.state_prefix_cache import StatePrefixCache
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models import architecture_uses_state_cache
from mini_infer.scheduler import (
    ContinuousScheduler,
    Request,
    RequestHandle,
    StateCacheContinuousScheduler,
)
from mini_infer.workers import PDScheduler

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Set to "1" / "true" / "yes" to back the HTTP API with the PD pipeline
# (`PrefillWorker` + `DecodeWorker` + `PDScheduler`) instead of the default
# `ContinuousScheduler`. Same `/v1/completions` surface either way; the
# pick is opaque to clients.
_USE_PD = os.environ.get("MINI_INFER_USE_PD", "").lower() in {"1", "true", "yes"}
# When PD is enabled, pick the threading variant:
#   - "serial":   one engine thread runs prefill + decode in sequence
#   - "parallel": two engine threads (prefill + decode) with a bounded
#                 handoff queue between them; phases overlap on multi-GPU
# Default "parallel" because that's where the PD throughput win is.
# Override with MINI_INFER_PD_MODE=serial to compare or to debug.
_PD_MODE = os.environ.get("MINI_INFER_PD_MODE", "parallel").lower()
if _PD_MODE not in {"serial", "parallel"}:
    raise ValueError(f"MINI_INFER_PD_MODE must be 'serial' or 'parallel'; got {_PD_MODE!r}")
# Cross-request prefix sharing for the StateCache (DeepSeek-V4) path: snapshot
# each prompt's post-prefill state so a later prompt that extends it replays only
# the new suffix instead of re-prefilling the shared prefix. Output is unchanged;
# it trades a bounded (FIFO-capped) snapshot pool for skipped re-prefills. On by
# default for that path; set MINI_INFER_PREFIX_SHARING=0 to disable under memory
# pressure. No effect on the PagedKVCache path (which has its own prefix cache).
_PREFIX_SHARING = os.environ.get("MINI_INFER_PREFIX_SHARING", "1").lower() in {"1", "true", "yes"}

# Default bind is loopback; set MINI_INFER_HOST=0.0.0.0 (or a specific
# interface) to expose. Pairing with MINI_INFER_API_KEY for any non-loopback
# bind is strongly recommended; the server logs a warning at startup if you
# bind publicly without a key.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Polling interval for client-disconnect detection during SSE. 100 ms keeps
# the per-step overhead negligible while still cancelling within a typical
# token's wall-clock budget.
_DISCONNECT_POLL_SECONDS = 0.1

logger = logging.getLogger(__name__)
_API_KEY = os.environ.get("MINI_INFER_API_KEY")
# `auto_error=False` keeps the Bearer scheme parsing optional so an unset
# MINI_INFER_API_KEY (the default dev case) doesn't reject every request.
# The actual key check happens inside `_verify_auth`.
_AUTH_SCHEME = HTTPBearer(auto_error=False)


async def _verify_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_AUTH_SCHEME),
) -> None:
    """Bearer-token gate; only active when MINI_INFER_API_KEY is set.

    Returns None on success; raises 401 on missing or wrong token. When the
    env var is unset (typical dev case), this dependency is a no-op so
    requests pass through.
    """
    if _API_KEY is None:
        return
    if creds is None or creds.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid api key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_name = os.environ.get("MINI_INFER_MODEL", DEFAULT_MODEL)
    scheduler: ContinuousScheduler | PDScheduler | StateCacheContinuousScheduler
    if architecture_uses_state_cache(model_name):
        # StateCache models (DeepSeek-V4) don't use PagedKVCache; serve them via
        # the StateCacheContinuousScheduler, which decodes a running batch of
        # requests at their own positions through one ragged forward per step
        # (dynamic admit / evict). PD / packed-varlen continuous batching don't
        # apply to the per-request StateCache path.
        logger.info(
            "Backing /v1/completions with StateCacheContinuousScheduler for %s (prefix_sharing=%s)",
            model_name,
            _PREFIX_SHARING,
        )
        scheduler = StateCacheContinuousScheduler(
            StateCacheGenerator.from_pretrained(model_name),
            prefix_cache=StatePrefixCache() if _PREFIX_SHARING else None,
        )
    else:
        # MINI_INFER_NUM_BLOCKS / MINI_INFER_BLOCK_SIZE let benchmarks size
        # the KV pool to fit the offered load. The runner defaults
        # (DEFAULT_NUM_BLOCKS=1024, DEFAULT_BLOCK_SIZE=16) hold ~16K token
        # slots, which is fine for small-concurrency dev but OOMs under a
        # rate sweep that puts dozens of long-prompt requests in flight.
        runner_kwargs: dict[str, int] = {}
        env_num_blocks = os.environ.get("MINI_INFER_NUM_BLOCKS")
        env_block_size = os.environ.get("MINI_INFER_BLOCK_SIZE")
        if env_num_blocks:
            runner_kwargs["num_blocks"] = int(env_num_blocks)
        if env_block_size:
            runner_kwargs["block_size"] = int(env_block_size)
        runner = ModelRunner.from_pretrained(model_name, **runner_kwargs)
        if _USE_PD:
            logger.info(
                "Backing /v1/completions with PDScheduler (mode=%s; MINI_INFER_USE_PD set)",
                _PD_MODE,
            )
            scheduler = PDScheduler(runner, mode=_PD_MODE)  # type: ignore[arg-type]
        else:
            scheduler = ContinuousScheduler(runner)
    scheduler.start()
    app.state.scheduler = scheduler
    try:
        yield
    finally:
        scheduler.stop()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request: FastAPIRequest, exc: Exception) -> JSONResponse:
    """Return a sanitized 503 instead of FastAPI's default traceback page.

    The exception is logged server-side with full context; the client gets
    only an opaque error code. Prevents accidentally leaking model paths,
    HF tokens, or internal state via tracebacks in error responses.
    """
    logger.exception("unhandled error in request handler", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={"error": {"type": "internal_error", "message": "engine error"}},
    )


@app.post("/v1/completions", response_model=None)
async def completions(
    req: CompletionRequest,
    fastapi_req: FastAPIRequest,
    _auth: None = Depends(_verify_auth),
) -> StreamingResponse | CompletionResponse:
    scheduler: ContinuousScheduler | PDScheduler | StateCacheContinuousScheduler = (
        fastapi_req.app.state.scheduler
    )
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
        handle = scheduler.submit(internal)

        async def event_stream() -> AsyncIterator[str]:
            """Stream generation steps as SSE.

            A background watcher polls `fastapi_req.is_disconnected` every
            100 ms; on client drop it calls `handle.cancel()`, which makes
            the engine emit a terminal step and unblocks the consumer's
            `get_step()` so resources (KV blocks, batch slot) are freed
            promptly.
            """
            watcher = asyncio.create_task(_watch_disconnect(fastapi_req, handle))
            try:
                while True:
                    step = await asyncio.to_thread(handle.get_step)
                    chunk = CompletionChunk(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[
                            CompletionChoice(text=step.text, finish_reason=step.finish_reason),
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    if step.finish_reason is not None:
                        break
                yield "data: [DONE]\n\n"
            finally:
                watcher.cancel()
                handle.cancel()

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


async def _watch_disconnect(fastapi_req: FastAPIRequest, handle: RequestHandle) -> None:
    """Poll the client's connection; cancel the request when it drops.

    Runs as a background task during streaming. When the client closes
    the SSE connection, calling `handle.cancel()` makes the engine emit a
    terminal step at its next safe point, which unblocks the main stream
    loop's `get_step()` call so the response generator returns and FastAPI
    cleans up.
    """
    try:
        while True:
            await asyncio.sleep(_DISCONNECT_POLL_SECONDS)
            if await fastapi_req.is_disconnected():
                handle.cancel()
                return
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MINI_INFER_HOST", DEFAULT_HOST)
    port = int(os.environ.get("MINI_INFER_PORT", str(DEFAULT_PORT)))
    if host != "127.0.0.1" and _API_KEY is None:
        logger.warning(
            "binding to %s without MINI_INFER_API_KEY set; the API is unauthenticated",
            host,
        )
    uvicorn.run("mini_infer.api.server:app", host=host, port=port)
