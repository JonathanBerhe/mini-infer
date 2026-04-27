"""Continuous batching scheduler with a dedicated engine thread.

The scheduler accepts requests asynchronously, admits them when the KV block pool
has room, drives them through chunked prefill and decode, and frees blocks when
each completes. A single engine thread owns the model and the running batch;
API threads only enqueue requests and drain per-request output queues.

Each step does:
1. Admit any newly arrived requests that fit in the block pool.
2. For each prefilling request, advance its prefill by ONE chunk (default 256
   tokens) into its own temporary single-slot cache. This is the head-of-line
   blocking fix: a long prompt's prefill is amortized across many steps so
   short decoders aren't blocked behind it for the entire prefill cost.
3. Run ONE batched decode forward over all currently-decoding requests.
4. Reap finished requests (free blocks, shift batch indices).

When the last chunk of a prefill lands, the request's prefill cache is merged
into the scheduler's shared batched cache and the request transitions to
DECODING. The first decode token is sampled from the last chunk's logits.
"""

import logging
import math
import queue
import threading
from collections.abc import Iterator

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import sample
from mini_infer.scheduler.request_state import (
    FinishReason,
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RequestState,
    RunningRequest,
)

logger = logging.getLogger(__name__)


class ContinuousScheduler:
    """FIFO multi-request scheduler with admission control and batched decode.

    A single engine thread owns the model and one shared `PagedKVCache` for the
    running batch. Submitted requests join a waiting queue; each engine step
    admits new requests if the block pool has capacity, prefills new admits and
    merges their cache state into the shared batched cache, runs ONE forward
    pass over all DECODING requests, samples a token per request, and reaps any
    that have finished.

    Lifecycle: call `start()` once before `submit()`; call `stop()` at shutdown
    to join the engine thread. The FastAPI lifespan does this for the HTTP
    server; tests do it via a fixture.
    """

    # Per-request output queue cap. Each item is a small GenerationStep; in
    # normal operation the consumer (SSE stream / wait()) drains as fast as the
    # engine produces, so the queue rarely holds more than 1-2 items. 256 is
    # generous headroom for transient consumer hiccups (network blip on a
    # streaming HTTP client) without burning meaningful memory.
    DEFAULT_OUTPUT_QUEUE_SIZE = 256

    # Extra blocks reserved beyond the prompt at admission time, so a request
    # has decode runway before the engine hits the pool again. 8 blocks * 16
    # tokens/block = 128 tokens of decode capacity. If a request needs more, we
    # currently allocate during decode (and may OOM if the pool is full).
    # Production engines preempt instead; that's tracked as a follow-up.
    DEFAULT_DECODE_HEADROOM_BLOCKS = 8

    # Engine-thread idle poll interval. When there's no work, the loop waits
    # this long before checking again. 5 ms = 200 Hz: low enough that newly
    # submitted requests start within ~5 ms (imperceptible for HTTP), high
    # enough not to spin the CPU. A signal-based wake on submit would be better
    # but adds complexity for negligible payoff at our scale.
    IDLE_SLEEP_SECONDS = 0.005

    # Prompt tokens processed per prefilling request per scheduler step. 256 is
    # vLLM/SGLang's default and balances two concerns: small enough that a long
    # prefill doesn't monopolize the engine for too long (so decoders make
    # progress), large enough that the per-step overhead (Python-side input
    # build + model.forward call) doesn't dominate the per-token cost. Tuneable
    # via the constructor; benchmarks should sweep this value.
    DEFAULT_CHUNK_SIZE = 256

    def __init__(
        self,
        runner: ModelRunner,
        max_concurrent: int = 16,
        decode_headroom_blocks: int = DEFAULT_DECODE_HEADROOM_BLOCKS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._runner = runner
        self._max_concurrent = max_concurrent
        self._decode_headroom = decode_headroom_blocks
        self._chunk_size = chunk_size
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._running: list[RunningRequest] = []
        self._batched_cache: PagedKVCache | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the engine thread. Idempotent if the thread is already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the engine thread to exit and join it (with the given timeout)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, request: Request) -> RequestHandle:
        """Enqueue a request and return a handle the caller drains for output."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("scheduler not running; call start() before submit()")
        running = RunningRequest(
            request=request,
            output_queue=queue.Queue(maxsize=self.DEFAULT_OUTPUT_QUEUE_SIZE),
        )
        self._waiting.put(running)
        return RequestHandle(running)

    def run(self, request: Request) -> GenerationResult:
        """Submit and block until the request completes; returns the aggregated result."""
        return self.submit(request).wait()

    def stream(self, request: Request) -> Iterator[GenerationStep]:
        """Submit and yield each generation step as the engine produces it."""
        yield from self.submit(request).steps()

    def _engine_loop(self) -> None:
        """Engine-thread main loop; runs `_step()` until `stop()` is signaled."""
        try:
            while not self._stop_event.is_set():
                self._step()
                # When fully idle, wait briefly so we don't spin the CPU.
                if not self._running and self._waiting.empty():
                    self._stop_event.wait(timeout=self.IDLE_SLEEP_SECONDS)
        except Exception:
            logger.exception("engine thread crashed")
            raise

    def _step(self) -> None:
        """One scheduler iteration: admit, advance prefills, batched decode, reap."""
        self._admit_waiting()
        prefilling = [
            r
            for r in self._running
            if r.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING)
        ]
        for req in prefilling:
            self._advance_chunked_prefill(req)
        decoding = [r for r in self._running if r.state == RequestState.DECODING]
        if decoding:
            self._batched_decode_step(decoding)
        self._reap_done()

    def _admit_waiting(self) -> None:
        """Move waiting requests into the running batch while the block pool has room."""
        while len(self._running) < self._max_concurrent:
            try:
                running = self._waiting.get_nowait()
            except queue.Empty:
                return

            running.prompt_token_ids = self._runner.tokenizer.encode(running.request.prompt)
            block_size = self._runner.block_pool.block_size
            required_blocks = (
                math.ceil(len(running.prompt_token_ids) / block_size) + self._decode_headroom
            )

            if self._runner.block_pool.num_free_blocks < required_blocks:
                # Pool is too full to safely admit this request. Re-enqueue and
                # stop admitting this step. Note: the re-enqueue breaks strict
                # FIFO if the queue contains smaller requests behind this one;
                # acceptable trade for now, can be revisited if it matters.
                self._waiting.put(running)
                return

            # Allocate the per-request prefill cache eagerly so subsequent chunk
            # dispatches just call prefill_chunk against an existing single-slot cache.
            running.prefill_cache = PagedKVCache(self._runner.block_pool)
            running.prefill_cache.add_request_slot()
            running.state = RequestState.PREFILLING
            self._running.append(running)

    def _advance_chunked_prefill(self, req: RunningRequest) -> None:
        """Dispatch one chunk's worth of prefill; merge + transition to DECODING on the last chunk.

        The chunk size is fixed at `self._chunk_size` (configurable on the
        scheduler). On the final chunk (the one that brings `tokens_prefilled`
        up to `len(prompt_token_ids)`), the request's temp prefill cache is
        merged into the shared batched cache and the last-position logits are
        retained as the seed for the next decode step.
        """
        assert req.prefill_cache is not None
        chunk_start = req.tokens_prefilled
        chunk_end = min(chunk_start + self._chunk_size, len(req.prompt_token_ids))
        chunk_tokens = req.prompt_token_ids[chunk_start:chunk_end]

        _, logits = self._runner.prefill_chunk(
            req.prefill_cache, chunk_tokens, position_offset=chunk_start
        )
        req.tokens_prefilled = chunk_end

        if chunk_end == len(req.prompt_token_ids):
            # Final chunk: merge prefill cache into the shared batched cache.
            if self._batched_cache is None:
                self._batched_cache = req.prefill_cache
                req.batch_idx = 0
            else:
                req.batch_idx = self._batched_cache.merge_request(req.prefill_cache)
            req.prefill_cache = None
            req.last_logits = logits
            req.state = RequestState.DECODING
        else:
            req.state = RequestState.CHUNKED_PREFILLING

    def _batched_decode_step(self, decoding: list[RunningRequest]) -> None:
        """Sample-and-emit for each request, then ONE batched forward over the survivors.

        Sampling is per-request (each draws from its own logits + sampling params);
        emission is per-request (decode-and-diff streaming). After sampling, any
        request that hit EOS or max_tokens transitions to DONE here. The forward
        runs over the rest in batched form via `runner.decode_batch`, producing
        each survivor's logits for the next step.
        """
        assert self._batched_cache is not None
        tokenizer = self._runner.tokenizer

        survivors: list[RunningRequest] = []
        survivor_tokens: list[int] = []
        for req in decoding:
            assert req.last_logits is not None
            next_token = sample(req.last_logits, req.request.sampling_params)
            if next_token == tokenizer.eos_token_id:
                self._finish(req, "stop")
                continue
            req.tokens_generated.append(next_token)
            # Decode-and-diff: re-decode all tokens so far and emit the new suffix.
            # Handles multi-byte UTF-8 sequences correctly at the cost of O(n) work
            # per step (one tokenizer call, one string slice).
            current_text = tokenizer.decode(req.tokens_generated)
            delta = current_text[len(req.last_text) :]
            req.last_text = current_text
            req.output_queue.put(GenerationStep(text=delta))
            if len(req.tokens_generated) >= req.request.max_tokens:
                self._finish(req, "length")
                continue
            survivors.append(req)
            survivor_tokens.append(next_token)

        if not survivors:
            return

        # The cache still holds slots for any just-finished requests until
        # `_reap_done()` runs. Build a token list for ALL slots; survivors get
        # their sampled token, others get a placeholder. We discard their logits.
        last_tokens_full = self._build_full_token_list(survivors, survivor_tokens)
        _, all_logits = self._runner.decode_batch(self._batched_cache, last_tokens_full)
        for req in survivors:
            assert req.batch_idx is not None
            req.last_logits = all_logits[req.batch_idx]

    def _build_full_token_list(
        self, survivors: list[RunningRequest], survivor_tokens: list[int]
    ) -> list[int]:
        """Build a length-batch_size token list; placeholder for soon-to-be-reaped slots."""
        assert self._batched_cache is not None
        tokens: list[int] = [0] * self._batched_cache.batch_size
        for req, token in zip(survivors, survivor_tokens, strict=True):
            assert req.batch_idx is not None
            tokens[req.batch_idx] = token
        return tokens

    def _reap_done(self) -> None:
        """Free blocks for finished requests; shift batch_idx for survivors."""
        # Process highest batch_idx first so each removal shifts only indices we
        # haven't yet processed; the survivor decrement keeps tracking variables
        # in sync.
        done = sorted(
            [r for r in self._running if r.state == RequestState.DONE],
            key=lambda r: r.batch_idx if r.batch_idx is not None else -1,
            reverse=True,
        )
        for finished in done:
            if finished.batch_idx is not None and self._batched_cache is not None:
                self._batched_cache.remove_request(finished.batch_idx)
                for surviving in self._running:
                    if (
                        surviving is not finished
                        and surviving.batch_idx is not None
                        and surviving.batch_idx > finished.batch_idx
                    ):
                        surviving.batch_idx -= 1
            self._running.remove(finished)

        if not self._running and self._batched_cache is not None:
            # Nothing in flight: drop the empty cache so the next admit starts clean.
            self._batched_cache = None

    def _finish(self, req: RunningRequest, reason: FinishReason) -> None:
        """Mark a request DONE and emit the terminal step with its finish_reason."""
        req.finish_reason = reason
        req.state = RequestState.DONE
        req.output_queue.put(GenerationStep(text="", finish_reason=reason))
