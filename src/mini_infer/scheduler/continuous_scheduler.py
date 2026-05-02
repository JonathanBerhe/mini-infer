"""Continuous batching scheduler with a dedicated engine thread.

The scheduler accepts requests asynchronously, admits them when the KV block pool
has room, drives them through chunked prefill and decode, and frees blocks when
each completes. A single engine thread owns the model and the running batch;
API threads only enqueue requests and drain per-request output queues.

Each step does:
1. Admit any newly arrived requests that fit in the block pool. New requests
   get a slot in the shared `_batched_cache` immediately (no per-request temp
   cache); their first chunk lands in the next forward.
2. For DECODING requests, sample the next token from prior `last_logits`,
   emit it, and check finish conditions. Some requests transition to DONE here.
3. Reap any DONE requests (free blocks, shift batch indices). This happens
   BEFORE the forward so the forward only runs over the alive set.
4. Build packed inputs over the alive in-flight set: each prefilling request
   contributes its next chunk's tokens, each decoding request contributes the
   one token sampled in step 2. ONE `runner.forward_step(...)` call per step.
5. Distribute per-request last-position logits and advance prefill state.

Throughput-wise this is "Approach 2": prefill chunks and decode tokens share
the same forward pass. The matmul cost amortizes across all in-flight q-tokens.
"""

import contextlib
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
    """FIFO multi-request scheduler with chunked prefill and packed forwards.

    A single engine thread owns the model and one shared `PagedKVCache`. The
    cache holds a slot for every in-flight request (prefilling or decoding).
    Each engine step admits new requests if the pool has capacity, samples
    decoders, reaps any that just finished, then runs ONE batched forward over
    the alive set via `runner.forward_step(...)`.

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
        """One scheduler iteration: admit, sample decoders, reap, packed forward."""
        self._cancel_pending()
        self._admit_waiting()
        sampled_decode_tokens = self._sample_decoders()
        self._reap_done()
        alive = [
            r
            for r in self._running
            if r.state
            in (
                RequestState.PREFILLING,
                RequestState.CHUNKED_PREFILLING,
                RequestState.DECODING,
            )
        ]
        if alive:
            self._packed_forward(alive, sampled_decode_tokens)

    def _cancel_pending(self) -> None:
        """Mark any cancellation-flagged in-flight requests as DONE.

        Cancellation is set on the request handle from the API thread (e.g.
        the SSE generator detecting client disconnect). Picking it up here,
        before sampling and the forward pass, lets the next `_reap_done`
        free the slot's blocks at the same boundary as a natural finish.
        """
        for req in self._running:
            if req.cancel_event.is_set() and req.state != RequestState.DONE:
                self._finish(req, "cancelled")

    def _admit_waiting(self) -> None:
        """Move waiting requests into the running batch while the block pool has room.

        New admits get a slot in the shared `_batched_cache` immediately; their
        first chunk lands in this step's forward. When a prefix cache is
        configured, the slot's blocks are pre-populated with cached prefix
        blocks (no new allocation needed for those positions), and evictable
        cached blocks count toward the admission budget since the pool can
        reclaim them on demand.
        """
        block_pool = self._runner.block_pool
        prefix_cache = block_pool.prefix_cache
        while len(self._running) < self._max_concurrent:
            try:
                running = self._waiting.get_nowait()
            except queue.Empty:
                return

            # API-side cancel before admission: drop without tokenizing.
            if running.cancel_event.is_set():
                self._finish(running, "cancelled")
                self._running.append(running)
                continue

            running.prompt_token_ids = self._runner.tokenizer.encode(running.request.prompt)
            block_size = block_pool.block_size
            required_blocks = (
                math.ceil(len(running.prompt_token_ids) / block_size) + self._decode_headroom
            )

            # Permanent rejection: if the request needs more blocks than the
            # pool can EVER provide, no amount of waiting helps. Reject with
            # finish_reason="cancelled" instead of unconditionally re-enqueuing
            # (which would otherwise let one oversized request block every
            # smaller request behind it forever).
            if required_blocks > block_pool.num_blocks:
                logger.warning(
                    "rejecting request: required_blocks=%d exceeds pool capacity=%d",
                    required_blocks,
                    block_pool.num_blocks,
                )
                self._finish(running, "cancelled")
                self._running.append(running)
                continue

            available_blocks = block_pool.num_free_blocks
            if prefix_cache is not None:
                # Evictable cached blocks are reclaimable by the pool on
                # demand; they count toward what we can give this request.
                available_blocks += prefix_cache.num_evictable
            if available_blocks < required_blocks:
                # Pool is full RIGHT NOW but the request fits in principle.
                # Re-enqueue and try next step once running requests free
                # blocks. Note: the re-enqueue breaks strict FIFO if the
                # queue contains smaller requests behind this one; acceptable
                # trade for now, can be revisited if it matters.
                self._waiting.put(running)
                return

            if self._batched_cache is None:
                self._batched_cache = PagedKVCache(block_pool)
            running.batch_idx = self._batched_cache.add_request_slot(
                prompt_token_ids=running.prompt_token_ids
            )
            # If the prefix cache pre-populated this slot, tokens_prefilled
            # starts non-zero. The PREFILLING -> CHUNKED_PREFILLING transition
            # in `_packed_forward` picks up from there.
            running.tokens_prefilled = self._batched_cache.seq_lens_list()[running.batch_idx]
            running.state = RequestState.PREFILLING
            self._running.append(running)

    def _sample_decoders(self) -> dict[int, int]:
        """Sample the next token for each DECODING request and emit it.

        Requests that hit EOS or max_tokens transition to DONE; the rest stay
        DECODING and the sampled token is returned in the result dict (keyed
        by `id(req)` so the caller can look it up after `_reap_done` shifts
        `batch_idx`).
        """
        sampled: dict[int, int] = {}
        tokenizer = self._runner.tokenizer
        for req in [r for r in self._running if r.state == RequestState.DECODING]:
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
            self._emit(req, GenerationStep(text=delta))
            if len(req.tokens_generated) >= req.request.max_tokens:
                self._finish(req, "length")
                continue
            sampled[id(req)] = next_token
        return sampled

    def _packed_forward(
        self, alive: list[RunningRequest], sampled_decode_tokens: dict[int, int]
    ) -> None:
        """Build packed inputs over the alive in-flight set and run ONE forward.

        Order in `alive` matches the cache's slot order (both are derived from
        `self._running` after the reap). `position_offsets` therefore line up
        with `cache.seq_lens_list()`.
        """
        assert self._batched_cache is not None

        packed_input_ids: list[int] = []
        cu_seqlens_q: list[int] = [0]
        position_offsets: list[int] = []
        cache_seq_lens = self._batched_cache.seq_lens_list()

        for req in alive:
            assert req.batch_idx is not None
            if req.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING):
                chunk_start = req.tokens_prefilled
                chunk_end = min(chunk_start + self._chunk_size, len(req.prompt_token_ids))
                chunk_tokens = req.prompt_token_ids[chunk_start:chunk_end]
                packed_input_ids.extend(chunk_tokens)
                cu_seqlens_q.append(cu_seqlens_q[-1] + len(chunk_tokens))
                position_offsets.append(chunk_start)
            else:
                next_token = sampled_decode_tokens[id(req)]
                packed_input_ids.append(next_token)
                cu_seqlens_q.append(cu_seqlens_q[-1] + 1)
                position_offsets.append(cache_seq_lens[req.batch_idx])

        per_request_logits = self._runner.forward_step(
            self._batched_cache, packed_input_ids, cu_seqlens_q, position_offsets
        )

        # Distribute logits and advance prefill state.
        for index, req in enumerate(alive):
            if req.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING):
                tokens_added = cu_seqlens_q[index + 1] - cu_seqlens_q[index]
                req.tokens_prefilled += tokens_added
                if req.tokens_prefilled == len(req.prompt_token_ids):
                    req.state = RequestState.DECODING
                    req.last_logits = per_request_logits[index]
                else:
                    req.state = RequestState.CHUNKED_PREFILLING
            else:
                req.last_logits = per_request_logits[index]

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
        """Mark a request DONE and emit the terminal step with its finish_reason.

        The terminal step is the consumer's signal that the stream is over,
        so this MUST go through; if the queue is full we drop the most
        recent intermediate step to make room rather than block the engine
        thread.
        """
        req.finish_reason = reason
        req.state = RequestState.DONE
        terminal = GenerationStep(text="", finish_reason=reason)
        try:
            req.output_queue.put_nowait(terminal)
        except queue.Full:
            # Make room by dropping one buffered intermediate step. The
            # consumer sees at most a one-step gap; the terminal step still
            # arrives so the stream terminates.
            with contextlib.suppress(queue.Empty):
                req.output_queue.get_nowait()
            try:
                req.output_queue.put_nowait(terminal)
            except queue.Full:
                logger.error(
                    "output_queue stuck full even after drop; consumer is gone, "
                    "terminal step will be missed (request leaked)"
                )

    def _emit(self, req: RunningRequest, step: GenerationStep) -> None:
        """Non-blocking emit of an intermediate step.

        Engine-thread safety: we never block on `output_queue.put`. A stuck
        consumer (e.g. a streaming HTTP client that disconnected without
        the engine noticing yet) must not be able to wedge the engine. If
        the queue is full we drop the oldest buffered step and put the new
        one; the consumer will see a small gap, which is strictly better
        than a hung engine.
        """
        try:
            req.output_queue.put_nowait(step)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                req.output_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                req.output_queue.put_nowait(step)
