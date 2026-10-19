"""Continuous batching scheduler with a dedicated engine thread.

The scheduler accepts requests asynchronously, admits them when the KV block pool
has room, drives them through chunked prefill and decode, and frees blocks when
each completes. A single engine thread owns the model and the running batch;
API threads only enqueue requests and drain per-request output queues.

Each step does:
1. Mark cancelled requests DONE. For DECODING requests, sample the next token
   from prior `last_logits`, emit it, and check finish conditions. Some
   requests transition to DONE here. After a recoverable OOM, a request feeds
   its already emitted tokens again.
2. Reap any DONE requests (free blocks, shift batch indices): those that
   finished in the last verify, by a cancel, or in step 1.
3. Admit waiting requests into the freed slots while the block pool has room.
   New requests get a slot in the shared `_batched_cache` immediately (no
   per-request temp cache); their first chunk lands in this step's forward.
4. Build packed inputs over the alive in-flight set: each prefilling request
   contributes its next chunk's tokens, each decoding request contributes the
   tokens from step 1. ONE `runner.forward_step_packed(...)` call per step.
5. Distribute per-request last-position logits and advance prefill state.

With `dspark` set, admission decides once whether a request speculates. A
speculator feeds `[anchor, *offered]`: its last committed token, then a drafted
block that the same forward verifies. Plain requests share the batch.
Speculators add two phases:

1b. `_draft_speculators`: draft a block from the anchor and the injected
    context (target hidden states at the tap layers).
5b. `_verify_speculators`: accept a prefix of the draft plus one bonus token,
    emit them, roll the slot back to the committed length, and take the next
    injected context from this forward.

(INV): the target slot length equals the draft cache length plus the context
rows. See ADR-031.

Throughput-wise this is "Approach 2": prefill chunks and decode tokens share
the same forward pass. The matmul cost amortizes across all in-flight q-tokens.
"""

import contextlib
import json
import logging
import math
import queue
import threading
import time
from collections.abc import Iterator

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.dspark.spec import (
    DSparkSpec,
    DSparkStats,
    SpecDecodeState,
    context_window,
    verify_block,
)
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import sample
from mini_infer.exceptions import EngineNotRunningError, OutOfMemoryError
from mini_infer.scheduler.request_state import (
    FinishReason,
    GenerationResult,
    GenerationStep,
    Request,
    RequestHandle,
    RequestState,
    RunningRequest,
    StepQueue,
)

logger = logging.getLogger(__name__)


class ContinuousScheduler:
    """FIFO multi-request scheduler with chunked prefill and packed forwards.

    A single engine thread owns the model and one shared `PagedKVCache`. The
    cache holds a slot for every in-flight request (prefilling or decoding).
    Each engine step samples decoders, reaps the requests that finished, admits
    new requests into the freed slots if the pool has capacity, then runs ONE
    batched forward over the alive set via `runner.forward_step_packed(...)`.

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
        trace_out: str | None = None,
        dspark: DSparkSpec | None = None,
    ) -> None:
        # Prefix-cache hits skip the forward, so they give no tapped hidden
        # states, and the short context breaks (INV). See ADR-031.
        if dspark is not None and runner.block_pool.prefix_cache is not None:
            raise ValueError(
                "DSpark speculative decoding cannot run with a prefix cache: "
                "cached prompt positions never produce the tapped hidden states "
                "the drafter needs. Construct the runner with prefix_cache=False."
            )
        # Without this guard, the first speculating forward raises a TypeError
        # that kills the engine thread and every in-flight request.
        if dspark is not None and not runner.supports_hidden_state_taps:
            raise ValueError(
                f"DSpark speculative decoding needs a target whose forward accepts "
                f"tap_layers/hidden_state_sink; {type(runner).__name__}'s model does not. "
                "Qwen3 is the family that supports it today."
            )
        self._runner = runner
        self._dspark = dspark
        self._max_concurrent = max_concurrent
        self._decode_headroom = decode_headroom_blocks
        self._chunk_size = chunk_size
        self._waiting: queue.Queue[RunningRequest] = queue.Queue()
        self._running: list[RunningRequest] = []
        self._batched_cache: PagedKVCache | None = None
        # Slot lengths before the packed forward, for `_roll_back_partial_append`.
        # Not None only while a forward runs or after one raised.
        self._pre_forward_seq_lens: list[int] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # True while `stop()` runs. Handles then count the engine as alive,
        # because `stop()` sends the terminal steps after the thread exits.
        self._stopping = False
        # Set when an exception kills the engine thread. `stop()` reads it after
        # the join, so that a crash does not become 'cancelled'. See ADR-028.
        self._crashed = False
        # Chrome Trace Event recording. When `trace_out` is set, each engine
        # step that runs a forward appends a duration event to `_trace_events`;
        # `stop()` writes them out as JSON. `_trace_t0` anchors `ts` at zero
        # for the first recorded step so the timeline starts at the origin.
        self._trace_out = trace_out
        self._trace_events: list[dict[str, object]] | None = [] if trace_out is not None else None
        self._trace_t0: float | None = None

    @property
    def is_running(self) -> bool:
        """Whether the engine thread is alive.

        False before `start()`, after `stop()`, and after a crash (any exception
        other than `OutOfMemoryError`).
        """
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Spawn the engine thread. Idempotent if the thread is already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._crashed:
            # No `stop()` ended the requests of the dead thread. Their handles
            # watch that thread, and their state can be half-updated.
            self._end_in_flight(crashed=True)
        self._stop_event.clear()
        self._crashed = False
        self._thread = threading.Thread(
            target=self._engine_loop, name="mini-infer-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the engine thread to exit, join it, and end the requests left.

        After a clean join, each leftover request ends as 'cancelled'. After a
        crash, they get no terminal step, so their handles raise
        `EngineNotRunningError`. Both cases free the blocks, so a later `start()`
        begins empty. If the join times out, `stop()` ends nothing, because the
        thread can still change the state. The handles then fail through the
        liveness check.
        """
        self._stopping = True
        try:
            self._stop_event.set()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=timeout)
                self._thread = None
                if not thread.is_alive():
                    self._end_in_flight(crashed=self._crashed)
        finally:
            self._stopping = False
        if self._trace_events is not None and self._trace_out is not None:
            with open(self._trace_out, "w") as f:
                json.dump({"traceEvents": self._trace_events}, f)

    def submit(self, request: Request) -> RequestHandle:
        """Enqueue a request and return a handle the caller drains for output.

        The handle watches the engine thread that runs now, not a later one. It
        raises `EngineNotRunningError` if that thread dies.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            raise EngineNotRunningError("scheduler not running; call start() before submit()")
        running = RunningRequest(
            request=request,
            output_queue=StepQueue(maxsize=self.DEFAULT_OUTPUT_QUEUE_SIZE),
        )
        self._waiting.put(running)
        # Read the thread before `_stopping`. Then a dead thread with `_stopping`
        # False means a crash or a `stop()` that already pushed its terminal steps.
        thread_alive = thread.is_alive
        return RequestHandle(running, engine_alive=lambda: thread_alive() or self._stopping)

    def _end_in_flight(self, *, crashed: bool) -> None:
        """End every request still running or waiting, and free their blocks.

        Callers run this only after the engine thread exited, so no thread
        changes this state. A `submit()` during `stop()` can enqueue after the
        drain. Its handle then fails through the liveness check.
        """
        leftover = [r for r in self._running if r.state != RequestState.DONE]
        while True:
            try:
                leftover.append(self._waiting.get_nowait())
            except queue.Empty:
                break
        for req in leftover:
            if crashed:
                req.state = RequestState.DONE
            else:
                self._finish(req, "cancelled")
        if not crashed:
            self._reap_done()
            return
        try:
            self._reap_done()
        except Exception:
            # A crash mid-step can leave the cache half-updated. Only these blocks leak.
            logger.exception("stop(): could not free the KV blocks after the crash")
            self._running.clear()
            self._batched_cache = None

    def run(self, request: Request) -> GenerationResult:
        """Submit and block until the request completes; returns the aggregated result."""
        return self.submit(request).wait()

    def stream(self, request: Request) -> Iterator[GenerationStep]:
        """Submit and yield each generation step as the engine produces it."""
        yield from self.submit(request).steps()

    def _engine_loop(self) -> None:
        """Engine-thread main loop; runs `_step()` until `stop()` is signaled.

        Mid-step OOM is recoverable: the pool can hand out blocks during a
        forward (decode growth, partial prefill chunk needing a new block)
        and may run out even after admission accepted everyone. Killing the
        engine thread on that would take down every in-flight request and
        every future one. Instead we preempt the youngest still-prefilling
        request and let the loop retry; the cancelled request returns
        `finish_reason="cancelled"` to its client. Other exceptions still
        crash the thread loudly, because they signal real bugs.

        The crash handler wraps the whole loop, so an OOM-handler error is a crash too.
        """
        try:
            while not self._stop_event.is_set():
                try:
                    self._step()
                except OutOfMemoryError:
                    logger.warning("engine OOM during step; preempting youngest in-flight request")
                    self._preempt_on_oom()
                # When fully idle, wait briefly so we don't spin the CPU.
                if not self._running and self._waiting.empty():
                    self._stop_event.wait(timeout=self.IDLE_SLEEP_SECONDS)
        except Exception:
            logger.exception("engine thread crashed")
            self._crashed = True
            raise

    def _preempt_on_oom(self) -> None:
        """Cancel the youngest in-flight request and reap so the loop can retry.

        Picks a prefiller over a decoder when possible: prefillers haven't
        emitted any user-visible output yet, so the user-facing damage is
        smaller. Among prefillers we cancel the most recently admitted one
        (FILO), matching the intuition that the most recent admission is
        what pushed us past the watermark.

        The survivors repeat the failed step exactly: slots roll back, plain
        decoders resend `pending_decode_tokens`, and speculators reuse their
        round. So the client never gets a token twice.
        """
        # Roll back before the reap. The snapshot uses the slot indices of the
        # failed forward, and the reap shifts them.
        self._roll_back_partial_append()
        candidates = [
            r
            for r in self._running
            if r.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING)
        ]
        if not candidates:
            candidates = [r for r in self._running if r.state == RequestState.DECODING]
        if not candidates:
            return
        # `_running` preserves admission order; youngest is at the back.
        victim = candidates[-1]
        self._finish(victim, "cancelled")
        self._reap_done()

    def _roll_back_partial_append(self) -> None:
        """Return every slot to the length it had before the forward that raised.

        On layer 0, `append_kv_packed` advances each slot's count and allocates
        its blocks before it writes any K/V. An OOM at slot j leaves slots
        0..j-1 with claimed positions that hold no data. Without this rollback,
        the retry writes a plain decoder's token at the wrong position silently.
        A speculator fails its start-vs-slot assertion and kills the engine.

        Speculator state needs no undo: drafting ends before the forward, and
        the retry reuses the proposal. A draft cache rollback on that reuse path
        breaks (INV). Every slot can take `truncate_to`, because it is a no-op at
        the current length. Publication happens on the last layer, so a layer-0
        failure leaves no new published block for `truncate_to` to cut into. The
        slot that raised keeps its new blocks until the retry reuses them or
        `remove_request` frees them. See ADR-031.
        """
        snapshot = self._pre_forward_seq_lens
        self._pre_forward_seq_lens = None
        if snapshot is None or self._batched_cache is None:
            return
        cache = self._batched_cache
        assert len(snapshot) == cache.batch_size, (
            f"saved {len(snapshot)} slot lengths for a batch of {cache.batch_size}; "
            "a slot was added or removed between the forward and the rollback"
        )
        for batch_idx, length in enumerate(snapshot):
            cache.truncate_to(batch_idx, length)

    def _step(self) -> None:
        """One scheduler iteration: sample/draft, reap, admit, forward, verify.

        The reap comes before admission, so that a request that finished (in
        the last verify, by a cancel, or in this step's sampling) frees its
        slot and blocks for a waiting request in this same forward.
        """
        step_start = time.perf_counter() if self._trace_events is not None else 0.0

        self._cancel_pending()
        sampled_decode_tokens = self._sample_decoders()
        sampled_decode_tokens.update(self._draft_speculators())
        self._reap_done()
        self._admit_waiting()
        # Admission ends rejected, empty, and already-cancelled requests on the
        # spot. Reap them now, so `_running` holds only requests in flight after
        # the step and a rejected speculator's state goes at once.
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

        # Snapshot the phase BEFORE `_packed_forward` mutates request states
        # (e.g. PREFILLING -> DECODING when the last chunk lands).
        trace_phase: str | None = None
        n_verify = 0
        if alive and self._trace_events is not None:
            trace_phase = self._classify_phase(alive)
            # Count verify windows now, because the verify phase clears `offered`.
            n_verify = sum(1 for r in alive if r.spec is not None and r.spec.offered is not None)

        total_q = 0
        if alive:
            packed_logits, cu_seqlens_q, sink = self._packed_forward(alive, sampled_decode_tokens)
            total_q = cu_seqlens_q[-1]
            self._verify_speculators(alive, packed_logits, cu_seqlens_q, sink)

        if alive and self._trace_events is not None:
            if self._trace_t0 is None:
                self._trace_t0 = step_start
            assert trace_phase is not None
            self._trace_events.append(
                {
                    "name": "step",
                    "ph": "X",
                    "ts": (step_start - self._trace_t0) * 1e6,
                    "dur": (time.perf_counter() - step_start) * 1e6,
                    "pid": 0,
                    "tid": 0,
                    # Step cost follows total_q (q-tokens), not B (requests).
                    # Prefill chunks and verify windows make total_q exceed B.
                    "args": {
                        "B": len(alive),
                        "phase": trace_phase,
                        "total_q": total_q,
                        "n_verify": n_verify,
                    },
                }
            )

    @staticmethod
    def _classify_phase(alive: list[RunningRequest]) -> str:
        """Label the step as prefill / decode / mixed for the trace timeline."""
        states = {r.state for r in alive}
        if states <= {RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING}:
            return "prefill"
        if states == {RequestState.DECODING}:
            return "decode"
        return "mixed"

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
        # Count seated requests only. A request that admission ends on the spot
        # (cancelled, empty, or rejected) holds no slot and must not block the
        # waiter behind it.
        seated = sum(1 for r in self._running if r.state != RequestState.DONE)
        while seated < self._max_concurrent:
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

            # An empty prompt fails an assertion in `_packed_forward`, which kills
            # the engine thread. Finish it here with no output, as
            # StateCacheContinuousScheduler does.
            if not running.prompt_token_ids:
                logger.info("empty prompt: finishing request without generating")
                self._finish(running, "stop")
                self._running.append(running)
                continue

            running.spec = self._spec_state_for(running)

            block_size = block_pool.block_size
            plain_blocks = (
                math.ceil(len(running.prompt_token_ids) / block_size) + self._decode_headroom
            )
            required_blocks = plain_blocks
            if running.spec is not None:
                assert self._dspark is not None
                # Truncation frees the rejected positions of each round, so only
                # the verify window in flight needs extra blocks. See ADR-031.
                spec_blocks = math.ceil((self._dspark.block_size + 1) / block_size) + 1
                if plain_blocks + spec_blocks > block_pool.num_blocks:
                    # Speculation must never refuse a request that the plain path can serve.
                    logger.info(
                        "request not speculating: pool capacity %d cannot hold its prompt "
                        "plus %d blocks of verify-window headroom",
                        block_pool.num_blocks,
                        spec_blocks,
                    )
                    running.spec = None
                else:
                    required_blocks = plain_blocks + spec_blocks

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
            seated += 1

    def _spec_state_for(self, running: RunningRequest) -> SpecDecodeState | None:
        """Decide at admission whether this request speculates. None means plain.

        A request with top-k or top-p runs plain, because the spec path applies
        temperature only, not the filter. A prompt over `max_spec_prompt_tokens`
        runs plain, because its injected context grows with the prompt outside
        the block pool. Both log at INFO: they change latency, not output.
        """
        if self._dspark is None:
            return None
        params = running.request.sampling_params
        if params.top_k > 0 or params.top_p < 1.0:
            logger.info(
                "request not speculating: top_k=%d top_p=%.3f need the plain sampling path",
                params.top_k,
                params.top_p,
            )
            return None
        if len(running.prompt_token_ids) > self._dspark.max_spec_prompt_tokens:
            logger.info(
                "request not speculating: prompt length %d exceeds max_spec_prompt_tokens=%d",
                len(running.prompt_token_ids),
                self._dspark.max_spec_prompt_tokens,
            )
            return None
        return SpecDecodeState(draft_cache=self._dspark.new_draft_cache(), stats=DSparkStats())

    def _sample_decoders(self) -> dict[int, list[int]]:
        """Sample the next token for each DECODING request and emit it.

        Requests that hit EOS or max_tokens transition to DONE; the rest stay
        DECODING and their next tokens are returned in the result dict (keyed
        by `id(req)` so the caller can look them up after `_reap_done` shifts
        `batch_idx`).

        Each value is a one-token list, the type that speculators also use.
        After a recoverable OOM, a request resends `pending_decode_tokens` and
        does NOT sample again: it already emitted them, and the stream cannot
        take a token back.
        """
        sampled: dict[int, list[int]] = {}
        tokenizer = self._runner.tokenizer
        for req in [
            r for r in self._running if r.state == RequestState.DECODING and r.spec is None
        ]:
            if req.pending_decode_tokens is not None:
                sampled[id(req)] = req.pending_decode_tokens
                continue
            assert req.last_logits is not None
            next_token = sample(req.last_logits, req.request.sampling_params)
            if next_token == tokenizer.eos_token_id:
                self._finish(req, "stop")
                continue
            req.tokens_generated.append(next_token)
            self._emit_new_text(req, 1)
            if len(req.tokens_generated) >= req.request.max_tokens:
                self._finish(req, "length")
                continue
            req.pending_decode_tokens = [next_token]
            sampled[id(req)] = req.pending_decode_tokens
        return sampled

    def _emit_new_text(self, req: RunningRequest, num_tokens: int) -> None:
        """Decode-and-diff: re-decode all tokens so far and emit the new suffix.

        Handles multi-byte UTF-8 correctly at O(n) cost per call. One call emits
        ONE step, and `num_tokens` says how many tokens it carries.
        """
        current_text = self._runner.tokenizer.decode(req.tokens_generated)
        delta = current_text[len(req.last_text) :]
        req.last_text = current_text
        self._emit(req, GenerationStep(text=delta, num_tokens=num_tokens))

    def _draft_speculators(self) -> dict[int, list[int]]:
        """Draft one block per speculating DECODING request. Return verify inputs.

        Each entry is `[anchor, *offered]`: the last committed token, then the
        clamped drafted block. A request with no anchor yet (prefill ended last
        step) samples and emits its anchor like a plain decoder, then drafts in
        the same step. A retry after a recoverable OOM reuses its proposal,
        because a second draft folds the context in twice and breaks (INV).
        """
        if self._dspark is None:
            return {}
        out: dict[int, list[int]] = {}
        tokenizer = self._runner.tokenizer
        for req in [
            r for r in self._running if r.state == RequestState.DECODING and r.spec is not None
        ]:
            spec = req.spec
            assert spec is not None and self._batched_cache is not None
            assert req.batch_idx is not None

            if spec.anchor is None:
                # The tapped rows of the prompt chunks become the round-1 context.
                assert spec.context_parts, "spec request reached DECODING with no tapped context"
                spec.context = (
                    torch.cat(spec.context_parts, dim=1)
                    if len(spec.context_parts) > 1
                    else spec.context_parts[0]
                )
                spec.context_parts.clear()

                assert req.last_logits is not None
                anchor = sample(req.last_logits, req.request.sampling_params)
                if anchor == tokenizer.eos_token_id:
                    self._finish(req, "stop")
                    continue
                req.tokens_generated.append(anchor)
                self._emit_new_text(req, 1)
                if len(req.tokens_generated) >= req.request.max_tokens:
                    self._finish(req, "length")
                    continue
                spec.anchor = anchor
                spec.start = self._batched_cache.seq_lens_list()[req.batch_idx]

            assert spec.start is not None and spec.context is not None
            assert spec.anchor is not None
            target_len = self._batched_cache.seq_lens_list()[req.batch_idx]
            assert spec.start == target_len, (
                f"spec.start={spec.start} disagrees with target slot length {target_len}"
            )

            if not (spec.drafted_for_start == spec.start and spec.offered is not None):
                # The drafter's RoPE uses an unsliced cos/sin table, so (INV) is a
                # precondition of drafting. On the reuse path the fold into the
                # draft cache already happened, so the check lives only here.
                # A violation fails later with no recovery.
                assert target_len == spec.draft_cache.get_seq_length() + spec.context.shape[1], (
                    f"(INV) target_slot_seq_len={target_len} != "
                    f"draft_cache_len={spec.draft_cache.get_seq_length()} + "
                    f"context_rows={spec.context.shape[1]}"
                )
                room = req.request.max_tokens - len(req.tokens_generated)
                offered, confidence, draft_probs = self._dspark.propose(
                    anchor=spec.anchor,
                    context=spec.context,
                    start=spec.start,
                    draft_cache=spec.draft_cache,
                    temperature=req.request.sampling_params.temperature,
                )
                # Clamp so the verify forward appends no K/V that cannot commit.
                # The bonus token takes one place of `room`.
                spec.offered = offered[: max(0, room - 1)]
                spec.room_clamped = len(spec.offered) < len(offered)
                spec.confidence = confidence
                spec.draft_probs = draft_probs
                spec.drafted_for_start = spec.start
                spec.stats.n_drafter_forwards += 1

            assert spec.offered is not None
            out[id(req)] = [spec.anchor, *spec.offered]
        return out

    def _packed_forward(
        self, alive: list[RunningRequest], sampled_decode_tokens: dict[int, list[int]]
    ) -> tuple[torch.Tensor, list[int], dict[int, torch.Tensor] | None]:
        """Build packed inputs over the alive in-flight set and run ONE forward.

        Order in `alive` matches the cache's slot order (both are derived from
        `self._running` after the reap). `position_offsets` therefore line up
        with `cache.seq_lens_list()`.

        A decoder's q-length is `len(tokens)`. Request `index` has its
        last-position logits at row `cu_seqlens_q[index + 1] - 1`. Returns
        `(packed_logits, cu_seqlens_q, sink)` for `_verify_speculators`. `sink`
        is None when no request speculates. If the forward raises,
        `pending_decode_tokens` stays set, so the retry feeds the same tokens.
        """
        assert self._batched_cache is not None

        packed_input_ids: list[int] = []
        cu_seqlens_q: list[int] = [0]
        position_offsets: list[int] = []
        cache_seq_lens = self._batched_cache.seq_lens_list()

        for index, req in enumerate(alive):
            # `index` is both the packed position and the slot, because a reap keeps
            # `_running` in slot order. A mismatch writes K/V at another request's positions.
            assert req.batch_idx is not None and req.batch_idx == index
            if req.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING):
                chunk_start = req.tokens_prefilled
                # The slot length and the prefill counter advance together. A slot
                # ahead of its counter holds positions that an append claimed and
                # did not write. This chunk then stores its K/V after a gap, silently.
                assert chunk_start == cache_seq_lens[req.batch_idx], (
                    f"prefill at {chunk_start} disagrees with slot length "
                    f"{cache_seq_lens[req.batch_idx]}"
                )
                chunk_end = min(chunk_start + self._chunk_size, len(req.prompt_token_ids))
                chunk_tokens = req.prompt_token_ids[chunk_start:chunk_end]
                # With no rows, the last-position slice reads a neighbor's logits.
                # Admission and the prefix cache keep at least one token here.
                assert chunk_tokens, "prefilling request contributed no tokens"
                packed_input_ids.extend(chunk_tokens)
                cu_seqlens_q.append(cu_seqlens_q[-1] + len(chunk_tokens))
                position_offsets.append(chunk_start)
            else:
                next_tokens = sampled_decode_tokens[id(req)]
                # With no rows, the last-position slice reads a neighbor's logits.
                assert next_tokens, "decoding request contributed no tokens"
                packed_input_ids.extend(next_tokens)
                cu_seqlens_q.append(cu_seqlens_q[-1] + len(next_tokens))
                position_offsets.append(cache_seq_lens[req.batch_idx])

        # Tap hidden states only when a request speculates. The sink covers every
        # request's rows. With no speculator, no tap kwargs go in, so other models
        # keep working.
        sink: dict[int, torch.Tensor] | None = None
        self._pre_forward_seq_lens = cache_seq_lens
        if self._dspark is not None and any(r.spec is not None for r in alive):
            sink = {}
            packed_logits = self._runner.forward_step_packed(
                self._batched_cache,
                packed_input_ids,
                cu_seqlens_q,
                position_offsets,
                tap_layers=self._dspark.tap_layers,
                hidden_state_sink=sink,
            )
        else:
            packed_logits = self._runner.forward_step_packed(
                self._batched_cache, packed_input_ids, cu_seqlens_q, position_offsets
            )
        # Every slot now holds its K/V. Clear the snapshot BEFORE anything else
        # can fail, or a later rollback undoes real writes.
        self._pre_forward_seq_lens = None

        # Distribute logits and advance prefill state.
        for index, req in enumerate(alive):
            last_pos = cu_seqlens_q[index + 1] - 1
            if req.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING):
                tokens_added = cu_seqlens_q[index + 1] - cu_seqlens_q[index]
                req.tokens_prefilled += tokens_added
                if req.spec is not None:
                    # `context_window` clones the rows, so they do not pin the
                    # whole packed activation.
                    assert sink is not None and self._dspark is not None
                    req.spec.context_parts.append(
                        context_window(
                            sink,
                            self._dspark.ordered_taps,
                            cu_seqlens_q[index],
                            cu_seqlens_q[index + 1],
                        )
                    )
                if req.tokens_prefilled == len(req.prompt_token_ids):
                    req.state = RequestState.DECODING
                    req.last_logits = packed_logits[0, last_pos, :]
                else:
                    req.state = RequestState.CHUNKED_PREFILLING
            elif req.spec is not None:
                # The last row follows an OFFERED token that verify can reject.
                req.last_logits = None
            else:
                req.last_logits = packed_logits[0, last_pos, :]
                req.pending_decode_tokens = None
        return packed_logits, cu_seqlens_q, sink

    def _verify_speculators(
        self,
        alive: list[RunningRequest],
        packed_logits: torch.Tensor,
        cu_seqlens_q: list[int],
        sink: dict[int, torch.Tensor] | None,
    ) -> None:
        """Judge each proposal in flight against the forward that just ran.

        Select by `spec.offered is not None`, never by state: a request that just
        finished prefill is DECODING with no proposal, and verifying it emits
        garbage. Unlike `run_greedy`, the scheduler does not emit EOS. Like it,
        the scheduler clamps the commit to `max_tokens`.
        """
        if self._dspark is None:
            return
        tokenizer = self._runner.tokenizer
        for index, req in enumerate(alive):
            spec = req.spec
            if spec is None or spec.offered is None:
                continue
            assert spec.start is not None and spec.draft_probs is not None
            assert req.batch_idx is not None and self._batched_cache is not None

            lo = cu_seqlens_q[index]
            width = 1 + len(spec.offered)
            # A wrong offset reads ANOTHER request's logits and emits wrong tokens silently.
            assert cu_seqlens_q[index + 1] - lo == width, (
                f"verify window mismatch: cu_seqlens_q spans "
                f"{cu_seqlens_q[index + 1] - lo} rows, expected {width}"
            )
            window = packed_logits[:, lo : lo + width, :]

            accepted, bonus = verify_block(
                window,
                offered=spec.offered,
                draft_probs=spec.draft_probs,
                temperature=req.request.sampling_params.temperature,
            )

            if spec.confidence is not None:
                conf_probs = spec.confidence[0].sigmoid().tolist()
                for i in range(len(spec.offered)):
                    spec.stats.confidence_observations.append((float(conf_probs[i]), i < accepted))
            spec.stats.proposal_lengths.append(len(spec.offered))
            spec.stats.accepted_draft_lengths.append(accepted)
            spec.stats.acceptance_lengths.append(accepted + 1)
            if spec.room_clamped:
                spec.stats.room_clamped_rounds += 1
                spec.stats.room_clamped_committed += accepted + 1

            committed = [*spec.offered[:accepted], bonus]
            room = req.request.max_tokens - len(req.tokens_generated)
            emitted: list[int] = []
            hit_eos = False
            for tok in committed[:room]:
                if tok == tokenizer.eos_token_id:
                    hit_eos = True
                    break
                emitted.append(tok)
            if emitted:
                req.tokens_generated.extend(emitted)
                self._emit_new_text(req, len(emitted))

            if hit_eos:
                self._finish(req, "stop")
                continue
            if len(req.tokens_generated) >= req.request.max_tokens:
                self._finish(req, "length")
                continue
            # Roll the slot back to the committed prefix. The next round rewrites
            # the rejected tail. A finished request skips this, because the reap
            # frees its blocks.
            new_start = spec.start + len(emitted)
            self._batched_cache.truncate_to(req.batch_idx, new_start)

            # The next context is the committed rows only (anchor plus accepted).
            # It REPLACES the old context, as in `deepspec`.
            assert sink is not None and self._dspark is not None
            spec.context = context_window(sink, self._dspark.ordered_taps, lo, lo + accepted + 1)
            spec.anchor = bonus
            spec.start = new_start
            spec.offered = None
            spec.draft_probs = None
            spec.confidence = None
            spec.drafted_for_start = None
            spec.room_clamped = False

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
            # The API handle keeps the RunningRequest alive after completion,
            # and the draft cache and context must not live that long.
            finished.spec = None
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

        Summarize speculator stats before the reap drops `spec` and before the
        terminal-step put, which publishes the summary to the API thread.
        """
        if req.spec is not None:
            assert self._dspark is not None
            req.spec_summary = req.spec.stats.summary(self._dspark.block_size)
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
