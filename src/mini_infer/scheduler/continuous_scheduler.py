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
   A request whose tokens a previous step sampled but could not feed (its
   forward hit a recoverable OOM) hands those back instead of sampling again.
3. Reap any DONE requests (free blocks, shift batch indices). This happens
   BEFORE the forward so the forward only runs over the alive set.
4. Build packed inputs over the alive in-flight set: each prefilling request
   contributes its next chunk's tokens, each decoding request contributes the
   token list step 2 produced for it. ONE `runner.forward_step_packed(...)`
   call per step.
5. Slice each request's last-position logits out of the packed result and
   advance prefill state.

A decoding request's contribution is a LIST, not a single token, so requests
with different q-lengths ride one forward. Plain sampling fills it with exactly
one token; DSpark speculative decoding (opt-in via the `dspark` constructor
argument) fills it with `[anchor, *offered]`, the drafted block the same
forward then verifies. The speculative phases wrap steps 2 and 5:

2b. `_draft_speculators`: for each speculating DECODING request, draft a block
    off the last committed token (the anchor) and the injected target context.
    Requests whose forward failed last step reuse their pending proposal.
5b. `_verify_speculators`: slice each speculator's window out of the packed
    logits, accept the longest matching prefix plus the target's own bonus
    token, emit, roll the KV slot back to the committed length, and refresh
    the drafter's context from this forward's tapped hidden states.

A request speculates only if it was eligible at admission (greedy or plain
temperature sampling, prompt under the configured cap); everything else in
the same batch runs the plain path, so mixed batches are the normal case,
not a special one.

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
from mini_infer.exceptions import OutOfMemoryError
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
    the alive set via `runner.forward_step_packed(...)`.

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
        # A prefix-cache hit pre-populates a slot with positions that never
        # pass through a forward, so those positions produce no tapped hidden
        # states and the drafter's injected context comes up short of the
        # target slot length. That violates the RoPE invariant the drafter
        # rotates against and kills the engine thread. Refusing the
        # combination at construction is the loudest failure available;
        # injecting only the post-prefix suffix would feed the drafter less
        # context than it trained on, which is a measurement, not a fallback.
        if dspark is not None and runner.block_pool.prefix_cache is not None:
            raise ValueError(
                "DSpark speculative decoding cannot run with a prefix cache: "
                "cached prompt positions never produce the tapped hidden states "
                "the drafter needs. Construct the runner with prefix_cache=False."
            )
        # The drafter is fed the target's hidden states at tapped layers, which
        # only a model whose forward accepts the tap kwargs can produce. Without
        # this check the mismatch surfaces as a TypeError inside the first
        # forward that carries a speculator, which `_engine_loop` does not catch
        # and which therefore kills every in-flight request rather than one.
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
        # Per-slot cache lengths saved right before the packed forward and
        # cleared the moment it returns. Non-None only while a forward is in
        # flight or after one raised, which is exactly when
        # `_roll_back_partial_append` needs it.
        self._pre_forward_seq_lens: list[int] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
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

        False after `stop()`, and also after the thread died on an exception
        other than `OutOfMemoryError`. In that second case no request in
        flight ever receives its terminal step, so a caller that waits on
        results can poll this to fail fast instead of waiting out a timeout.
        """
        return self._thread is not None and self._thread.is_alive()

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
        if self._trace_events is not None and self._trace_out is not None:
            with open(self._trace_out, "w") as f:
                json.dump({"traceEvents": self._trace_events}, f)

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
        """Engine-thread main loop; runs `_step()` until `stop()` is signaled.

        Mid-step OOM is recoverable: the pool can hand out blocks during a
        forward (decode growth, partial prefill chunk needing a new block)
        and may run out even after admission accepted everyone. Killing the
        engine thread on that would take down every in-flight request and
        every future one. Instead we preempt the youngest still-prefilling
        request and let the loop retry; the cancelled request returns
        `finish_reason="cancelled"` to its client. Other exceptions still
        crash the thread loudly, because they signal real bugs.
        """
        while not self._stop_event.is_set():
            try:
                self._step()
            except OutOfMemoryError:
                logger.warning("engine OOM during step; preempting youngest in-flight request")
                self._preempt_on_oom()
            except Exception:
                logger.exception("engine thread crashed")
                raise
            # When fully idle, wait briefly so we don't spin the CPU.
            if not self._running and self._waiting.empty():
                self._stop_event.wait(timeout=self.IDLE_SLEEP_SECONDS)

    def _preempt_on_oom(self) -> None:
        """Cancel the youngest in-flight request and reap so the loop can retry.

        Picks a prefiller over a decoder when possible: prefillers haven't
        emitted any user-visible output yet, so the user-facing damage is
        smaller. Among prefillers we cancel the most recently admitted one
        (FILO), matching the intuition that the most recent admission is
        what pushed us past the watermark.

        Survivors resume the failed step exactly. Every slot's cache length is
        first rolled back to what it was before the forward
        (`_roll_back_partial_append`). Plain decoders keep the tokens they
        already sampled and emitted in `pending_decode_tokens`, and speculators
        keep their whole round (anchor, proposal, and drafter cache), so the
        retry re-feeds the same inputs at the same positions and the client
        never sees a token twice.
        """
        # Before the reap: the saved lengths are indexed by the batch_idx
        # values the failed forward used, and the reap shifts them.
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

        `append_kv_packed` works in two phases. On layer 0 it first advances
        each slot's token count and allocates its blocks, slot by slot, and
        only then writes any K/V. When the pool runs dry at slot j, slots
        0..j-1 already claim positions that hold no data. The retry takes its
        position offsets from those counts, so without this a plain decoder
        writes its token at the wrong position (silent corruption) and a
        speculator fails its start-vs-slot assertion (a dead engine thread,
        because only `OutOfMemoryError` is caught).

        Speculator state needs no undo here. Drafting finishes before the
        forward starts, so each speculator's drafter cache and proposal are
        complete and consistent at `spec.start`, and the retry reuses them
        through the `drafted_for_start` guard. Rolling the drafter cache back
        as well would break (INV) on that reuse path.

        Safe to apply to every slot unconditionally. `truncate_to` only
        shrinks and is a no-op at the current length, and it cannot land inside
        a published prefix block: publication happens on the last layer, after
        every write, so a layer-0 failure publishes nothing new.

        Blocks: every slot whose count advanced gets the blocks it claimed past
        the boundary freed. The slot whose `allocate` raised is different. It
        adds blocks to its list before it advances its count, so it can keep
        blocks it received before the raise while its count stays put, and
        `truncate_to` leaves them alone. They do not leak: the retry feeds that
        slot the same q-length, so the append reuses exactly those blocks, and
        `remove_request` frees them if the slot is the victim or is cancelled.
        Between this rollback and the retry, that one slot holds more blocks
        than its length needs.
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
        """One scheduler iteration: admit, sample/draft, reap, forward, verify."""
        step_start = time.perf_counter() if self._trace_events is not None else 0.0

        self._cancel_pending()
        self._admit_waiting()
        sampled_decode_tokens = self._sample_decoders()
        sampled_decode_tokens.update(self._draft_speculators())
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
            # Requests carrying a verify window in this forward. `offered` is
            # set by the draft phase and cleared by verify, so it must be read
            # here, before the forward.
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
                    # B counts requests, total_q counts the q-tokens of the
                    # forward. Plain decoding makes them equal; prefill chunks
                    # and verify windows make total_q larger, which is the
                    # quantity a step's cost actually follows.
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

            # A prompt that tokenizes to nothing has no q-tokens to contribute,
            # so admitting it would put a zero-length window in the packed
            # forward: `cu_seqlens_q` would not advance for its slot and the
            # last-position slice would read index -1 of an empty logits
            # tensor. That IndexError is not `OutOfMemoryError`, so it kills the
            # engine thread and every other in-flight request with it. Finish it
            # here with nothing generated, as StateCacheContinuousScheduler does.
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
                # A round appends `1 + len(offered)` positions but commits only
                # `accepted + 1`, and truncation returns the difference, so
                # steady-state blocks per EMITTED token are unchanged by
                # speculation. What is new is only the intra-step transient of
                # the in-flight verify window; multiplying the whole headroom
                # by the block length would throttle achievable concurrency.
                spec_blocks = math.ceil((self._dspark.block_size + 1) / block_size) + 1
                if plain_blocks + spec_blocks > block_pool.num_blocks:
                    # Speculation is an optimization, never a reason to refuse
                    # work the plain path could serve. Downgrade instead of
                    # falling through to the permanent rejection below: turning
                    # a servable request into finish_reason="cancelled" purely
                    # because dspark is configured would be a regression from
                    # the non-speculative engine.
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

    def _spec_state_for(self, running: RunningRequest) -> SpecDecodeState | None:
        """Decide at admission whether this request speculates; None runs plain.

        Decided once, here, so the rest of the engine never re-evaluates
        eligibility mid-request. Two downgrades, both logged at INFO because
        they silently change latency, not output:

        - top-k / top-p sampling: the spec path builds distributions through
          `logits_to_probs`, which applies temperature only. Silently ignoring
          the caller's filtering would change sampling semantics, so those
          requests run the plain path instead (mixed batches are the normal
          case). Threading top-k/top-p through both sides is deferred.
        - prompts over `max_spec_prompt_tokens`: the injected context buffers
          tapped hidden states per prompt token outside the block pool's
          accounting (see `DSparkSpec`).
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

        Each value is the list of tokens that request feeds into the next
        forward. Plain sampling always yields exactly one; the list is what
        lets a request whose next forward spans several positions share the
        packed forward with everyone else.

        A request whose tokens were sampled but never consumed (its forward
        raised a recoverable OOM) is NOT sampled again: it hands the same tokens
        back. Sampling emits, and an emitted token cannot be recalled from the
        client's stream, so re-sampling would duplicate it.
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

        Handles multi-byte UTF-8 sequences correctly at the cost of O(n) work
        per call (one tokenizer call, one string slice). A call covering
        several newly appended tokens emits ONE step carrying their whole
        delta, which is how the speculative path streams a committed block.
        `num_tokens` says how many tokens that one step carries, so a consumer
        can count tokens without counting steps.
        """
        current_text = self._runner.tokenizer.decode(req.tokens_generated)
        delta = current_text[len(req.last_text) :]
        req.last_text = current_text
        self._emit(req, GenerationStep(text=delta, num_tokens=num_tokens))

    def _draft_speculators(self) -> dict[int, list[int]]:
        """Draft one block per speculating DECODING request; return verify inputs.

        Each entry is `[anchor, *offered]`: the last committed token plus the
        drafted block after confidence truncation and `max_tokens` clamping.
        The packed forward appends K/V for exactly those positions, and
        `_verify_speculators` judges them after it runs.

        Three cases per request, in order:

        - No anchor yet (the request finished prefill last step): sample it
          from `last_logits` like any plain decoder, emit it, run the finish
          checks, and finalize the injected context accumulated across the
          prompt's chunks. The draft then hangs off that anchor in the same
          step, so speculation starts without a wasted round.
        - A proposal already exists for this exact `start` (the previous
          forward raised recoverably and the engine retried): reuse it. The
          proposal is a pure function of `(anchor, context, start)`, and
          re-drafting would double-mutate the draft cache.
        - Otherwise draft, clamping the proposal so a fully accepted block
          plus its bonus can never overshoot `max_tokens`.
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
                # The prompt's tapped hidden states, accumulated chunk by
                # chunk, become round 1's injected context.
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
                # The invariant everything hinges on: the drafter's RoPE
                # rotates keys against an unsliced cos/sin table, so the
                # injected context plus the drafter's own cache must land
                # exactly at the target slot length. It is a PREcondition of
                # proposing (the forward folds the context into the cache and
                # truncates back to `start`), which is why it sits inside the
                # draft branch: on the reuse path the fold already happened.
                # Asserted rather than trusted, because every known violation
                # route fails unrecoverably later (INV).
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
                # Clamp at draft time so the verify forward does not append
                # K/V for positions that can never commit: the bonus token
                # occupies one slot of the remaining room, so at most
                # `room - 1` draft tokens are worth verifying.
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

        A decoder's q-length this step is `len(tokens)`, so `cu_seqlens_q`
        advances by that instead of by 1. The forward returns one packed
        `(1, total_q, vocab)` tensor and each request's last-position logits
        are sliced out of it at `cu_seqlens_q[index + 1] - 1`.

        Returns `(packed_logits, cu_seqlens_q, sink)` so `_verify_speculators`
        can slice each speculator's window out of the same forward; `sink` is
        None when nothing speculated.

        Reaching the end of this method is what marks a decoder's tokens as
        consumed. A forward that raises leaves `pending_decode_tokens` set, and
        the retry re-feeds them rather than sampling and emitting again.
        """
        assert self._batched_cache is not None

        packed_input_ids: list[int] = []
        cu_seqlens_q: list[int] = [0]
        position_offsets: list[int] = []
        cache_seq_lens = self._batched_cache.seq_lens_list()

        for index, req in enumerate(alive):
            # The packed lists are built in `alive` order and the cache is read
            # per slot; `index` serves as both only because a reap leaves
            # `_running` in slot order. With variable q-lengths a mismatch would
            # write one request's K/V at another's positions, so pin it here.
            assert req.batch_idx is not None and req.batch_idx == index
            if req.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING):
                chunk_start = req.tokens_prefilled
                # A prefiller's slot length and its prefill counter advance
                # together, chunk by chunk (and a prefix-cache hit sets both at
                # admission). A slot ahead of its counter means an append that
                # claimed positions without writing them; feeding the chunk
                # anyway would store its K/V past a hole, silently. The
                # speculators' start-vs-slot assertion guards the same failure.
                assert chunk_start == cache_seq_lens[req.batch_idx], (
                    f"prefill at {chunk_start} disagrees with slot length "
                    f"{cache_seq_lens[req.batch_idx]}"
                )
                chunk_end = min(chunk_start + self._chunk_size, len(req.prompt_token_ids))
                chunk_tokens = req.prompt_token_ids[chunk_start:chunk_end]
                # Same hazard as the decode branch below: no rows of its own
                # means the last-position slice reads a neighbour. Unreachable
                # now that admission rejects empty prompts and the prefix
                # cache's last-token rule always leaves a token unprocessed.
                assert chunk_tokens, "prefilling request contributed no tokens"
                packed_input_ids.extend(chunk_tokens)
                cu_seqlens_q.append(cu_seqlens_q[-1] + len(chunk_tokens))
                position_offsets.append(chunk_start)
            else:
                next_tokens = sampled_decode_tokens[id(req)]
                # A slot contributing zero tokens gets no row of its own in the
                # packed forward, and the last-position slice below would then
                # silently read the PREVIOUS request's logits.
                assert next_tokens, "decoding request contributed no tokens"
                packed_input_ids.extend(next_tokens)
                cu_seqlens_q.append(cu_seqlens_q[-1] + len(next_tokens))
                position_offsets.append(cache_seq_lens[req.batch_idx])

        # Tapped hidden states are collected only when some request in this
        # batch speculates: the sink covers EVERY request's rows, and each
        # speculator later slices its own window out by `cu_seqlens_q`. A
        # batch with no speculators skips the tap kwargs entirely, which is
        # also what keeps non-Qwen3 models working when dspark is off.
        sink: dict[int, torch.Tensor] | None = None
        # Saved for `_roll_back_partial_append`: if the forward raises
        # mid-append, these are the lengths every slot must return to.
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
        # The forward landed, so every slot now legitimately holds the K/V it
        # appended. Clear the snapshot BEFORE anything else can fail: a stale
        # one would let a later rollback undo writes that really happened.
        self._pre_forward_seq_lens = None

        # Distribute logits and advance prefill state.
        for index, req in enumerate(alive):
            last_pos = cu_seqlens_q[index + 1] - 1
            if req.state in (RequestState.PREFILLING, RequestState.CHUNKED_PREFILLING):
                tokens_added = cu_seqlens_q[index + 1] - cu_seqlens_q[index]
                req.tokens_prefilled += tokens_added
                if req.spec is not None:
                    # This chunk's tapped hidden states become part of round
                    # 1's injected context; cloned so the buffer does not pin
                    # the whole packed activation.
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
                # A speculator's last row is the logits after its last OFFERED
                # token, which may be rejected; the verify phase derives the
                # real continuation, so leave nothing tempting behind.
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
        """Judge each in-flight proposal against the forward that just ran.

        Membership is keyed off `spec.offered is not None`, never off
        `RequestState`: `_packed_forward` flips PREFILLING -> DECODING
        mid-iteration, so a request that just finished prefill is DECODING by
        now while having no proposal, and reading a verify window for it would
        take the argmax at its chunk's first position and emit garbage.

        Emission follows the scheduler's conventions, which differ from the
        batch-1 runner's in two deliberate ways: EOS is dropped, not emitted
        (`run_greedy` appends it), and the commit is clamped so
        `len(tokens_generated)` can never overshoot `max_tokens`.
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
            # A wrong offset here reads ANOTHER request's logits and emits
            # plausible wrong tokens with no error at all, so the window is
            # pinned to the request's own packed extent.
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
                # The scheduler's convention: stop BEFORE the first EOS and do
                # not emit it. The opposite of `run_greedy`, which appends it.
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
            # A finished request skips the rollback: its blocks are about to
            # be freed wholesale by the reaper. A continuing one rolls its
            # slot back to exactly the committed prefix; the verify forward
            # wrote K/V for every offered token and the rejected tail is
            # wrong, so the next round rewrites those positions.
            new_start = spec.start + len(emitted)
            self._batched_cache.truncate_to(req.batch_idx, new_start)

            # Next round's context is the target's hidden states for the
            # committed window only: rows 0..accepted of this request's verify
            # window, i.e. the anchor plus the accepted tokens. REPLACES the
            # previous context rather than appending, mirroring `deepspec`.
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
            # Drop speculative state with the request: the API handle keeps
            # the RunningRequest alive after completion, and the drafter
            # cache plus injected context must not outlive the generation.
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

        A speculator's stats are summarized here, not in the reap. The reap
        drops `spec` and runs after this, and the summary must be in place
        before the terminal step goes out: the queue put is what makes it
        visible to the API thread that reads `GenerationResult`.
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
