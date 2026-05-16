"""Decode side of the disaggregated PD pipeline.

Two layers of API on the same machinery:

  - **`DecodeSession`** is the per-step primitive that the `PDScheduler`
    drives. A session owns a `PagedKVCache` and tracks per-slot state
    for every handoff currently in the decode pool. The scheduler
    interleaves `add_handoff` / `step` / `remove_slot` calls to
    multiplex many requests through a single decode loop.
  - **`DecodeWorker.decode(handoff)`** and **`decode_batch([handoffs])`**
    are convenience wrappers built on top of the session: they open a
    fresh session, push everything through, drain results. Single-call
    APIs for callers that don't want a long-lived scheduler.

Per-request termination (EOS, `max_tokens`, cancellation) lives in the
caller, not the session. The session just runs forwards; the caller
decides when each slot is done and calls `remove_slot`.

The "first token comes from the handoff, not from this worker's model"
detail matters for parity: prefill saw the full prompt context when it
sampled, so it produced the same token target-alone would produce. The
decode worker continues from there. Token-for-token greedy parity vs
`ContinuousScheduler` falls out by construction.

Limitations match the prefill side:
  - `kv_quant` must be `None`.
  - The handoff's tensors are moved to this worker's device on entry.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import sample
from mini_infer.workers.kv_handoff import KVHandoff

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _SlotState:
    """Per-slot state maintained by a `DecodeSession`.

    The session needs `handoff.sampling_params` to call `sample(...)`
    and `last_token` to feed the next forward. EOS / max_tokens /
    cancellation logic lives in the caller, not here.
    """

    handoff: KVHandoff
    last_token: int


class DecodeSession:
    """Per-session decode state: paged cache + per-slot bookkeeping.

    A session is the per-step primitive that drives batched decode for
    multiple concurrent requests. The lifecycle:

      session = DecodeSession(runner)
      slot_a = session.add_handoff(handoff_a)
      slot_b = session.add_handoff(handoff_b)
      while ...:
          tokens = session.step()  # {slot_id: new_token}
          # caller checks each slot for termination
      session.remove_slot(slot_a)
      session.remove_slot(slot_b)

    Slot IDs are stable for the lifetime of the slot. The underlying
    `PagedKVCache.batch_idx` may shift when other slots are removed
    (cache compacts), but the slot_id doesn't change — the session
    maintains the slot_id ↔ batch_idx mapping internally.

    Why this exists separate from `decode_batch`: `decode_batch` runs
    the entire decode loop in one call, with no way to add or remove
    slots mid-loop. The `PDScheduler` wants to interleave operations
    (new handoff arrives mid-decode; old slot terminates; both happen
    between forwards). The session exposes the smaller primitives.
    """

    def __init__(self, runner: ModelRunner) -> None:
        if runner.block_pool.kv_quant is not None:
            raise NotImplementedError(
                f"DecodeSession requires kv_quant=None (got "
                f"kv_quant={runner.block_pool.kv_quant!r}); KV-quant + PD is not wired."
            )
        self._runner = runner
        self._cache = PagedKVCache(runner.block_pool)
        # Monotonic slot id generator. Stable per slot; cache batch_idx
        # is given by `self._ordered.index(slot_id)`.
        self._next_slot_id = 0
        self._slots: dict[int, _SlotState] = {}
        self._ordered: list[int] = []

    @property
    def num_active_slots(self) -> int:
        return len(self._ordered)

    def is_active(self, slot_id: int) -> bool:
        """True iff `slot_id` is still in the session (not yet removed)."""
        return slot_id in self._slots

    def add_handoff(self, handoff: KVHandoff) -> int:
        """Materialize `handoff` into a fresh cache slot; return slot_id.

        The slot_id is stable for the lifetime of the slot. Use it to
        identify this request in `step()` output dicts and in
        `remove_slot()`.
        """
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        self._cache.add_request_slot()
        self._ordered.append(slot_id)
        self._slots[slot_id] = _SlotState(
            handoff=handoff,
            last_token=handoff.first_sampled_token_id,
        )
        self._materialize_handoff_into_last_slot(handoff)
        return slot_id

    def step(self) -> dict[int, int]:
        """Run one batched decode forward; return `{slot_id: sampled_token}`.

        The returned tokens are also stored as each slot's `last_token`
        so the next `step()` feeds them automatically.

        Empty pool (no slots added, or all removed): returns `{}` without
        running a forward.
        """
        if not self._ordered:
            return {}
        last_tokens = [self._slots[sid].last_token for sid in self._ordered]
        _, logits_list = self._runner.decode_batch(self._cache, last_tokens)
        out: dict[int, int] = {}
        for batch_idx, slot_id in enumerate(self._ordered):
            state = self._slots[slot_id]
            next_token = sample(logits_list[batch_idx], state.handoff.sampling_params)
            state.last_token = next_token
            out[slot_id] = next_token
        return out

    def remove_slot(self, slot_id: int) -> None:
        """Free the slot's blocks and drop it from the session.

        After this call, `slot_id` is invalid; calling `step()` will no
        longer produce a token for it. Later slot_ids keep their
        identity (the session re-maps to the cache's new batch_idx
        layout internally).
        """
        if slot_id not in self._slots:
            raise KeyError(f"slot_id={slot_id} is not active in this session")
        batch_idx = self._ordered.index(slot_id)
        self._cache.remove_request(batch_idx)
        self._ordered.pop(batch_idx)
        del self._slots[slot_id]

    def _materialize_handoff_into_last_slot(self, handoff: KVHandoff) -> None:
        """Write the handoff's per-stream KV into the most-recently-added slot.

        `append_stream_packed` writes only into slots whose
        `cu_seqlens_q_new` increment is non-zero, so we mark just the
        target slot. Block allocation happens on the first (layer 0,
        first-stream) call; subsequent (layer, stream) pairs just write
        into the already-allocated blocks.

        Stream order: iterate `pool.stream_names(layer_idx)` so the
        first-stream-of-layer-0 trigger fires correctly. Standard
        models declare `["k", "v"]`; MLA declares `["kv_latent",
        "k_rope"]`; V4 has its own list.
        """
        pool = self._runner.block_pool
        prefill_len = handoff.prefill_len
        target_slot = self._cache.batch_size - 1
        device = pool.storage_for_stream(0, pool.stream_names(0)[0]).device
        cu_seqlens_list = [0] * (self._cache.batch_size + 1)
        for r in range(self._cache.batch_size):
            if r == target_slot:
                cu_seqlens_list[r + 1] = cu_seqlens_list[r] + prefill_len
            else:
                cu_seqlens_list[r + 1] = cu_seqlens_list[r]
        cu_seqlens_q_new = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

        if pool.num_layers != handoff.num_layers:
            raise ValueError(
                f"handoff has {handoff.num_layers} layers but pool has {pool.num_layers}"
            )
        for layer_idx in range(pool.num_layers):
            expected_streams = pool.stream_names(layer_idx)
            handoff_streams = handoff.kv_streams_per_layer[layer_idx]
            if set(expected_streams) != set(handoff_streams.keys()):
                raise ValueError(
                    f"handoff layer {layer_idx} streams {sorted(handoff_streams.keys())} "
                    f"don't match pool streams {sorted(expected_streams)}"
                )
            for stream_name in expected_streams:
                stream_packed = handoff_streams[stream_name]
                if stream_packed.device != device:
                    stream_packed = stream_packed.to(device=device)
                if stream_packed.shape[0] != prefill_len:
                    raise ValueError(
                        f"handoff layer {layer_idx} stream {stream_name!r} has "
                        f"{stream_packed.shape[0]} positions; expected {prefill_len}"
                    )
                self._cache.append_stream_packed(
                    stream_packed=stream_packed,
                    cu_seqlens_q_new=cu_seqlens_q_new,
                    layer_idx=layer_idx,
                    stream_name=stream_name,
                )


class DecodeWorker:
    """Owns a `ModelRunner`; ingests a `KVHandoff` and streams decoded tokens.

    Two API tiers:

      - `decode(handoff) -> Iterator[int]` and
        `decode_batch([handoffs]) -> [[int]]` for callers that just want
        decoded tokens out of one or more handoffs.
      - `start_session() -> DecodeSession` for the `PDScheduler`, which
        needs to interleave operations across many concurrent requests.

    The session-based path is the primitive; `decode` and `decode_batch`
    are convenience wrappers over a freshly-opened session.
    """

    def __init__(self, runner: ModelRunner) -> None:
        if runner.block_pool.kv_quant is not None:
            raise NotImplementedError(
                f"DecodeWorker requires kv_quant=None (got "
                f"kv_quant={runner.block_pool.kv_quant!r}); KV-quant + PD is not wired."
            )
        self._runner = runner

    @property
    def runner(self) -> ModelRunner:
        return self._runner

    def start_session(self) -> DecodeSession:
        """Open a fresh `DecodeSession`. Used by the `PDScheduler`."""
        return DecodeSession(self._runner)

    def decode(self, handoff: KVHandoff) -> Iterator[int]:
        """Yield decoded token ids one at a time, starting with the handoff's first token.

        Stops at the first of:
          - `len(emitted) == handoff.max_tokens`,
          - emitted token equals `handoff.eos_token_id`.

        The first yielded token is `handoff.first_sampled_token_id` (the
        token sampled from the last prefill logit, already determined by
        the prefill worker). Subsequent tokens come from this worker's
        decode steps.

        Internally: opens a fresh `DecodeSession`, drives one slot
        through it until termination. The session-based API is the
        primitive; this is the convenience wrapper.
        """
        if handoff.max_tokens <= 0:
            return
        session = self.start_session()
        slot_id: int | None = None
        try:
            slot_id = session.add_handoff(handoff)
            first = handoff.first_sampled_token_id
            yield first
            if handoff.max_tokens <= 1:
                return
            if handoff.eos_token_id is not None and first == handoff.eos_token_id:
                return
            emitted = 1
            while emitted < handoff.max_tokens:
                tokens = session.step()
                next_token = tokens[slot_id]
                yield next_token
                emitted += 1
                if handoff.eos_token_id is not None and next_token == handoff.eos_token_id:
                    return
        finally:
            if slot_id is not None and session.is_active(slot_id):
                session.remove_slot(slot_id)

    def decode_batch(self, handoffs: list[KVHandoff]) -> list[list[int]]:
        """Run decode for a batch of handoffs in batched-decode forwards.

        All B handoffs are materialized into a shared session, then the
        decode loop runs ONE `session.step()` per iteration (B tokens
        per step, one per slot). Per-request termination (EOS or
        `max_tokens`) is tracked individually; finished slots still
        participate in subsequent forwards (their outputs are discarded)
        until every request has terminated.

        Returns a list of B token-id lists, one per input handoff, each
        starting with `handoff.first_sampled_token_id`.

        Equivalent in output to calling `decode(handoff)` sequentially
        per handoff (same emitted tokens, same termination behaviour),
        but uses one forward per step instead of B per step. Parity is
        validated against the sequential loop in
        `tests/unit/test_workers_batch.py`.

        Internally: opens a fresh `DecodeSession`, adds all handoffs,
        drives the loop with per-slot alive tracking. The session-based
        API is the primitive; this is the convenience wrapper.

        Caller invariants:
          - Single-call: don't share a `DecodeWorker` across concurrent
            `decode_batch` calls.
          - All handoffs must come from a prefill worker whose pool has
            the same stream topology as this worker's pool.
        """
        if not handoffs:
            return []
        session = self.start_session()
        slot_ids: list[int] = []
        try:
            for handoff in handoffs:
                slot_ids.append(session.add_handoff(handoff))

            outputs: list[list[int]] = []
            alive: list[bool] = []
            for handoff in handoffs:
                first = handoff.first_sampled_token_id
                outputs.append([first])
                is_eos = handoff.eos_token_id is not None and first == handoff.eos_token_id
                alive.append(handoff.max_tokens > 1 and not is_eos)

            while any(alive):
                tokens = session.step()
                for r, slot_id in enumerate(slot_ids):
                    if not alive[r]:
                        # Slot still participates in the forward; we discard its output.
                        continue
                    handoff = handoffs[r]
                    next_token = tokens[slot_id]
                    outputs[r].append(next_token)
                    if len(outputs[r]) >= handoff.max_tokens:
                        alive[r] = False
                        continue
                    if handoff.eos_token_id is not None and next_token == handoff.eos_token_id:
                        alive[r] = False
            return outputs
        finally:
            # Drain remaining slots in reverse order so each batch_idx stays valid.
            for slot_id in reversed(slot_ids):
                if session.is_active(slot_id):
                    session.remove_slot(slot_id)
