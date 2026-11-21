"""Publish KV-cache events over ZMQ in vLLM's wire format.

A KV-cache-aware router (for example the llm-d router) subscribes to these
events and keeps an index of the blocks each server holds. The format follows
vLLM's `ZmqEventPublisher`, so a router that reads vLLM servers reads this one
with its vLLM settings.

Wire format:

- One ZMQ multipart message per batch: `[topic, seq, payload]`.
  - `topic`: the configured topic, UTF-8. The llm-d router reads
    `kv@<pod>@<model>`, or `kv@<model>` when it subscribes to each pod itself.
  - `seq`: an 8-byte big-endian counter, one per batch, from 0. A router sees a
    smaller number after a restart.
  - `payload`: msgpack array `[timestamp, [event, ...]]`, timestamp in seconds.
- Each event is a msgpack array with its tag first:
  - `["BlockStored", block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium]`
  - `["BlockRemoved", block_hashes, medium]`
  - `["AllBlocksCleared"]`

  Hashes are msgpack bin values. `lora_id` and `medium` are nil: mini-infer has
  no LoRA adapters, and a nil medium means the GPU tier to the router.

The publisher sends `AllBlocksCleared` first, so a router drops what it knew
about an earlier run under the same topic. PUB/SUB drops messages that are sent
before a subscriber joins, so this first batch is best effort.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from mini_infer.cache.kv_events import AllBlocksCleared, BlockRemoved, BlockStored, KVEvent

logger = logging.getLogger(__name__)

# Events that one batch holds at most. The engine thread emits a few events per
# step, so a batch is normally small. The cap bounds one message after a stall.
_MAX_EVENTS_PER_BATCH = 1024
# Messages that ZMQ queues for a slow subscriber before it drops new ones. The
# same value as vLLM's publisher.
_SEND_HIGH_WATER_MARK = 100_000
# Time that closing the socket waits to send the messages still queued.
_CLOSE_LINGER_MS = 1000

_INSTALL_HINT = "KV-cache events need pyzmq and msgpack: install mini-infer[kv-events]"


def _import_msgpack() -> Any:
    # Typed as Any: msgpack ships no type information.
    try:
        import msgpack
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return msgpack


def _import_zmq() -> ModuleType:
    try:
        import zmq
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return zmq


def encode_event(event: KVEvent) -> list[Any]:
    """Return the msgpack array of one event, in vLLM's field order."""
    if isinstance(event, BlockStored):
        return [
            "BlockStored",
            list(event.block_hashes),
            event.parent_block_hash,
            list(event.token_ids),
            event.block_size,
            None,  # lora_id
            None,  # medium
        ]
    if isinstance(event, BlockRemoved):
        return ["BlockRemoved", list(event.block_hashes), None]
    if isinstance(event, AllBlocksCleared):
        return ["AllBlocksCleared"]
    raise TypeError(f"not a KV-cache event: {event!r}")


def encode_event_batch(timestamp: float, events: Sequence[KVEvent]) -> bytes:
    """Return the msgpack payload of one batch."""
    msgpack = _import_msgpack()
    payload: bytes = msgpack.packb(
        [float(timestamp), [encode_event(event) for event in events]], use_bin_type=True
    )
    return payload


def binds(endpoint: str) -> bool:
    """Whether the publisher binds `endpoint` (True) or connects to it (False).

    This is vLLM's rule: a wildcard address or a local transport binds, and a
    concrete remote address connects. Bind `tcp://*:5557` when the router
    subscribes to each server. Connect to the router's address when the router
    binds one socket that every server publishes to.
    """
    return "*" in endpoint or "::" in endpoint or endpoint.startswith(("ipc://", "inproc://"))


class ZmqKVEventPublisher:
    """Send KV-cache events to a ZMQ PUB socket from a background thread.

    The instance is a `KVEventListener`: pass it to
    `PrefixCache.set_event_listener`. A call only puts the event on a queue, so
    the engine thread never waits for the network. The thread sends all the
    events that wait on the queue as one batch.
    """

    def __init__(self, endpoint: str, topic: str) -> None:
        self._zmq = _import_zmq()
        _import_msgpack()
        self._endpoint = endpoint
        self._topic = topic.encode("utf-8")
        self._queue: queue.SimpleQueue[KVEvent | None] = queue.SimpleQueue()
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._setup_error: BaseException | None = None
        self._last_endpoint = endpoint
        self._queue.put(AllBlocksCleared())
        self._thread = threading.Thread(target=self._run, name="mini-infer-kv-events", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._setup_error is not None:
            raise RuntimeError(
                f"cannot open the KV-event socket at {endpoint}"
            ) from self._setup_error

    @property
    def endpoint(self) -> str:
        """The socket's endpoint, with the real port when the configured one was a wildcard."""
        return self._last_endpoint

    def __call__(self, event: KVEvent) -> None:
        if self._closed.is_set():
            return
        self._queue.put(event)

    def close(self, timeout: float = 5.0) -> None:
        """Send the events already queued, then close the socket."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        zmq = self._zmq
        # A ZMQ socket must stay on one thread, so this thread creates it.
        context = zmq.Context()
        try:
            socket = context.socket(zmq.PUB)
            socket.setsockopt(zmq.SNDHWM, _SEND_HIGH_WATER_MARK)
            socket.setsockopt(zmq.LINGER, _CLOSE_LINGER_MS)
            if binds(self._endpoint):
                socket.bind(self._endpoint)
            else:
                socket.connect(self._endpoint)
            self._last_endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        except Exception as exc:
            self._setup_error = exc
            self._closed.set()
            self._ready.set()
            context.destroy(linger=0)
            return
        self._ready.set()
        logger.info(
            "publishing KV-cache events on %s (topic %r)",
            self._last_endpoint,
            self._topic.decode("utf-8"),
        )
        seq = 0
        try:
            stop = False
            while not stop:
                first = self._queue.get()
                if first is None:
                    break
                batch = [first]
                while len(batch) < _MAX_EVENTS_PER_BATCH:
                    try:
                        event = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if event is None:
                        stop = True
                        break
                    batch.append(event)
                payload = encode_event_batch(time.time(), batch)
                # A PUB socket does not block: it drops the message for a
                # subscriber that is over the high-water mark.
                socket.send_multipart([self._topic, seq.to_bytes(8, "big"), payload])
                seq += 1
        except Exception:
            # Stop taking events, so the queue does not grow without a reader.
            self._closed.set()
            logger.exception("the KV-event publisher stopped; the router index is now stale")
        finally:
            socket.close()
            context.term()
