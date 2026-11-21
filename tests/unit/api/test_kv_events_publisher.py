"""The ZMQ KV-event publisher and its vLLM wire format.

The expected layouts come from the llm-d router's vLLM decoder
(`pkg/kvevents/engineadapter/vllm_msgpack.go`): a batch is
`[timestamp, events]`, and each event is an array with its tag first.
"""

import time

import msgpack
import pytest
import zmq

from mini_infer.api.kv_events_publisher import (
    ZmqKVEventPublisher,
    binds,
    encode_event,
    encode_event_batch,
)
from mini_infer.cache.kv_events import AllBlocksCleared, BlockRemoved, BlockStored

H1 = bytes(range(16))
H2 = bytes(range(16, 32))
STORED = BlockStored(block_hashes=(H2,), parent_block_hash=H1, token_ids=(5, 6, 7, 8), block_size=4)


def test_block_stored_uses_vllm_field_order() -> None:
    assert encode_event(STORED) == ["BlockStored", [H2], H1, [5, 6, 7, 8], 4, None, None]


def test_first_block_has_a_nil_parent() -> None:
    root = BlockStored(block_hashes=(H1,), parent_block_hash=None, token_ids=(1,), block_size=1)
    assert encode_event(root)[2] is None


def test_block_removed_and_all_cleared_layouts() -> None:
    assert encode_event(BlockRemoved(block_hashes=(H1, H2))) == ["BlockRemoved", [H1, H2], None]
    assert encode_event(AllBlocksCleared()) == ["AllBlocksCleared"]


def test_batch_payload_is_msgpack_with_binary_hashes() -> None:
    payload = encode_event_batch(12.5, [STORED, AllBlocksCleared()])
    decoded = msgpack.unpackb(payload, raw=False)
    assert decoded == [
        12.5,
        [["BlockStored", [H2], H1, [5, 6, 7, 8], 4, None, None], ["AllBlocksCleared"]],
    ]
    # Hashes must travel as msgpack bin (bytes), not str: the router reads bin or int.
    assert isinstance(decoded[1][0][1][0], bytes)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("tcp://*:5557", True),
        ("tcp://[::]:5557", True),
        ("ipc:///tmp/kv", True),
        ("inproc://kv", True),
        ("tcp://10.0.0.7:5557", False),
        ("tcp://epp.default.svc:5557", False),
    ],
)
def test_bind_or_connect_follows_vllm(endpoint: str, expected: bool) -> None:
    assert binds(endpoint) is expected


def _receive(sub: zmq.Socket[bytes], timeout_s: float) -> list[bytes] | None:
    if sub.poll(int(timeout_s * 1000)) == 0:
        return None
    return sub.recv_multipart()


def test_publisher_sends_topic_sequence_and_payload_frames() -> None:
    """End to end over TCP: a SUB socket binds, the publisher connects to it."""
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"kv@")
    port = sub.bind_to_random_port("tcp://127.0.0.1")
    publisher = ZmqKVEventPublisher(f"tcp://127.0.0.1:{port}", "kv@pod-a@my-model")
    try:
        # PUB drops messages until the subscription reaches it, so repeat a
        # probe until one arrives. Each probe is one batch with its own number.
        frames = None
        deadline = time.monotonic() + 10
        while frames is None and time.monotonic() < deadline:
            publisher(AllBlocksCleared())
            frames = _receive(sub, timeout_s=0.1)
        assert frames is not None, "no KV-event batch arrived"
        last_seq = int.from_bytes(frames[1], "big")

        publisher(STORED)
        frames = _receive(sub, timeout_s=5)
        while frames is not None and msgpack.unpackb(frames[2])[1] == [["AllBlocksCleared"]]:
            last_seq = int.from_bytes(frames[1], "big")
            frames = _receive(sub, timeout_s=5)
        assert frames is not None

        topic, seq, payload = frames
        assert topic == b"kv@pod-a@my-model"
        assert len(seq) == 8
        assert int.from_bytes(seq, "big") == last_seq + 1
        timestamp, events = msgpack.unpackb(payload, raw=False)
        assert isinstance(timestamp, float)
        assert events == [encode_event(STORED)]
    finally:
        publisher.close()
        sub.close(linger=0)
        context.term()


def test_publisher_resolves_a_wildcard_port() -> None:
    publisher = ZmqKVEventPublisher("tcp://127.0.0.1:*", "kv@m")
    try:
        assert publisher.endpoint.startswith("tcp://127.0.0.1:")
        assert not publisher.endpoint.endswith("*")
    finally:
        publisher.close()


def test_publisher_drops_events_after_close() -> None:
    publisher = ZmqKVEventPublisher("inproc://closed-publisher", "kv@m")
    publisher.close()
    publisher(STORED)  # must not raise or queue
    publisher.close()  # idempotent


def test_a_bad_endpoint_fails_at_startup() -> None:
    with pytest.raises(RuntimeError, match="cannot open the KV-event socket"):
        ZmqKVEventPublisher("not-an-endpoint", "kv@m")
