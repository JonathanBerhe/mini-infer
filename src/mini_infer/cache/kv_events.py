"""KV-cache events: what the prefix cache reports when it stores or drops a block.

A KV-cache-aware router (for example the llm-d router) keeps an index of the
blocks that each server holds, and sends a request to the server that already
has the longest part of its prompt. The router builds that index from these
events. The event set and its fields follow vLLM's KV-cache events, so a router
that reads vLLM events can read ours. `mini_infer.api.kv_events_publisher`
encodes them in vLLM's wire format.

The prefix cache emits the events from the engine thread, through a
`KVEventListener` callback. A listener must return quickly and must not raise.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class BlockStored:
    """New full blocks entered the prefix cache.

    `block_hashes[i]` is the chained hash of block `i`, and `token_ids` holds the
    tokens of all the blocks in order (`block_size` tokens per block).
    `parent_block_hash` is the hash of the block before the first one, or None
    when the first block starts the sequence. A router recomputes its own keys
    from `token_ids`, and it finds the prefix of those tokens through the parent.
    """

    block_hashes: tuple[bytes, ...]
    parent_block_hash: bytes | None
    token_ids: tuple[int, ...]
    block_size: int


@dataclasses.dataclass(frozen=True)
class BlockRemoved:
    """Blocks left the prefix cache (LRU eviction)."""

    block_hashes: tuple[bytes, ...]


@dataclasses.dataclass(frozen=True)
class AllBlocksCleared:
    """The prefix cache is empty. A publisher sends it at startup."""


KVEvent = BlockStored | BlockRemoved | AllBlocksCleared
KVEventListener = Callable[[KVEvent], None]
