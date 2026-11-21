"""A snapshot of a scheduler's load, for the `/metrics` endpoint."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class SchedulerLoad:
    """Queue lengths and KV-cache use at one moment.

    `kv_cache_usage` is the fraction of KV blocks in use, from 0.0 to 1.0. Cached
    blocks that no request holds count as free, because the pool can take them
    back at once. vLLM counts them the same way.
    """

    num_waiting: int
    num_running: int
    kv_cache_usage: float
    block_size: int
    num_blocks: int
