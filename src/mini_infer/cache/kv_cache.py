from transformers import DynamicCache


class KVCache(DynamicCache):  # type: ignore[misc]
    """Per-request KV cache; Phase 2 PagedAttention will replace internals."""
