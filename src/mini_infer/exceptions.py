class OutOfMemoryError(Exception):
    """Raised when the engine cannot satisfy a memory request (e.g., no free KV blocks)."""
