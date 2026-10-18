class OutOfMemoryError(Exception):
    """Raised when the engine cannot satisfy a memory request (e.g., no free KV blocks)."""


class EngineNotRunningError(RuntimeError):
    """Raised when a request needs an engine thread that is not running.

    `submit()` raises it before the scheduler is started or after its engine
    thread is gone. A `RequestHandle` raises it when its queue is empty and the
    thread that owned the request died on an exception (a clean `stop()`
    finishes the requests in flight as 'cancelled' instead). No further step
    will arrive, so a caller that kept waiting would wait forever.

    It subclasses `RuntimeError` because every scheduler except `PDScheduler`
    raised a plain `RuntimeError` from `submit()` for the same condition
    before this type existed.
    """
