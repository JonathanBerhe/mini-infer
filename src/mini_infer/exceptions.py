class OutOfMemoryError(Exception):
    """Raised when the engine cannot satisfy a memory request (e.g., no free KV blocks)."""


class EngineNotRunningError(RuntimeError):
    """Raised when a request needs an engine thread that is not running.

    `submit()` raises it before `start()` and after the thread is gone. A handle
    raises it when its queue is empty and its thread died, so the caller does not
    wait forever. A clean `stop()` ends requests as 'cancelled' instead. See ADR-028.
    """


class RequestFailedError(RuntimeError):
    """Raised by a `RequestHandle` when the engine failed this one request.

    The engine logged the cause and keeps serving other requests. The cause can
    hold internal state (paths, shapes, devices), so the client gets a fixed message.
    """
