"""Single-process orchestrator for the disaggregated PD pipeline.

The orchestrator is the request entry point. It owns a `PrefillWorker`
and a `DecodeWorker`, and per request:

  1. Hands the request to the prefill worker.
  2. Receives a `KVHandoff`.
  3. Hands the handoff to the decode worker.
  4. Streams the decoded tokens back to the caller (as an iterator) or
     accumulates them (as a full-text return).

Both workers may share a `ModelRunner` (caller's choice). The
orchestrator does not own model loading — callers build the workers,
the orchestrator wires them. This mirrors `SpeculativeRunner`'s
constructor pattern and keeps the orchestrator unit-testable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from mini_infer.scheduler.request_state import Request
from mini_infer.workers.decode_worker import DecodeWorker
from mini_infer.workers.prefill_worker import PrefillWorker

logger = logging.getLogger(__name__)


class Orchestrator:
    """Routes a request through prefill -> handoff -> decode and streams tokens."""

    def __init__(
        self,
        *,
        prefill_worker: PrefillWorker,
        decode_worker: DecodeWorker,
    ) -> None:
        self._prefill_worker = prefill_worker
        self._decode_worker = decode_worker

    def run_stream(self, request: Request) -> Iterator[int]:
        """Yield decoded token ids one at a time.

        This is the streaming entrypoint. The first yielded value is the
        token sampled from the last prefill logit
        (= `handoff.first_sampled_token_id`), which is what
        `ContinuousScheduler` emits first too — so streaming parity is
        preserved.
        """
        handoff = self._prefill_worker.prefill(request)
        yield from self._decode_worker.decode(handoff)

    def run(self, request: Request) -> list[int]:
        """Run the request to completion and return the full output token list.

        Convenience for tests and batch (non-streaming) callers.
        """
        return list(self.run_stream(request))
