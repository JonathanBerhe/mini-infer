"""Disaggregated prefill/decode workers (PD disaggregation).

In a "mixed" (non-disaggregated) inference engine, prefill and decode share
the same GPU. They have different resource profiles:

  - Prefill: compute-bound, single forward over many tokens, peak FLOPS.
  - Decode:  memory-bound, many forwards each with 1 token, peak HBM BW.

Mixing them means neither phase runs at optimal occupancy: prefill steals
HBM bandwidth from decode, decode steals SMs from prefill. Production
inference engines (vLLM, SGLang, TensorRT-LLM, the DistServe paper) all
support a "disaggregated" mode where prefill and decode run on separate
GPU workers connected by a KV-cache transfer mechanism.

This module ships mini-infer's disaggregated path:

  - `PrefillWorker`: runs prefill on the prompt, samples the first output
    token, extracts the per-layer KV into a `KVHandoff` and returns it.
  - `DecodeWorker`: accepts a `KVHandoff`, materializes the KV into its
    own paged cache, runs the decode loop, yields tokens.
  - `Orchestrator`: routes requests through prefill -> handoff -> decode.

Workers and orchestrator run in the same Python process today; both
workers may share a `ModelRunner`. The greedy-parity contract against
`ContinuousScheduler` is the correctness gate: disaggregation must not
change output distribution.
"""

from mini_infer.workers.decode_worker import DecodeSession, DecodeWorker
from mini_infer.workers.kv_handoff import KVHandoff
from mini_infer.workers.kv_transfer import recv_handoff, send_handoff
from mini_infer.workers.multi_process import (
    DECODE_RANK,
    PREFILL_RANK,
    pd_two_process_target,
)
from mini_infer.workers.orchestrator import Orchestrator
from mini_infer.workers.pd_scheduler import PDScheduler
from mini_infer.workers.prefill_worker import PrefillWorker

__all__ = [
    "DECODE_RANK",
    "PREFILL_RANK",
    "DecodeSession",
    "DecodeWorker",
    "KVHandoff",
    "Orchestrator",
    "PDScheduler",
    "PrefillWorker",
    "pd_two_process_target",
    "recv_handoff",
    "send_handoff",
]
