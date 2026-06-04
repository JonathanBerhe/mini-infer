"""Per-rank entry point for the two-process PD pipeline.

`pd_two_process_target` is the function each worker process runs:

  - Rank 0 (`PREFILL_RANK`): loads a `ModelRunner`, runs
    `PrefillWorker.prefill(request)`, and ships the resulting `KVHandoff`
    to rank 1 via `kv_transfer.send_handoff`. Returns `None`.
  - Rank 1 (`DECODE_RANK`): loads a `ModelRunner`, receives the handoff
    via `kv_transfer.recv_handoff`, hands it to a `DecodeWorker`, and
    returns the decoded token list.

This module deliberately does not contain spawn / process-group
lifecycle code; that's the caller's responsibility (e.g., `mp.spawn`
in production, the `run_multi_process` helper in the test harness).
That separation lets the per-rank function be unit-tested and lets
different launchers (torchrun, mp.spawn, k8s job specs) use the same
worker logic.

Continuous batching within each rank, a long-lived serve loop, and a
request queue are intentionally out of scope here; they belong with the
in-worker scheduler integration that's a follow-up to this slice.
"""

from __future__ import annotations

import logging
import os

import torch

from mini_infer.distributed.group import replica_scope
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.request_state import Request
from mini_infer.workers.decode_worker import DecodeWorker
from mini_infer.workers.kv_transfer import recv_handoff, send_handoff
from mini_infer.workers.prefill_worker import PrefillWorker

logger = logging.getLogger(__name__)

PREFILL_RANK = 0
DECODE_RANK = 1


def _limit_child_threads(num_threads: int = 1) -> None:
    """Cap PyTorch + BLAS thread counts in child processes.

    Two PyTorch processes each defaulting to `# CPUs` intra-op threads
    create heavy thread contention on a shared-memory host. We cap to a
    small value in the child entry point before any model code runs.
    Real multi-host deployments don't hit this (one process per host).
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def pd_two_process_target(
    rank: int,
    world_size: int,
    *,
    model_name: str,
    prompt: str,
    sampling_temperature: float,
    sampling_top_k: int,
    sampling_top_p: float,
    max_tokens: int,
    device: str,
    dtype_str: str,
) -> list[int] | None:
    """Per-rank entry. Module-scope so `mp.spawn` can pickle it.

    Args (besides the launcher-supplied rank/world_size):
      - `model_name`: HF model id; loaded independently per rank.
      - `prompt`: prompt text. Rank 0 tokenizes + prefills it; rank 1 ignores it.
      - `sampling_*`: `SamplingParams` reconstituted on each rank.
      - `max_tokens`: total generation budget.
      - `device` / `dtype_str`: where + at what precision to load. The
        torch dtype is passed as a string (e.g. `"float32"`) so that
        non-CUDA child processes can resolve it independently of CUDA
        context state on the parent.
    """
    if world_size != 2:
        raise ValueError(f"pd_two_process_target requires world_size=2, got {world_size}")
    _limit_child_threads()
    dtype = getattr(torch, dtype_str)

    # PD runs a FULL model on each rank; the two ranks talk only via the
    # explicit `send_handoff` / `recv_handoff` below, never via TP collectives.
    # `replica_scope` makes the model's TP-aware layers see world_size=1 so they
    # build + run with no all-reduce, which would otherwise deadlock against the
    # handoff (rank 0's prefill all-reduce waits for rank 1, which is in recv).
    # The handoff uses `dist.send`/`dist.recv` directly, unaffected by the scope.
    with replica_scope():
        runner = ModelRunner.from_pretrained(model_name, device=device, dtype=dtype)
        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            top_k=sampling_top_k,
            top_p=sampling_top_p,
        )

        if rank == PREFILL_RANK:
            prefill_worker = PrefillWorker(runner)
            request = Request(prompt=prompt, sampling_params=sampling_params, max_tokens=max_tokens)
            send_handoff(prefill_worker.prefill(request), dst_rank=DECODE_RANK)
            return None

        if rank == DECODE_RANK:
            decode_worker = DecodeWorker(runner)
            handoff = recv_handoff(src_rank=PREFILL_RANK, pool=runner.block_pool)
            return list(decode_worker.decode(handoff))

    raise ValueError(f"unexpected rank {rank}; expected 0 or 1")
