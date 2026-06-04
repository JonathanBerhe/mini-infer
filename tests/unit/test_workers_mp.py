"""Multi-process end-to-end test for the disaggregated PD pipeline.

This is the full integration shape: spawns two child processes (rank 0 =
prefill, rank 1 = decode) under a gloo `torch.distributed` group, each
loads its own `ModelRunner`, runs a single request through the pipeline
with a `dist.send/recv` KV transfer in between, and asserts the output
matches the single-process `Orchestrator` token-for-token.

Marked `slow` + `requires_model` (downloads Qwen2.5-0.5B, ~1 GB), so it
runs locally / on demand, not in the fast CI lane. The lightweight
wire-protocol coverage lives in `test_kv_transfer_mp.py` (proves
`send_handoff` / `recv_handoff` round-trip across processes in ~4 s
using synthetic KV tensors).

Previously skipped: a deadlock
------------------------------

This test was skipped on the belief that the two-process forward
"stalls on CPU / under pytest" but "runs cleanly on multi-GPU CUDA."
That was a misdiagnosis. The real cause was a topology conflation: each
PD rank built its model under the 2-rank process group, so the model's
TP-aware layers inserted `all_reduce` collectives. Rank 0's prefill
blocked on the very first collective (the token embedding) waiting for
rank 1, which was parked in `recv_handoff`: a hard deadlock, on CPU
*and* CUDA alike (the 2x H100 smoke hit the same hang). The fix is
`replica_scope()` in `pd_two_process_target`: each rank builds + runs a
full replica (world_size=1, no collectives) and the ranks communicate
only via the explicit `send_handoff` / `recv_handoff`. With that in
place this test runs end-to-end, so it's no longer skipped.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.request_state import Request
from mini_infer.workers import DECODE_RANK, DecodeWorker, Orchestrator, PrefillWorker
from mini_infer.workers.multi_process import pd_two_process_target
from tests.unit._distributed_test_utils import (
    is_multi_process_available,
    run_multi_process,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _single_process_pd_baseline(prompt: str, max_tokens: int) -> list[int]:
    """Run the request through the in-process Orchestrator; return token ids."""
    runner = ModelRunner.from_pretrained(MODEL_NAME, device="cpu", dtype=torch.float32)
    orchestrator = Orchestrator(
        prefill_worker=PrefillWorker(runner),
        decode_worker=DecodeWorker(runner),
    )
    return orchestrator.run(
        Request(
            prompt=prompt,
            sampling_params=SamplingParams(),
            max_tokens=max_tokens,
        )
    )


@pytest.mark.requires_model
@pytest.mark.slow
def test_pd_two_process_matches_single_process() -> None:
    """Multi-process PD output equals single-process Orchestrator output."""
    if not is_multi_process_available():
        pytest.skip("multi-process / gloo unavailable in this environment")

    prompt = "The capital of France is"
    max_tokens = 8

    baseline = _single_process_pd_baseline(prompt, max_tokens)

    results = run_multi_process(
        2,
        pd_two_process_target,
        model_name=MODEL_NAME,
        prompt=prompt,
        sampling_temperature=0.0,
        sampling_top_k=0,
        sampling_top_p=1.0,
        max_tokens=max_tokens,
        device="cpu",
        dtype_str="float32",
        timeout_sec=240.0,
    )
    decode_tokens = results[DECODE_RANK]
    assert decode_tokens is not None
    assert decode_tokens == baseline, (
        f"multi-process PD diverged from single-process PD:\n"
        f"  multi-process : {decode_tokens}\n"
        f"  single-process: {baseline}"
    )
