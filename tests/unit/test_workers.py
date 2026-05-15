"""Slice 1 tests for the disaggregated PD pipeline.

Two layers:

  1. Fast unit tests (no model load): KVHandoff shape + DecodeWorker
     argument validation. Run in CI.
  2. Real-model parity (`@pytest.mark.requires_model`): the headline
     test for Slice 1. PD's greedy output for a small prompt is
     token-for-token equal to `ContinuousScheduler`'s greedy output on
     the same prompt + same model. Skipped in CI; opt in locally with
     `uv run pytest tests/unit/test_workers.py -v`.

The parity test is the contract: PD is allowed to add latency / shift
work between workers, but greedy output must be identical to the
non-disaggregated path. If this test ever drifts, the bug is in PD
(handoff materialization, cache wiring, or sampling ordering), not in
the model.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
from mini_infer.scheduler.request_state import Request
from mini_infer.workers import (
    DecodeWorker,
    KVHandoff,
    Orchestrator,
    PrefillWorker,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_runner() -> ModelRunner:
    """Single module-scoped runner; cheap to load and shared across tests."""
    return ModelRunner.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# Fast unit tests (no model load)
# ---------------------------------------------------------------------------


def test_kvhandoff_basic_shape() -> None:
    """`KVHandoff.num_layers` / `stream_names_for_layer` match the construction."""
    handoff = KVHandoff(
        request_id="x",
        kv_streams_per_layer=[
            {"k": torch.zeros(4, 2, 8), "v": torch.zeros(4, 2, 8)},
            {"k": torch.zeros(4, 2, 8), "v": torch.zeros(4, 2, 8)},
        ],
        prefill_len=4,
        first_sampled_token_id=42,
        sampling_params=SamplingParams(),
        max_tokens=8,
        eos_token_id=0,
    )
    assert handoff.num_layers == 2
    assert handoff.stream_names_for_layer(0) == ["k", "v"]
    assert handoff.stream_names_for_layer(1) == ["k", "v"]


def test_kvhandoff_carries_eos_optional() -> None:
    handoff = KVHandoff(
        request_id="x",
        kv_streams_per_layer=[{"k": torch.zeros(1, 1, 1), "v": torch.zeros(1, 1, 1)}],
        prefill_len=1,
        first_sampled_token_id=7,
        sampling_params=SamplingParams(),
        max_tokens=4,
    )
    assert handoff.eos_token_id is None


def test_kv_transfer_sampling_params_round_trip() -> None:
    """The fixed-point header encoding round-trips SamplingParams values.

    The wire format encodes temperature and top_p as int64 micros
    (multiplied by 1e6). This test exercises the encode/decode pair
    without spinning up a process group.
    """
    from mini_infer.workers.kv_transfer import (
        _decode_sampling_params,
        _encode_sampling_params,
    )

    cases = [
        SamplingParams(temperature=0.0, top_k=0, top_p=1.0),  # greedy
        SamplingParams(temperature=0.7, top_k=50, top_p=0.95),
        SamplingParams(temperature=1.234567, top_k=0, top_p=0.9),
    ]
    for original in cases:
        t, k, p = _encode_sampling_params(original)
        restored = _decode_sampling_params(t, k, p)
        # Fixed-point encoding loses precision past 1e-6; assert within that.
        assert abs(restored.temperature - original.temperature) <= 1e-6
        assert restored.top_k == original.top_k
        assert abs(restored.top_p - original.top_p) <= 1e-6


# ---------------------------------------------------------------------------
# Real-model parity vs ContinuousScheduler (the headline Slice-1 contract)
# ---------------------------------------------------------------------------


def _continuous_scheduler_greedy_tokens(
    runner: ModelRunner, prompt: str, max_tokens: int
) -> list[int]:
    """Drive `ContinuousScheduler` for a single greedy request; return token ids."""
    scheduler = ContinuousScheduler(runner)
    scheduler.start()
    try:
        result = scheduler.run(
            Request(
                prompt=prompt,
                sampling_params=SamplingParams(),  # temperature=0 -> greedy
                max_tokens=max_tokens,
            )
        )
        return list(result.tokens)
    finally:
        scheduler.stop()


@pytest.mark.requires_model
def test_pd_greedy_matches_continuous_scheduler(qwen_runner: ModelRunner) -> None:
    """PD's greedy output is token-for-token identical to the non-disaggregated path.

    This is the Slice-1 contract: disaggregation must not change output
    distribution. Greedy is the cleanest knob to assert this on.

    Both workers share the same `ModelRunner` (same model state, same pool).
    Each call creates a fresh `PagedKVCache(pool)`; the pool sees alloc/free
    events but no two callers overlap, so this is safe within a single
    process.
    """
    prompt = "The capital of France is"
    max_tokens = 12

    # Run ContinuousScheduler first so its `_batched_cache` is fully freed
    # by `stop()` before PD's workers allocate against the same pool.
    baseline = _continuous_scheduler_greedy_tokens(qwen_runner, prompt, max_tokens)

    prefill_worker = PrefillWorker(qwen_runner)
    decode_worker = DecodeWorker(qwen_runner)
    orchestrator = Orchestrator(
        prefill_worker=prefill_worker,
        decode_worker=decode_worker,
    )
    pd_tokens = orchestrator.run(
        Request(
            prompt=prompt,
            sampling_params=SamplingParams(),
            max_tokens=max_tokens,
        )
    )

    assert pd_tokens == baseline, (
        f"PD greedy diverged from ContinuousScheduler greedy:\n"
        f"  PD       : {pd_tokens}\n"
        f"  baseline : {baseline}"
    )


@pytest.mark.requires_model
def test_pd_first_yielded_token_matches_handoff(qwen_runner: ModelRunner) -> None:
    """The first token the orchestrator yields equals the prefill worker's first sample.

    Spec check on the streaming path: in PD, the first emitted token is
    determined entirely by prefill (prefill samples from the last-prefill
    logit). The decode worker yields that token verbatim, then runs decode
    starting from step 2.
    """
    prefill_worker = PrefillWorker(qwen_runner)
    decode_worker = DecodeWorker(qwen_runner)
    orchestrator = Orchestrator(prefill_worker=prefill_worker, decode_worker=decode_worker)

    # Call prefill directly to capture the handoff, then drive the decode
    # worker so we can inspect both sides without re-running prefill.
    request = Request(
        prompt="Once upon a time",
        sampling_params=SamplingParams(),
        max_tokens=4,
    )
    handoff = prefill_worker.prefill(request)
    decoded = list(decode_worker.decode(handoff))
    assert decoded, "decoder yielded no tokens"
    assert decoded[0] == handoff.first_sampled_token_id

    # And the full orchestrator path produces the same first token.
    full = orchestrator.run(request)
    assert full[0] == handoff.first_sampled_token_id


@pytest.mark.requires_model
def test_pd_releases_blocks_on_completion(qwen_runner: ModelRunner) -> None:
    """Block pool is fully free before and after a PD run.

    Catches leaked blocks from a missing `remove_request` (the most likely
    bug surface): if either worker leaks a slot, repeated runs would
    starve the pool. We assert directly on `num_free_blocks` since the
    pool is shared between workers in this single-process slice.
    """
    pool = qwen_runner.block_pool
    free_before = pool.num_free_blocks

    orchestrator = Orchestrator(
        prefill_worker=PrefillWorker(qwen_runner),
        decode_worker=DecodeWorker(qwen_runner),
    )
    _ = orchestrator.run(
        Request(
            prompt="The capital of France is",
            sampling_params=SamplingParams(),
            max_tokens=6,
        )
    )
    free_after = pool.num_free_blocks
    assert free_after == free_before, (
        f"PD leaked blocks: {free_before} free before, {free_after} after"
    )
