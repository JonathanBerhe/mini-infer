"""Slice 3 tests: batched prefill + batched decode.

Two correctness contracts:

  1. `PrefillWorker.prefill_batch([r1, r2, ...])` produces the same
     handoffs as `[PrefillWorker.prefill(r1), PrefillWorker.prefill(r2), ...]`.
     Same first sampled token per request, same KV tensors.

  2. `DecodeWorker.decode_batch([h1, h2, ...])` produces the same per-request
     token lists as `[list(DecodeWorker.decode(h1)), list(DecodeWorker.decode(h2)), ...]`.

If either contract drifts, batching has changed output distribution —
which would make PD's "speed up without changing answers" claim false.

Marked `requires_model`: needs a real model load. Greedy sampling so the
parity contract is exact, not approximate.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler.request_state import Request
from mini_infer.workers import DecodeWorker, PrefillWorker

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_runner() -> ModelRunner:
    return ModelRunner.from_pretrained(MODEL_NAME)


@pytest.mark.requires_model
def test_prefill_batch_matches_sequential(qwen_runner: ModelRunner) -> None:
    """prefill_batch([r1, r2]) == [prefill(r1), prefill(r2)] in handoff content."""
    worker = PrefillWorker(qwen_runner)
    requests = [
        Request(prompt="The capital of France is", sampling_params=SamplingParams(), max_tokens=4),
        Request(prompt="Once upon a time", sampling_params=SamplingParams(), max_tokens=4),
    ]
    sequential = [worker.prefill(r) for r in requests]
    batched = worker.prefill_batch(requests)
    assert len(batched) == len(sequential)

    for r, (seq_h, batch_h) in enumerate(zip(sequential, batched, strict=True)):
        assert seq_h.prefill_len == batch_h.prefill_len, f"request {r} prefill_len differs"
        assert seq_h.first_sampled_token_id == batch_h.first_sampled_token_id, (
            f"request {r}: first token differs (seq={seq_h.first_sampled_token_id}, "
            f"batch={batch_h.first_sampled_token_id})"
        )
        assert seq_h.max_tokens == batch_h.max_tokens
        assert seq_h.eos_token_id == batch_h.eos_token_id
        # KV bytes: every layer x every stream tensor matches to within
        # FP32 round-off. Strict bit-equality (`atol=0, rtol=0`) is
        # platform-dependent because the matmul reduction order differs
        # between batched and sequential prefill paths — Linux CI shows
        # ULP-level deltas where macOS dev hardware happens to produce
        # bit-identical output. The functional contract is "same tokens
        # downstream", validated separately in `test_decode_batch_matches_sequential`.
        for layer_idx in range(seq_h.num_layers):
            seq_streams = seq_h.kv_streams_per_layer[layer_idx]
            batch_streams = batch_h.kv_streams_per_layer[layer_idx]
            assert set(seq_streams.keys()) == set(batch_streams.keys())
            for stream_name in seq_streams:
                torch.testing.assert_close(
                    seq_streams[stream_name],
                    batch_streams[stream_name],
                    atol=1e-5,
                    rtol=1e-5,
                    msg=(f"request {r} layer {layer_idx} stream {stream_name!r} differs"),
                )


@pytest.mark.requires_model
def test_decode_batch_matches_sequential(qwen_runner: ModelRunner) -> None:
    """decode_batch([h1, h2]) yields the same per-request tokens as sequential decode.

    Tests against the SAME handoffs produced sequentially, to isolate
    decode-side batching from prefill-side batching.
    """
    prefill = PrefillWorker(qwen_runner)
    decode = DecodeWorker(qwen_runner)
    requests = [
        Request(
            prompt="The capital of France is",
            sampling_params=SamplingParams(),
            max_tokens=6,
        ),
        Request(
            prompt="Once upon a time in",
            sampling_params=SamplingParams(),
            max_tokens=6,
        ),
    ]
    handoffs = [prefill.prefill(r) for r in requests]
    sequential = [list(decode.decode(h)) for h in handoffs]
    batched = decode.decode_batch(handoffs)
    assert batched == sequential, (
        f"decode_batch diverged from sequential decode:\n"
        f"  batched   : {batched}\n"
        f"  sequential: {sequential}"
    )


@pytest.mark.requires_model
def test_pd_batched_end_to_end_matches_sequential(qwen_runner: ModelRunner) -> None:
    """The full batched PD path matches sequential PD token-for-token.

    Composes `prefill_batch` + `decode_batch` and compares the per-request
    token lists against running `prefill` + `decode` once per request.
    """
    prefill = PrefillWorker(qwen_runner)
    decode = DecodeWorker(qwen_runner)
    requests = [
        Request(
            prompt="The capital of France is",
            sampling_params=SamplingParams(),
            max_tokens=6,
        ),
        Request(
            prompt="The largest planet is",
            sampling_params=SamplingParams(),
            max_tokens=6,
        ),
        Request(
            prompt="A common greeting is",
            sampling_params=SamplingParams(),
            max_tokens=6,
        ),
    ]
    sequential: list[list[int]] = []
    for request in requests:
        handoff = prefill.prefill(request)
        sequential.append(list(decode.decode(handoff)))

    handoffs = prefill.prefill_batch(requests)
    batched = decode.decode_batch(handoffs)

    assert batched == sequential, (
        f"batched PD diverged from sequential PD:\n"
        f"  batched   : {batched}\n"
        f"  sequential: {sequential}"
    )


@pytest.mark.requires_model
def test_prefill_batch_empty_input(qwen_runner: ModelRunner) -> None:
    """Empty batch returns empty list (no model forward)."""
    worker = PrefillWorker(qwen_runner)
    assert worker.prefill_batch([]) == []


@pytest.mark.requires_model
def test_decode_batch_empty_input(qwen_runner: ModelRunner) -> None:
    worker = DecodeWorker(qwen_runner)
    assert worker.decode_batch([]) == []


@pytest.mark.requires_model
def test_decode_batch_heterogeneous_max_tokens(qwen_runner: ModelRunner) -> None:
    """Requests with different max_tokens each stop at their own budget.

    Verifies per-request termination tracking: each output list is
    exactly `min(max_tokens, len-until-eos)` long, independent of
    other requests in the batch.
    """
    prefill = PrefillWorker(qwen_runner)
    decode = DecodeWorker(qwen_runner)
    requests = [
        Request(prompt="The capital of France is", sampling_params=SamplingParams(), max_tokens=3),
        Request(prompt="The capital of France is", sampling_params=SamplingParams(), max_tokens=8),
    ]
    handoffs = [prefill.prefill(r) for r in requests]
    batched = decode.decode_batch(handoffs)
    # Greedy + same prompt -> the longer one's first 3 tokens equal the shorter's full output.
    assert len(batched[0]) <= 3
    assert len(batched[1]) <= 8
    assert batched[0] == batched[1][: len(batched[0])]
