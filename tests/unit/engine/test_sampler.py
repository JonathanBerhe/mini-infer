import math

import pytest
import torch

from mini_infer.engine.sampler import (
    SamplingParams,
    _apply_top_k,
    _apply_top_k_rows,
    _apply_top_p,
    _apply_top_p_rows,
    sample,
    sample_batch,
    sample_tokens,
)


def test_greedy_picks_argmax() -> None:
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    assert sample(logits, SamplingParams(temperature=0.0)) == 1


def test_top_k_one_is_greedy_even_with_high_temperature() -> None:
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    torch.manual_seed(0)
    assert sample(logits, SamplingParams(temperature=10.0, top_k=1)) == 1


def test_seeded_sampling_is_reproducible() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.manual_seed(42)
    a = sample(logits, SamplingParams(temperature=1.0))
    torch.manual_seed(42)
    b = sample(logits, SamplingParams(temperature=1.0))
    assert a == b


def test_top_p_small_falls_back_to_top_token() -> None:
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    torch.manual_seed(0)
    # very small top_p means only the highest-prob token survives
    assert sample(logits, SamplingParams(temperature=1.0, top_p=0.001)) == 1


def test_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-0.1)


def test_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=-1)


def test_invalid_top_p_above_one_raises() -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=1.5)


def test_invalid_top_p_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=-0.1)


# ---- sample_batch: one host copy for the whole batch ----


def test_batch_greedy_matches_sample_on_each_row() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 50)
    logits[3, [4, 9]] = 10.0  # a tie: both pick the first index
    params = [SamplingParams(temperature=0.0)] * 8
    expected = [sample(row, SamplingParams(temperature=0.0)) for row in logits]
    assert sample_batch(logits, params) == expected
    assert expected[3] == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch_greedy_rows_ignore_the_sampled_rows(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, 50).to(dtype)
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0),
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.7, top_k=5, top_p=0.9),
        SamplingParams(temperature=0.0, top_k=3),  # top-k without temperature is greedy
        SamplingParams(temperature=2.0, top_p=0.5),
    ]
    tokens = sample_batch(logits, params)
    for row in (0, 2, 4):
        assert tokens[row] == sample(logits[row], params[row])


def test_batch_top_k_one_and_tiny_top_p_are_greedy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    params = [
        SamplingParams(temperature=10.0, top_k=1),
        SamplingParams(temperature=1.0, top_p=0.001),
        SamplingParams(temperature=10.0, top_k=1, top_p=0.5),
        SamplingParams(temperature=0.0),
    ]
    assert sample_batch(logits, params) == logits.argmax(dim=-1).tolist()


def test_batch_top_k_filter_matches_the_scalar_filter() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 40)
    top_k = [0, 1, 3, 7, 39]
    filtered = _apply_top_k_rows(logits, torch.tensor(top_k), max(top_k))
    for row, k in enumerate(top_k):
        expected = logits[row] if k == 0 else _apply_top_k(logits[row], k)
        assert torch.equal(filtered[row], expected)


def test_batch_top_p_filter_matches_the_scalar_filter() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 40) * 3
    top_p = [1.0, 0.95, 0.5, 0.1, 0.0]
    filtered = _apply_top_p_rows(logits, torch.tensor(top_p))
    for row, p in enumerate(top_p):
        expected = logits[row] if p == 1.0 else _apply_top_p(logits[row], p)
        assert torch.equal(filtered[row], expected)


def test_batch_draws_stay_in_each_row_top_k() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 50).repeat(200, 1)  # rows alternate between two distributions
    params = [SamplingParams(temperature=1.0, top_k=3 if i % 2 else 5) for i in range(400)]
    tokens = sample_batch(logits, params)
    for parity, k in ((0, 5), (1, 3)):
        allowed = set(torch.topk(logits[parity], k).indices.tolist())
        assert set(tokens[parity::2]) == allowed


def test_batch_draws_follow_the_softmax() -> None:
    """Index 1 has three times the probability of index 0."""
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, math.log(3.0)]]).repeat(20000, 1)
    tokens = sample_batch(logits, [SamplingParams(temperature=1.0)] * 20000)
    assert sum(tokens) / len(tokens) == pytest.approx(0.75, abs=0.01)


def test_batch_seeded_sampling_is_reproducible() -> None:
    logits = torch.randn(16, 50, generator=torch.Generator().manual_seed(1))
    params = [SamplingParams(temperature=1.0, top_p=0.9)] * 16
    torch.manual_seed(42)
    first = sample_batch(logits, params)
    torch.manual_seed(42)
    assert sample_batch(logits, params) == first


def test_batch_needs_one_params_per_row() -> None:
    with pytest.raises(ValueError, match="one row per params"):
        sample_batch(torch.zeros(3, 10), [SamplingParams()] * 2)
    with pytest.raises(ValueError, match="one row per params"):
        sample_batch(torch.zeros(10), [SamplingParams()])


@pytest.mark.requires_cuda
def test_sample_tokens_does_not_sync_with_the_host() -> None:
    torch.manual_seed(0)
    logits = torch.randn(16, 129280, device="cuda", dtype=torch.bfloat16)
    params = [SamplingParams(temperature=0.0)] * 8 + [
        SamplingParams(temperature=0.8, top_k=50, top_p=0.95)
    ] * 8
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        tokens = sample_tokens(logits, params)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert tokens[:8].tolist() == logits[:8].argmax(dim=-1).tolist()
