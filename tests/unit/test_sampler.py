import pytest
import torch

from mini_infer.engine.sampler import SamplingParams, sample


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
