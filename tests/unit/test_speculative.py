"""Unit tests for the speculative decoder.

A few tests use a mock `ModelRunner` (no model loading) to exercise the
accept-reject loop and stats bookkeeping in isolation. The real-model
parity tests live in `tests/stress/test_speculative_load.py` because they
need a model load.
"""

from typing import Any
from unittest.mock import patch

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.speculative import SpecStats, SpeculativeRunner


class _FakeTokenizer:
    """Tokenizer stub: identity encoder, no EOS hit unless we ask."""

    def __init__(self, vocab_size: int = 1000, eos_token_id: int = 0) -> None:
        self._vocab_size = vocab_size
        self._eos = eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos

    def encode(self, text: str) -> list[int]:
        # Each character -> a deterministic token in [1, vocab).
        return [(ord(c) % (self._vocab_size - 1)) + 1 for c in text]


def _make_fake_runner(vocab_size: int = 1000, eos_token_id: int = 0) -> Any:
    """Build a ModelRunner-like object whose tokenizer satisfies the spec runner contract.

    We don't need the actual model machinery for the init/contract tests;
    we only need `tokenizer.vocab_size`. Methods that the runner would call
    are mocked at use-time per test.
    """
    runner = ModelRunner.__new__(ModelRunner)
    runner._tokenizer = _FakeTokenizer(vocab_size=vocab_size, eos_token_id=eos_token_id)  # type: ignore[attr-defined]
    return runner


def test_spec_runner_rejects_mismatched_vocab() -> None:
    """Building with two runners that have different vocab sizes raises clearly."""
    target = _make_fake_runner(vocab_size=1000)
    draft = _make_fake_runner(vocab_size=999)
    with pytest.raises(ValueError, match="vocab mismatch"):
        SpeculativeRunner(target, draft)


def test_spec_runner_rejects_k_below_one() -> None:
    target = _make_fake_runner()
    draft = _make_fake_runner()
    with pytest.raises(ValueError, match="K must be >= 1"):
        SpeculativeRunner(target, draft, K=0)


def test_spec_stats_mean_acceptance_handles_zero_iterations() -> None:
    """`mean_acceptance_per_iter` must be 0.0 (not div-by-zero) on an empty run."""
    stats = SpecStats()
    assert stats.mean_acceptance_per_iter == 0.0


def test_spec_stats_mean_acceptance_basic() -> None:
    stats = SpecStats(n_iterations=4, n_accepted_total=10)
    assert stats.mean_acceptance_per_iter == pytest.approx(2.5)


@pytest.mark.requires_model
def test_same_model_target_and_draft_gives_full_acceptance_and_matches_baseline() -> None:
    """Target == draft -> every draft token is accepted; output matches target-alone greedy.

    The same-model trick collapses the math: the draft and target argmax are
    bit-equal at every position, so the accept loop accepts all K candidates
    on every iteration. This validates the draft loop, verify pack, and cache
    truncation without needing a real big-target/small-draft pair.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    target = ModelRunner.from_pretrained(model_name)
    draft = ModelRunner.from_pretrained(model_name)
    spec = SpeculativeRunner(target, draft, K=3)
    prompt = "The capital of France is"
    max_tokens = 6

    spec_tokens, stats = spec.run_greedy(prompt, max_tokens=max_tokens)

    # Acceptance should equal K on every iteration that ran a full verify.
    assert stats.n_iterations >= 1, "expected at least one verify iteration"
    assert stats.mean_acceptance_per_iter == pytest.approx(spec.K), (
        f"mean acceptance {stats.mean_acceptance_per_iter} != K={spec.K} "
        "(same-model trick should produce 100%)"
    )

    # Compare with target-alone greedy via target's own forward path.
    baseline_tokens = _greedy_tokens_target_alone(target, prompt, max_tokens)
    assert spec_tokens == baseline_tokens, (
        f"spec output {spec_tokens} differs from target-alone greedy {baseline_tokens}"
    )


@pytest.mark.requires_model
def test_spec_decode_synthetic_divergent_draft_emits_correct_tokens() -> None:
    """Force the draft to disagree with the target on every position; spec output
    must still match target-alone greedy (the bonus mechanism corrects each step).
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    target = ModelRunner.from_pretrained(model_name)
    draft = ModelRunner.from_pretrained(model_name)
    spec = SpeculativeRunner(target, draft, K=3)
    prompt = "The capital of France is"
    max_tokens = 5

    # Wrap draft.forward_step so every returned logits row argmaxes to a
    # token we know the target won't pick (token id 0; usually a non-natural
    # character / EOS-ish, but we won't actually emit it because target's
    # bonus replaces the rejected token at every step).
    real_draft_forward = draft.forward_step

    def _bad_draft_forward(*args: Any, **kwargs: Any) -> Any:
        result = real_draft_forward(*args, **kwargs)
        # Force argmax to token 999 (an unlikely-natural token in Qwen vocab).
        # Returning entirely synthetic logits keeps the cache state consistent
        # with the actual real forward_step (which still ran).
        import torch as _torch

        forced = []
        for logits in result:
            scrambled = logits.clone()
            min_val = _torch.finfo(logits.dtype).min
            scrambled[:] = min_val
            scrambled[999] = 0.0
            forced.append(scrambled)
        return forced

    with patch.object(draft, "forward_step", side_effect=_bad_draft_forward):
        spec_tokens, stats = spec.run_greedy(prompt, max_tokens=max_tokens)

    # Acceptance should be (near) zero since draft never matches target.
    assert stats.mean_acceptance_per_iter <= 0.5, (
        f"unexpected high acceptance {stats.mean_acceptance_per_iter} from a "
        "deliberately-wrong draft"
    )
    # Output still matches target-alone greedy: every rejected step's bonus
    # is target's own argmax, which is what target-alone would have produced.
    baseline_tokens = _greedy_tokens_target_alone(target, prompt, max_tokens)
    assert spec_tokens == baseline_tokens, (
        f"divergent-draft spec output {spec_tokens} != target-alone {baseline_tokens}"
    )


def _greedy_tokens_target_alone(target: ModelRunner, prompt: str, max_tokens: int) -> list[int]:
    """Greedy-decode `max_tokens` from `target` without speculation (reference oracle)."""
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    cache = PagedKVCache(target.block_pool)
    batch_idx = cache.add_request_slot()
    eos = target.tokenizer.eos_token_id
    try:
        prompt_ids = target.tokenizer.encode(prompt)
        packed = target.forward_step_packed(cache, prompt_ids, [0, len(prompt_ids)], [0])
        next_token = int(packed[0, -1, :].argmax().item())
        out: list[int] = [next_token]
        seq_len = len(prompt_ids)
        while len(out) < max_tokens and next_token != eos:
            logits_list = target.forward_step(cache, [next_token], [0, 1], [seq_len])
            next_token = int(logits_list[0].argmax().item())
            out.append(next_token)
            seq_len += 1
        return out[:max_tokens]
    finally:
        cache.remove_request(batch_idx)
