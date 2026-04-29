"""Stress tests for the speculative decoder on a real Qwen2.5-0.5B model.

These run multi-iteration spec-decode against target-alone greedy across
several prompts and longer generations to exercise the iteration loop and
cache-truncation paths more thoroughly than the single-prompt unit tests.

Marked `@pytest.mark.slow` to keep them out of CI; run locally with
`uv run pytest tests/stress/test_speculative_load.py -v`.
"""

import pytest

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.speculative import SpeculativeRunner

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _greedy_target_alone(target: ModelRunner, prompt: str, max_tokens: int) -> list[int]:
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


@pytest.mark.requires_model
@pytest.mark.slow
def test_spec_decode_matches_target_alone_across_prompts() -> None:
    """Same-model spec-decode produces target-alone greedy output token-for-token
    across multiple prompts and a longer max_tokens than the unit tests use.
    """
    prompts = [
        "The capital of France is",
        "Once upon a time in a faraway land,",
        "def fibonacci(n):",
    ]
    max_tokens = 12

    target = ModelRunner.from_pretrained(MODEL_NAME)
    draft = ModelRunner.from_pretrained(MODEL_NAME)
    spec = SpeculativeRunner(target, draft, K=4)

    diffs: list[str] = []
    for prompt in prompts:
        baseline = _greedy_target_alone(target, prompt, max_tokens)
        spec_tokens, stats = spec.run_greedy(prompt, max_tokens=max_tokens)
        if spec_tokens != baseline:
            diffs.append(
                f"  {prompt!r}: spec={spec_tokens} baseline={baseline} "
                f"(iters={stats.n_iterations}, mean_acc={stats.mean_acceptance_per_iter:.2f})"
            )
    assert not diffs, "spec-decode diverged from target-alone:\n" + "\n".join(diffs)


@pytest.mark.requires_model
@pytest.mark.slow
def test_spec_decode_target_cache_returns_to_pool() -> None:
    """After a spec-decode run completes, both block pools are back to fully free.

    A leaked block (refcount/free imbalance from cache truncation) is the
    most plausible bug surface; this test catches it directly.
    """
    target = ModelRunner.from_pretrained(MODEL_NAME)
    draft = ModelRunner.from_pretrained(MODEL_NAME)
    spec = SpeculativeRunner(target, draft, K=3)

    target_free_before = target.block_pool.num_free_blocks
    draft_free_before = draft.block_pool.num_free_blocks

    spec.run_greedy("The quick brown fox", max_tokens=10)

    assert target.block_pool.num_free_blocks == target_free_before, (
        f"target pool leaked {target_free_before - target.block_pool.num_free_blocks} blocks"
    )
    assert draft.block_pool.num_free_blocks == draft_free_before, (
        f"draft pool leaked {draft_free_before - draft.block_pool.num_free_blocks} blocks"
    )
