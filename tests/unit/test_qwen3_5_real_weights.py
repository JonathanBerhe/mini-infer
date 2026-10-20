"""Real-weight smoke for the Qwen hybrid family on `Qwen/Qwen3.5-4B`.

The 4B is the smallest checkpoint with the `Qwen3_5ForConditionalGeneration`
architecture that Qwen3.8-27B ships. It loads through the production route,
`StateCacheGenerator.from_pretrained`, which streams the weights in one shard
at a time, and it decodes through the state-cache scheduler.

The download is 9.3 GB and the model holds about 8.5 GB in half precision, so
the tests are marked `large_model` as well as `requires_model` and skip unless
the run passes `--run-large-models`:

    uv run pytest tests/unit/test_qwen3_5_real_weights.py --run-large-models -v

The bit-parity workflow does not pass the flag, because its runner and its
model cache cannot hold the checkpoint.
"""

from __future__ import annotations

import pytest

from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.qwen3_5 import Qwen35ForConditionalGeneration
from mini_infer.scheduler import Request, StateCacheContinuousScheduler

pytestmark = [pytest.mark.requires_model, pytest.mark.large_model]

MODEL_NAME = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


@pytest.fixture(scope="module")
def generator() -> StateCacheGenerator:
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        MODEL_NAME,
        revision=REVISION,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
    )
    return StateCacheGenerator.from_pretrained(local_dir)


def test_qwen35_4b_loads_through_registry(generator: StateCacheGenerator) -> None:
    """The 4B routes to our class with its 3:1 layout, 2x head ratio and tied embeddings."""
    model = generator.model
    assert isinstance(model, Qwen35ForConditionalGeneration)
    kinds = model.cfg.layer_types
    assert kinds.count("linear_attention") == 24
    assert kinds.count("full_attention") == 8
    assert model.cfg.linear_head_ratio == 2
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_qwen35_4b_decodes_paris_for_france_prompt(generator: StateCacheGenerator) -> None:
    """Greedy decode through the state-cache scheduler gives a coherent completion."""
    scheduler = StateCacheContinuousScheduler(generator, max_batch_size=1, max_seq_len=64)
    scheduler.start()
    try:
        result = scheduler.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=12,
            )
        )
    finally:
        scheduler.stop()
    assert "Paris" in result.text, f"unexpected output: {result.text!r}"
    assert result.finish_reason in {"stop", "length"}
