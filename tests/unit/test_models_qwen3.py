"""Qwen3 owned-model loads and runs through the registry.

Uses `Qwen/Qwen3-0.6B` (ungated, ~600M params). Validates Qwen3-specific
deltas vs Qwen2: no QKV biases + per-head Q/K norm + tied embeddings.
"""

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.models.qwen3 import Qwen3ForCausalLM
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.mark.requires_model
def test_qwen3_loads_through_registry() -> None:
    """Qwen3-0.6B routes to our `Qwen3ForCausalLM` and constructs Q/K norm."""
    runner = ModelRunner.from_pretrained(MODEL_NAME)
    assert isinstance(runner._model, Qwen3ForCausalLM)
    assert runner._model.cfg.tie_word_embeddings is True
    # Q/K norm must be constructed (Qwen3 ships them in safetensors).
    layer0_attn = runner._model.model.layers[0].self_attn
    assert layer0_attn.q_norm is not None
    assert layer0_attn.k_norm is not None


@pytest.mark.requires_model
def test_qwen3_decodes_paris_for_france_prompt() -> None:
    """Greedy decode through Qwen3-0.6B produces a coherent completion."""
    runner = ModelRunner.from_pretrained(MODEL_NAME)
    sched = ContinuousScheduler(runner)
    sched.start()
    try:
        result = sched.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=12,
            )
        )
    finally:
        sched.stop()
    assert "Paris" in result.text, f"unexpected output: {result.text!r}"
    assert result.finish_reason in {"stop", "length"}
