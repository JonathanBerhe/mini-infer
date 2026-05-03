"""Llama-shape models load and run through the owned-model registry.

Uses an ungated Llama-shape checkpoint (SmolLM2-135M-Instruct, HF
architecture string `LlamaForCausalLM`) so the test runs without HF
auth. Confirms the model loads, the Llama branch of the registry
fires, and greedy decode produces non-degenerate output.
"""

import pytest

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.models.llama import LlamaForCausalLM
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.requires_model
def test_smollm2_loads_as_llama_for_causal_lm() -> None:
    """SmolLM2-135M (HF arch=LlamaForCausalLM) routes to our LlamaForCausalLM."""
    runner = ModelRunner.from_pretrained(MODEL_NAME)
    assert isinstance(runner._model, LlamaForCausalLM)
    assert runner._model.cfg.tie_word_embeddings is True


@pytest.mark.requires_model
def test_smollm2_decodes_paris_for_france_prompt() -> None:
    """Greedy decode on a Llama-shape model produces a coherent completion."""
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
    assert len(result.tokens) > 0
    assert result.finish_reason in {"stop", "length"}
