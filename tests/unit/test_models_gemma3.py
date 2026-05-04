"""Gemma 3 owned-model loads and runs through the registry.

Uses `unsloth/gemma-3-1b-it` (ungated mirror of Gemma 3 1B-it). Confirms
the registry routes the HF arch string to our class, the per-layer
attention pattern reflects Gemma 3's 5-sliding:1-global cadence, and
greedy decode produces a coherent completion.
"""

import pytest
import torch

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.models.blocks import GemmaRMSNorm
from mini_infer.models.gemma3 import Gemma3ForCausalLM
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "unsloth/gemma-3-1b-it"


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().to(torch.float32), b.flatten().to(torch.float32), dim=0
        ).item()
    )


def test_gemma_rmsnorm_matches_hf_reference() -> None:
    """`GemmaRMSNorm` matches HF's `Gemma3RMSNorm` (block-parity, no model load)."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

    torch.manual_seed(0)
    hidden = 64
    x = torch.randn(1, 8, hidden, dtype=torch.float32)

    ours = GemmaRMSNorm(hidden, eps=1e-6)
    theirs = Gemma3RMSNorm(hidden, eps=1e-6)
    with torch.no_grad():
        # HF Gemma3RMSNorm stores the parameter as `weight` like ours.
        theirs.weight.copy_(ours.weight)

    out_ours = ours(x)
    out_theirs = theirs(x)
    assert _cos_sim(out_ours, out_theirs) > 0.999
    assert torch.allclose(out_ours, out_theirs, atol=1e-5)


@pytest.mark.requires_model
def test_gemma3_loads_through_registry() -> None:
    """Gemma 3 1B-it routes to our `Gemma3ForCausalLM` and exposes the SWA pattern."""
    runner = ModelRunner.from_pretrained(MODEL_NAME)
    assert isinstance(runner._model, Gemma3ForCausalLM)
    layer_attention = runner._model.per_layer_attention()
    assert len(layer_attention) == runner._model.cfg.num_hidden_layers
    # Gemma 3 1B-it has the 5:1 sliding:full cadence — at least one of each.
    assert any(spec == "full" for spec in layer_attention)
    assert any(isinstance(spec, tuple) and spec[0] == "sliding" for spec in layer_attention)


@pytest.mark.requires_model
def test_gemma3_decodes_paris_for_france_prompt() -> None:
    """Greedy decode through Gemma 3 produces a coherent completion."""
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
