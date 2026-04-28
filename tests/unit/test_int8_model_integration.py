"""End-to-end INT8 quantization tests on a real Qwen2.5-0.5B model.

These verify that quantizing the model in place doesn't break the forward
pass, that the right modules are replaced, and that quantized logits are close
enough to the fp reference (cosine similarity > 0.99).

Marked `@pytest.mark.requires_model` so they only run when the HF model is
locally available.
"""

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.quant import Int8Linear
from mini_infer.scheduler import ContinuousScheduler, Request

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _count_modules(model: torch.nn.Module) -> tuple[int, int]:
    """Return (n_nn_linear, n_int8_linear)."""
    n_nn = sum(
        1
        for _, m in model.named_modules()
        if isinstance(m, nn.Linear) and not isinstance(m, Int8Linear)
    )
    n_int8 = sum(1 for _, m in model.named_modules() if isinstance(m, Int8Linear))
    return n_nn, n_int8


@pytest.mark.requires_model
def test_quantizes_qwen2_linears_excluding_lm_head_by_default() -> None:
    """Default `quant='int8'` replaces all Linears except `lm_head`."""
    runner = ModelRunner.from_pretrained(MODEL_NAME, quant="int8")
    n_nn, n_int8 = _count_modules(runner._model)
    # Qwen2.5-0.5B has 169 Linears total: 168 in layers + 1 lm_head.
    assert n_int8 == 168, f"expected 168 Int8Linear, got {n_int8}"
    assert n_nn == 1, f"expected exactly 1 remaining nn.Linear (lm_head), got {n_nn}"
    assert isinstance(runner._model.lm_head, nn.Linear)
    assert not isinstance(runner._model.lm_head, Int8Linear)


@pytest.mark.requires_model
def test_quantize_lm_head_flag_replaces_lm_head() -> None:
    """`quant_lm_head=True` flips lm_head to Int8Linear too."""
    runner = ModelRunner.from_pretrained(MODEL_NAME, quant="int8", quant_lm_head=True)
    n_nn, n_int8 = _count_modules(runner._model)
    assert n_int8 == 169, f"expected 169 Int8Linear (incl. lm_head), got {n_int8}"
    assert n_nn == 0, f"expected 0 remaining nn.Linear, got {n_nn}"
    assert isinstance(runner._model.lm_head, Int8Linear)


@pytest.mark.requires_model
def test_quantized_logits_close_to_fp_reference() -> None:
    """Quantized model's last-token logits cosine-sim > 0.99 against fp reference.

    The benchmark threshold for weight-only INT8 with per-channel scales on
    Qwen-class models is typically > 0.999; we set 0.99 as a generous floor
    with comfortable margin against unlucky weight distributions.
    """
    prompt = "The quick brown fox jumps over the lazy dog. Their"

    fp_runner = ModelRunner.from_pretrained(MODEL_NAME, quant=None)
    int8_runner = ModelRunner.from_pretrained(MODEL_NAME, quant="int8")

    input_ids = fp_runner.tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=fp_runner.device, dtype=torch.long)
    with torch.inference_mode():
        fp_logits = fp_runner._model(input_tensor, use_cache=False).logits[0, -1, :]
        int8_logits = int8_runner._model(input_tensor, use_cache=False).logits[0, -1, :]

    cos = torch.nn.functional.cosine_similarity(
        fp_logits.float().flatten(), int8_logits.float().flatten(), dim=0
    ).item()
    assert cos > 0.99, f"cosine sim {cos:.4f} below 0.99"


@pytest.mark.requires_model
def test_quantized_model_decodes_paris() -> None:
    """A quantized model still produces sensible text on the Paris prompt."""
    runner = ModelRunner.from_pretrained(MODEL_NAME, quant="int8")
    sched = ContinuousScheduler(runner)
    sched.start()
    try:
        result = sched.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=8,
            )
        )
    finally:
        sched.stop()
    assert "Paris" in result.text, f"expected 'Paris' in output, got {result.text!r}"


@pytest.mark.requires_model
def test_quantized_weight_storage_smaller_than_fp() -> None:
    """Quantized linears occupy ~half the bytes of their fp counterparts."""
    fp_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    fp_linear_bytes = sum(
        m.weight.numel() * m.weight.element_size()
        + (m.bias.numel() * m.bias.element_size() if m.bias is not None else 0)
        for _, m in fp_model.named_modules()
        if isinstance(m, nn.Linear) and not isinstance(m, Int8Linear)
    )

    runner = ModelRunner.from_pretrained(MODEL_NAME, quant="int8", dtype=torch.float16)
    int8_linear_bytes = 0
    for _, m in runner._model.named_modules():
        if isinstance(m, Int8Linear):
            int8_linear_bytes += m.weight.numel() * m.weight.element_size()
            int8_linear_bytes += m.scales.numel() * m.scales.element_size()
            if m.bias is not None:
                int8_linear_bytes += m.bias.numel() * m.bias.element_size()
        elif isinstance(m, nn.Linear):
            int8_linear_bytes += m.weight.numel() * m.weight.element_size()
            if m.bias is not None:
                int8_linear_bytes += m.bias.numel() * m.bias.element_size()

    saved = fp_linear_bytes - int8_linear_bytes
    saved_pct = saved / fp_linear_bytes
    # Default skip leaves lm_head fp16 (~28% of Linear params on 0.5B). The
    # remaining 72% gets ~50% smaller, so the model-wide savings are ~36%.
    assert saved_pct >= 0.30, f"only saved {saved_pct:.1%} of Linear weight bytes; expected >= 30%"
