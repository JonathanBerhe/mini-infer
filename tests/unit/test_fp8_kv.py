"""Parity tests for the FP8 KV cache mode (`kv_quant="fp8"`).

CPU mechanics (BlockPool wiring, error paths) run anywhere; the actual
fp8 storage and FlashInfer attention path require CUDA + Hopper-class
hardware (`@pytest.mark.requires_cuda`).
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool


def _has_fp8() -> bool:
    """Whether torch.float8_e4m3fn allocation works on this build."""
    try:
        torch.empty((1,), dtype=torch.float8_e4m3fn, device="cpu")
    except (RuntimeError, AttributeError):
        return False
    return True


def test_block_pool_rejects_fp8_without_flashinfer_backend() -> None:
    """`kv_quant="fp8"` requires `attention_backend="flashinfer"`."""
    if not _has_fp8():
        pytest.skip("torch build lacks float8_e4m3fn support")
    with pytest.raises(ValueError, match="kv_quant='fp8' requires attention_backend='flashinfer'"):
        BlockPool(
            num_blocks=2,
            block_size=8,
            num_layers=1,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.bfloat16,
            device="cpu",
            kv_quant="fp8",
            attention_backend="flash_attn",
        )


def test_block_pool_allocates_fp8_storage() -> None:
    if not _has_fp8():
        pytest.skip("torch build lacks float8_e4m3fn support")
    pool = BlockPool(
        num_blocks=2,
        block_size=8,
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        device="cpu",
        kv_quant="fp8",
        attention_backend="flashinfer",
    )
    assert pool._fp8_storage is not None
    assert pool._fp8_storage.shape == (2, 2, 2, 8, 2, 32)
    assert pool._fp8_storage.dtype == torch.float8_e4m3fn
    assert pool._fp8_scales is not None
    assert pool._fp8_scales.shape == (2, 2, 2)
    assert pool._fp8_scales.dtype == torch.float32
    assert pool._fp8_scales_initialized is not None
    assert pool._fp8_scales_initialized.shape == (2, 2)
    assert pool._fp8_scales_initialized.dtype == torch.bool
    # Compressed-mode tensors must stay None for the fp8 path.
    assert pool._compressed_storage is None
    assert pool._radii_storage is None
    assert pool._rotation is None


# ─────────────────────────────────────────────────────────────────────
# CUDA: end-to-end FP8 vs bf16 parity via the FlashInfer attention path
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.requires_cuda
@pytest.mark.requires_model
def test_qwen_05b_fp8_logits_close_to_bf16() -> None:
    """First-token logits with `kv_quant="fp8"` should track bf16 within
    cosine sim > 0.99. FP8 e4m3fn is lossy but the loss should be small
    on a 0.5B model with reasonable per-head scales."""
    from mini_infer.engine.model_runner import ModelRunner

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "The capital of France is"

    def _first_token_logits(**kwargs: object) -> torch.Tensor:
        runner = ModelRunner.from_pretrained(model_name, **kwargs)
        input_ids = runner.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], device=runner.device, dtype=torch.long)
        with torch.inference_mode():
            return runner._model(x, use_cache=False).logits[0, -1, :].float().cpu()

    bf16_logits = _first_token_logits()
    fp8_logits = _first_token_logits(kv_quant="fp8", attention_backend="flashinfer")

    cos = float(
        torch.nn.functional.cosine_similarity(
            bf16_logits.flatten(), fp8_logits.flatten(), dim=0
        ).item()
    )
    assert cos > 0.99, f"fp8 vs bf16 logit cosine sim {cos:.6f} below 0.99"
