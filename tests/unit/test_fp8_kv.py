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


def test_prefill_wrapper_key_separates_kv_storage_classes() -> None:
    """fp8 and bf16 pools must never share a FlashInfer prefill wrapper.

    FlashInfer resolves backend="auto" once per wrapper, from the first
    plan's dtypes, and writes the choice back onto the wrapper. A shared
    wrapper therefore pins later pools to the first pool's backend; on
    Hopper that pinned fa3 onto an fp8 plan, whose sm90 template does not
    exist, and crashed the engine mid-forward. The key function is the
    entire mechanism keeping the classes apart, so pin it model-free.
    """
    from mini_infer.cache.flashinfer_backend import _wrapper_key

    keys = {_wrapper_key(None), _wrapper_key("fp8"), _wrapper_key("nvfp4")}
    assert len(keys) == 3, f"kv storage classes collide on wrapper keys: {keys}"


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
    on a 0.5B model with reasonable per-head scales.

    Driven through `runner.prefill`, which is what writes K/V into the paged
    cache and reads it back through the attention path. A forward that
    bypassed the cache would compare two identical bf16 runs and pass without
    touching fp8 storage at all.

    This exact sequence, a bf16 runner and then an fp8 runner in one process,
    is also a regression test for the wrapper-poisoning bug it once exposed on
    Hopper: FlashInfer resolves backend="auto" once per wrapper, from the
    FIRST plan's dtypes, so a prefill wrapper shared across pools got pinned
    to fa3 by the bf16 half and then JIT-compiled fa3's nonexistent
    bf16-q x e4m3-kv sm90 template for the fp8 half (cutlass "No eligible
    GMMA operator", H100, 0.6.16rc4, 2026-07-29). The backend now keys its
    prefill wrapper by KV storage class so FlashInfer's dtype-aware selection
    runs per class: fa2 for the fp8 mix, fa3 for bf16. On pre-Hopper parts
    both resolve to fa2 and the sharing never mattered, which is why the L4
    GLM validation and the A10 run of this test passed before the fix.
    """
    from mini_infer.cache.flashinfer_backend import supports_flashinfer_backend
    from mini_infer.engine.model_runner import ModelRunner

    if not supports_flashinfer_backend("cuda"):
        pytest.skip("flashinfer-python not installed")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "The capital of France is"

    def _first_token_logits(**kwargs: object) -> torch.Tensor:
        runner = ModelRunner.from_pretrained(model_name, **kwargs)
        _, logits = runner.prefill(runner.tokenizer.encode(prompt))
        return logits.float().cpu()

    bf16_logits = _first_token_logits()
    fp8_logits = _first_token_logits(kv_quant="fp8", attention_backend="flashinfer")

    cos = float(
        torch.nn.functional.cosine_similarity(
            bf16_logits.flatten(), fp8_logits.flatten(), dim=0
        ).item()
    )
    assert cos > 0.99, f"fp8 vs bf16 logit cosine sim {cos:.6f} below 0.99"
