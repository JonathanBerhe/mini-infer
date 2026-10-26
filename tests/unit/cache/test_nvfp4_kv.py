"""Parity tests for the NVFP4 KV cache mode (`kv_quant="nvfp4"`).

CPU mechanics (BlockPool wiring, error paths) run anywhere; the actual
nvfp4 quantization and FlashInfer attention path require CUDA + Blackwell
(`@pytest.mark.requires_cuda` + a runtime SM_100+ check inside the test).
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


def test_block_pool_rejects_nvfp4_without_flashinfer_backend() -> None:
    if not _has_fp8():
        pytest.skip("torch build lacks float8_e4m3fn support")
    with pytest.raises(
        ValueError, match="kv_quant='nvfp4' requires attention_backend='flashinfer'"
    ):
        BlockPool(
            num_blocks=2,
            block_size=8,
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            device="cpu",
            kv_quant="nvfp4",
            attention_backend="flash_attn",
        )


def test_block_pool_rejects_nvfp4_with_bad_block_size() -> None:
    if not _has_fp8():
        pytest.skip("torch build lacks float8_e4m3fn support")
    with pytest.raises(ValueError, match="block_size %% 4 == 0"):
        BlockPool(
            num_blocks=2,
            block_size=6,  # not divisible by 4
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            device="cpu",
            kv_quant="nvfp4",
            attention_backend="flashinfer",
        )


def test_block_pool_rejects_nvfp4_with_bad_head_dim() -> None:
    if not _has_fp8():
        pytest.skip("torch build lacks float8_e4m3fn support")
    with pytest.raises(ValueError, match="head_dim %% 64 == 0"):
        BlockPool(
            num_blocks=2,
            block_size=8,
            num_layers=1,
            num_kv_heads=2,
            head_dim=32,  # not divisible by 64
            dtype=torch.bfloat16,
            device="cpu",
            kv_quant="nvfp4",
            attention_backend="flashinfer",
        )


def test_block_pool_allocates_nvfp4_storage() -> None:
    if not _has_fp8():
        pytest.skip("torch build lacks float8_e4m3fn support")
    pool = BlockPool(
        num_blocks=2,
        block_size=8,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        device="cpu",
        kv_quant="nvfp4",
        attention_backend="flashinfer",
    )
    assert pool._nvfp4_storage is not None
    # (num_layers, 2 sides, num_blocks, block_size, num_kv_heads, head_dim // 2)
    assert pool._nvfp4_storage.shape == (2, 2, 2, 8, 2, 32)
    assert pool._nvfp4_storage.dtype == torch.uint8
    assert pool._nvfp4_block_scales is not None
    # (num_layers, 2 sides, num_blocks, block_size, num_kv_heads, head_dim // 16)
    assert pool._nvfp4_block_scales.shape == (2, 2, 2, 8, 2, 4)
    assert pool._nvfp4_block_scales.dtype == torch.float8_e4m3fn
    assert pool._nvfp4_global_sf is not None
    assert pool._nvfp4_global_sf.shape == (2, 2)
    assert pool._nvfp4_global_sf.dtype == torch.float32
    assert pool._nvfp4_initialized is not None
    assert pool._nvfp4_initialized.shape == (2, 2)
    assert pool._nvfp4_initialized.dtype == torch.bool
    # Other modes' storage must stay None for the nvfp4 path.
    assert pool._fp8_storage is None
    assert pool._compressed_storage is None
    assert pool._radii_storage is None
    assert pool._rotation is None


def test_block_pool_rejects_nvfp4_with_prefix_cache() -> None:
    if not _has_fp8():
        pytest.skip("torch build lacks float8_e4m3fn support")
    from mini_infer.cache.prefix_cache import PrefixCache

    cache = PrefixCache(block_size=8)
    with pytest.raises(ValueError, match="prefix caching"):
        BlockPool(
            num_blocks=2,
            block_size=8,
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            device="cpu",
            kv_quant="nvfp4",
            attention_backend="flashinfer",
            prefix_cache=cache,
        )


# ─────────────────────────────────────────────────────────────────────
# CUDA: end-to-end NVFP4 vs bf16 parity via the FlashInfer attention path
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.requires_cuda
@pytest.mark.requires_model
def test_qwen_15b_nvfp4_logits_close_to_bf16() -> None:
    """First-token logits with `kv_quant="nvfp4"` should track bf16 within
    cosine sim > 0.97. NVFP4 is lossier than FP8 but should still preserve
    the relative ordering of top tokens on a 1.5B model. Requires Blackwell
    (SM_100+); skipped automatically on Hopper or Ampere."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 KV requires Blackwell (SM_100+)")

    from mini_infer.engine.model_runner import ModelRunner

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    prompt = "The capital of France is"

    def _first_token_logits(**kwargs: object) -> torch.Tensor:
        runner = ModelRunner.from_pretrained(model_name, **kwargs)
        input_ids = runner.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], device=runner.device, dtype=torch.long)
        with torch.inference_mode():
            return runner._model(x, use_cache=False).logits[0, -1, :].float().cpu()

    bf16_logits = _first_token_logits()
    nvfp4_logits = _first_token_logits(kv_quant="nvfp4", attention_backend="flashinfer")

    cos = float(
        torch.nn.functional.cosine_similarity(
            bf16_logits.flatten(), nvfp4_logits.flatten(), dim=0
        ).item()
    )
    assert cos > 0.97, f"nvfp4 vs bf16 logit cosine sim {cos:.6f} below 0.97"
