"""The Qwen hybrid's FlashAttention path over its contiguous K/V buffers.

On CUDA with flash-attn installed, the full-attention layers call
`flash_attn_with_kvcache` instead of masked SDPA over the whole buffer. This
machine has no FlashAttention, so the test swaps in a PyTorch stand-in that
follows the documented semantics of the call (bottom-right causal alignment,
grouped-query heads, `cache_seqlens` per row) and requires the same logits as
the SDPA path. That pins what the model passes: the layout, the lengths and
the causal flag. The real kernel runs in the GPU gate of
`scripts/qwen/modal_qwen38_lsa_gate.py`.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from mini_infer.cache import contiguous_attention as contiguous
from mini_infer.cache.contiguous_attention import supports_contiguous_kernel
from mini_infer.models.qwen import qwen3_5
from mini_infer.models.qwen.qwen3_5 import Qwen35Config, Qwen35ForConditionalGeneration


def _model() -> Qwen35ForConditionalGeneration:
    torch.manual_seed(0)
    cfg = Qwen35Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        tie_word_embeddings=False,
        max_position_embeddings=256,
    )
    return Qwen35ForConditionalGeneration(cfg).eval()


def _flash_attn_with_kvcache(q, k_cache, v_cache, *, cache_seqlens, softmax_scale, causal):
    """Stand-in for `flash_attn.flash_attn_with_kvcache` with `k=v=None`."""
    assert causal and cache_seqlens.dtype == torch.int32
    _, span, heads, _ = q.shape
    group = heads // k_cache.shape[2]
    out = torch.empty_like(q)
    for row, length in enumerate(cache_seqlens.tolist()):
        keys = k_cache[row, :length].repeat_interleave(group, dim=1).transpose(0, 1)
        values = v_cache[row, :length].repeat_interleave(group, dim=1).transpose(0, 1)
        # Bottom-right alignment: the last query sits at the last key.
        allowed = torch.arange(length) <= torch.arange(span).view(span, 1) + (length - span)
        out[row] = functional.scaled_dot_product_attention(
            q[row].transpose(0, 1), keys, values, attn_mask=allowed, scale=softmax_scale
        ).transpose(0, 1)
    return out


@torch.inference_mode()
def _run(model, ids: torch.Tensor, prefill: int) -> torch.Tensor:
    """Chunked prefill of `ids[:, :prefill]`, then one decode step per token."""
    cache = model.build_state_cache(max_seq_len=128, batch_size=ids.shape[0])
    logits = []
    for start in range(0, prefill, 13):
        chunk = ids[:, start : min(start + 13, prefill)]
        logits.append(model.forward_prefill_with_cache(chunk, state_cache=cache))
        cache.advance_start_pos(chunk.shape[1])
    for pos in range(prefill, ids.shape[1]):
        logits.append(
            model.forward_decode_with_cache_ragged(
                ids[:, pos : pos + 1],
                positions=torch.full((ids.shape[0],), pos),
                state_cache=cache,
            )
        )
    return torch.cat(logits, dim=1)


def test_the_flash_path_gives_the_sdpa_logits(monkeypatch):
    ids = torch.randint(0, 64, (2, 60), generator=torch.Generator().manual_seed(1))
    expected = _run(_model(), ids, prefill=40)

    monkeypatch.setattr(contiguous, "_FLASH_ATTN_AVAILABLE", True)
    monkeypatch.setattr(contiguous, "flash_attn_with_kvcache", _flash_attn_with_kvcache)
    monkeypatch.setattr(contiguous, "require_cuda_device", lambda device, what: None)
    monkeypatch.setattr(qwen3_5, "supports_contiguous_kernel", lambda k_buffer: True)
    calls: list[list[int]] = []

    def recorded(query, k_buffer, v_buffer, lengths, scale):
        calls.append(lengths.tolist())
        return contiguous.contiguous_attention(query, k_buffer, v_buffer, lengths, scale)

    monkeypatch.setattr(qwen3_5, "contiguous_attention", recorded)
    flash = _run(_model(), ids, prefill=40)

    torch.testing.assert_close(flash, expected, atol=1e-5, rtol=0)
    # Each call passes every row's length after its span: 13, 26, 39, 40,
    # then one more per decode step.
    assert calls[:5] == [[13, 13], [26, 26], [39, 39], [40, 40], [41, 41]]


def test_the_flash_path_is_off_without_cuda():
    assert not supports_contiguous_kernel(torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16))


def test_the_flash_path_needs_half_precision(monkeypatch):
    monkeypatch.setattr(contiguous, "_FLASH_ATTN_AVAILABLE", True)
    monkeypatch.setattr(contiguous, "is_cuda_device", lambda device: True)
    assert not supports_contiguous_kernel(torch.zeros(1, 1, 1, 1, dtype=torch.float32))
    assert supports_contiguous_kernel(torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16))
