"""Parity tests for the FlashInfer attention backend.

CPU mechanics (predicate, dispatcher fallback, BlockPool wiring) run
anywhere. Numerical parity vs `flash_attn_varlen_func` is gated on
`@pytest.mark.requires_cuda`; locally those skip and only run on Modal.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
    )


# ─────────────────────────────────────────────────────────────────────
# CPU mechanics: predicate, dispatcher gating, BlockPool wiring
# ─────────────────────────────────────────────────────────────────────


def test_supports_flashinfer_backend_returns_false_on_cpu() -> None:
    """The dispatcher predicate must report not-supported on CPU."""
    from mini_infer.cache.flashinfer_backend import supports_flashinfer_backend

    assert supports_flashinfer_backend("cpu") is False
    assert supports_flashinfer_backend(torch.device("cpu")) is False


def test_block_pool_default_attention_backend_is_flash_attn() -> None:
    pool = BlockPool(
        num_blocks=4,
        block_size=8,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert pool.attention_backend == "flash_attn"


def test_block_pool_accepts_flashinfer_attention_backend() -> None:
    pool = BlockPool(
        num_blocks=4,
        block_size=8,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        device="cpu",
        attention_backend="flashinfer",
    )
    assert pool.attention_backend == "flashinfer"


def test_block_pool_rejects_unknown_attention_backend() -> None:
    with pytest.raises(ValueError, match="unsupported attention_backend"):
        BlockPool(
            num_blocks=4,
            block_size=8,
            num_layers=1,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.bfloat16,
            device="cpu",
            attention_backend="not-a-real-backend",
        )


def test_block_pool_rejects_flashinfer_with_compressed_pool() -> None:
    """FlashInfer reads bf16 paged storage directly; combining with
    `kv_quant` (compressed pool) is invalid until FP8/NVFP4 modes ship."""
    with pytest.raises(ValueError, match="requires kv_quant=None"):
        BlockPool(
            num_blocks=4,
            block_size=8,
            num_layers=1,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.bfloat16,
            device="cpu",
            kv_quant="turbo3",
            attention_backend="flashinfer",
        )


def test_flashinfer_attention_forward_raises_on_cpu_pool() -> None:
    """Calling the wrapper on a CPU pool must raise; the dispatcher's
    `supports_flashinfer_backend` check routes around this in production."""
    from mini_infer.cache.flashinfer_backend import flashinfer_attention_forward

    pool = BlockPool(
        num_blocks=2,
        block_size=8,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        device="cpu",
        attention_backend="flashinfer",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache._block_ids[0] = [0]
    cache._num_tokens[0] = 4

    q = torch.empty((1, 4, 32), dtype=torch.bfloat16, device="cpu")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cpu")

    with pytest.raises(RuntimeError):
        # Either "flashinfer-python not installed" (M1 / no wheel) or
        # "requires CUDA" (Linux without GPU); both are valid CPU
        # failure modes.
        flashinfer_attention_forward(
            q, cache, layer_idx=0, cu_seqlens_q=cu_seqlens_q, softmax_scale=1.0
        )


# ─────────────────────────────────────────────────────────────────────
# CUDA parity: FlashInfer output matches flash-attn within cosine sim
# ─────────────────────────────────────────────────────────────────────


def _build_uncompressed_cache(
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: list[int],
    seed: int,
    device: str,
    attention_backend: str,
) -> tuple[BlockPool, PagedKVCache]:
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device=device,
        attention_backend=attention_backend,
    )
    cache = PagedKVCache(pool)
    torch.manual_seed(seed)
    for slot_idx, seq_len in enumerate(seq_lens):
        cache.add_request_slot()
        if seq_len == 0:
            continue
        num_blocks_used = (seq_len + block_size - 1) // block_size
        block_ids = [pool.allocate() for _ in range(num_blocks_used)]
        cache._block_ids[slot_idx] = block_ids
        cache._num_tokens[slot_idx] = seq_len
        for layer_idx in range(num_layers):
            for bid in block_ids:
                k = (
                    (torch.randn(block_size, num_kv_heads, head_dim, dtype=torch.float32) * 0.1)
                    .to(torch.bfloat16)
                    .to(device)
                )
                v = (
                    (torch.randn(block_size, num_kv_heads, head_dim, dtype=torch.float32) * 0.1)
                    .to(torch.bfloat16)
                    .to(device)
                )
                pool.storage[layer_idx, 0, bid] = k
                pool.storage[layer_idx, 1, bid] = v
    return pool, cache


@pytest.mark.requires_cuda
def test_flashinfer_decode_matches_flash_attn_qwen_05b_shape() -> None:
    """Decode-only batch (Qwen2.5-0.5B shape): cosine sim > 0.999 vs flash-attn."""
    from mini_infer.cache.packed_attention import packed_attention_forward

    _, fi_cache = _build_uncompressed_cache(
        num_layers=2,
        num_blocks=16,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[24, 17, 32, 8],
        seed=101,
        device="cuda",
        attention_backend="flashinfer",
    )
    _, fa_cache = _build_uncompressed_cache(
        num_layers=2,
        num_blocks=16,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[24, 17, 32, 8],
        seed=101,
        device="cuda",
        attention_backend="flash_attn",
    )

    batch_size = fi_cache.batch_size
    torch.manual_seed(202)
    q = torch.randn(batch_size, 14, 64, dtype=torch.bfloat16, device="cuda") * 0.1
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    out_flashinfer = packed_attention_forward(q, fi_cache, layer_idx=1, cu_seqlens_q=cu_seqlens_q)
    out_flash_attn = packed_attention_forward(q, fa_cache, layer_idx=1, cu_seqlens_q=cu_seqlens_q)

    assert out_flashinfer.shape == out_flash_attn.shape
    cos = _cosine_sim(out_flashinfer, out_flash_attn)
    assert cos > 0.999, f"FlashInfer vs flash-attn decode cosine sim {cos:.6f} below 0.999"


@pytest.mark.requires_cuda
def test_flashinfer_prefill_matches_flash_attn_qwen_05b_shape() -> None:
    """Prefill batch (multi-token Q per request): cosine sim > 0.999 vs flash-attn."""
    from mini_infer.cache.packed_attention import packed_attention_forward

    seq_lens = [12, 8]
    _, fi_cache = _build_uncompressed_cache(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=seq_lens,
        seed=303,
        device="cuda",
        attention_backend="flashinfer",
    )
    _, fa_cache = _build_uncompressed_cache(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=seq_lens,
        seed=303,
        device="cuda",
        attention_backend="flash_attn",
    )

    # Multi-token Q: one prefill chunk per request (q_len == seq_len so the
    # request has no prior history; this exercises the prefill wrapper).
    torch.manual_seed(404)
    q_lens = list(seq_lens)
    total_q = sum(q_lens)
    q = torch.randn(total_q, 14, 64, dtype=torch.bfloat16, device="cuda") * 0.1
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )

    out_flashinfer = packed_attention_forward(q, fi_cache, layer_idx=0, cu_seqlens_q=cu_seqlens_q)
    out_flash_attn = packed_attention_forward(q, fa_cache, layer_idx=0, cu_seqlens_q=cu_seqlens_q)

    cos = _cosine_sim(out_flashinfer, out_flash_attn)
    assert cos > 0.999, f"FlashInfer vs flash-attn prefill cosine sim {cos:.6f} below 0.999"


@pytest.mark.requires_cuda
def test_flashinfer_decode_qwen_7b_shape() -> None:
    """Larger GQA shape (head_dim=128, num_kv_heads=4, num_q_heads=28)."""
    from mini_infer.cache.packed_attention import packed_attention_forward

    _, fi_cache = _build_uncompressed_cache(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_kv_heads=4,
        head_dim=128,
        seq_lens=[16, 32, 8],
        seed=505,
        device="cuda",
        attention_backend="flashinfer",
    )
    _, fa_cache = _build_uncompressed_cache(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_kv_heads=4,
        head_dim=128,
        seq_lens=[16, 32, 8],
        seed=505,
        device="cuda",
        attention_backend="flash_attn",
    )

    batch_size = fi_cache.batch_size
    torch.manual_seed(606)
    q = torch.randn(batch_size, 28, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    out_flashinfer = packed_attention_forward(q, fi_cache, layer_idx=1, cu_seqlens_q=cu_seqlens_q)
    out_flash_attn = packed_attention_forward(q, fa_cache, layer_idx=1, cu_seqlens_q=cu_seqlens_q)

    cos = _cosine_sim(out_flashinfer, out_flash_attn)
    assert cos > 0.999, f"7B-shape FlashInfer vs flash-attn cosine sim {cos:.6f} below 0.999"
