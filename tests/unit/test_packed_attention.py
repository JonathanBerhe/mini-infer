import math

import pytest
import torch

from mini_infer.cache.packed_attention import (
    packed_attention_torch,
    supports_packed_kernel,
)
from tests.unit._kernel_test_utils import sdpa_reference


def _build_packed_inputs(
    seq_lens_q: list[int],
    seq_lens_k: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build random packed Q/K/V plus cu_seqlens for varlen attention tests."""
    g = torch.Generator().manual_seed(seed)
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)
    q = torch.randn(total_q, num_q_heads, head_dim, generator=g)
    k = torch.randn(total_k, num_kv_heads, head_dim, generator=g)
    v = torch.randn(total_k, num_kv_heads, head_dim, generator=g)
    cu_q = torch.tensor([0, *list(_cumsum(seq_lens_q))], dtype=torch.int32)
    cu_k = torch.tensor([0, *list(_cumsum(seq_lens_k))], dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def _cumsum(values: list[int]) -> list[int]:
    out = []
    running = 0
    for v in values:
        running += v
        out.append(running)
    return out


def test_torch_packed_matches_per_request_sdpa() -> None:
    """Per-request causal SDPA reference produces the same output as the packed call."""
    seq_lens_q = [3, 1, 2]
    seq_lens_k = [3, 8, 2]  # request 1 has prefilled K/V from before
    num_q_heads, num_kv_heads, head_dim = 4, 4, 8
    q, k, v, cu_q, cu_k = _build_packed_inputs(
        seq_lens_q, seq_lens_k, num_q_heads, num_kv_heads, head_dim, seed=1
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_packed = packed_attention_torch(q, k, v, cu_q, cu_k, softmax_scale)

    # Build a per-request reference manually: per-token causal SDPA.
    out_ref = torch.empty_like(q)
    for batch_idx, (q_len, k_len) in enumerate(zip(seq_lens_q, seq_lens_k, strict=True)):
        q_start = int(cu_q[batch_idx])
        k_start = int(cu_k[batch_idx])
        q_b = q[q_start : q_start + q_len]
        k_b = k[k_start : k_start + k_len]
        v_b = v[k_start : k_start + k_len]
        q_offset = k_len - q_len
        for i in range(q_len):
            allowed_keys = q_offset + i + 1
            # scores: (num_heads, allowed_keys)
            scores = (
                torch.einsum("hd,khd->hk", q_b[i].float(), k_b[:allowed_keys].float())
                * softmax_scale
            )
            weights = torch.softmax(scores, dim=-1)
            out_ref[q_start + i] = torch.einsum(
                "hk,khd->hd", weights, v_b[:allowed_keys].float()
            ).to(q.dtype)

    assert torch.allclose(out_packed, out_ref, atol=1e-4)


def test_torch_packed_handles_gqa() -> None:
    """GQA with 14 Q heads / 2 KV heads (Qwen2.5-0.5B's ratio)."""
    seq_lens_q = [4, 1]
    seq_lens_k = [4, 6]
    num_q_heads, num_kv_heads, head_dim = 14, 2, 8
    q, k, v, cu_q, cu_k = _build_packed_inputs(
        seq_lens_q, seq_lens_k, num_q_heads, num_kv_heads, head_dim, seed=2
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out = packed_attention_torch(q, k, v, cu_q, cu_k, softmax_scale)
    assert out.shape == (5, num_q_heads, head_dim)
    # Sanity: not NaN, not zero (random inputs should produce non-trivial output).
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0


def test_torch_packed_isolates_requests() -> None:
    """Request 0's queries do not attend to request 1's keys (cross-request isolation)."""
    # Build inputs where request 1's K/V is huge magnitude; if request 0 leaked
    # to it the output would be dominated by those values.
    num_q_heads, num_kv_heads, head_dim = 2, 2, 4
    q = torch.cat(
        [
            torch.ones(2, num_q_heads, head_dim),  # request 0
            torch.ones(1, num_q_heads, head_dim),  # request 1
        ]
    )
    k_req0 = torch.ones(2, num_kv_heads, head_dim)
    v_req0 = torch.ones(2, num_kv_heads, head_dim)
    k_req1 = torch.full((1, num_kv_heads, head_dim), 1000.0)  # huge magnitude
    v_req1 = torch.full((1, num_kv_heads, head_dim), 1000.0)
    k = torch.cat([k_req0, k_req1])
    v = torch.cat([v_req0, v_req1])
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 3], dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out = packed_attention_torch(q, k, v, cu_q, cu_k, softmax_scale)
    # Request 0 attended only to its own K/V (all 1.0). Output should be ~1.0,
    # nowhere near 1000.0 — proving request 1's huge values didn't leak.
    assert torch.allclose(out[:2], torch.ones_like(out[:2]), atol=1e-4)
    # Request 1 attended only to its own (huge) K/V. Output ~1000.0.
    assert torch.allclose(out[2:], torch.full_like(out[2:], 1000.0), atol=1e-2)


def test_torch_packed_causal_within_request() -> None:
    """Within a request, position i must not attend to position j > i."""
    # Single request, 4 q-tokens against its own 4-token K/V history. Last
    # K/V position has huge magnitude. If position 0 attended to it, the
    # output at position 0 would be ~1000; with proper causal masking it should be ~1.
    num_q_heads, num_kv_heads, head_dim = 2, 2, 4
    q = torch.ones(4, num_q_heads, head_dim)
    k = torch.ones(4, num_kv_heads, head_dim)
    v = torch.ones(4, num_kv_heads, head_dim)
    k[3] = 1000.0
    v[3] = 1000.0
    cu_q = torch.tensor([0, 4], dtype=torch.int32)
    cu_k = torch.tensor([0, 4], dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out = packed_attention_torch(q, k, v, cu_q, cu_k, softmax_scale)
    # Position 0 attends only to position 0 → output[0] = V[0] = 1.0
    # Position 1 attends to positions 0, 1 → output[1] = average of (1.0, 1.0) = 1.0
    # Position 2 attends to positions 0, 1, 2 → 1.0
    # Position 3 attends to all four → mostly the huge V[3] dominates softmax.
    assert torch.allclose(out[0], torch.ones_like(out[0]), atol=1e-4)
    assert torch.allclose(out[1], torch.ones_like(out[1]), atol=1e-4)
    assert torch.allclose(out[2], torch.ones_like(out[2]), atol=1e-4)
    # Position 3 should be nowhere near 1.0; the huge V[3] dominates.
    assert out[3].abs().min() > 100.0


def test_torch_packed_b_equals_one_matches_ordinary_sdpa() -> None:
    """Single-request packed call should match an ordinary SDPA over the same Q/K/V."""
    seq_len_q = 5
    seq_len_k = 5
    num_q_heads, num_kv_heads, head_dim = 4, 4, 8
    g = torch.Generator().manual_seed(7)
    q = torch.randn(seq_len_q, num_q_heads, head_dim, generator=g)
    k = torch.randn(seq_len_k, num_kv_heads, head_dim, generator=g)
    v = torch.randn(seq_len_k, num_kv_heads, head_dim, generator=g)
    cu_q = torch.tensor([0, seq_len_q], dtype=torch.int32)
    cu_k = torch.tensor([0, seq_len_k], dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_packed = packed_attention_torch(q, k, v, cu_q, cu_k, softmax_scale)

    # Reference: causal SDPA over (1, num_q_heads, seq_len_q, head_dim) etc.
    out_ref = torch.empty_like(q)
    for i in range(seq_len_q):
        out_ref[i] = sdpa_reference(
            q[i : i + 1].unsqueeze(0).transpose(1, 2).reshape(1, num_q_heads, head_dim),
            k[: i + 1],
            v[: i + 1],
        ).reshape(num_q_heads, head_dim)

    assert torch.allclose(out_packed, out_ref, atol=1e-4)


def test_torch_packed_rejects_q_longer_than_k() -> None:
    """A request can't have more queries than keys (would imply attending to the future)."""
    q = torch.zeros(5, 2, 4)
    k = torch.zeros(3, 2, 4)
    v = torch.zeros(3, 2, 4)
    cu_q = torch.tensor([0, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="q_len=5 > k_len=3"):
        packed_attention_torch(q, k, v, cu_q, cu_k, 1.0)


@pytest.mark.requires_cuda
def test_paged_flash_matches_torch_reference() -> None:
    """Paged FA varlen output must match the PyTorch reference within cosine sim > 0.99.

    Builds a small `PagedKVCache`, appends K/V for two requests with different
    seq_lens, then compares `packed_attention_forward` (paged FA on CUDA) to
    `packed_attention_torch` (per-request SDPA reference).
    """
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.packed_attention import packed_attention_forward
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    device = torch.device("cuda")
    if not supports_packed_kernel(device):
        pytest.skip("flash-attn not available on this device")

    seq_lens_q = [3, 1, 2]
    seq_lens_k = [3, 8, 2]  # request 1 has 7 tokens of prior history
    num_q_heads, num_kv_heads, head_dim = 4, 2, 16
    # FA's paged varlen requires block_size divisible by 256.
    block_size = 256

    q_cpu, k_cpu, v_cpu, cu_q_cpu, cu_k_cpu = _build_packed_inputs(
        seq_lens_q, seq_lens_k, num_q_heads, num_kv_heads, head_dim, seed=11
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Set up the cache and append all per-request K/V (this simulates the cache
    # state right before attention runs in the engine flow).
    pool = BlockPool(
        num_blocks=8,
        block_size=block_size,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    cache = PagedKVCache(pool)
    for _ in seq_lens_k:
        cache.add_request_slot()
    # Append all of each request's K/V at once (cu_seqlens_q matches seq_lens_k here).
    cache.append_kv_packed(
        k_cpu.to(device).to(torch.float16),
        v_cpu.to(device).to(torch.float16),
        cu_k_cpu.to(device),
        layer_idx=0,
    )

    # The "queries" for attention are the LAST seq_lens_q tokens of each request,
    # consistent with how the engine builds them (chunked prefill or decode).
    q_gpu = q_cpu.to(device).to(torch.float16)
    cu_q_gpu = cu_q_cpu.to(device)
    out_flash = packed_attention_forward(
        q_gpu, cache, layer_idx=0, cu_seqlens_q=cu_q_gpu, softmax_scale=softmax_scale
    )

    out_torch = packed_attention_torch(
        q_cpu.float(), k_cpu.float(), v_cpu.float(), cu_q_cpu, cu_k_cpu, softmax_scale
    )

    sim = torch.nn.functional.cosine_similarity(
        out_flash.float().cpu().flatten(), out_torch.flatten(), dim=0
    ).item()
    assert sim > 0.99, f"cosine sim {sim} below threshold"
