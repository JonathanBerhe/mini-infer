"""The fused RoPE-and-write kernel against the PyTorch path of the attention block.

On CUDA q and the cached k agree with `apply_rotary_pos_emb` bit for bit. Under
the Triton interpreter on the CPU, fp32 agrees bit for bit and bf16 within a
few bf16 steps, because the interpreter truncates fp32 to bf16 instead of
rounding it.
"""

from __future__ import annotations

import os

import pytest
import torch

from mini_infer.models.blocks import rope_write_kernel
from mini_infer.models.blocks.rope import apply_rotary_pos_emb
from mini_infer.models.blocks.rope_write_kernel import (
    rope_and_write_kv,
    supports_rope_write_kernel,
)

needs_triton = pytest.mark.skipif(not rope_write_kernel._TRITON_AVAILABLE, reason="needs triton")


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _tables(
    tokens: int, head_dim: int, dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` of `(tokens, head_dim)`, halves repeated as in `RotaryEmbedding`."""
    angles = torch.rand(tokens, head_dim // 2, generator=torch.Generator().manual_seed(1)) * 6.0
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)


def _torch_path(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`apply_rotary_pos_emb` on `(1, heads, tokens, head_dim)`, back to the packed layout."""
    q_rot, k_rot = apply_rotary_pos_emb(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        cos.unsqueeze(0),
        sin.unsqueeze(0),
    )
    return q_rot.transpose(1, 2).squeeze(0), k_rot.transpose(1, 2).squeeze(0)


def _assert_same(got: torch.Tensor, want: torch.Tensor, device: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape
    if device == "cuda" or want.dtype == torch.float32:
        assert torch.equal(got, want)
    else:
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-6, atol=2**-5)


def _inputs(
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    *,
    wide: bool = False,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    extra = 8 if wide else 0
    q = torch.randn(tokens, num_q_heads, head_dim + extra, generator=generator)
    k = torch.randn(tokens, num_kv_heads, head_dim + extra, generator=generator)
    v = torch.randn(tokens, num_kv_heads, head_dim + extra, generator=generator)
    q, k, v = (t.to(device=device, dtype=dtype)[..., :head_dim] for t in (q, k, v))
    cos, sin = _tables(tokens, head_dim, dtype, device)
    return q, k, v, cos, sin


@needs_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("tokens", "num_q_heads", "num_kv_heads", "head_dim", "wide"),
    [(5, 4, 2, 16, False), (3, 7, 1, 96, False), (1, 2, 2, 64, False), (4, 4, 2, 16, True)],
)
def test_kernel_matches_the_torch_path(
    tokens: int, num_q_heads: int, num_kv_heads: int, head_dim: int, wide: bool, dtype: torch.dtype
) -> None:
    device = _device()
    q, k, v, cos, sin = _inputs(
        tokens, num_q_heads, num_kv_heads, head_dim, dtype, device, wide=wide
    )
    assert (not wide) or not q.is_contiguous()
    num_blocks, block_size = 6, 4
    k_pool = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_pool = torch.zeros_like(k_pool)
    block_ids = torch.tensor([5, 0, 3, 3, 1][:tokens], dtype=torch.int32, device=device)
    slots = torch.tensor([2, 0, 1, 3, 0][:tokens], dtype=torch.int64, device=device)

    q_got = rope_and_write_kv(q, k, v, cos, sin, k_pool, v_pool, block_ids, slots)

    q_want, k_want = _torch_path(q, k, cos, sin)
    assert q_got.is_contiguous()
    _assert_same(q_got, q_want, device)
    written = set()
    for t in range(tokens):
        block, slot = int(block_ids[t]), int(slots[t])
        written.add((block, slot))
        _assert_same(k_pool[block, slot], k_want[t], device)
        assert torch.equal(v_pool[block, slot], v[t])
    for block in range(num_blocks):
        for slot in range(block_size):
            if (block, slot) not in written:
                assert not k_pool[block, slot].any() and not v_pool[block, slot].any()


def test_gate_takes_only_matching_cuda_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    q, k, v = torch.zeros(3, 4, 16), torch.zeros(3, 2, 16), torch.zeros(3, 2, 16)
    cos, sin = torch.zeros(3, 16), torch.zeros(3, 16)
    assert not supports_rope_write_kernel(q, k, v, cos, sin)  # not CUDA
    monkeypatch.setattr(rope_write_kernel, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(rope_write_kernel, "is_cuda_device", lambda device: True)
    assert supports_rope_write_kernel(q, k, v, cos, sin)
    assert not supports_rope_write_kernel(q, k, v, cos.double(), sin)  # one dtype for all
    assert not supports_rope_write_kernel(q, k, v, cos[:2], sin)  # tables per token
    assert not supports_rope_write_kernel(q, k, torch.zeros(3, 2, 8), cos, sin)  # v like k
    assert not supports_rope_write_kernel(q, torch.zeros(2, 2, 16), v, cos, sin)  # k per token
    odd = torch.zeros(3, 4, 15)
    assert not supports_rope_write_kernel(odd, odd[:, :2], odd[:, :2], odd[:, 0], odd[:, 0])
    strided = torch.zeros(3, 4, 32)[..., ::2]
    assert not supports_rope_write_kernel(strided, k, v, cos, sin)  # unit last stride
    assert not supports_rope_write_kernel(q.clone().requires_grad_(), k, v, cos, sin)
    with torch.no_grad():
        assert supports_rope_write_kernel(q.clone().requires_grad_(), k, v, cos, sin)
    monkeypatch.setattr(rope_write_kernel, "_DISABLED_FOR_BENCH", True)
    assert not supports_rope_write_kernel(q, k, v, cos, sin)
