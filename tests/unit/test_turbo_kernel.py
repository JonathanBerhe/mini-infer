"""Parity tests for the fused TurboQuant V3 dequant Triton kernel.

CPU-side mechanics (codebook caching, dispatcher fallback, error
handling) run anywhere. The actual kernel runs only on CUDA + Triton;
those tests are gated on `@pytest.mark.requires_cuda`. The numerical
oracle is `BlockPool.read_compressed_block` going through the existing
Python `polar_dequantize_block` path (`turbo_quant.py:303-369`).
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.turbo_quant import _LLOYD_MAX_GAUSSIAN_3BIT, _LLOYD_MAX_GAUSSIAN_4BIT


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
    )


def _make_turbo3_pool(
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> BlockPool:
    return BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        kv_quant="turbo3",
    )


# ─────────────────────────────────────────────────────────────────────
# CPU-side mechanics: dispatch, codebook caching, fallback path
# ─────────────────────────────────────────────────────────────────────


def test_supports_fused_kernel_returns_false_on_cpu() -> None:
    """The dispatcher must report not-supported on CPU so the Python loop runs."""
    from mini_infer.cache.turbo_kernel import supports_fused_kernel

    assert supports_fused_kernel("cpu") is False
    assert supports_fused_kernel(torch.device("cpu")) is False


def test_pool_caches_lloyd_max_codebooks_when_turbo3() -> None:
    """BlockPool(kv_quant='turbo3') must expose Lloyd-Max codebooks on the pool.

    The fused kernel reads them as device tensors; reallocating per call
    would cost a CUDA H2D copy on every materialize.
    """
    pool = _make_turbo3_pool(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert pool._k_codebook is not None
    assert pool._v_codebook is not None
    assert pool._k_codebook.shape == (8,)
    assert pool._v_codebook.shape == (16,)
    assert pool._k_codebook.dtype == torch.float32
    assert pool._v_codebook.dtype == torch.float32
    # Values match the canonical Lloyd-Max tables from turbo_quant.py.
    assert torch.allclose(
        pool._k_codebook.cpu(),
        torch.tensor(_LLOYD_MAX_GAUSSIAN_3BIT, dtype=torch.float32),
    )
    assert torch.allclose(
        pool._v_codebook.cpu(),
        torch.tensor(_LLOYD_MAX_GAUSSIAN_4BIT, dtype=torch.float32),
    )


def test_pool_does_not_cache_codebooks_when_uncompressed() -> None:
    """Uncompressed pools shouldn't hold codebook tensors at all."""
    pool = BlockPool(
        num_blocks=4,
        block_size=8,
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        device="cpu",
        kv_quant=None,
    )
    assert pool._k_codebook is None
    assert pool._v_codebook is None


def test_fused_wrapper_raises_on_cpu_pool() -> None:
    """Calling the wrapper on a CPU pool must raise; the dispatcher gate
    in materialize_packed_kv routes to the Python loop instead.
    """
    from mini_infer.cache.turbo_kernel import (
        _TRITON_AVAILABLE,
        fused_materialize_packed_kv,
    )

    pool = _make_turbo3_pool(
        num_layers=1,
        num_blocks=2,
        block_size=8,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        device="cpu",
    )
    key_out = torch.empty((4, 2, 32), dtype=torch.bfloat16, device="cpu")
    value_out = torch.empty_like(key_out)
    task_block_ids = torch.tensor([0], dtype=torch.int32, device="cpu")
    task_offsets = torch.tensor([0], dtype=torch.int32, device="cpu")
    task_valid = torch.tensor([4], dtype=torch.int32, device="cpu")

    with pytest.raises(RuntimeError):
        # Either "Triton not available" (M1 wheels) or "requires CUDA"
        # depending on the host; both are valid CPU failure modes.
        fused_materialize_packed_kv(
            pool,
            layer_idx=0,
            task_block_ids=task_block_ids,
            task_offsets=task_offsets,
            task_valid=task_valid,
            key_out=key_out,
            value_out=value_out,
        )
    # Make sure we exercised the right code path on hosts WITH Triton:
    # the M1 case goes through the import-guard error, which is fine too.
    _ = _TRITON_AVAILABLE  # touched for clarity; no assertion needed


def test_python_loop_fallback_unchanged_on_cpu_turbo3() -> None:
    """When Triton/CUDA isn't available, materialize_packed_kv must keep
    using the per-block Python loop and produce the same numerical
    output it did before this change.

    Regression guard: importing turbo_kernel (and adding the new branch
    to the dispatcher) must not perturb the CPU fallback.
    """
    pool = _make_turbo3_pool(
        num_layers=1,
        num_blocks=4,
        block_size=8,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        device="cpu",
    )
    cache = PagedKVCache(pool)
    batch_idx = cache.add_request_slot()

    # Allocate two blocks for this slot and write known data.
    block_a = pool.allocate()
    block_b = pool.allocate()
    cache._block_ids[batch_idx] = [block_a, block_b]
    cache._num_tokens[batch_idx] = 10  # 8 + 2: full block + partial

    torch.manual_seed(7)
    block_data_k = torch.randn(8, 2, 32, dtype=torch.float32) * 0.1
    block_data_v = torch.randn(8, 2, 32, dtype=torch.float32) * 0.1
    pool.write_compressed_block(0, 0, block_a, block_data_k.to(torch.bfloat16))
    pool.write_compressed_block(0, 1, block_a, block_data_v.to(torch.bfloat16))
    pool.write_compressed_block(0, 0, block_b, block_data_k.to(torch.bfloat16))
    pool.write_compressed_block(0, 1, block_b, block_data_v.to(torch.bfloat16))

    k_packed, v_packed, _cu_seqlens_k, max_seqlen_k = cache.materialize_packed_kv(0)
    assert k_packed.shape == (10, 2, 32)
    assert v_packed.shape == (10, 2, 32)
    assert max_seqlen_k == 10

    # Round-trip cosine sim should be high (turbo3 is lossy but coherent).
    expected_k = pool.read_compressed_block(0, 0, block_a)[:8]
    cos = _cosine_sim(k_packed[:8], expected_k)
    assert cos > 0.999, f"CPU fallback K cosine sim {cos:.4f} below 0.999"


# ─────────────────────────────────────────────────────────────────────
# CUDA parity: fused kernel matches the Python-loop reference
# ─────────────────────────────────────────────────────────────────────


def _materialize_two_paths(
    pool: BlockPool,
    cache: PagedKVCache,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run materialize once with the fused kernel ON, once with it forced
    OFF, return both K/V tensors.
    """
    from mini_infer.cache import turbo_kernel

    k_fused, v_fused, _, _ = cache.materialize_packed_kv(layer_idx)

    saved = turbo_kernel._FUSED_DISABLED_FOR_BENCH
    turbo_kernel._FUSED_DISABLED_FOR_BENCH = True
    try:
        k_python, v_python, _, _ = cache.materialize_packed_kv(layer_idx)
    finally:
        turbo_kernel._FUSED_DISABLED_FOR_BENCH = saved
    return k_fused, v_fused, k_python, v_python


def _build_populated_turbo3(
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: list[int],
    seed: int,
    device: str,
) -> tuple[BlockPool, PagedKVCache]:
    """Build a turbo3 pool, attach a cache with `len(seq_lens)` slots, and
    write random K/V data into all referenced blocks.
    """
    pool = _make_turbo3_pool(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device=device,
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
            for block_id in block_ids:
                k_block = (
                    (torch.randn(block_size, num_kv_heads, head_dim, dtype=torch.float32) * 0.1)
                    .to(torch.bfloat16)
                    .to(device)
                )
                v_block = (
                    (torch.randn(block_size, num_kv_heads, head_dim, dtype=torch.float32) * 0.1)
                    .to(torch.bfloat16)
                    .to(device)
                )
                pool.write_compressed_block(layer_idx, 0, block_id, k_block)
                pool.write_compressed_block(layer_idx, 1, block_id, v_block)
    return pool, cache


@pytest.mark.requires_cuda
def test_fused_matches_python_loop_qwen_05b_shape() -> None:
    """Qwen2.5-0.5B shape: head_dim=64, num_kv_heads=2."""
    pool, cache = _build_populated_turbo3(
        num_layers=2,
        num_blocks=16,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[24, 17, 0, 32],
        seed=11,
        device="cuda",
    )
    k_fused, v_fused, k_python, v_python = _materialize_two_paths(pool, cache, layer_idx=1)

    assert k_fused.shape == k_python.shape
    assert v_fused.shape == v_python.shape
    cos_k = _cosine_sim(k_fused, k_python)
    cos_v = _cosine_sim(v_fused, v_python)
    assert cos_k > 0.999, f"K cosine sim {cos_k:.4f} below 0.999"
    assert cos_v > 0.999, f"V cosine sim {cos_v:.4f} below 0.999"


@pytest.mark.requires_cuda
def test_fused_matches_python_loop_qwen_7b_shape() -> None:
    """Qwen2.5-7B shape: head_dim=128, num_kv_heads=8 (head_dim=128 is the
    upper end of what the kernel must support; rotation tile is 32 KB SMEM)."""
    pool, cache = _build_populated_turbo3(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_kv_heads=8,
        head_dim=128,
        seq_lens=[16, 32, 8],
        seed=23,
        device="cuda",
    )
    k_fused, v_fused, k_python, v_python = _materialize_two_paths(pool, cache, layer_idx=0)
    cos_k = _cosine_sim(k_fused, k_python)
    cos_v = _cosine_sim(v_fused, v_python)
    assert cos_k > 0.999, f"7B-shape K cosine sim {cos_k:.4f} below 0.999"
    assert cos_v > 0.999, f"7B-shape V cosine sim {cos_v:.4f} below 0.999"


@pytest.mark.requires_cuda
def test_fused_handles_empty_batch_and_partial_blocks() -> None:
    """Edge cases: batch with seq_len < block_size (full mask), batch with
    seq_len exactly on block boundary, plus a slot with seq_len=0.
    """
    pool, cache = _build_populated_turbo3(
        num_layers=1,
        num_blocks=8,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[16, 0, 1, 31],  # exact, empty, single token, just-under-2-blocks
        seed=42,
        device="cuda",
    )
    k_fused, v_fused, k_python, v_python = _materialize_two_paths(pool, cache, layer_idx=0)
    cos_k = _cosine_sim(k_fused, k_python)
    cos_v = _cosine_sim(v_fused, v_python)
    assert cos_k > 0.999
    assert cos_v > 0.999


@pytest.mark.requires_cuda
def test_fused_block_size_64() -> None:
    """Larger block_size=64 to exercise BLOCK_SIZE constexpr scaling."""
    pool, cache = _build_populated_turbo3(
        num_layers=1,
        num_blocks=4,
        block_size=64,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[100, 50],
        seed=77,
        device="cuda",
    )
    k_fused, v_fused, k_python, v_python = _materialize_two_paths(pool, cache, layer_idx=0)
    cos_k = _cosine_sim(k_fused, k_python)
    cos_v = _cosine_sim(v_fused, v_python)
    assert cos_k > 0.999
    assert cos_v > 0.999


# ─────────────────────────────────────────────────────────────────────
# Dequant parity on a real Qwen2.5-0.5B turbo3 cache
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.requires_cuda
@pytest.mark.requires_model
def test_qwen_05b_turbo3_prefill_materialize_fused_matches_python() -> None:
    """Fused and Python dequant must agree, per layer, on a cache a REAL
    prefill populated.

    Same assertion as `test_fused_matches_python_loop_*` above, but the pool
    state comes from an actual Qwen2.5-0.5B forward rather than hand-written
    blocks: real strides and dims, radii from real activations rather than
    randn*0.1, and a partial final block whose tail slots hold the quantized
    ZERO vector (a real append leaves them dequant-of-nothing; the fixtures
    fill whole blocks with random data before shrinking `_num_tokens`, so
    they never store that state).

    This test used to assert fused-vs-python FIRST-TOKEN LOGIT cosine > 0.999.
    That quantity is not well-defined for turbo3, so the bar was retired, not
    loosened, on this evidence (A10, 2026-07-29: measured 0.805):

    - The two dequant paths legitimately differ by bf16-LSB reduction order
      (`tl.dot` vs `torch.matmul`). The write path then re-quantizes every
      appended block through `searchsorted` at EVERY layer
      (`paged_kv_cache.py`, `_write_packed_kv_compressed`), so a layer-L read
      perturbation crosses a quantization step boundary at layer L+1 and grows
      to whole codec steps by layer 24. Logits diverge macroscopically with
      both paths correct.
    - The record agrees: the 2026-05-02 bench run's greedy decode diverged at
      generated token 0, which IS the argmax of the prefill logits
      (`docs/benchmarks/2026-05-02-turboquant-v2a.md`). Its E2E comparison is
      explicitly informational and computes no logit cosine; the 0.999 logit
      bar was never measured by anything. The prior form of this test passed
      `use_cache=False`, which bypassed the paged cache, so it compared two
      unquantized bf16 runs and validated nothing.

    Comparing what the two paths read back from the SAME stored bytes is the
    level at which they are defined to agree, and it is deterministic.
    """
    from mini_infer.cache import turbo_kernel
    from mini_infer.engine.model_runner import ModelRunner

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "The capital of France is"

    runner = ModelRunner.from_pretrained(model_name, kv_quant="turbo3")
    if not turbo_kernel.supports_fused_kernel(runner.device):
        pytest.skip("fused turbo kernel unavailable; both halves would be the Python loop")

    # One real prefill writes compressed K/V for every layer; whichever read
    # path it used internally, the stored bytes are now fixed, and both halves
    # below read those same bytes.
    pool_cache, logits = runner.prefill(runner.tokenizer.encode(prompt))
    assert torch.isfinite(logits).all(), "prefill produced non-finite logits"

    for layer_idx in range(runner.block_pool.num_layers):
        k_fused, v_fused, k_python, v_python = _materialize_two_paths(
            runner.block_pool, pool_cache, layer_idx
        )
        for name, fused, python in (("K", k_fused, k_python), ("V", v_fused, v_python)):
            cos = _cosine_sim(fused, python)
            max_abs = (fused.float() - python.float()).abs().max().item()
            assert cos > 0.999, (
                f"layer {layer_idx} {name}: fused vs python cosine {cos:.6f} "
                f"below 0.999 (max_abs_diff={max_abs:.6f})"
            )


# ─────────────────────────────────────────────────────────────────────
# V2b: fully-fused attention kernel parity vs V2a materialized path
# ─────────────────────────────────────────────────────────────────────


def test_supports_fused_attn_kernel_returns_false_on_cpu() -> None:
    """V2b predicate must report not-supported on CPU (and respect both toggles)."""
    from mini_infer.cache.turbo_kernel import supports_fused_attn_kernel

    assert supports_fused_attn_kernel("cpu") is False
    assert supports_fused_attn_kernel(torch.device("cpu")) is False


def _attention_two_paths(
    pool: BlockPool,
    cache: PagedKVCache,
    *,
    layer_idx: int,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention via V2b (fused-attn ON) and via V2a materialized (V2b OFF).

    Returns `(out_v2b, out_v2a)` of shape `(total_q, num_q_heads, head_dim)`.

    V2b is now off by default (V2a is faster at 7B+); the test explicitly
    flips the toggle to compare the two paths.
    """
    from mini_infer.cache import turbo_kernel
    from mini_infer.cache.packed_attention import packed_attention_forward

    saved = turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH

    # Force V2b on for the first call.
    turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = False
    try:
        out_v2b = packed_attention_forward(q, cache, layer_idx, cu_seqlens_q)
    finally:
        turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = saved

    # Force V2b off (V2a path) for the second call.
    turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = True
    try:
        out_v2a = packed_attention_forward(q, cache, layer_idx, cu_seqlens_q)
    finally:
        turbo_kernel._FUSED_ATTN_DISABLED_FOR_BENCH = saved
    return out_v2b, out_v2a


@pytest.mark.requires_cuda
def test_fused_attn_decode_matches_materialized_qwen_05b_shape() -> None:
    """V2b decode-only attention output matches V2a materialized at cos sim > 0.999.

    Qwen2.5-0.5B shape (head_dim=64, num_kv_heads=2, num_q_heads=14).
    Each request has q_len=1 — the decode-only contract V2b enforces.
    """
    pool, cache = _build_populated_turbo3(
        num_layers=2,
        num_blocks=16,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[24, 17, 32, 8],
        seed=101,
        device="cuda",
    )
    batch_size = cache.batch_size
    num_q_heads = 14
    head_dim = 64

    torch.manual_seed(202)
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    out_v2b, out_v2a = _attention_two_paths(
        pool, cache, layer_idx=1, q=q, cu_seqlens_q=cu_seqlens_q
    )
    assert out_v2b.shape == out_v2a.shape == (batch_size, num_q_heads, head_dim)
    cos = _cosine_sim(out_v2b, out_v2a)
    assert cos > 0.999, f"V2b vs V2a attention output cosine sim {cos:.6f} below 0.999"


@pytest.mark.requires_cuda
def test_fused_attn_decode_matches_materialized_qwen_7b_shape() -> None:
    """Qwen2.5-7B shape (head_dim=128, num_kv_heads=4, num_q_heads=28, group_size=7).

    Validates the kernel at the larger head_dim where the rotation
    matrix tile in SMEM is 32 KB on A10.
    """
    pool, cache = _build_populated_turbo3(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_kv_heads=4,
        head_dim=128,
        seq_lens=[16, 32, 8],
        seed=303,
        device="cuda",
    )
    batch_size = cache.batch_size
    num_q_heads = 28
    head_dim = 128

    torch.manual_seed(404)
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    out_v2b, out_v2a = _attention_two_paths(
        pool, cache, layer_idx=0, q=q, cu_seqlens_q=cu_seqlens_q
    )
    assert out_v2b.shape == out_v2a.shape == (batch_size, num_q_heads, head_dim)
    cos = _cosine_sim(out_v2b, out_v2a)
    assert cos > 0.999, f"7B-shape V2b vs V2a cosine sim {cos:.6f} below 0.999"
