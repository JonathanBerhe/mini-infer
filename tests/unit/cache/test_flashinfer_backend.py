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


def test_block_pool_rejects_flashinfer_with_turboquant_pool() -> None:
    """FlashInfer's bf16/fp8 paged storage layouts don't match the
    TurboQuant nibble-packed layout, so the combo is rejected."""
    with pytest.raises(ValueError, match="kv_quant in \\(None, 'fp8', 'nvfp4'\\)"):
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


class _FakePrefillWrapper:
    """Records plan() calls and runs attention as identity."""

    def __init__(self) -> None:
        self.plans: list[dict[str, object]] = []
        self.runs = 0

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.plans.append(
            {
                "qo_indptr": qo_indptr.tolist(),
                "indptr": paged_kv_indptr.tolist(),
                "indices": paged_kv_indices.tolist(),
                "last_page_len": paged_kv_last_page_len.tolist(),
                "window_left": kwargs["window_left"],
                "non_blocking": kwargs.get("non_blocking"),
                "tensors": (qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len),
            }
        )

    def run(self, q: torch.Tensor, kv_cache: object, **kwargs: object) -> torch.Tensor:
        self.runs += 1
        return q


def test_plan_runs_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every layer of a step reuses one plan; a new step, other parameters or a
    layout change plan again. Runs on the CPU with a fake wrapper."""
    from mini_infer.cache import flashinfer_backend

    fake = _FakePrefillWrapper()
    monkeypatch.setattr(flashinfer_backend, "_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(flashinfer_backend, "require_cuda_device", lambda device, what: None)
    monkeypatch.setattr(
        flashinfer_backend, "_ensure_wrappers", lambda device, kv_quant: (fake, None)
    )
    monkeypatch.setattr(flashinfer_backend, "_last_plan", None)
    forward = flashinfer_backend.flashinfer_attention_forward

    num_layers, block_size = 3, 4
    pool = BlockPool(
        num_blocks=16,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="flashinfer",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cache.add_request_slot()

    def step(new_tokens: list[int]) -> None:
        cu_seqlens = torch.tensor([0, new_tokens[0], sum(new_tokens)], dtype=torch.int32)
        total = sum(new_tokens)
        for layer_idx in range(num_layers):
            kv = torch.randn(total, 2, 8)
            cache.append_kv_packed(kv, kv, cu_seqlens, layer_idx)
            q = torch.randn(total, 4, 8)
            assert forward(q, cache, layer_idx, cu_seqlens, softmax_scale=0.5) is q

    step([5, 3])
    assert fake.runs == num_layers
    assert len(fake.plans) == 1
    expected_indices = cache.block_ids_for_request(0) + cache.block_ids_for_request(1)
    assert fake.plans[0]["qo_indptr"] == [0, 5, 8]
    assert fake.plans[0]["indptr"] == [0, 2, 3]
    assert fake.plans[0]["indices"] == expected_indices
    assert fake.plans[0]["last_page_len"] == [1, 3]
    assert fake.plans[0]["window_left"] == -1

    step([1, 1])
    assert fake.runs == 2 * num_layers
    assert len(fake.plans) == 2
    assert fake.plans[1]["last_page_len"] == [2, 4]

    # A layer with a sliding window in the same step plans again, with the window.
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32)
    forward(torch.randn(2, 4, 8), cache, 1, cu_seqlens, softmax_scale=0.5, window=3)
    assert len(fake.plans) == 3
    assert fake.plans[2]["window_left"] == 2

    # A layout change between steps plans again, on the new index.
    cache.remove_request(0)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    forward(torch.randn(1, 4, 8), cache, 0, cu_seqlens, softmax_scale=0.5)
    assert len(fake.plans) == 4
    assert fake.plans[3]["indptr"] == [0, 1]
    assert fake.plans[3]["last_page_len"] == [4]


def test_static_step_plans_before_the_forward_on_a_fixed_buffer_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first run of a static step plans on the graph wrapper and records the
    parameters; from then on `plan_static_step` plans before the forward and the
    layers plan nothing. An eager step afterwards plans on the eager wrapper again."""
    from mini_infer.cache import flashinfer_backend
    from mini_infer.cache.paged_kv_cache import StaticStep

    eager_fake, graph_fake = _FakePrefillWrapper(), _FakePrefillWrapper()
    monkeypatch.setattr(flashinfer_backend, "_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(flashinfer_backend, "require_cuda_device", lambda device, what: None)
    monkeypatch.setattr(
        flashinfer_backend, "_ensure_wrappers", lambda device, kv_quant: (eager_fake, None)
    )
    monkeypatch.setattr(
        flashinfer_backend, "_graph_wrapper", lambda device, kv_quant, step: graph_fake
    )
    monkeypatch.setattr(flashinfer_backend, "_last_plan", None)
    forward = flashinfer_backend.flashinfer_attention_forward
    device = torch.device("cpu")

    num_layers = 3
    pool = BlockPool(
        num_blocks=16,
        block_size=4,
        num_layers=num_layers,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="flashinfer",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cu_prefill = torch.tensor([0, 3], dtype=torch.int32)
    for layer_idx in range(num_layers):
        kv = torch.randn(3, 2, 8)
        cache.append_kv_packed(kv, kv, cu_prefill, layer_idx)
    step = StaticStep.allocate(
        rows=2, slots=2, scratch_block=pool.allocate(), num_blocks=16, device=device
    )

    # First static run: no parameters yet, so the forward plans and records them.
    cache.begin_static_decode(step, active=1)
    flashinfer_backend.plan_static_step(cache, device)
    assert graph_fake.plans == []
    for layer_idx in range(num_layers):
        q = torch.randn(2, 4, 8)
        assert forward(q, cache, layer_idx, step.qo_indptr, softmax_scale=0.5) is q
    cache.end_static_step()
    assert len(graph_fake.plans) == 1 and eager_fake.plans == []
    assert graph_fake.plans[0]["qo_indptr"] == [0, 1, 2]
    assert graph_fake.plans[0]["indptr"] == [0, 1, 2]
    assert graph_fake.plans[0]["last_page_len"] == [4, 1]
    assert step.plan_params is not None and step.plan_params[:3] == (4, 2, 8)

    # Second static run: the plan happens before the forward, the layers reuse it.
    cache.begin_static_decode(step, active=1)
    flashinfer_backend.plan_static_step(cache, device)
    assert len(graph_fake.plans) == 2
    assert graph_fake.plans[1]["indptr"] == [0, 2, 3]
    assert graph_fake.plans[1]["last_page_len"] == [1, 1]
    # The pre-replay plan gets the pinned host views for the boundaries, the
    # device buffer for the indices, and asks for non-blocking copies.
    planned = graph_fake.plans[1]["tensors"]
    assert isinstance(planned, tuple)
    assert planned[0] is step.host_qo_indptr
    assert planned[1] is step.host_indptr
    assert planned[2] is step.indices
    assert planned[3] is step.host_last_page_len
    assert graph_fake.plans[1]["non_blocking"] is True
    for layer_idx in range(num_layers):
        forward(torch.randn(2, 4, 8), cache, layer_idx, step.qo_indptr, softmax_scale=0.5)
    cache.end_static_step()
    assert len(graph_fake.plans) == 2 and graph_fake.runs == 2 * num_layers

    # An eager step afterwards plans once on the eager wrapper.
    cu_decode = torch.tensor([0, 1], dtype=torch.int32)
    for layer_idx in range(num_layers):
        kv = torch.randn(1, 2, 8)
        cache.append_kv_packed(kv, kv, cu_decode, layer_idx)
        forward(torch.randn(1, 4, 8), cache, layer_idx, cu_decode, softmax_scale=0.5)
    assert len(eager_fake.plans) == 1 and len(graph_fake.plans) == 2
    assert eager_fake.plans[0]["indptr"] == [0, 2]
