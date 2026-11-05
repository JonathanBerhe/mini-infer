"""FlashInfer attention over the contiguous K/V buffers of the state caches.

On CUDA with FlashInfer and no flash-attn, the full-attention layers of
Qwen3.8 and Nemotron-H view each row's buffer as whole pages and read only
the pages that hold its keys. This machine has no FlashInfer, so the model
tests swap in a PyTorch stand-in that follows FlashInfer's documented paged
semantics (NHD pages, CSR page index, bottom-right causal alignment,
grouped-query heads) and require the same logits as the masked SDPA path.
That pins what the models pass: the page index, the page view, the packing
and one plan per step. The real kernel runs in the GPU A/B of
`scripts/bench/modal_state_cache_attention_ab.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.nn import functional

from mini_infer.cache import contiguous_attention as contiguous
from mini_infer.cache.contiguous_attention import (
    PAGE_SIZE,
    contiguous_backend,
    padded_buffer_len,
    page_index,
)
from mini_infer.models.nemotron_h import nemotron_h
from mini_infer.models.nemotron_h.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from mini_infer.models.qwen import qwen3_5
from mini_infer.models.qwen.qwen3_5 import Qwen35Config, Qwen35ForConditionalGeneration

VOCAB = 64


def test_page_index_lists_each_rows_own_pages() -> None:
    indptr, indices, last = page_index([1, 16, 17, 40], pages_per_row=4)
    assert indptr == [0, 1, 2, 4, 7]
    assert indices.tolist() == [0, 4, 8, 9, 12, 13, 14]
    assert last == [1, 16, 1, 8]
    assert indices.dtype == torch.int32


@pytest.mark.parametrize("length", [0, 4 * PAGE_SIZE + 1])
def test_page_index_rejects_a_length_outside_the_row(length: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        page_index([length], pages_per_row=4)


def test_buffers_round_up_to_whole_pages_and_keep_their_capacity() -> None:
    assert padded_buffer_len(1) == PAGE_SIZE
    assert padded_buffer_len(PAGE_SIZE) == PAGE_SIZE
    assert padded_buffer_len(100) == 112
    cache = _qwen().build_state_cache(max_seq_len=100, batch_size=2)
    attention = cache.layer(1)
    assert attention.k.shape[1] == 112
    cache.check_capacity(100)
    with pytest.raises(ValueError, match="exceeds"):
        cache.check_capacity(101)


def test_the_backend_is_off_on_the_cpu_and_in_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    assert contiguous_backend(torch.zeros(1, 16, 1, 1, dtype=torch.bfloat16)) is None
    monkeypatch.setattr(contiguous, "is_cuda_device", lambda device: True)
    monkeypatch.setattr(contiguous, "_FLASH_ATTN_AVAILABLE", False)
    monkeypatch.setattr(contiguous, "_FLASHINFER_AVAILABLE", True)
    assert contiguous_backend(torch.zeros(1, 16, 1, 1, dtype=torch.float32)) is None
    assert contiguous_backend(torch.zeros(1, 16, 1, 1, dtype=torch.bfloat16)) == "flashinfer"
    # A buffer that is not whole pages cannot be viewed as pages.
    assert contiguous_backend(torch.zeros(1, 17, 1, 1, dtype=torch.bfloat16)) is None
    monkeypatch.setattr(contiguous, "_FLASH_ATTN_AVAILABLE", True)
    assert contiguous_backend(torch.zeros(1, 16, 1, 1, dtype=torch.bfloat16)) == "flash_attn"
    monkeypatch.setattr(contiguous, "_DISABLED_FOR_BENCH", True)
    assert contiguous_backend(torch.zeros(1, 16, 1, 1, dtype=torch.bfloat16)) is None


class _FakeWrapper:
    """`flashinfer.BatchPrefillWithPagedKVCacheWrapper` in PyTorch, NHD layout."""

    def __init__(self, workspace: torch.Tensor, kv_layout: str, **kwargs: Any) -> None:
        assert kv_layout == "NHD"
        self.use_cuda_graph = bool(kwargs.get("use_cuda_graph", False))
        self.planned: list[list[int]] = []

    def plan(self, qo_indptr, kv_indptr, kv_indices, last_page_len, *args, **kwargs) -> None:
        page_size = args[3]  # after num_qo_heads, num_kv_heads, head_dim
        assert kwargs["causal"] and kwargs["pos_encoding_mode"] == "NONE"
        self.qo = qo_indptr.tolist()
        self.kv = kv_indptr.tolist()
        self.indices = kv_indices.tolist()
        self.last = last_page_len.tolist()
        self.page_size = page_size
        self.scale = kwargs["sm_scale"]
        lengths = [
            (self.kv[b + 1] - self.kv[b] - 1) * page_size + self.last[b]
            for b in range(len(self.last))
        ]
        self.planned.append(lengths)

    def run(self, q: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        k_pages, v_pages = kv
        kv_heads, head_dim = k_pages.shape[2], k_pages.shape[3]
        group = q.shape[1] // kv_heads
        out = torch.empty_like(q)
        for row in range(len(self.last)):
            pages = self.indices[self.kv[row] : self.kv[row + 1]]
            length = (len(pages) - 1) * self.page_size + self.last[row]
            keys = k_pages[pages].reshape(-1, kv_heads, head_dim)[:length]
            values = v_pages[pages].reshape(-1, kv_heads, head_dim)[:length]
            span_q = q[self.qo[row] : self.qo[row + 1]]
            span = span_q.shape[0]
            # Bottom-right alignment: the last query sits at the last key.
            allowed = torch.arange(length) <= torch.arange(span).view(span, 1) + (length - span)
            out[self.qo[row] : self.qo[row + 1]] = functional.scaled_dot_product_attention(
                span_q.transpose(0, 1),
                keys.repeat_interleave(group, dim=1).transpose(0, 1),
                values.repeat_interleave(group, dim=1).transpose(0, 1),
                attn_mask=allowed,
                scale=self.scale,
            ).transpose(0, 1)
        return out


@pytest.fixture
def fake_flashinfer(monkeypatch: pytest.MonkeyPatch) -> list[_FakeWrapper]:
    """Route both models through the stand-in. Returns the wrappers it makes."""
    made: list[_FakeWrapper] = []

    def factory(*args: Any, **kwargs: Any) -> _FakeWrapper:
        made.append(_FakeWrapper(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(
        contiguous, "flashinfer", SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=factory)
    )
    monkeypatch.setattr(contiguous, "_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(contiguous, "_WORKSPACE_BYTES", 8)
    monkeypatch.setattr(contiguous, "_workspace", {})
    monkeypatch.setattr(contiguous, "_prefill_wrappers", {})
    monkeypatch.setattr(contiguous, "_decode_wrappers", {})
    for module in (contiguous, qwen3_5, nemotron_h):
        monkeypatch.setattr(module, "contiguous_backend", lambda k_buffer: "flashinfer")
    return made


def _qwen() -> Qwen35ForConditionalGeneration:
    torch.manual_seed(0)
    cfg = Qwen35Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=16,
        # Two attention layers, so that a shared plan is visible.
        layer_types=["linear_attention", "full_attention"] * 2,
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


def _nemotron() -> NemotronHForCausalLM:
    torch.manual_seed(0)
    cfg = NemotronHConfig.from_hf(
        SimpleNamespace(
            vocab_size=VOCAB,
            hidden_size=32,
            layers_block_type=["mamba", "attention", "moe", "attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            mamba_num_heads=4,
            mamba_head_dim=8,
            ssm_state_size=8,
            n_groups=2,
            conv_kernel=4,
            chunk_size=4,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            moe_shared_expert_intermediate_size=24,
            routed_scaling_factor=2.5,
            max_position_embeddings=128,
        )
    )
    model = NemotronHForCausalLM(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("dt_bias"):
                param.uniform_(0.5, 1.5)
            elif not name.endswith("A_log"):
                param.uniform_(-0.3, 0.3)
    return model.eval()


@torch.inference_mode()
def _serve(model: Any, ids: torch.Tensor, prefill: int) -> torch.Tensor:
    """Chunked prefill of `ids[:, :prefill]`, one decode step per later token,
    then one ragged step with the rows at different positions."""
    cache = model.build_state_cache(max_seq_len=100, batch_size=ids.shape[0])
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
    end = ids.shape[1]
    logits.append(
        model.forward_decode_with_cache_ragged(
            ids[:, :1], positions=torch.tensor([end, end - 21]), state_cache=cache
        )
    )
    return torch.cat(logits, dim=1)


@pytest.mark.parametrize("build", [_qwen, _nemotron], ids=["qwen3.8", "nemotron-h"])
def test_flashinfer_path_gives_the_sdpa_logits(
    build: Any, fake_flashinfer: list[_FakeWrapper], monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = torch.randint(0, VOCAB, (2, 48), generator=torch.Generator().manual_seed(1))
    with monkeypatch.context() as off:
        for module in (contiguous, qwen3_5, nemotron_h):
            off.setattr(module, "contiguous_backend", lambda k_buffer: None)
        expected = _serve(build(), ids, prefill=40)
    assert not fake_flashinfer

    got = _serve(build(), ids, prefill=40)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)
    prefill_wrapper, decode_wrapper = fake_flashinfer
    assert not prefill_wrapper.use_cuda_graph and decode_wrapper.use_cuda_graph
    # One plan per step, shared by both attention layers. The last prefill
    # chunk holds one token, so it runs on the one-token wrapper.
    assert prefill_wrapper.planned == [[13, 13], [26, 26], [39, 39]]
    assert decode_wrapper.planned == [[p, p] for p in range(40, 49)] + [[49, 28]]


def test_a_missing_plan_raises(fake_flashinfer: list[_FakeWrapper]) -> None:
    model = _qwen()
    cache = model.build_state_cache(max_seq_len=32, batch_size=1)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="plan the step first"):
        model.decode_step_static(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1).long(), cache)
