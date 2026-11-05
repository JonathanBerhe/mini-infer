"""The attention backend that really runs, and the report line that shows it.

The packed attention dispatcher falls back to a slower backend without an
error when FlashInfer or flash-attn is missing. These tests pin the resolution
rule and the report line that Modal scripts print, on CPU.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mini_infer.cache import flashinfer_backend, packed_attention
from mini_infer.cache.packed_attention import effective_attention_backend
from mini_infer.fast_paths import fast_path_report, fast_paths


@pytest.fixture
def cuda_libraries(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Both attention libraries present on a CUDA device, without a GPU."""
    monkeypatch.setattr(flashinfer_backend, "_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(packed_attention, "_FLASH_ATTN_AVAILABLE", True)
    return monkeypatch


def test_torch_request_runs_torch(cuda_libraries: pytest.MonkeyPatch) -> None:
    assert effective_attention_backend("torch", "cuda") == "torch"


def test_requests_run_as_asked_when_the_libraries_are_present(
    cuda_libraries: pytest.MonkeyPatch,
) -> None:
    assert effective_attention_backend("flashinfer", "cuda") == "flashinfer"
    assert effective_attention_backend("flashinfer", "cuda", kv_quant="fp8") == "flashinfer"
    assert effective_attention_backend("flash_attn", "cuda") == "flash_attn"


def test_flashinfer_without_flashinfer_runs_flash_attn(
    cuda_libraries: pytest.MonkeyPatch,
) -> None:
    cuda_libraries.setattr(flashinfer_backend, "_FLASHINFER_AVAILABLE", False)
    assert effective_attention_backend("flashinfer", "cuda") == "flash_attn"


def test_flashinfer_with_a_turbo_cache_runs_flash_attn(
    cuda_libraries: pytest.MonkeyPatch,
) -> None:
    assert effective_attention_backend("flashinfer", "cuda", kv_quant="turbo3") == "flash_attn"


def test_flash_attn_without_flash_attn_runs_torch(cuda_libraries: pytest.MonkeyPatch) -> None:
    cuda_libraries.setattr(packed_attention, "_FLASH_ATTN_AVAILABLE", False)
    assert effective_attention_backend("flash_attn", "cuda") == "torch"
    cuda_libraries.setattr(flashinfer_backend, "_FLASHINFER_AVAILABLE", False)
    assert effective_attention_backend("flashinfer", "cuda") == "torch"


def test_cpu_runs_torch(cuda_libraries: pytest.MonkeyPatch) -> None:
    assert effective_attention_backend("flashinfer", "cpu") == "torch"
    assert effective_attention_backend("flash_attn", "cpu") == "torch"


def test_report_line_shows_a_fallback() -> None:
    paths = fast_paths("flash_attn")
    assert list(paths) == [
        "gpu",
        "cc",
        "torch",
        "triton",
        "flashinfer",
        "flash_attn",
        "gluon",
        "attention",
    ]
    if paths["gpu"] == "none":
        # On CPU every request falls back to the torch reference path.
        assert paths["attention"] == "flash_attn->torch"
        assert paths["gluon"] == "False"
    assert fast_path_report("torch").endswith("attention=torch")


def test_server_passes_the_attention_backend_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    from mini_infer.api import server

    seen: dict[str, Any] = {}

    def fake_from_pretrained(model_name: str, **kwargs: Any) -> object:
        seen.update(kwargs)
        return object()

    class FakeScheduler:
        def __init__(self, runner: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(server, "architecture_uses_state_cache", lambda name: False)
    monkeypatch.setattr(server.ModelRunner, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(server, "ContinuousScheduler", FakeScheduler)

    async def enter_lifespan() -> None:
        async with server.lifespan(server.app):
            pass

    monkeypatch.delenv("MINI_INFER_ATTENTION_BACKEND", raising=False)
    asyncio.run(enter_lifespan())
    assert seen["attention_backend"] == "flash_attn"

    monkeypatch.setenv("MINI_INFER_ATTENTION_BACKEND", "flashinfer")
    asyncio.run(enter_lifespan())
    assert seen["attention_backend"] == "flashinfer"


def test_paged_graph_refuses_a_flashinfer_pool_without_flashinfer(
    cuda_libraries: pytest.MonkeyPatch,
) -> None:
    import torch

    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.engine.paged_decode_graph import paged_decode_graph_blocker

    pool = BlockPool(
        num_blocks=8,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="flashinfer",
    )
    assert paged_decode_graph_blocker(pool, "cuda") is None
    cuda_libraries.setattr(flashinfer_backend, "_FLASHINFER_AVAILABLE", False)
    assert "cannot import" in str(paged_decode_graph_blocker(pool, "cuda"))
