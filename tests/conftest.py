"""Test-collection hooks + auto-cleanup for model-loading tests.

`requires_cuda` skip: Tests marked `@pytest.mark.requires_cuda` use Triton
kernels or CUDA-specific flash-attn paths that don't run on CPU or MPS.
Without this hook every such test would fail at the first CUDA call rather
than skipping cleanly.

`requires_model` cleanup: Tests marked `@pytest.mark.requires_model` each
load a HF checkpoint (~0.1-2 GB) onto the device. Across the suite we load
SmolLM2 + Gemma 3 + Qwen2.5 + Qwen3 + int8/turbo variants — accumulated
MPS allocation on M1 (16 GB unified) gets close enough to saturation that
late tests have observed silently-zero weight tensors after `from_pretrained`
returns. Forcing `mps.empty_cache()` + a `gc.collect()` between requires_model
tests keeps the device's allocator from fragmenting catastrophically.
"""

import gc
from collections.abc import Iterator

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="requires CUDA; not available on this host")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(autouse=True)
def _cleanup_after_model_test(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if "requires_model" not in request.keywords:
        return
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
