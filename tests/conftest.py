"""Auto-skip CUDA-only tests on non-CUDA hosts.

Tests marked `@pytest.mark.requires_cuda` use Triton kernels or CUDA-specific
flash-attn paths that don't run on CPU or MPS. Without this hook every such
test would fail at the first CUDA call rather than skipping cleanly. Tests
that ALSO want to skip on systems without an actual GPU available (CI, M1)
need only the marker; this hook handles the runtime gate.
"""

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="requires CUDA; not available on this host")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_cuda)
