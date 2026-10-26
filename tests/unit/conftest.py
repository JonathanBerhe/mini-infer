"""Pytest fixtures available to all unit tests.

`reference_module` lazily imports the vendored DeepSeek-V4 inference
code with the tilelang kernel module replaced by PyTorch stubs. Used
by the V4 parity tests; harmless for other tests.

`kimi_reference` does the same for the vendored Kimi Linear remote code,
with the FLA kernel package replaced by its naive-reference semantics.
"""

from tests.unit.models.deepseek_v4._v4_reference_helpers import reference_module
from tests.unit.models.kimi_linear._kimi_reference_helpers import kimi_reference

__all__ = ["kimi_reference", "reference_module"]
