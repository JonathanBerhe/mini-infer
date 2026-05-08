"""Pytest fixtures available to all unit tests.

`reference_module` lazily imports the vendored DeepSeek-V4 inference
code with the tilelang kernel module replaced by PyTorch stubs. Used
by the V4 parity tests; harmless for other tests.
"""

from tests.unit._v4_reference_helpers import reference_module

__all__ = ["reference_module"]
