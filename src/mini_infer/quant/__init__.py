"""Weight quantization primitives for inference.

Currently exposes weight-only INT8 (W8A16): symmetric, per-output-channel
scales, no calibration. Activations stay in their original dtype.

Public API:
    Int8Linear              — drop-in replacement for nn.Linear with INT8 weights.
    quantize_per_channel    — pure tensor op: float weight -> (int8 weight, scales).
    dequantize_per_channel  — inverse, useful for tests and the naive forward.
    quantize_model_to_int8  — walks an nn.Module and replaces Linear in place.
"""

from mini_infer.quant.int8 import (
    Int8Linear,
    dequantize_per_channel,
    quantize_model_to_int8,
    quantize_per_channel,
)

__all__ = [
    "Int8Linear",
    "dequantize_per_channel",
    "quantize_model_to_int8",
    "quantize_per_channel",
]
