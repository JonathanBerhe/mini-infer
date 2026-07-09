"""JAX/Pallas TPU kernels for mini-infer.

This package holds hand-written TPU kernels expressed in JAX with the
`jax.experimental.pallas` DSL, which lowers through Mosaic to the TPU. It is
the TPU arm of the cross-accelerator scope decided in ADR-023
(docs/decisions/ADR-023-cross-accelerator-scope.md): backend-specific code is
isolated here and never leaks into the scheduler, cache, API, or model layers.

JAX is an OPTIONAL, import-guarded dependency. It lives in the `tpu` extra
(`uv sync --extra tpu` or `uv add --optional tpu 'jax>=0.4'`), NOT in the base
install. Importing `mini_infer` on a machine without JAX (a plain M1 / CPU box,
or CI) must never fail. To keep that guarantee, this `__init__` does NOT import
any submodule that touches JAX at package-import time. Reach into a specific
kernel module (for example `mini_infer.backends.tpu.pallas_softmax`) only when
you actually intend to run it; those modules guard `import jax` behind
try/except and expose a `supports_*` predicate that reports False when JAX is
absent, mirroring the CUDA modules' `supports_fused_kernel` /
`supports_flashinfer_backend` pattern.

Vendor path first (ADR-023 rule 2): most TPU work should ride XLA via plain
JAX ops. A Pallas kernel is warranted only where XLA cannot express the memory
pattern we need (the motivating case is paged attention, whose non-contiguous
KV gather fights XLA's static-shape compiler). Every hand-written kernel here
is held to the same parity bar as the CUDA kernels: cosine similarity > 0.99
against the reference path (ADR-023 rule 3).

Why plain `jax`, not `jax[tpu]`: plain JAX runs Pallas kernels in *interpret*
mode on CPU (`pallas_call(..., interpret=True)`), which is enough to develop
and parity-test the kernels with no TPU hardware. Real TPU execution needs
`jax[tpu]` on a TPU host, installed there separately.
"""

from __future__ import annotations

__all__: list[str] = []
