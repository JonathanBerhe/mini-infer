"""Per-accelerator backend packages (CUDA/Triton, TPU/Pallas, Trainium/NKI).

This package is the home for accelerator-specific kernels that cannot live
in the vendor-agnostic core. It exists to enforce the isolation rule from
ADR-023 (docs/decisions/ADR-023-cross-accelerator-scope.md): backend code
lives here and never leaks into the scheduler, cache, API, or model layers.
A new architecture must run on the CPU/MPS reference path without importing
anything from this package.

Each subpackage guards its accelerator SDK behind an optional dependency
(for example `jax` for `tpu`), so importing `mini_infer` on a machine that
lacks that SDK must never fail. Subpackages expose a `supports_X(device)`
predicate (mirroring the `supports_fused_kernel` / `supports_flashinfer_backend`
pattern in the CUDA modules) that a dispatcher checks before routing work to
a hand-written kernel.

Nothing is imported eagerly here. Pulling in `mini_infer.backends` must stay
free of heavy SDK imports; callers reach into a specific subpackage
(`mini_infer.backends.tpu`) only when they intend to use that accelerator.
"""

from __future__ import annotations

__all__: list[str] = []
