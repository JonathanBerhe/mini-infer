"""Selection and routing for the TPU attention backend (ADR-023).

Two jobs: decide whether the Pallas TPU kernels should run on this host, and
route an attention call to the dense (`pallas_attention`) or paged
(`pallas_paged_attention`) kernel.

This is the seam a future engine integration would call. Today mini-infer's
runner is PyTorch on CUDA/CPU/MPS, so no real request is routed here yet: wiring
a JAX execution path into the runner is a separate, larger change that must keep
to ADR-023's backend-isolation rule (backend code stays under `backends/`, never
in the scheduler, cache, API, or model layers). The dispatcher plus its golden
parity test (tests/unit/test_tpu_attention_golden.py) exist so the TPU path is
held to the same numerical bar as the CUDA path: it must match a PyTorch
reference at temperature 0 (deterministic forward), the same ground truth the
project's golden tests rest on.

Import-safe without JAX, like the kernel modules: importing this module never
fails; `tpu_backend_available()` returns False and `dispatch_attention` raises a
clear error when JAX is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .pallas_attention import pallas_attention, supports_pallas_attention
from .pallas_paged_attention import (
    pallas_paged_attention,
    supports_pallas_paged_attention,
)

try:
    import jax

    _JAX_AVAILABLE = True
except ImportError:
    jax = None
    _JAX_AVAILABLE = False

if TYPE_CHECKING:
    from jax import Array


def tpu_backend_available(*, require_device: bool = False) -> bool:
    """Whether the TPU attention backend can run on this host.

    With ``require_device`` False (default), returns True when JAX imported: the
    kernels run in interpret mode on CPU, which is enough for parity testing.
    With ``require_device`` True, also requires a physical TPU device to be
    present, the gate for real hardware execution (as opposed to interpret mode).

    Distinct from the per-kernel ``supports_*`` predicates, which only report
    whether JAX imported; this is the backend-level, optionally device-aware gate
    an engine would consult before routing work to the TPU path.
    """
    if not (_JAX_AVAILABLE and supports_pallas_attention() and supports_pallas_paged_attention()):
        return False
    if require_device:
        try:
            return len(jax.devices("tpu")) > 0
        except RuntimeError:
            # jax.devices raises if no backend of that kind is configured.
            return False
    return True


def dispatch_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    block_tables: Any = None,
    lengths: Any = None,
    scale: float | None = None,
    causal: bool = False,
    interpret: bool = False,
) -> Array:
    """Route an attention call to the paged or dense Pallas TPU kernel.

    If ``block_tables`` is given, the paged decode kernel is used: ``k`` and
    ``v`` are the page pools ``(num_pages, page_size, num_kv_heads, head_dim)``
    and ``block_tables`` / ``lengths`` select each sequence's pages and context
    length. Otherwise the dense kernel is used: ``k`` and ``v`` are contiguous
    ``(heads, seq, head_dim)`` (or 4D with a batch axis).

    ``causal`` applies only to the dense path; the paged decode path has a single
    query per sequence that attends over all of its cached (past) context, so it
    is causal by construction and ``causal`` is ignored there.

    Raises:
        RuntimeError: if the TPU backend is unavailable (JAX not installed).
        ValueError: if ``block_tables`` is given without ``lengths``.
    """
    if not tpu_backend_available():
        raise RuntimeError("TPU backend unavailable; install the 'tpu' extra (uv sync --extra tpu)")
    if block_tables is not None:
        if lengths is None:
            raise ValueError("paged attention requires both block_tables and lengths")
        return pallas_paged_attention(
            q, k, v, block_tables, lengths, scale=scale, interpret=interpret
        )
    return pallas_attention(q, k, v, scale=scale, causal=causal, interpret=interpret)
