"""Row-wise (last-axis) softmax as a Pallas TPU kernel.

This is the M0 scaffold kernel for the TPU backend (ADR-023): the smallest
non-trivial Pallas kernel that exercises the whole path (grid, BlockSpec
tiling, VMEM refs, a numerically-stable reduction) without yet being the real
paged-attention kernel. Softmax is a good first target because it is the
reduction at the heart of attention, so the numerics and the tiling mental
model carry straight over to M1 (dense attention).

Numerically it computes the standard stable softmax over the last axis of a
2D `(rows, cols)` array:

    m       = max_j x[i, j]
    e[i, j] = exp(x[i, j] - m)
    out     = e[i, j] / sum_j e[i, j]

Subtracting the row max before `exp` is the usual guard against overflow; it
is mandatory here for the same reason `torch.softmax` does it internally.

TPU mental model (and how it differs from CUDA SIMT)
----------------------------------------------------
A CUDA kernel is written from the point of view of one thread out of many:
thousands of scalar threads run the same code (SIMT), you index global memory
by `pid * BLOCK + tid`, and you hand-manage shared memory and `__syncthreads`.
Pallas-on-TPU is different in three ways that shape this kernel:

- VMEM scratchpad, not registers/shared. A TPU core has a small on-chip
  scratchpad (VMEM). Pallas stages each block of the input into VMEM and hands
  the kernel body `Ref`s into that scratchpad. We read the whole block with
  `x_ref[...]`, compute on it as a normal `jnp` array, and write the result
  with `out_ref[...] = ...`. There is no per-thread indexing: the kernel body
  operates on the entire block at once, and Mosaic (the TPU compiler backend)
  vectorizes it onto the hardware. This is closer to "array program on a tile"
  than "one thread's slice".

- 8x128 lane alignment. The TPU's vector unit and MXU work on 2D tiles whose
  last dimension is 128 lanes wide and second-to-last is 8 sublanes. Shapes
  that are multiples of (8, 128) map cleanly onto the hardware; ragged
  trailing tiles get padded, wasting lanes. We therefore tile along ROWS only
  and keep every column of a row inside one block, so the last-axis reduction
  never has to cross a block boundary (a cross-block reduction on TPU would
  need a second pass or scratch accumulation). Row-block sizing toward a
  multiple of 8 keeps the sublane dimension aligned. This 8x128 discipline has
  no analogue in CUDA SIMT, where the warp size is 32 and there is no
  compiler-enforced 2D tile shape.

- Sequential lexicographic grid. Unlike a CUDA grid of blocks scheduled
  concurrently across SMs, a Pallas TPU grid iterates sequentially in row-major
  (lexicographic) order over the grid axes, reusing the same core. Each grid
  step `i = pl.program_id(0)` maps, via the BlockSpec `index_map`, to the i-th
  row-block of the input and output. Because our reduction is fully contained
  within a block, the steps are independent and the sequential order does not
  matter for correctness here; for kernels with carried state (for example
  a flash-attention running max/sum) the lexicographic order is exactly the
  schedule you reason about.

JAX is optional and import-guarded (see the package docstring). On a machine
without JAX, `supports_pallas_softmax()` returns False and the kernel entry
point raises rather than crashing at import. With plain `jax` (no `jax[tpu]`),
call the entry point with `interpret=True` to run the kernel on CPU: that is
how the parity test validates it with no TPU hardware.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    _JAX_AVAILABLE = True
except ImportError:  # plain CPU / M1 / CI installs typically lack jax
    _JAX_AVAILABLE = False
    jax = None
    # `import jax.numpy as jnp` binds a submodule alias that mypy resolves even
    # under follow_imports=skip, so the None fallback needs a scoped ignore
    # (the plain `jax` / `pl` fallbacks do not). Matches the targeted per-line
    # ignores used in the Triton kernel modules.
    jnp = None  # type: ignore[assignment]
    pl = None

if TYPE_CHECKING:  # for type checkers only; never executed at runtime
    import jax as jax
    from jax import Array


# Default rows-per-block. Kept a multiple of 8 so the block's sublane
# dimension aligns with the TPU's 8x128 tile geometry (see module docstring).
# Small by design: this is a scaffold kernel, not a tuned production kernel.
_DEFAULT_ROW_BLOCK = 8


def supports_pallas_softmax() -> bool:
    """Whether the Pallas softmax kernel can run in this process.

    Today this only checks that JAX imported. It does NOT require a physical
    TPU: with plain `jax`, the kernel runs in interpret mode on CPU, which is
    enough for parity testing. A future revision can extend this to probe for
    a real TPU device when we want to gate hardware execution specifically.
    Mirrors the `supports_fused_kernel` / `supports_flashinfer_backend`
    predicate style in the CUDA modules.
    """
    return _JAX_AVAILABLE


if _JAX_AVAILABLE:

    def _softmax_row_kernel(x_ref: Any, out_ref: Any) -> None:
        """Kernel body: stable last-axis softmax over one row-block.

        `x_ref` and `out_ref` are VMEM `Ref`s into the current block, shaped
        `(row_block, cols)` as declared by the BlockSpecs. We read the whole
        block at once (no per-thread indexing, unlike CUDA SIMT), compute on
        it as an ordinary `jnp` array, and write the block back.

        `keepdims=True` on the reductions preserves the trailing axis so the
        subtract and divide broadcast cleanly across columns, and confines the
        reduction to the last axis (which lives entirely inside this block).
        """
        x = x_ref[...]
        row_max = jnp.max(x, axis=-1, keepdims=True)
        # Subtract the per-row max before exp: standard overflow guard, same
        # reason torch.softmax does it internally.
        shifted = x - row_max
        numerator = jnp.exp(shifted)
        denom = jnp.sum(numerator, axis=-1, keepdims=True)
        out_ref[...] = numerator / denom

    def pallas_softmax(x: Array, *, row_block: int | None = None, interpret: bool = False) -> Array:
        """Row-wise (last-axis) softmax of a 2D array via a Pallas TPU kernel.

        Args:
            x: 2D array `(rows, cols)`. The softmax is taken over `cols` (the
                last axis), matching `jax.nn.softmax(x, axis=-1)`.
            row_block: rows processed per grid step. Defaults to
                `_DEFAULT_ROW_BLOCK` (8, aligned to the TPU sublane dimension).
                Must divide `rows` evenly so every grid step sees a full block;
                pad `rows` externally otherwise. Each column of a row stays
                inside one block, so the last-axis reduction never crosses a
                block boundary.
            interpret: run the kernel in Pallas interpret mode on the host CPU
                instead of compiling for a TPU. Set True to develop and
                parity-test without TPU hardware (plain `jax` supports this);
                leave False on a real TPU host.

        Returns:
            An array the same shape and dtype as `x`, row-wise softmaxed.

        Raises:
            RuntimeError: if JAX is not installed.
            ValueError: if `x` is not 2D or `rows` is not divisible by the
                row-block size.
        """
        if not _JAX_AVAILABLE:
            raise RuntimeError(
                "JAX is not available; install the 'tpu' extra "
                "(uv sync --extra tpu) to use the Pallas softmax kernel"
            )
        if x.ndim != 2:
            raise ValueError(f"pallas_softmax expects a 2D array, got shape {x.shape}")

        rows, cols = x.shape
        block = row_block if row_block is not None else _DEFAULT_ROW_BLOCK
        if rows % block != 0:
            raise ValueError(
                f"rows={rows} must be divisible by row_block={block}; "
                "pad the input or pass a divisor as row_block"
            )

        num_row_blocks = rows // block

        # BlockSpec maps grid step i -> the i-th row-block. block_shape is the
        # VMEM tile handed to the kernel body; index_map returns the block's
        # (row, col) coordinates in units of blocks. Column block == full width
        # (cols), so a whole row lives in one block and the last-axis reduction
        # is local. The grid is 1D and iterates lexicographically over the row
        # blocks (sequential on one core, unlike a concurrent CUDA grid).
        block_spec = pl.BlockSpec((block, cols), lambda i: (i, 0))

        out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        result: Array = pl.pallas_call(
            _softmax_row_kernel,
            grid=(num_row_blocks,),
            in_specs=[block_spec],
            out_specs=block_spec,
            out_shape=out_shape,
            interpret=interpret,
        )(x)
        return result

else:

    def pallas_softmax(x: Any, *, row_block: int | None = None, interpret: bool = False) -> Any:
        """Fallback when JAX is absent: raise a clear error rather than crash on import.

        The real implementation is defined only when JAX imported. Keeping a
        stub here means `from mini_infer.backends.tpu.pallas_softmax import
        pallas_softmax` still works without JAX; calling it explains what to
        install. Callers should gate on `supports_pallas_softmax()` first.
        """
        raise RuntimeError(
            "JAX is not available; install the 'tpu' extra "
            "(uv sync --extra tpu) to use the Pallas softmax kernel"
        )
