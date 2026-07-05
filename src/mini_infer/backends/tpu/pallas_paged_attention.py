"""Paged, ragged decode attention as a Pallas TPU kernel (M2 for the TPU backend).

This is the M2 kernel for the TPU backend (ADR-023), the step from the dense M1
attention (pallas_attention.py) toward the Ragged Paged Attention design
(arXiv 2604.15464). It computes one decode step of attention: a single query per
sequence attends over that sequence's key/value cache, where the cache does not
live in one contiguous buffer but in fixed-size PAGES scattered through a shared
pool, addressed per sequence by a BLOCK TABLE, and each sequence has its own
context LENGTH (the batch is ragged).

Why this is the load-bearing RPA piece
--------------------------------------
On a GPU, a data-dependent KV gather (read page block_table[s, p] from the pool)
is cheap: threads issue independent indexed loads. On a TPU the compiler wants
statically shaped, contiguous access, so a per-step page index that is only known
at runtime is the exact thing that does not lower well through plain XLA. Pallas
handles it with SCALAR PREFETCH: the block table and the lengths are prefetched
into SMEM and made available to each BlockSpec's `index_map`, so the physical
page fetched for grid step `(s, h, p)` is `block_tables[s, p]`, computed at run
time. That is the "fine-grained dynamic slicing over ragged memory" from the RPA
paper, expressed in the one place the TPU toolchain accepts it.

Online softmax and the sequential grid
--------------------------------------
The recurrence is the same FlashAttention online softmax as M1 (running max `m`,
denominator `l`, accumulator `acc` in VMEM scratch), and it relies on the same
TPU property: the grid runs sequentially in lexicographic order, so the page axis
is the INNER axis and successive grid steps are successive recurrence iterations
over one (sequence, head). See pallas_attention.py for the full derivation of the
recurrence; here the query is a single row and the KV block is one page.

Ragged masking
--------------
Grid step `(s, h, p)` covers absolute key positions `[p*page_size, (p+1)*page_size)`.
A position is valid only if it is `< lengths[s]`; the rest are set to a large
negative sentinel before exp, so a partial final page and any page slot past the
sequence's real pages contribute nothing. Slots past a sequence's pages still
gather some in-bounds dummy page (the caller sets them to a valid index); masking
makes their contribution zero, so the result is independent of the dummy.

Scope: this kernel supports grouped-query attention (num_kv_heads divides
num_heads; query head h reads kv head h // (num_heads // num_kv_heads)) and runs
in interpret mode. On-TPU execution is a follow-up (see ADR-023).

JAX is optional and import-guarded exactly as in pallas_softmax.py /
pallas_attention.py. With plain `jax` (no `jax[tpu]`), pass `interpret=True` to
run on CPU; that is how the parity test validates it with no TPU hardware.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    _JAX_AVAILABLE = True
except ImportError:  # plain CPU / M1 / CI installs typically lack jax
    _JAX_AVAILABLE = False
    jax = None
    # See pallas_attention.py: the jnp submodule alias needs a scoped ignore.
    jnp = None  # type: ignore[assignment]
    pl = None
    pltpu = None

if TYPE_CHECKING:  # for type checkers only; never executed at runtime
    from jax import Array

# Masked-out scores get a large finite negative sentinel (not -inf) so every
# intermediate stays finite in interpret mode, matching pallas_attention.py.
_MASK_NEG = -1e30


def supports_pallas_paged_attention() -> bool:
    """Whether the paged attention kernel can run in this process.

    Only checks that JAX imported; it does not require a physical TPU (plain
    `jax` runs the kernel in interpret mode on CPU). Mirrors
    `supports_pallas_attention` / `supports_pallas_softmax`.
    """
    return _JAX_AVAILABLE


if _JAX_AVAILABLE:

    def _paged_decode_launch(
        q: Array,
        k_pages: Array,
        v_pages: Array,
        block_tables: Array,
        lengths: Array,
        *,
        scale: float,
        interpret: bool,
    ) -> Array:
        """Launch the paged decode attention over canonical shapes.

        Shapes:
            q            (num_seqs, num_heads, head_dim)   one query per sequence
            k_pages      (num_pages, page_size, num_heads, head_dim)
            v_pages      same as k_pages
            block_tables (num_seqs, max_pages) int32       physical page per slot
            lengths      (num_seqs,) int32                 context length per seq
        """
        num_seqs, num_heads, head_dim = q.shape
        page_size = k_pages.shape[1]
        num_kv_heads = k_pages.shape[2]
        max_pages = block_tables.shape[1]
        # Grouped-query attention: each block of q_per_kv query heads shares one
        # kv head. q_per_kv == 1 is plain multi-head attention.
        q_per_kv = num_heads // num_kv_heads

        # Grid outer -> inner: (sequence, head, page). The page axis is INNER so
        # successive sequential grid steps are successive online-softmax
        # iterations over one (sequence, head); scratch is reset on the first
        # page and finalised on the last.
        grid = (num_seqs, num_heads, max_pages)

        def kernel(
            block_tables_ref: Any,
            lengths_ref: Any,
            q_ref: Any,
            k_ref: Any,
            v_ref: Any,
            out_ref: Any,
            m_scratch: Any,
            l_scratch: Any,
            acc_scratch: Any,
        ) -> None:
            del block_tables_ref  # used only by the index_maps, not the body
            seq_index = pl.program_id(0)
            page_index = pl.program_id(2)
            num_page_steps = pl.num_programs(2)

            @pl.when(page_index == 0)  # type: ignore[untyped-decorator]
            def _init() -> None:
                m_scratch[...] = jnp.full_like(m_scratch, _MASK_NEG)
                l_scratch[...] = jnp.zeros_like(l_scratch)
                acc_scratch[...] = jnp.zeros_like(acc_scratch)

            length = lengths_ref[seq_index]  # scalar context length for this seq
            # Blocks carry singleton sequence/head axes; reshape to 2D tiles so
            # the recurrence reads cleanly.
            q_vec = q_ref[...].reshape(1, head_dim)  # (1, head_dim)
            k = k_ref[...].reshape(page_size, head_dim)  # (page_size, head_dim)
            v = v_ref[...].reshape(page_size, head_dim)  # (page_size, head_dim)

            scores = jax.lax.dot_general(
                q_vec,
                k,
                (((1,), (1,)), ((), ())),  # contract head_dim
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale  # (1, page_size)

            # Ragged mask: absolute key positions for this page, valid < length.
            positions = page_index * page_size + jax.lax.broadcasted_iota(
                jnp.int32, (1, page_size), 1
            )
            scores = jnp.where(positions < length, scores, _MASK_NEG)

            # Online-softmax fold (single query row). Same recurrence as M1.
            m_prev = m_scratch[...]  # (1, 1)
            l_prev = l_scratch[...]  # (1, 1)
            acc_prev = acc_scratch[...]  # (1, head_dim)

            m_cur = jnp.max(scores, axis=-1, keepdims=True)  # (1, 1)
            m_new = jnp.maximum(m_prev, m_cur)
            p = jnp.exp(scores - m_new)  # (1, page_size)
            correction = jnp.exp(m_prev - m_new)  # (1, 1)
            l_new = correction * l_prev + jnp.sum(p, axis=-1, keepdims=True)
            acc_new = correction * acc_prev + jax.lax.dot_general(
                p,
                v,
                (((1,), (0,)), ((), ())),  # contract page_size
                preferred_element_type=jnp.float32,
            )

            m_scratch[...] = m_new
            l_scratch[...] = l_new
            acc_scratch[...] = acc_new

            @pl.when(page_index == num_page_steps - 1)  # type: ignore[untyped-decorator]
            def _finalize() -> None:
                out_ref[...] = (
                    (acc_scratch[...] / l_scratch[...]).reshape(out_ref.shape).astype(out_ref.dtype)
                )

        # index_maps receive the grid indices followed by the scalar-prefetch
        # arrays (block table, lengths). The KV specs read block_tables to fetch
        # the physical page for this (sequence, page): this is the paged gather.
        q_spec = pl.BlockSpec((1, 1, head_dim), lambda s, h, p, bt, ln: (s, h, 0))
        out_spec = pl.BlockSpec((1, 1, head_dim), lambda s, h, p, bt, ln: (s, h, 0))
        kv_block = (1, page_size, 1, head_dim)
        # The kv head for query head h is h // q_per_kv (grouped-query mapping).
        k_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln: (bt[s, p], 0, h // q_per_kv, 0))
        v_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln: (bt[s, p], 0, h // q_per_kv, 0))

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,  # block_tables, lengths
            grid=grid,
            in_specs=[q_spec, k_spec, v_spec],
            out_specs=out_spec,
            scratch_shapes=[
                pltpu.VMEM((1, 1), jnp.float32),
                pltpu.VMEM((1, 1), jnp.float32),
                pltpu.VMEM((1, head_dim), jnp.float32),
            ],
        )

        result: Array = pl.pallas_call(
            kernel,
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            interpret=interpret,
        )(block_tables, lengths, q, k_pages, v_pages)
        return result

    def pallas_paged_attention(
        q: Array,
        k_pages: Array,
        v_pages: Array,
        block_tables: Array,
        lengths: Array,
        *,
        scale: float | None = None,
        interpret: bool = False,
    ) -> Array:
        """Paged, ragged decode attention via a Pallas TPU kernel.

        One decode step: each sequence contributes a single query that attends
        over its own paged KV cache, up to its own context length.

        Args:
            q: queries, `(num_seqs, num_heads, head_dim)` (one token per seq).
            k_pages: key cache pool, `(num_pages, page_size, num_kv_heads, head_dim)`
                (num_kv_heads may be < num_heads for grouped-query attention).
            v_pages: value cache pool, same shape as `k_pages`.
            block_tables: `(num_seqs, max_pages)` int array; `block_tables[s, p]`
                is the physical page index in the pool for sequence `s`'s logical
                page `p`. Entries for slots past a sequence's real pages must
                still be valid in-bounds indices (they are fully masked out).
            lengths: `(num_seqs,)` int array of per-sequence context lengths.
            scale: softmax scale on `q @ k^T`. Defaults to `1/sqrt(head_dim)`.
            interpret: run in Pallas interpret mode on CPU (no TPU needed). Set
                True to parity-test with plain `jax`; leave False on a TPU host.

        Returns:
            Attention output, `(num_seqs, num_heads, head_dim)`, dtype of `q`.

        Raises:
            RuntimeError: if JAX is not installed.
            ValueError: on rank/shape mismatch, or if num_heads is not a multiple
                of num_kv_heads (grouped-query requires a clean head grouping).
        """
        if not _JAX_AVAILABLE:
            raise RuntimeError(
                "JAX is not available; install the 'tpu' extra "
                "(uv sync --extra tpu) to use the Pallas paged attention kernel"
            )
        if q.ndim != 3:
            raise ValueError(
                f"q must be (num_seqs, num_heads, head_dim) for decode; got rank {q.ndim}"
            )
        if k_pages.ndim != 4 or v_pages.shape != k_pages.shape:
            raise ValueError(
                "k_pages and v_pages must be (num_pages, page_size, num_heads, "
                f"head_dim) and identical; got {k_pages.shape} and {v_pages.shape}"
            )
        if block_tables.ndim != 2 or block_tables.shape[0] != q.shape[0]:
            raise ValueError(
                "block_tables must be (num_seqs, max_pages) matching q's num_seqs; "
                f"got {block_tables.shape} for num_seqs={q.shape[0]}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != q.shape[0]:
            raise ValueError(
                f"lengths must be (num_seqs,); got {lengths.shape} for num_seqs={q.shape[0]}"
            )

        num_heads, head_dim = q.shape[1], q.shape[2]
        num_kv_heads = k_pages.shape[2]
        if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                "num_heads must be a positive multiple of num_kv_heads for "
                f"grouped-query attention; got {num_heads} query heads and "
                f"{num_kv_heads} kv heads"
            )
        if k_pages.shape[3] != head_dim:
            raise ValueError(f"head_dim mismatch: q has {head_dim}, k_pages has {k_pages.shape[3]}")

        effective_scale = scale if scale is not None else 1.0 / (head_dim**0.5)
        # Scalar-prefetch inputs must be integer and live in SMEM; normalise dtype.
        block_tables = block_tables.astype(jnp.int32)
        lengths = lengths.astype(jnp.int32)

        return _paged_decode_launch(
            q,
            k_pages,
            v_pages,
            block_tables,
            lengths,
            scale=effective_scale,
            interpret=interpret,
        )

else:

    def pallas_paged_attention(
        q: Any,
        k_pages: Any,
        v_pages: Any,
        block_tables: Any,
        lengths: Any,
        *,
        scale: float | None = None,
        interpret: bool = False,
    ) -> Any:
        """Fallback when JAX is absent: raise a clear error rather than crash on import.

        Keeps the import working without JAX; callers should gate on
        `supports_pallas_paged_attention()` first.
        """
        raise RuntimeError(
            "JAX is not available; install the 'tpu' extra "
            "(uv sync --extra tpu) to use the Pallas paged attention kernel"
        )
