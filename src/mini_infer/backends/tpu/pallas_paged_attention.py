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

Scope: this module holds the paged kernel family: single-token decode, uniform
prefill / multi-query, and mixed prefill/decode batches. All support
grouped-query attention (num_kv_heads divides num_heads; query head h reads kv
head h // (num_heads // num_kv_heads)) and run in interpret mode on CPU.
On-TPU validation goes through scripts/run_tpu_pallas_kernels.py (results in
docs/benchmarks/, design in ADR-024).

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
            k_pages      (num_kv_heads, num_pages, page_size, head_dim)
            v_pages      same as k_pages
            block_tables (num_seqs, max_pages) int32       physical page per slot
            lengths      (num_seqs,) int32                 context length per seq

        The pools are HEADS-FIRST. Real-TPU lowering requires each block's last
        two dims to be divisible by (8, 128) or equal the array dims (a rule
        interpret mode does not check; we hit it on hardware). Heads-first makes
        every KV block exactly (page_size, head_dim) == the array's last two
        dims, legal for any page size; JAX's production paged kernel makes the
        same choice. q gains a singleton axis for the same reason.
        """
        num_seqs, num_heads, head_dim = q.shape
        num_kv_heads, _num_pages, page_size, _ = k_pages.shape
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
        # q carries a singleton third axis so its block's last two dims equal
        # the array's (the TPU tiling rule above).
        q_spec = pl.BlockSpec((1, 1, 1, head_dim), lambda s, h, p, bt, ln: (s, h, 0, 0))
        out_spec = pl.BlockSpec((1, 1, 1, head_dim), lambda s, h, p, bt, ln: (s, h, 0, 0))
        kv_block = (1, 1, page_size, head_dim)
        # The kv head for query head h is h // q_per_kv (grouped-query mapping).
        k_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln: (h // q_per_kv, bt[s, p], 0, 0))
        v_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln: (h // q_per_kv, bt[s, p], 0, 0))

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

        q4 = q[:, :, None, :]  # singleton axis so the block equals the array's last two dims
        result: Array = pl.pallas_call(
            kernel,
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct(q4.shape, q.dtype),
            interpret=interpret,
        )(block_tables, lengths, q4, k_pages, v_pages)
        return result[:, :, 0, :]

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
            k_pages: key cache pool, heads-first
                `(num_kv_heads, num_pages, page_size, head_dim)` (num_kv_heads may
                be < num_heads for grouped-query attention; heads-first keeps the
                TPU blocks tile-legal, see `_paged_decode_launch`).
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
                "k_pages and v_pages must be heads-first (num_kv_heads, num_pages, "
                f"page_size, head_dim) and identical; got {k_pages.shape} and {v_pages.shape}"
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
        num_kv_heads = k_pages.shape[0]
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

    def _paged_prefill_launch(
        q: Array,
        k_pages: Array,
        v_pages: Array,
        block_tables: Array,
        lengths: Array,
        *,
        scale: float,
        interpret: bool,
    ) -> Array:
        """Launch paged prefill / multi-query attention.

        Like the decode launch but each sequence has q_len query tokens
        (q shape (num_seqs, q_len, num_heads, head_dim)), placed at the last
        q_len absolute positions of the context, attending causally.
        """
        num_seqs, q_len, num_heads, head_dim = q.shape
        num_kv_heads, _num_pages, page_size, _ = k_pages.shape
        max_pages = block_tables.shape[1]
        q_per_kv = num_heads // num_kv_heads

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
            del block_tables_ref  # used only by the index_maps
            seq_index = pl.program_id(0)
            page_index = pl.program_id(2)
            num_page_steps = pl.num_programs(2)

            @pl.when(page_index == 0)  # type: ignore[untyped-decorator]
            def _init() -> None:
                m_scratch[...] = jnp.full_like(m_scratch, _MASK_NEG)
                l_scratch[...] = jnp.zeros_like(l_scratch)
                acc_scratch[...] = jnp.zeros_like(acc_scratch)

            length = lengths_ref[seq_index]
            q_mat = q_ref[...].reshape(q_len, head_dim)  # (q_len, head_dim)
            k = k_ref[...].reshape(page_size, head_dim)  # (page_size, head_dim)
            v = v_ref[...].reshape(page_size, head_dim)  # (page_size, head_dim)

            scores = jax.lax.dot_general(
                q_mat,
                k,
                (((1,), (1,)), ((), ())),  # contract head_dim
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale  # (q_len, page_size)

            # Causal-by-absolute-position mask. Query row t sits at absolute
            # position (length - q_len + t) and may attend to key j only if
            # j <= that position. This subsumes the ragged length mask (keys past
            # the context sit beyond every query) and reduces to the decode mask
            # when q_len == 1.
            q_pos = (length - q_len) + jax.lax.broadcasted_iota(jnp.int32, (q_len, page_size), 0)
            k_pos = page_index * page_size + jax.lax.broadcasted_iota(
                jnp.int32, (q_len, page_size), 1
            )
            scores = jnp.where(k_pos <= q_pos, scores, _MASK_NEG)

            m_prev = m_scratch[...]  # (q_len, 1)
            l_prev = l_scratch[...]  # (q_len, 1)
            acc_prev = acc_scratch[...]  # (q_len, head_dim)

            m_cur = jnp.max(scores, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_prev, m_cur)
            p = jnp.exp(scores - m_new)
            correction = jnp.exp(m_prev - m_new)
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

        # q is carried transposed as (num_seqs, num_heads, q_len, head_dim) so the
        # block's last two dims (q_len, head_dim) equal the array's; the pools are
        # heads-first for the same TPU tiling rule (see the decode launch).
        q_spec = pl.BlockSpec((1, 1, q_len, head_dim), lambda s, h, p, bt, ln: (s, h, 0, 0))
        out_spec = pl.BlockSpec((1, 1, q_len, head_dim), lambda s, h, p, bt, ln: (s, h, 0, 0))
        kv_block = (1, 1, page_size, head_dim)
        k_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln: (h // q_per_kv, bt[s, p], 0, 0))
        v_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln: (h // q_per_kv, bt[s, p], 0, 0))

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=grid,
            in_specs=[q_spec, k_spec, v_spec],
            out_specs=out_spec,
            scratch_shapes=[
                pltpu.VMEM((q_len, 1), jnp.float32),
                pltpu.VMEM((q_len, 1), jnp.float32),
                pltpu.VMEM((q_len, head_dim), jnp.float32),
            ],
        )

        qt = jnp.swapaxes(q, 1, 2)  # (num_seqs, num_heads, q_len, head_dim)
        result: Array = pl.pallas_call(
            kernel,
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct(qt.shape, q.dtype),
            interpret=interpret,
        )(block_tables, lengths, qt, k_pages, v_pages)
        return jnp.swapaxes(result, 1, 2)

    def pallas_paged_prefill_attention(
        q: Array,
        k_pages: Array,
        v_pages: Array,
        block_tables: Array,
        lengths: Array,
        *,
        scale: float | None = None,
        interpret: bool = False,
    ) -> Array:
        """Paged prefill / multi-query attention via a Pallas TPU kernel.

        Generalises the decode kernel to q_len >= 1 query tokens per sequence:
        the case for prefill (a chunk of new tokens) and speculative decode (a
        small candidate window). Each sequence's q_len queries occupy the last
        q_len absolute positions of its context and attend causally: query token
        t attends to keys at positions j <= (length - q_len + t). Separate from
        decode, matching RPA's specialized-per-workload compilation.

        Args:
            q: queries, `(num_seqs, q_len, num_heads, head_dim)`.
            k_pages: key cache pool, heads-first
                `(num_kv_heads, num_pages, page_size, head_dim)`.
            v_pages: value cache pool, same shape as `k_pages`.
            block_tables: `(num_seqs, max_pages)` int page table (see the decode kernel).
            lengths: `(num_seqs,)` int total context length per sequence,
                including the q_len query tokens themselves.
            scale: softmax scale. Defaults to `1/sqrt(head_dim)`.
            interpret: run in Pallas interpret mode on CPU (no TPU needed).

        Returns:
            Attention output, `(num_seqs, q_len, num_heads, head_dim)`, dtype of q.

        Raises:
            RuntimeError: if JAX is not installed.
            ValueError: on rank/shape mismatch or an unclean head grouping.
        """
        if not _JAX_AVAILABLE:
            raise RuntimeError(
                "JAX is not available; install the 'tpu' extra "
                "(uv sync --extra tpu) to use the Pallas paged attention kernel"
            )
        if q.ndim != 4:
            raise ValueError(
                f"q must be (num_seqs, q_len, num_heads, head_dim) for prefill; got rank {q.ndim}"
            )
        if k_pages.ndim != 4 or v_pages.shape != k_pages.shape:
            raise ValueError(
                "k_pages and v_pages must be heads-first (num_kv_heads, num_pages, "
                f"page_size, head_dim) and identical; got {k_pages.shape} and {v_pages.shape}"
            )
        num_seqs = q.shape[0]
        if block_tables.ndim != 2 or block_tables.shape[0] != num_seqs:
            raise ValueError(
                "block_tables must be (num_seqs, max_pages) matching q's num_seqs; "
                f"got {block_tables.shape} for num_seqs={num_seqs}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != num_seqs:
            raise ValueError(
                f"lengths must be (num_seqs,); got {lengths.shape} for num_seqs={num_seqs}"
            )

        num_heads, head_dim = q.shape[2], q.shape[3]
        num_kv_heads = k_pages.shape[0]
        if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                "num_heads must be a positive multiple of num_kv_heads for "
                f"grouped-query attention; got {num_heads} query heads and "
                f"{num_kv_heads} kv heads"
            )
        if k_pages.shape[3] != head_dim:
            raise ValueError(f"head_dim mismatch: q has {head_dim}, k_pages has {k_pages.shape[3]}")

        effective_scale = scale if scale is not None else 1.0 / (head_dim**0.5)
        block_tables = block_tables.astype(jnp.int32)
        lengths = lengths.astype(jnp.int32)

        return _paged_prefill_launch(
            q,
            k_pages,
            v_pages,
            block_tables,
            lengths,
            scale=effective_scale,
            interpret=interpret,
        )

    def _paged_mixed_launch(
        q: Array,
        k_pages: Array,
        v_pages: Array,
        block_tables: Array,
        lengths: Array,
        q_lens: Array,
        *,
        scale: float,
        interpret: bool,
    ) -> Array:
        """Launch paged attention over a mixed prefill/decode batch.

        Like the prefill launch, but the number of real query tokens is a
        PER-SEQUENCE runtime value (q_lens, scalar-prefetched next to the block
        table): row t of sequence s is real only if t < q_lens[s]. Real rows sit
        at the last q_lens[s] absolute positions of the context and attend
        causally; padding rows have no visible keys and finalize to exact zeros.
        """
        num_seqs, max_q_len, num_heads, head_dim = q.shape
        num_kv_heads, _num_pages, page_size, _ = k_pages.shape
        max_pages = block_tables.shape[1]
        q_per_kv = num_heads // num_kv_heads

        grid = (num_seqs, num_heads, max_pages)

        def kernel(
            block_tables_ref: Any,
            lengths_ref: Any,
            q_lens_ref: Any,
            q_ref: Any,
            k_ref: Any,
            v_ref: Any,
            out_ref: Any,
            m_scratch: Any,
            l_scratch: Any,
            acc_scratch: Any,
        ) -> None:
            del block_tables_ref  # used only by the index_maps
            seq_index = pl.program_id(0)
            page_index = pl.program_id(2)
            num_page_steps = pl.num_programs(2)

            @pl.when(page_index == 0)  # type: ignore[untyped-decorator]
            def _init() -> None:
                m_scratch[...] = jnp.full_like(m_scratch, _MASK_NEG)
                l_scratch[...] = jnp.zeros_like(l_scratch)
                acc_scratch[...] = jnp.zeros_like(acc_scratch)

            length = lengths_ref[seq_index]
            q_len_s = q_lens_ref[seq_index]  # real query rows for this sequence
            q_mat = q_ref[...].reshape(max_q_len, head_dim)
            k = k_ref[...].reshape(page_size, head_dim)
            v = v_ref[...].reshape(page_size, head_dim)

            scores = jax.lax.dot_general(
                q_mat,
                k,
                (((1,), (1,)), ((), ())),  # contract head_dim
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale  # (max_q_len, page_size)

            # The prefill kernel's causal-by-absolute-position mask, with q_len
            # now a per-sequence runtime scalar, plus a row-validity term:
            # padding rows (t >= q_lens[s]) see no keys at all.
            row_ids = jax.lax.broadcasted_iota(jnp.int32, (max_q_len, page_size), 0)
            q_pos = (length - q_len_s) + row_ids
            k_pos = page_index * page_size + jax.lax.broadcasted_iota(
                jnp.int32, (max_q_len, page_size), 1
            )
            valid = (k_pos <= q_pos) & (row_ids < q_len_s)
            scores = jnp.where(valid, scores, _MASK_NEG)

            m_prev = m_scratch[...]  # (max_q_len, 1)
            l_prev = l_scratch[...]  # (max_q_len, 1)
            acc_prev = acc_scratch[...]  # (max_q_len, head_dim)

            m_cur = jnp.max(scores, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_prev, m_cur)
            p = jnp.exp(scores - m_new)
            # A fully-masked row keeps its running max at the sentinel, so
            # exp(score - max) is exp(0) == 1 at every masked position; without
            # this zeroing, padding rows would silently average the gathered
            # (meaningless) values instead of staying empty.
            p = jnp.where(valid, p, 0.0)
            correction = jnp.exp(m_prev - m_new)
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
                l_final = l_scratch[...]
                # Padding rows accumulated nothing (l == 0); divide those by 1
                # so they finalize to exact zeros instead of 0/0.
                safe_l = jnp.where(l_final > 0.0, l_final, 1.0)
                out_ref[...] = (
                    (acc_scratch[...] / safe_l).reshape(out_ref.shape).astype(out_ref.dtype)
                )

        # Same layouts as the prefill launch (q carried transposed, pools
        # heads-first, for the TPU tiling rule); the index_maps gain the q_lens
        # prefetch argument.
        q_spec = pl.BlockSpec((1, 1, max_q_len, head_dim), lambda s, h, p, bt, ln, ql: (s, h, 0, 0))
        out_spec = pl.BlockSpec(
            (1, 1, max_q_len, head_dim), lambda s, h, p, bt, ln, ql: (s, h, 0, 0)
        )
        kv_block = (1, 1, page_size, head_dim)
        k_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln, ql: (h // q_per_kv, bt[s, p], 0, 0))
        v_spec = pl.BlockSpec(kv_block, lambda s, h, p, bt, ln, ql: (h // q_per_kv, bt[s, p], 0, 0))

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,  # block_tables, lengths, q_lens
            grid=grid,
            in_specs=[q_spec, k_spec, v_spec],
            out_specs=out_spec,
            scratch_shapes=[
                pltpu.VMEM((max_q_len, 1), jnp.float32),
                pltpu.VMEM((max_q_len, 1), jnp.float32),
                pltpu.VMEM((max_q_len, head_dim), jnp.float32),
            ],
        )

        qt = jnp.swapaxes(q, 1, 2)  # (num_seqs, num_heads, max_q_len, head_dim)
        result: Array = pl.pallas_call(
            kernel,
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct(qt.shape, q.dtype),
            interpret=interpret,
        )(block_tables, lengths, q_lens, qt, k_pages, v_pages)
        return jnp.swapaxes(result, 1, 2)

    def pallas_paged_mixed_attention(
        q: Array,
        k_pages: Array,
        v_pages: Array,
        block_tables: Array,
        lengths: Array,
        q_lens: Array,
        *,
        scale: float | None = None,
        interpret: bool = False,
    ) -> Array:
        """Paged attention over a mixed prefill/decode batch, in one launch.

        Continuous batching schedules prefill chunks and single-token decode
        steps in the same iteration. The decode and prefill kernels above each
        serve one of those shapes, which would force the engine to split every
        step's batch into two launches. This kernel serves both at once, the
        distribution-aware case the Ragged Paged Attention design targets: the
        batch is ragged in the QUERY axis too, so per-sequence query counts ride
        next to the block table as scalar prefetch.

        Every sequence's queries are padded to the batch-wide max_q_len. Row t
        of sequence s is real only if t < q_lens[s]: a decode sequence has
        q_lens[s] == 1, a prefill chunk q_lens[s] > 1, and q_lens[s] == 0 marks
        an inactive batch slot. Real rows occupy the last q_lens[s] absolute
        positions of the context (row t sits at position
        lengths[s] - q_lens[s] + t) and attend causally, exactly as in the
        prefill kernel. Padding rows come back as exact zeros.

        Args:
            q: queries padded on the row axis,
                `(num_seqs, max_q_len, num_heads, head_dim)`; rows at or past
                `q_lens[s]` are ignored.
            k_pages: key cache pool, heads-first
                `(num_kv_heads, num_pages, page_size, head_dim)`.
            v_pages: value cache pool, same shape as `k_pages`.
            block_tables: `(num_seqs, max_pages)` int page table (see the decode
                kernel).
            lengths: `(num_seqs,)` int total context length per sequence,
                including that sequence's `q_lens[s]` query tokens.
            q_lens: `(num_seqs,)` int count of real query rows per sequence.
                Values must satisfy `0 <= q_lens[s] <= min(max_q_len, lengths[s])`.
                That is a caller contract: the values are runtime data (they may
                be traced), so it cannot be validated here.
            scale: softmax scale. Defaults to `1/sqrt(head_dim)`.
            interpret: run in Pallas interpret mode on CPU (no TPU needed).

        Returns:
            `(num_seqs, max_q_len, num_heads, head_dim)`, dtype of `q`; rows at
            or past `q_lens[s]` are exact zeros.

        Raises:
            RuntimeError: if JAX is not installed.
            ValueError: on rank/shape mismatch or an unclean head grouping.
        """
        if not _JAX_AVAILABLE:
            raise RuntimeError(
                "JAX is not available; install the 'tpu' extra "
                "(uv sync --extra tpu) to use the Pallas paged attention kernel"
            )
        if q.ndim != 4:
            raise ValueError(
                "q must be (num_seqs, max_q_len, num_heads, head_dim) for a mixed "
                f"batch; got rank {q.ndim}"
            )
        if k_pages.ndim != 4 or v_pages.shape != k_pages.shape:
            raise ValueError(
                "k_pages and v_pages must be heads-first (num_kv_heads, num_pages, "
                f"page_size, head_dim) and identical; got {k_pages.shape} and {v_pages.shape}"
            )
        num_seqs = q.shape[0]
        if block_tables.ndim != 2 or block_tables.shape[0] != num_seqs:
            raise ValueError(
                "block_tables must be (num_seqs, max_pages) matching q's num_seqs; "
                f"got {block_tables.shape} for num_seqs={num_seqs}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != num_seqs:
            raise ValueError(
                f"lengths must be (num_seqs,); got {lengths.shape} for num_seqs={num_seqs}"
            )
        if q_lens.ndim != 1 or q_lens.shape[0] != num_seqs:
            raise ValueError(
                f"q_lens must be (num_seqs,); got {q_lens.shape} for num_seqs={num_seqs}"
            )

        num_heads, head_dim = q.shape[2], q.shape[3]
        num_kv_heads = k_pages.shape[0]
        if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                "num_heads must be a positive multiple of num_kv_heads for "
                f"grouped-query attention; got {num_heads} query heads and "
                f"{num_kv_heads} kv heads"
            )
        if k_pages.shape[3] != head_dim:
            raise ValueError(f"head_dim mismatch: q has {head_dim}, k_pages has {k_pages.shape[3]}")

        effective_scale = scale if scale is not None else 1.0 / (head_dim**0.5)
        block_tables = block_tables.astype(jnp.int32)
        lengths = lengths.astype(jnp.int32)
        q_lens = q_lens.astype(jnp.int32)

        return _paged_mixed_launch(
            q,
            k_pages,
            v_pages,
            block_tables,
            lengths,
            q_lens,
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

    def pallas_paged_prefill_attention(
        q: Any,
        k_pages: Any,
        v_pages: Any,
        block_tables: Any,
        lengths: Any,
        *,
        scale: float | None = None,
        interpret: bool = False,
    ) -> Any:
        """Fallback when JAX is absent; gate on `supports_pallas_paged_attention()`."""
        raise RuntimeError(
            "JAX is not available; install the 'tpu' extra "
            "(uv sync --extra tpu) to use the Pallas paged attention kernel"
        )

    def pallas_paged_mixed_attention(
        q: Any,
        k_pages: Any,
        v_pages: Any,
        block_tables: Any,
        lengths: Any,
        q_lens: Any,
        *,
        scale: float | None = None,
        interpret: bool = False,
    ) -> Any:
        """Fallback when JAX is absent; gate on `supports_pallas_paged_attention()`."""
        raise RuntimeError(
            "JAX is not available; install the 'tpu' extra "
            "(uv sync --extra tpu) to use the Pallas paged attention kernel"
        )
