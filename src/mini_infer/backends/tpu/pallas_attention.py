"""Dense scaled-dot-product attention forward as a Pallas TPU kernel.

This is the M1 kernel for the TPU backend (ADR-023): the first real attention
kernel, one step up from the M0 softmax scaffold (pallas_softmax.py). It
computes a single, dense (non-paged) attention forward,

    out = softmax(Q @ K^T / sqrt(d)) @ V

using the FLASH-ATTENTION online-softmax pattern rather than materialising the
full `(seq_q, seq_k)` score matrix. Instead of computing every score, taking a
global max, and then normalising (three passes over the scores, plus a large
intermediate), we stream over KV blocks in a single pass and carry three
running quantities per query row:

    m   running max of the scores seen so far        (for the stability shift)
    l   running sum of exp(score - m)                (the softmax denominator)
    acc running sum of exp(score - m) * V            (the unnormalised output)

When a new KV block yields a larger max, the block-local max `m_new` is folded
in and the previously accumulated `l` and `acc` are rescaled by
`exp(m_old - m_new)` so they are expressed relative to the new max. After the
last block, `out = acc / l`. This is the standard FlashAttention recurrence
(Dao et al.); the same shift-before-exp guard as the M0 softmax makes it stable
on large logits, because no score is ever exp'd without first subtracting a max.

Why the online softmax carries state across the grid (the load-bearing TPU
difference)
-------------------------------------------------------------------------------
On CUDA the flash-attention KV loop is a plain sequential `for` loop *inside*
one thread block, while the GRID of query blocks runs concurrently across SMs.
On Pallas-for-TPU it is the other way around: the grid itself is the loop. A
Pallas TPU grid iterates SEQUENTIALLY in row-major (lexicographic) order over
the same core (see pallas_softmax.py's docstring), so we make the KV-block axis
the INNER, fastest-moving grid axis and let successive grid steps be the
successive iterations of the flash recurrence. The running `m`, `l`, and `acc`
cannot live in Python (they must persist across compiled grid steps), so they
live in VMEM scratch buffers that Pallas threads through the steps for us
(`pltpu.VMEM` scratch shapes declared in `scratch_shapes`). We reset them on
the first KV step (`kv == 0`) and finalise (`acc / l`) on the last. This is the
concrete cash-out of "for kernels with carried state the lexicographic order is
exactly the schedule you reason about" from the M0 module: here the carried
state is real, and correctness depends on the KV steps running in order.

Layout
------
We pick ONE clean layout and stick to it: inputs are 3D `(num_heads, seq,
head_dim)`. A 4D `(batch, num_heads, seq, head_dim)` input is accepted at the
Python entry point and flattened to `(batch * num_heads, seq, head_dim)` before
the kernel, then un-flattened on the way out, so the kernel body only ever sees
one canonical rank. Heads (and batch*heads) are fully independent attention
problems, so they become the OUTER grid axis; the KV-block loop is the inner
axis. Each grid step owns one head, one query block, and one KV block.

TPU tiling
----------
The kernel tiles queries into blocks of `block_q` rows and keys/values into
blocks of `block_k` rows, each carrying the full `head_dim` so the `Q @ K^T`
and `P @ V` contractions stay inside a block (no cross-block reductions on the
contraction axis, same discipline as the M0 softmax's last-axis reduction).
Block sizes default to multiples of 8 to align with the TPU's 8x128 tile
geometry (8 sublanes x 128 lanes); `head_dim` is the natural 128-lane axis.
Query/key rows are the sublane axis. The running `m`/`l`/`acc` state sits in the
VMEM scratchpad, not registers or CUDA shared memory, which do not exist here.

JAX is optional and import-guarded exactly as in pallas_softmax.py. Without JAX,
`supports_pallas_attention()` returns False and the entry point raises rather
than crashing at import. With plain `jax` (no `jax[tpu]`), pass `interpret=True`
to run on CPU: that is how the parity test validates it with no TPU hardware.
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
    # `import jax.numpy as jnp` binds a submodule alias that mypy resolves even
    # under follow_imports=skip, so the None fallback needs a scoped ignore
    # (the plain `jax` / `pl` / `pltpu` fallbacks do not). Matches the targeted
    # per-line ignores used in the Triton kernel modules and pallas_softmax.
    jnp = None  # type: ignore[assignment]
    pl = None
    pltpu = None

if TYPE_CHECKING:  # for type checkers only; never executed at runtime
    import jax as jax
    from jax import Array


# Default query/key rows per block. Kept multiples of 8 so the block's sublane
# dimension aligns with the TPU's 8x128 tile geometry (see module docstring).
# Small by design: this is a first dense kernel, not a tuned production kernel.
_DEFAULT_BLOCK_Q = 128
_DEFAULT_BLOCK_K = 128

# Masked-out scores are set to a large negative sentinel before exp so they
# contribute ~0 to the running max and denominator. Not -inf: -inf would make a
# fully-masked query row (which cannot happen with a causal mask, since a query
# always attends to itself) produce NaN, and a finite sentinel keeps every
# intermediate finite in interpret mode.
_MASK_NEG = -1e30


def supports_pallas_attention() -> bool:
    """Whether the Pallas attention kernel can run in this process.

    Today this only checks that JAX imported. It does NOT require a physical
    TPU: with plain `jax`, the kernel runs in interpret mode on CPU, which is
    enough for parity testing. A future revision can extend this to probe for
    a real TPU device when we want to gate hardware execution specifically.
    Mirrors `supports_pallas_softmax` and the CUDA `supports_fused_kernel` /
    `supports_flashinfer_backend` predicate style.
    """
    return _JAX_AVAILABLE


if _JAX_AVAILABLE:

    def _attention_kernel(
        q_ref: Any,
        k_ref: Any,
        v_ref: Any,
        out_ref: Any,
        m_scratch: Any,
        l_scratch: Any,
        acc_scratch: Any,
        *,
        scale: float,
        causal: bool,
        block_q: int,
        block_k: int,
    ) -> None:
        """Kernel body: one flash-attention step over a single KV block.

        Refs are VMEM tiles for the current grid step, as declared by the
        BlockSpecs: `q_ref` is `(block_q, head_dim)`, `k_ref`/`v_ref` are
        `(block_k, head_dim)`, `out_ref` is `(block_q, head_dim)`. The scratch
        refs carry the running online-softmax state across the sequential KV
        steps: `m_scratch` `(block_q, 1)`, `l_scratch` `(block_q, 1)`,
        `acc_scratch` `(block_q, head_dim)`.

        The kernel runs once per grid step. `pl.program_id(2)` is the KV-block
        index (the inner, fastest-moving axis), so the recurrence advances one
        KV block per call. We reset the state on the first KV step and write the
        normalised output on the last; in between we only fold the block in.
        """
        kv_index = pl.program_id(2)
        num_kv_blocks = pl.num_programs(2)

        # First KV step for this (head, query-block): initialise the carried
        # state. Doing it here (rather than pre-filling the scratch) keeps the
        # reset inside the same sequential schedule Pallas threads the scratch
        # through, so it works identically in interpret mode and on real TPU.
        @pl.when(kv_index == 0)  # type: ignore[untyped-decorator]
        def _init() -> None:
            m_scratch[...] = jnp.full_like(m_scratch, _MASK_NEG)
            l_scratch[...] = jnp.zeros_like(l_scratch)
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        q = q_ref[...]  # (block_q, head_dim)
        k = k_ref[...]  # (block_k, head_dim)
        v = v_ref[...]  # (block_k, head_dim)

        # Scores for this query block against this KV block. Scaling by
        # 1/sqrt(head_dim) before softmax is the standard attention scale and a
        # numerical-stability requirement (CLAUDE.md); fold it into `scale`.
        scores = jax.lax.dot_general(
            q,
            k,
            (((1,), (1,)), ((), ())),  # contract head_dim, no batch dims
            preferred_element_type=jnp.float32,
        )
        scores = scores * scale  # (block_q, block_k)

        if causal:
            # A query at absolute position i may attend to key at absolute
            # position j only if j <= i. Absolute positions come from the block
            # indices: query rows start at program_id(1) * block_q, key rows at
            # kv_index * block_k. Masked entries get a large negative sentinel
            # so they vanish under exp (verified by the causal test).
            q_start = pl.program_id(1) * block_q
            k_start = kv_index * block_k
            q_pos = q_start + jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0)
            k_pos = k_start + jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1)
            scores = jnp.where(k_pos <= q_pos, scores, _MASK_NEG)

        # Online-softmax fold. Compare this block's row max against the running
        # max, rescale the carried denominator and accumulator to the new max,
        # then add this block's contribution. This is the FlashAttention
        # recurrence; every exp sees `score - m_new`, so nothing overflows.
        m_prev = m_scratch[...]  # (block_q, 1)
        l_prev = l_scratch[...]  # (block_q, 1)
        acc_prev = acc_scratch[...]  # (block_q, head_dim)

        m_cur = jnp.max(scores, axis=-1, keepdims=True)  # (block_q, 1)
        m_new = jnp.maximum(m_prev, m_cur)

        p = jnp.exp(scores - m_new)  # (block_q, block_k)
        # Rescale the previous state into the new max's frame before adding.
        correction = jnp.exp(m_prev - m_new)  # (block_q, 1)
        l_new = correction * l_prev + jnp.sum(p, axis=-1, keepdims=True)
        acc_new = correction * acc_prev + jax.lax.dot_general(
            p,
            v,
            (((1,), (0,)), ((), ())),  # contract block_k, no batch dims
            preferred_element_type=jnp.float32,
        )

        m_scratch[...] = m_new
        l_scratch[...] = l_new
        acc_scratch[...] = acc_new

        # Last KV step for this (head, query-block): normalise and write out.
        @pl.when(kv_index == num_kv_blocks - 1)  # type: ignore[untyped-decorator]
        def _finalize() -> None:
            out_ref[...] = (acc_scratch[...] / l_scratch[...]).astype(out_ref.dtype)

    def _attention_3d(
        q: Array,
        k: Array,
        v: Array,
        *,
        scale: float,
        causal: bool,
        block_q: int,
        block_k: int,
        interpret: bool,
    ) -> Array:
        """Run the Pallas attention over a canonical 3D `(heads, seq, dim)`.

        Kept separate from the public entry point so the batch flatten/unflatten
        logic stays out of the kernel-launch path and the launch always sees one
        rank. `heads` here is `batch * num_heads` after flattening.
        """
        num_heads, seq_q, head_dim = q.shape
        seq_k = k.shape[1]

        num_q_blocks = seq_q // block_q
        num_k_blocks = seq_k // block_k

        # Grid axes, outer -> inner: (head, query-block, kv-block). Heads and
        # query blocks are independent problems; the kv-block axis is INNER so
        # successive sequential grid steps are successive flash-recurrence
        # iterations over one query block (see module docstring). BlockSpec
        # index_maps return block coordinates in units of blocks.
        grid = (num_heads, num_q_blocks, num_k_blocks)

        q_spec = pl.BlockSpec((1, block_q, head_dim), lambda h, i, j: (h, i, 0))
        k_spec = pl.BlockSpec((1, block_k, head_dim), lambda h, i, j: (h, j, 0))
        v_spec = pl.BlockSpec((1, block_k, head_dim), lambda h, i, j: (h, j, 0))
        out_spec = pl.BlockSpec((1, block_q, head_dim), lambda h, i, j: (h, i, 0))

        out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)

        # The kernel body works on 2D tiles, but BlockSpecs above carry a
        # leading singleton head axis. Squeeze it inside a thin wrapper so the
        # recurrence code reads cleanly in 2D; the scratch shapes are 2D too.
        def _kernel(
            q_ref: Any,
            k_ref: Any,
            v_ref: Any,
            out_ref: Any,
            m_scratch: Any,
            l_scratch: Any,
            acc_scratch: Any,
        ) -> None:
            _attention_kernel(
                q_ref.at[0],
                k_ref.at[0],
                v_ref.at[0],
                out_ref.at[0],
                m_scratch,
                l_scratch,
                acc_scratch,
                scale=scale,
                causal=causal,
                block_q=block_q,
                block_k=block_k,
            )

        result: Array = pl.pallas_call(
            _kernel,
            grid=grid,
            in_specs=[q_spec, k_spec, v_spec],
            out_specs=out_spec,
            out_shape=out_shape,
            scratch_shapes=[
                pltpu.VMEM((block_q, 1), jnp.float32),
                pltpu.VMEM((block_q, 1), jnp.float32),
                pltpu.VMEM((block_q, head_dim), jnp.float32),
            ],
            interpret=interpret,
        )(q, k, v)
        return result

    def pallas_attention(
        q: Array,
        k: Array,
        v: Array,
        *,
        scale: float | None = None,
        causal: bool = False,
        block_q: int | None = None,
        block_k: int | None = None,
        interpret: bool = False,
    ) -> Array:
        """Dense scaled-dot-product attention forward via a Pallas TPU kernel.

        Computes `out = softmax(Q @ K^T * scale + mask) @ V` with the
        FlashAttention online-softmax recurrence (never materialises the full
        score matrix). Matches a plain-JAX reference within the ADR-023 parity
        bar (cosine similarity > 0.99).

        Args:
            q: queries, `(num_heads, seq_q, head_dim)` or
                `(batch, num_heads, seq_q, head_dim)`.
            k: keys, same rank as `q`, with `seq_k` in place of `seq_q`.
            v: values, same shape as `k`.
            scale: softmax scale applied to `Q @ K^T`. Defaults to
                `1 / sqrt(head_dim)`, the standard attention scale.
            causal: if True, a query at position i attends only to keys at
                positions j <= i (future positions are masked out).
            block_q: query rows per grid step. Defaults to `_DEFAULT_BLOCK_Q`
                (128, aligned to the TPU tile geometry). Must divide `seq_q`.
            block_k: key/value rows per grid step. Defaults to
                `_DEFAULT_BLOCK_K` (128). Must divide `seq_k`.
            interpret: run in Pallas interpret mode on the host CPU instead of
                compiling for a TPU. Set True to develop and parity-test without
                TPU hardware (plain `jax` supports this); leave False on a real
                TPU host.

        Returns:
            Attention output, same shape and dtype as `q`.

        Raises:
            RuntimeError: if JAX is not installed.
            ValueError: on rank mismatch, head_dim mismatch, incompatible
                seq_k between k and v, or a sequence length not divisible by its
                block size.
        """
        if not _JAX_AVAILABLE:
            raise RuntimeError(
                "JAX is not available; install the 'tpu' extra "
                "(uv sync --extra tpu) to use the Pallas attention kernel"
            )
        if not (q.ndim == k.ndim == v.ndim):
            raise ValueError(f"q, k, v must share rank; got {q.ndim}, {k.ndim}, {v.ndim}")
        if q.ndim not in (3, 4):
            raise ValueError(
                "pallas_attention expects 3D (num_heads, seq, head_dim) or 4D "
                f"(batch, num_heads, seq, head_dim) inputs, got rank {q.ndim}"
            )

        # Canonicalise to 3D (heads, seq, head_dim). A 4D input flattens
        # batch and heads into one leading axis (both are independent attention
        # problems and become the outer grid axis together); we restore the
        # original shape on the way out.
        original_shape = q.shape
        if q.ndim == 4:
            batch, num_heads = q.shape[0], q.shape[1]
            q = q.reshape(batch * num_heads, q.shape[2], q.shape[3])
            k = k.reshape(batch * num_heads, k.shape[2], k.shape[3])
            v = v.reshape(batch * num_heads, v.shape[2], v.shape[3])

        head_dim = q.shape[2]
        if k.shape[2] != head_dim or v.shape[2] != head_dim:
            raise ValueError(
                "q, k, v must share head_dim (last axis); got "
                f"{head_dim}, {k.shape[2]}, {v.shape[2]}"
            )
        if k.shape[1] != v.shape[1]:
            raise ValueError(f"k and v must share seq length; got {k.shape[1]} and {v.shape[1]}")
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            # Without this guard a head-count mismatch (e.g. unexpanded GQA K/V,
            # which the PAGED kernels accept) would not error: Pallas dynamic-slice
            # clamping maps the excess query heads onto the last kv head and
            # returns finite but wrong attention output.
            raise ValueError(
                "q, k, v must share the heads axis for dense attention; got "
                f"{q.shape[0]}, {k.shape[0]}, {v.shape[0]}. Grouped-query K/V "
                "(num_kv_heads < num_heads) is supported by the paged kernels, "
                "not the dense kernel."
            )

        seq_q = q.shape[1]
        seq_k = k.shape[1]
        bq = block_q if block_q is not None else _DEFAULT_BLOCK_Q
        bk = block_k if block_k is not None else _DEFAULT_BLOCK_K
        # Clamp block sizes to the sequence so short sequences run in one block
        # rather than tripping the divisibility guard for no reason.
        bq = min(bq, seq_q)
        bk = min(bk, seq_k)
        if seq_q % bq != 0:
            raise ValueError(
                f"seq_q={seq_q} must be divisible by block_q={bq}; "
                "pad the sequence or pass a divisor as block_q"
            )
        if seq_k % bk != 0:
            raise ValueError(
                f"seq_k={seq_k} must be divisible by block_k={bk}; "
                "pad the sequence or pass a divisor as block_k"
            )

        effective_scale = scale if scale is not None else 1.0 / (head_dim**0.5)

        out = _attention_3d(
            q,
            k,
            v,
            scale=effective_scale,
            causal=causal,
            block_q=bq,
            block_k=bk,
            interpret=interpret,
        )
        return out.reshape(original_shape)

else:

    def pallas_attention(
        q: Any,
        k: Any,
        v: Any,
        *,
        scale: float | None = None,
        causal: bool = False,
        block_q: int | None = None,
        block_k: int | None = None,
        interpret: bool = False,
    ) -> Any:
        """Fallback when JAX is absent: raise a clear error rather than crash on import.

        The real implementation is defined only when JAX imported. Keeping a
        stub here means `from mini_infer.backends.tpu.pallas_attention import
        pallas_attention` still works without JAX; calling it explains what to
        install. Callers should gate on `supports_pallas_attention()` first.
        """
        raise RuntimeError(
            "JAX is not available; install the 'tpu' extra "
            "(uv sync --extra tpu) to use the Pallas attention kernel"
        )
