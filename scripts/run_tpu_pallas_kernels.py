#!/usr/bin/env python
"""Run the Pallas TPU attention kernels on a real TPU and check parity.

The rest of the TPU backend (ADR-023) is validated in Pallas interpret mode on
CPU. This script is the on-hardware follow-up: it runs the dense, paged, and
grouped-query kernels with interpret=False on an actual TPU, checks each against
a NumPy reference (cosine similarity > 0.99, the ADR-023 bar), and prints a short
timing so Mosaic's lowering of the scalar-prefetch page gather and the VMEM
scratch state can be confirmed on real silicon.

Run on a free TPU (Google Colab TPU runtime, or a Kaggle TPU notebook):

    pip install -U "jax[tpu]"      # TPU wheels (interpret-mode plain `jax` is not enough here)
    pip install -e .               # so `import mini_infer` finds the kernels
    python scripts/run_tpu_pallas_kernels.py

With no TPU present it falls back to interpret mode on CPU and says so, so the
script is safe to smoke-test anywhere (that is all this environment can do).
"""

from __future__ import annotations

import time

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _dense_reference(q, k, v, scale, causal):
    scores = np.einsum("hqd,hkd->hqk", q, k) * scale
    if causal:
        seq_q, seq_k = q.shape[1], k.shape[1]
        i = np.arange(seq_q)[:, None]
        j = np.arange(seq_k)[None, :]
        scores = np.where(j <= i, scores, -1e30)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,hkd->hqd", weights, v)


def _paged_reference(q, k_pages, v_pages, block_tables, lengths, scale):
    num_seqs, num_heads, _ = q.shape
    num_kv_heads = k_pages.shape[2]
    q_per_kv = num_heads // num_kv_heads
    max_pages = block_tables.shape[1]
    out = np.zeros_like(q)
    for s in range(num_seqs):
        length = int(lengths[s])
        k_full = np.concatenate([k_pages[block_tables[s, pi]] for pi in range(max_pages)], axis=0)
        v_full = np.concatenate([v_pages[block_tables[s, pi]] for pi in range(max_pages)], axis=0)
        valid = np.arange(max_pages * k_pages.shape[1]) < length
        for h in range(num_heads):
            kv = h // q_per_kv
            scores = (k_full[:, kv, :] @ q[s, h]) * scale
            scores = np.where(valid, scores, -1e30)
            scores = scores - scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            out[s, h] = weights @ v_full[:, kv, :]
    return out


def _make_paged(num_seqs, num_heads, num_kv_heads, head_dim, page_size, max_pages, seed):
    rng = np.random.default_rng(seed)
    num_pages = num_seqs * max_pages + 2
    q = rng.standard_normal((num_seqs, num_heads, head_dim)).astype(np.float32)
    k_pages = rng.standard_normal((num_pages, page_size, num_kv_heads, head_dim)).astype(np.float32)
    v_pages = rng.standard_normal((num_pages, page_size, num_kv_heads, head_dim)).astype(np.float32)
    block_tables = np.zeros((num_seqs, max_pages), dtype=np.int32)
    lengths = np.zeros((num_seqs,), dtype=np.int32)
    cursor = 0
    for s in range(num_seqs):
        length = int(rng.integers(1, max_pages * page_size + 1))
        lengths[s] = length
        n_used = (length + page_size - 1) // page_size
        for pi in range(max_pages):
            block_tables[s, pi] = cursor if pi < n_used else 0
            if pi < n_used:
                cursor += 1
    return q, k_pages, v_pages, block_tables, lengths


def _timed(jax, fn, iters=20):
    out = fn()
    jax.block_until_ready(out)  # trigger compile / first run
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters * 1e3, out  # ms per call


def _report(name, got, ref, ms):
    got = np.asarray(got)
    cos = _cosine(got, ref)
    max_abs = float(np.max(np.abs(got - ref)))
    status = "PASS" if cos > 0.99 else "FAIL"
    print(f"  [{status}] {name:28s} cosine={cos:.7f}  max_abs_err={max_abs:.2e}  {ms:.3f} ms/call")
    return cos > 0.99


def main() -> int:
    import jax
    import jax.numpy as jnp

    from mini_infer.backends.tpu.pallas_attention import pallas_attention
    from mini_infer.backends.tpu.pallas_paged_attention import pallas_paged_attention

    print("jax", jax.__version__)
    try:
        tpus = jax.devices("tpu")
    except RuntimeError:
        tpus = []
    on_tpu = len(tpus) > 0
    interpret = not on_tpu
    if on_tpu:
        print(f"Running on TPU: {tpus}")
    else:
        print("No TPU found; running interpret mode on CPU (smoke test only)")

    all_ok = True

    # Dense attention (non-causal and causal).
    rng = np.random.default_rng(0)
    num_heads, seq, head_dim = 4, 128, 128
    q = rng.standard_normal((num_heads, seq, head_dim)).astype(np.float32)
    k = rng.standard_normal((num_heads, seq, head_dim)).astype(np.float32)
    v = rng.standard_normal((num_heads, seq, head_dim)).astype(np.float32)
    scale = 1.0 / (head_dim**0.5)
    qj, kj, vj = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    print("Dense attention:")
    for causal in (False, True):
        ms, out = _timed(
            jax,
            lambda causal=causal: pallas_attention(qj, kj, vj, causal=causal, interpret=interpret),
        )
        ref = _dense_reference(q, k, v, scale, causal)
        all_ok &= _report(f"dense causal={causal}", out, ref, ms)

    # Paged decode attention: multi-head and grouped-query.
    print("Paged decode attention:")
    for label, (nh, nkv) in {"MHA (8 heads)": (8, 8), "GQA (8q/2kv)": (8, 2)}.items():
        q2, kp, vp, bt, ln = _make_paged(
            num_seqs=8,
            num_heads=nh,
            num_kv_heads=nkv,
            head_dim=128,
            page_size=16,
            max_pages=8,
            seed=1,
        )
        args = (jnp.asarray(q2), jnp.asarray(kp), jnp.asarray(vp), jnp.asarray(bt), jnp.asarray(ln))
        ms, out = _timed(jax, lambda args=args: pallas_paged_attention(*args, interpret=interpret))
        ref = _paged_reference(q2, kp, vp, bt, ln, 1.0 / (128**0.5))
        all_ok &= _report(f"paged {label}", out, ref, ms)

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
