# ADR-018: Triton port of `hc_split_sinkhorn`

Date: 2026-05-17 (accepted 2026-06-02 after live validation)
Status: Accepted

## Context

DeepSeek-V4's Hyper-Connections sublayer (paper §2.5) wraps every
attention and FFN call with multi-residual mixing. The arithmetic core
is `hc_split_sinkhorn`, which produces three per-row outputs from a
single linear projection's output:

```
Input:
  mixes:    (B, T, (2 + hc) * hc)    raw scores
  hc_scale: (3,)                       per-chunk scale
  hc_base:  ((2 + hc) * hc,)           per-feature bias
  hc, sinkhorn_iters, eps

Output:
  pre:  (B, T, hc)            sigmoid + eps
  post: (B, T, hc)            2 * sigmoid
  comb: (B, T, hc, hc)        Sinkhorn-normalized (doubly-stochastic)
```

The V4 reference (`third_party/deepseek_v4_reference/kernel.py:371`)
implements this as a **tilelang** kernel: one thread block per
`(B*T)` row, a `(hc, hc)` matrix in fragment registers, an initial
row-softmax-with-eps plus column-normalize, then `sinkhorn_iters - 1`
alternating row/column normalizations.

Our previous implementation (`src/mini_infer/models/blocks/hyper_connections.py:hc_split_sinkhorn`)
was a pure-PyTorch transcription that produces bit-equivalent output
but expresses each step as a separate PyTorch op. For V4-Flash
(61 layers, 2 HC sublayers per layer), one forward pass triggers
roughly `122 * 50 = 6,100` kernel launches just for the Sinkhorn
arithmetic. The compute per call is trivial (a 16-element matrix for
`hc = 4`); the kernel-launch count dominates.

Two motivations to port:

1. **Paper-faithfulness.** The reference *is* a custom kernel.
   Reading our PyTorch transcription, a paper reader sees the math
   but loses the fusion property that makes the original a kernel in
   the first place. A Triton version restores that correspondence in
   readable code (the kernel module sits next to the PyTorch oracle).
2. **Decode-latency relief.** HC isn't the dominant cost in a V4
   decode (FlashAttention + GEMMs are), but it sits on the per-token
   critical path. Reducing 50 launches per HC sublayer to 1 removes
   one of the serial overhead sources per layer per token.

## Decision

Ship a Triton kernel `_hc_split_sinkhorn_kernel` in a new module
`src/mini_infer/models/blocks/hc_sinkhorn_kernel.py`, with a thin
Python entry point `hc_split_sinkhorn_triton` and a dispatch predicate
`supports_hc_kernel(device, hc_mult)`. The existing PyTorch body is
renamed `_hc_split_sinkhorn_torch` and kept verbatim as the numerical
oracle + CPU/MPS fallback.

The public `hc_split_sinkhorn` becomes a 5-line dispatcher:

```python
def hc_split_sinkhorn(mixes, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, eps):
    if supports_hc_kernel(mixes.device, hc_mult):
        return hc_split_sinkhorn_triton(...)
    return _hc_split_sinkhorn_torch(...)
```

No call sites change.

### Kernel shape

- **One Triton program per row of `(B*T, mix_hc)`.** The `(hc, hc)`
  comb matrix is held in registers for the entire Sinkhorn iteration;
  no HBM round-trip between iterations.
- **Scale-vector Sinkhorn loop** (see "Triton 3.1 compiler
  constraint" below). The loop carries two 1D vectors (`row_scale`,
  `col_scale`); the post-softmax matrix stays loop-invariant; each
  iteration recomputes the current matrix as
  `comb * row_scale[:, None] * col_scale[None, :]` and reduces that
  body-local value. Dividing every row j by `(row_sum[j] + eps)` is
  identical to folding `1 / (row_sum[j] + eps)` into `row_scale[j]`,
  so the sums each iteration sees equal the reference's sums over its
  sequentially divided matrix. The FP difference (multiplicative
  composition vs sequential division) measured 3.6e-7 max-abs on FP32
  at `hc=4, iters=20`.
- **`hc_mult` and `sinkhorn_iters` are constexpr.** The Triton
  compiler unrolls the `(HC, HC)` tile arithmetic; the Sinkhorn loop
  compiles as a normal loop over the constexpr trip count. JIT cache
  key includes both, so V4's `hc=4, iters=20` and any future variant
  get their own specializations.
- **Default `num_warps`.** A 16-element matrix doesn't benefit from
  intra-program parallelism either way; the parallelism is across
  rows. `num_warps=1` was tried and abandoned: it is an untested
  corner of the reduction codegen with zero measurable upside.
- **`hc_mult` constrained to a power of 2.** `tl.arange(0, HC)` and
  the `(HC, HC)` 2D tile work without masking when HC is a power of
  2. V4 uses `hc=4`. Non-power-of-2 hc (e.g. the `hc=3` case in the
  PyTorch shape-contract tests) falls back to the PyTorch path via
  the predicate.

### Triton 3.1 compiler constraint (found during bring-up)

The mathematically direct transcription of the reference loop:

```python
for _ in range(SINKHORN_ITERS - 1):
    comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
```

**segfaults the Triton 3.1.0 compiler** (SIGSEGV in the runner process
at JIT time, torch 2.5.1 + CUDA 12.4, L40S). Bisected via a
subprocess-isolated variant matrix
(`scripts/modal_hc_kernel_probe.py`):

| Variant | Loop | Reduction operand | Result |
|---|---|---|---|
| no loop (`iters=1` shape) | none | plain value | OK, diff ~1e-7 |
| direct loop, axis-0 sums | scf.for | **loop-carried tile** | SIGSEGV |
| direct loop, trans + axis-1 sums | scf.for | **loop-carried tile** | SIGSEGV |
| `tl.static_range(3)` unroll | unrolled | plain value | OK, diff ~1e-7 |
| `tl.static_range(19)` unroll | unrolled | plain value | compile hang (5+ min) |
| scale-vector, axis-0 sums | scf.for | body-local value | OK, diff 3.6e-7 |
| scale-vector, trans + axis-1 | scf.for | body-local value | OK, diff 2.4e-7 |

The crashing construct is a reduction whose operand chains back to a
loop-carried 2D tile, regardless of reduction axis. This also explains
why FlashAttention-style kernels (the heavily exercised path on this
stack) never hit it: their loops reduce values computed inside the
body (`qk` from loads) while the carried accumulator only ever
receives multiply-adds. The scale-vector formulation puts our kernel
on exactly that proven shape: carried 1D stats (like FA's `m` / `l`),
reductions on body-local 2D values.

A full unroll (`tl.static_range(19)`) avoids the carried value but
sends the compile into a 5+ minute hang at V4's iteration count, so
it's not a viable alternative.

### Numerical contract

- **FP32 throughout.** Matches the reference's tilelang signature
  (`FP32` annotations on every tensor). The Sinkhorn iteration's
  alternating normalization is sensitive to accumulation order;
  BF16 / FP16 variants would compound across 20 iterations and need
  their own parity story. Out of scope here.
- **Forward only.** mini-infer is an inference engine; no backward
  through Sinkhorn.
- **Parity vs PyTorch oracle: cosine sim > 0.999** on FP32 inputs,
  the codebase standard (matches `paged_attention`, `int8_kernel`).
  Plus a tighter `torch.testing.assert_close` on `pre`/`post`
  (rtol=1e-4, atol=1e-5; sigmoid outputs are small and cos-sim alone
  is too loose) and a slightly looser bound on `comb` (rtol=1e-3,
  atol=1e-4; 20 alternating normalizations compound any reduction-
  order divergence).
- **Doubly-stochastic invariant.** After 20 iterations, every row
  and column of `comb` sums to ~1 modulo the additive `eps`. Same
  invariant the PyTorch oracle asserts.

### Bench shape

`scripts/bench_hc_sinkhorn.py` measures median per-call latency
(after warmup) for PyTorch vs Triton at `hc=4`, `batch=2`,
`seqlen ∈ {1, 64, 256, 1024}`, `sinkhorn_iters=20`. Runs locally with
just the PyTorch path (no speedup story on CPU) or on Modal with both.

The expected pattern:

- **PyTorch path scales weakly with seqlen at small T.** Most of the
  per-call cost is fixed Python overhead + launch overhead. For
  `T=1024` the actual GPU compute starts to matter.
- **Triton path scales linearly with seqlen.** One launch per call;
  per-row work is constant in `hc`, so total time is `O(B*T)`.
- **Speedup is largest at small T** (where launch overhead
  dominates the PyTorch path) and shrinks as T grows (where the
  PyTorch path's compute catches up).

Measured numbers go in this ADR's "Consequences" section once the
Modal run lands.

## Alternatives Considered

1. **Keep the PyTorch path only.**
   - Pros: zero kernel complexity; one numerical implementation; works
     on M1 as the dev surface.
   - Cons: ~6,100 launches per V4 forward for the Sinkhorn arithmetic;
     loses the paper-faithfulness of the reference being a kernel;
     can't relieve the per-token launch overhead.
   - Rejected: roadmap item #11 explicitly identifies this kernel as
     the first paper-faithful Triton port. The pure-PyTorch version
     was always the placeholder.

2. **Fuse the upstream `linear + rsqrt` into the kernel.**
   - The HC `hc_pre` does `mixes = (x_flat @ fn.T) * rsqrt(...)`
     in PyTorch before calling `hc_split_sinkhorn`. Fusing the GEMM
     into the kernel saves an HBM round-trip on `mixes`.
   - Rejected for this ADR: the kernel shape would change
     substantially (different grid, different memory traffic pattern),
     and the cleanest split for review is "Sinkhorn fusion only" now,
     "GEMM-plus-Sinkhorn fusion" as a separate follow-up. The current
     Sinkhorn-only kernel already removes the launch-count blow-up.

3. **Port to CUDA C++ instead of Triton.**
   - Pros: tighter control over register allocation; potentially
     lower kernel-launch overhead than Triton's runtime.
   - Cons: every other kernel in this codebase is Triton; introducing
     a CUDA C++ build path adds toolchain surface (nvcc, build deps,
     CI image edits) for one kernel.
   - Rejected: not worth the toolchain delta for a microkernel.

4. **BF16/FP16 variant.**
   - Could nominally double throughput for HBM-bound stages.
   - Rejected: the inputs come from a PyTorch FP32 path (hc_pre's
     rsqrt and the upstream linear), the reference is FP32, and the
     20-iter Sinkhorn loop's numerical sensitivity to reduction order
     would need its own parity bar. Not the right first version.

## Consequences

### Trade-offs

- **+ Launch-count collapse.** Fifty-ish PyTorch ops per Sinkhorn call
  become one Triton launch. The Triton kernel itself is JIT-compiled
  on first call; subsequent calls hit the cache.
- **+ Aligned with the niche.** First Triton kernel in this repo
  whose reason for existing is "the paper has a kernel," not
  "performance." Reads as paper → reference → mini-infer kernel.
- **+ Predicate keeps the codebase honest.** `supports_hc_kernel`
  centralizes the "can we run this kernel" question. No
  `device.type == "cuda"` checks bleed into callers.
- **- Two implementations to maintain.** When the V4 reference
  changes the Sinkhorn iteration (it hasn't, but a future V5 might),
  both `_hc_split_sinkhorn_torch` and `_hc_split_sinkhorn_kernel`
  need to update in lockstep. The parity test is the regression
  guard.
- **- Triton compilation cost on first call.** A few hundred ms.
  Mostly invisible (it happens during the first decoded token, behind
  other warmup work) but worth noting for cold-start latency.
- **- Non-power-of-2 `hc_mult` falls back to PyTorch.** V4 uses
  `hc=4`. A future architecture with `hc=3` or `hc=5` would silently
  take the slower path until we extend the kernel to handle padded
  shapes. Documented in `supports_hc_kernel`'s docstring.

### Measurement (L40S, 2026-06-02)

```
device: NVIDIA L40S  (torch 2.5.1+cu124, triton 3.1.0, FP32)
hc_mult=4  batch=2  sinkhorn_iters=20
median per-call latency over 100 iters after 10-iter warmup

  seqlen |   pytorch us |    triton us |  speedup
-------------------------------------------------
       1 |       1006.6 |         70.7 |   14.25x
      64 |       1007.8 |         72.5 |   13.90x
     256 |       1092.7 |         73.6 |   14.84x
    1024 |       1020.9 |         73.6 |   13.87x
```

Reading the table:

- **The PyTorch column is flat at ~1 ms regardless of seqlen.** The
  per-call cost is op-dispatch overhead (~50 launches), not compute,
  exactly the motivating hypothesis.
- **The Triton column is flat at ~72 us.** One launch; rows process
  in parallel across the grid, so seqlen doesn't move the latency at
  these sizes.
- **~14x per call.** For V4-Flash decode (122 HC calls per token),
  this is the difference between ~123 ms and ~8.8 ms of HC-induced
  serial overhead per token at these isolated-call rates. In-model
  rates will differ (stream overlap with attention / GEMM work);
  the kernel-level measurement is the defensible claim.

Parity at the same run: all 18 configurations
(hc ∈ {2,4,8} x T ∈ {1,16,256} x iters ∈ {1,20}, raw randn inputs)
passed cosine-sim > 0.999 plus elementwise tolerances
(pre/post rtol=1e-4 atol=1e-5, comb rtol=1e-3 atol=1e-4), plus the
doubly-stochastic sanity on tempered inputs at the CPU test's
established 5e-3 tolerance.

### What this unlocks

- **Roadmap item #11** ("one real Triton kernel per quarter where
  the paper calls for it"): first ship.
- **A reusable pattern** for future paper-faithful kernels. The next
  candidate in the V4 reference is `sparse_attn`, though that kernel's
  job substantially overlaps FlashInfer's existing block-sparse
  attention so the niche fit is weaker.

### What this doesn't change

- **The PyTorch oracle stays canonical.** Tests on M1 / CI / any
  non-CUDA path continue to use `_hc_split_sinkhorn_torch`. The
  Triton kernel adds a fast path; it doesn't replace the reference.
- **The V4 reference parity test continues to pass via the same
  injection point.** `tests/unit/_v4_reference_helpers.py` patches
  `kernel.hc_split_sinkhorn = <our public dispatcher>`; on CPU that
  routes to the PyTorch oracle, identical numerics to before the
  port.

## Pointers

- **Triton kernel + dispatch predicate**: [src/mini_infer/models/blocks/hc_sinkhorn_kernel.py](../../src/mini_infer/models/blocks/hc_sinkhorn_kernel.py).
- **Dispatcher + PyTorch oracle**: [src/mini_infer/models/blocks/hyper_connections.py](../../src/mini_infer/models/blocks/hyper_connections.py).
- **Bench**: [scripts/bench_hc_sinkhorn.py](../../scripts/bench_hc_sinkhorn.py).
- **Parity tests**: [tests/unit/test_hc_sinkhorn_kernel.py](../../tests/unit/test_hc_sinkhorn_kernel.py).
- **V4 reference (tilelang)**: `third_party/deepseek_v4_reference/kernel.py:371`.
- **Plan**: [docs/plans/hc-sinkhorn-triton.md](../plans/hc-sinkhorn-triton.md).
- **Paper section**: DeepSeek-V4 §2.5 (Hyper-Connections).
