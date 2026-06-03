# Plan: Triton port of `hc_split_sinkhorn`

> Status: proposed.
> Date: 2026-05-17.
> Replaces the pure-PyTorch transcription of the V4 reference's
> `hc_split_sinkhorn` tilelang kernel with a Triton kernel that runs
> on CUDA, while keeping the PyTorch path as the numerical oracle.

## Context

DeepSeek-V4's Hyper-Connections sublayer (paper §2.5) wraps every
attention and FFN call with a multi-residual mixing scheme. The core
helper is `hc_split_sinkhorn`:

```
Input:
  mixes:    (B, T, (2 + hc) * hc)   raw scores from a linear
  hc_scale: (3,)                     per-chunk scale (pre / post / comb)
  hc_base:  ((2 + hc) * hc,)         per-feature additive bias
  hc, sinkhorn_iters, eps

Output:
  pre:  (B, T, hc)            sigmoid + eps              (per-row independent)
  post: (B, T, hc)            2 * sigmoid                (per-row independent)
  comb: (B, T, hc, hc)        Sinkhorn-normalized       (per-row independent)
```

The reference (`third_party/deepseek_v4_reference/kernel.py:371`) is a
**tilelang** kernel: one thread block per `(B*T)` row, one
`(hc, hc)` matrix in fragment registers, an initial
row-softmax-with-eps + column-normalize, then `sinkhorn_iters - 1`
alternating row/column normalizations.

Our current implementation
(`src/mini_infer/models/blocks/hyper_connections.py:58`,
`hc_split_sinkhorn`) is a **pure-PyTorch transcription**. It produces
bit-identical output on the same inputs by design, but it expresses each
operation as a separate PyTorch op. For V4-Flash (61 layers, 2 HC
sublayers per layer), one forward pass triggers `122 * (~50 ops per
sinkhorn call)` kernel launches just for the Sinkhorn portion. Kernel-
launch overhead dominates the actual compute, which is a `(hc, hc)`
matrix iteration where `hc = 4` (V4); a 16-element matrix that fits
in a single warp's registers.

This is the textbook "fuse 50 elementwise + reduction ops into one
launch" optimization. It's also paper-faithful in the strict sense:
the reference *is* a custom kernel, not a chain of PyTorch ops.

## Goal

After this plan completes:

1. **`hc_split_sinkhorn_triton`** is a Triton kernel that produces
   `pre`, `post`, `comb` from the same inputs as the PyTorch reference,
   within tolerance (cosine sim > 0.999 vs PyTorch on FP32 inputs).
2. **`supports_hc_kernel(device)`** is the dispatch predicate, mirroring
   `supports_paged_kernel` / `supports_fused_kernel`. Pure-PyTorch path
   is preserved verbatim as the CPU / non-CUDA fallback and the
   numerical oracle.
3. **`hc_split_sinkhorn(...)`** dispatches to the Triton path on CUDA,
   PyTorch path everywhere else. Callers (`HyperConnections.hc_pre`)
   don't change.
4. **Decode throughput improves measurably** on V4 with `hc_mult=4` on
   real hardware. Target: kernel-launch overhead from the HC sublayer
   drops by ≥10x. Absolute end-to-end speedup depends on what else is
   running in the layer (FlashAttention, GEMMs); HC is one of several
   serial dependencies per token.
5. **ADR-018** documents the kernel design, the parity contract, and
   the measured speedup.

## Approach

Mirror the codebase's existing Triton kernel modules
(`src/mini_infer/cache/paged_attention.py`,
`src/mini_infer/quant/int8_kernel.py`):

- One Python file per kernel, holding both the Triton kernel and the
  Python entry point.
- Triton imported under a `try / except ImportError` so M1 / non-CUDA
  installs still import cleanly.
- `supports_X_kernel(device)` predicate is the single source of truth
  for dispatch (per the codebase's centralized-device-check convention).
- The dispatcher in the user-facing module (here `hc_split_sinkhorn`
  in `hyper_connections.py`) takes the predicate path; pure-PyTorch
  body is unchanged.

The kernel itself maps cleanly from tilelang to Triton:

| tilelang construct | Triton equivalent |
|---|---|
| `T.Kernel(n, threads=64) as i` | `pid = tl.program_id(0)`; one program per row |
| `T.alloc_shared(mix_hc, FP32)` | implicit: `tl.load` rows into registers |
| `T.alloc_fragment((hc, hc), FP32)` | `(BLOCK_HC, BLOCK_HC)` register tile |
| `T.Parallel(hc, hc)` | scalar-mode within the program (hc=4 fits a warp) |
| `T.reduce_sum(..., dim=1)` | `tl.sum(..., axis=1)` |
| `T.reduce_max(..., dim=1)` | `tl.max(..., axis=1)` |
| `T.exp` / `T.sigmoid` | `tl.exp` / `tl.sigmoid` |

For V4, `hc = 4`, so `(hc, hc) = (4, 4) = 16` floats. One program
processes one row; the entire comb matrix lives in registers for the
duration of all 20 Sinkhorn iterations. No shared memory needed.

For larger `hc` (a hypothetical V5 with hc=8 means `(8,8)=64` floats,
still fits), the same kernel scales. We hardcode `BLOCK_HC = hc` at
compile time (Triton constexpr); the kernel JIT-compiles per `hc` value.

Parallelism: launch grid is `(B * T,)`. Each program processes one row
independently. No cross-program communication. Trivially scales with
batch and sequence length.

## Phased execution

### Slice 1: Triton kernel + parity test

Implement `hc_split_sinkhorn_triton` in a new file
`src/mini_infer/models/blocks/hc_sinkhorn_kernel.py`. The file holds:

- `_TRITON_AVAILABLE` import gate.
- `supports_hc_kernel(device) -> bool`.
- The `@triton.jit` kernel itself.
- The Python entry point that allocates outputs and launches the kernel.

The kernel is forward-only (we're an inference engine; no backward
through Sinkhorn). FP32 only (matches the reference; lower precision
would compound the Sinkhorn iteration's accumulation error).

**Deliverable**: ~250 LoC kernel + ~150 LoC unit tests. Single commit.

**Test contract**:
1. Shape contract: outputs have correct shapes for representative
   `(B, T, hc_mult)` combinations (V4's hc=4, plus hc=2 / hc=8 for
   robustness).
2. Parity vs PyTorch reference: cosine sim > 0.999 on random FP32
   inputs across multiple seeds. Absolute tolerance check on
   `pre` / `post` (small values, divisions tighter than cos-sim).
3. Doubly-stochastic invariant: comb's row sums and column sums are
   within `2 * eps` of 1 after 20 iterations. Same invariant the
   existing PyTorch test asserts.
4. Edge case: `hc_mult=1` (HC degenerates to standard residual);
   `sinkhorn_iters=1` (single iteration); large `eps`.

Tests are gated on `supports_hc_kernel(device)`: CI's CPU runners
skip; CUDA-equipped local runs and Modal runs execute the parity.

### Slice 2: Dispatcher wiring

Edit `src/mini_infer/models/blocks/hyper_connections.py`:

- Top of file imports `supports_hc_kernel` + `hc_split_sinkhorn_triton`
  under a CUDA-safe gate.
- `hc_split_sinkhorn(...)` becomes a 5-line dispatcher:

```python
def hc_split_sinkhorn(mixes, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, eps):
    if supports_hc_kernel(mixes.device):
        return hc_split_sinkhorn_triton(mixes, hc_scale, hc_base,
                                         hc_mult=hc_mult,
                                         sinkhorn_iters=sinkhorn_iters,
                                         eps=eps)
    return _hc_split_sinkhorn_torch(mixes, hc_scale, hc_base,
                                     hc_mult=hc_mult,
                                     sinkhorn_iters=sinkhorn_iters,
                                     eps=eps)
```

The existing PyTorch body is renamed to `_hc_split_sinkhorn_torch`.
No call sites change.

**Deliverable**: ~30 LoC rename + dispatcher. Same commit as Slice 1,
or a separate small commit if Slice 1 grows unexpectedly. Probably
fold into Slice 1 because the dispatcher is trivial without it.

**Test contract**: existing
`tests/unit/test_hyper_connections.py` continues to pass on CPU
(takes the PyTorch path). On CUDA, the kernel path is exercised
by the Slice 1 parity tests.

### Slice 3: Bench + ADR (Modal)

Add `scripts/bench_hc_sinkhorn.py`:

- Local CPU mode: prints PyTorch baseline latency. Useful for sanity
  but doesn't show the Triton win (no Triton on CPU).
- Modal mode (single H100 or L40S, ~$1-2): measures PyTorch vs Triton
  latency at representative shapes (`hc=4`, `T = {1, 64, 256, 1024}`).
  Reports speedup, kernel-launch count, and decode-throughput proxy.

Why Modal: M1 can't run Triton; the speedup story needs a CUDA
measurement. The kernel itself is small enough that a single GPU run
of a few seconds suffices.

ADR-018 documents:
- The kernel design (one program per row, register-resident comb
  matrix, JIT specialization on `hc_mult`).
- The parity contract (cosine sim > 0.999, absolute tolerance on
  pre/post).
- The measured speedup at the representative shapes.
- Why we still keep the PyTorch path as the oracle (M1 dev, CI parity,
  any future debugging where Triton diverges).
- Why FP32 only (Sinkhorn iteration accumulation in BF16/FP16 would
  be a separate experiment).

**Deliverable**: ~200 LoC bench script + ~250 LoC ADR-018. Single
commit (or two: bench + ADR).

## Decisions to confirm before implementation

1. **FP32 only.** Reference is FP32; Sinkhorn iteration is sensitive
   to accumulation order. Lower-precision variants are a separate
   experiment that needs its own parity story. Decision: stay FP32.
2. **No backward.** We're an inference engine; HC's training-time
   gradient is not our concern. Decision: forward only.
3. **`hc_mult` as a Triton constexpr.** JIT-compiles per `hc` value;
   one cache entry per unique `hc`. V4 uses `hc=4`, future variants
   maybe `hc=2 / hc=8`. The kernel-cache cost is negligible.
4. **`sinkhorn_iters` as a constexpr too.** V4 uses 20. Constexpr means
   the iteration is unrolled by the Triton compiler. If a future model
   uses a different iter count, it gets a new specialization. Cheap.
5. **Whether to fuse the upstream linear into the kernel.** Today
   `HyperConnections.hc_pre` computes `mixes = (x_flat @ fn.T) * rsqrt`
   in PyTorch, then calls `hc_split_sinkhorn(mixes, ...)`. Fusing the
   linear into the Triton kernel would save another HBM round-trip but
   complicates the kernel substantially (different program-grid shape,
   different per-program memory traffic). Decision: defer. Phase 1 is
   the Sinkhorn fusion only; the linear stays in PyTorch (cuBLAS).
6. **Tolerance for parity.** PyTorch and Triton can disagree at the
   1e-6 level on FP32 reductions due to summation order. cosine sim
   > 0.999 is the codebase's standard parity bar (matches `paged_attention`,
   `int8_kernel`). Decision: same bar.
7. **Whether to launch one program per row or one program per group
   of rows.** Triton's launch overhead is per-grid-program; if `B*T`
   is in the thousands, the overhead is in the kernel-launch count
   (low) not the per-program count (high, but parallel). Decision:
   one program per row. Simplest mapping from tilelang; revisit if a
   benchmark shows kernel-launch overhead dominates.

## Modal cost

- **Slice 1, 2**: zero Modal cost. The kernel can be written and
  parity-tested on a Modal sandbox in one short run (~$0.30 on an
  L40S), but the development happens on M1 against the PyTorch path;
  Modal is just the parity CI gate. Within the existing Modal CI
  budget; no incremental cost.
- **Slice 3 bench**: ~$1-2 for a single H100 (or ~$0.50 on L40S)
  measuring PyTorch vs Triton across shapes. Single run, no
  re-iteration needed once correctness is solid.

Total to ship Slices 1-3: ~$1-2 Modal spend, well under the per-slice
cap.

## Risks

1. **Triton compilation surface.** Triton's autotune and constexpr
   machinery have sharp edges; the kernel-cache key (which we don't
   control directly) can fire recompilation on shape variation. We
   mitigate by hardcoding `hc_mult` and `sinkhorn_iters` as constexprs
   so the cache key is small and stable.
2. **Numerical divergence on the Sinkhorn iteration.** 20 iterations
   of alternating row/col normalization is the kind of accumulation
   chain where Triton's reduction order can subtly diverge from
   PyTorch's. The parity test catches this; mitigation may involve
   matching PyTorch's exact reduction order (which is well-defined
   for the small matrix sizes here).
3. **Per-row independence assumption.** The tilelang reference's
   thread block per row is the design's load-bearing simplification.
   If a future V5/V6 changes that (e.g., cross-token Sinkhorn
   coupling), the kernel needs a redesign. Tracked as "not in scope"
   for the current port; current V4 / V4-Flash fit cleanly.
4. **FP32 forward in a model otherwise running BF16.** The kernel
   takes FP32 inputs (matching reference). Callers
   (`HyperConnections.hc_pre`) already upcast to FP32 internally
   before the call. No interface change; mention in the ADR.
5. **Triton ↔ PyTorch view-strides mismatch.** Triton expects tensors
   with predictable strides. `mixes` arrives as a `(B, T, mix_hc)`
   view of an upstream matmul output; we reshape to `(B*T, mix_hc)`
   before the launch (same as the tilelang reference does). Care needed
   that the reshape is contiguous; explicit `.contiguous()` if not.

## Files to create / modify

NEW source:

- `src/mini_infer/models/blocks/hc_sinkhorn_kernel.py`: the Triton
  kernel + `supports_hc_kernel` + the Python entry point.

EDIT source:

- `src/mini_infer/models/blocks/hyper_connections.py`: rename the
  existing PyTorch body to `_hc_split_sinkhorn_torch`; add the
  dispatcher to `hc_split_sinkhorn`; import the kernel module under
  the CUDA-safe gate.
- `src/mini_infer/models/blocks/__init__.py`: no change (the public
  API of `hc_split_sinkhorn` is unchanged).

NEW docs:

- `docs/decisions/ADR-018-hc-sinkhorn-triton-port.md`: the design ADR.

NEW scripts:

- `scripts/bench_hc_sinkhorn.py`: bench (Modal for the speedup
  measurement; CPU baseline locally).

NEW tests:

- `tests/unit/test_hc_sinkhorn_kernel.py`: gated on
  `supports_hc_kernel(device)`. Parity vs PyTorch reference.
- The existing `tests/unit/test_hyper_connections.py` already exercises
  `hc_split_sinkhorn` through `HyperConnections.hc_pre`; no edit needed
  beyond confirming it still passes on CPU.

EDIT docs:

- `docs/architectures/deepseek-v4.md`: update the "Hyper-Connections"
  section to mention the Triton path lives alongside the PyTorch
  oracle.
- `docs/plans/roadmap-2026.md`: once shipped, mark long-term item #11
  (Triton kernel per quarter) as having its first ship.

## Verification

Slice-by-slice:

1. **Slice 1**:
   - On a CUDA device, `pytest tests/unit/test_hc_sinkhorn_kernel.py`
     passes.
   - cosine sim > 0.999 vs `_hc_split_sinkhorn_torch` on multiple
     `(B, T, hc_mult)` configurations.
   - Doubly-stochastic invariant holds within `2 * eps`.
   - On CPU, the new test file skips (gated by predicate).
2. **Slice 2**:
   - `pytest tests/unit/test_hyper_connections.py` still passes on
     CPU (PyTorch path).
   - On CUDA, the same suite passes with the dispatch taking the
     Triton path (verifiable via a runtime assert or breakpoint in
     the dispatcher during development).
3. **Slice 3**:
   - `scripts/bench_hc_sinkhorn.py` prints PyTorch vs Triton latency
     at representative shapes; the Triton column is faster at every
     shape we benchmark.
   - ADR-018 records the speedup numbers and the kernel-launch-count
     reduction (50+ → 1 per call).

## What this gets us

After Slice 3:

- **The first Triton kernel where the reference implementation is
  itself a custom kernel.** Other Triton kernels in the codebase
  (`int8_kernel`, `paged_attention`, TurboQuant) are kernels we write
  for performance; this is a kernel because *the paper has a kernel*.
  Aligned squarely with the niche: paper-faithful kernels in readable
  code.
- **A demonstration of fusing a long elementwise + reduction chain
  into one launch.** The Sinkhorn iteration is a textbook example of
  "compute is trivial but kernel launch overhead dominates"; the
  perfect case study for a reader learning when fusion matters.
- **Material progress on roadmap item #11** ("one real Triton kernel
  per quarter where the paper calls for it"). This is the first ship
  under that goal.
- **A small decode-throughput win on V4.** HC isn't the bottleneck
  (FlashAttention + GEMMs dominate), but a 10x reduction in the
  HC-induced kernel-launch overhead removes one of the serial latency
  sources per layer per token.

Out of scope for this plan but worth noting:

- **Fusing the upstream `linear + rsqrt` into the kernel.** Would
  save the `mixes` allocation. Real win on top of the Sinkhorn fusion,
  but a different kernel shape. Tracked as a follow-up.
- **BF16 / FP16 variants.** Different numerical contract. Separate
  experiment.
- **A backward pass.** Not our scope; mini-infer is inference-only.
- **Other tilelang kernels in the reference.** `sparse_attn`,
  `fp8_gemm_kernel`, `fp4_gemm_kernel`, `act_quant_kernel` are also
  in `kernel.py`. Each one is a separate paper-faithful-Triton-port
  candidate, but they each have a different reason for being a kernel
  (FP8 GEMM ↔ cuBLAS, FP4 quantization ↔ vendor library). The
  sinkhorn one is uniquely "the paper specifies this exact algorithm";
  the others have more vendor-library substitutability.
