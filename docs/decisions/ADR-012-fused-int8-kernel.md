# ADR-012: Fused W8A16 Triton kernel for `Int8Linear`

Date: 2026-04-29
Status: Accepted

## Context

ADR-010 shipped weight-only INT8 (W8A16) with a naive `Int8Linear.forward`:

```python
def forward(self, x):
    w_dq = self.weight.to(x.dtype) * self.scales.unsqueeze(1)  # HBM round-trip
    return F.linear(x, w_dq, self.bias)
```

Memory savings (~30% on Qwen2.5-0.5B) landed cleanly, but throughput
was ~neutral because every forward materializes the bf16 weight matrix
to HBM and immediately reads it back for the matmul. ADR-010 explicitly
flagged "fused dequant-matmul kernel" as a follow-up.

This slice ships that kernel.

## Decision

1. **Triton W8A16 GEMM kernel** in `src/mini_infer/quant/int8_kernel.py`.
   One `@triton.jit` function (`_w8a16_gemm_kernel`) + a Python launch
   wrapper (`fused_w8a16_linear`). CUDA-only; the import is guarded the
   same way as our existing paged-attention Triton kernel.
2. **`Int8Linear.forward` dispatches** to the fused kernel on CUDA when
   Triton is available, falling back to the naive HBM-materialize path
   on CPU/MPS or when the kernel is unavailable. A
   `_FUSED_DISABLED_FOR_BENCH` toggle lets benchmarks A/B the two paths
   on the same model load.
3. **Kernel design**:
   - Tile-based GEMM with the canonical Triton matmul pattern: pre-
     compute pointers at the top, advance by `BLOCK_K * stride_k` per
     loop iter, accumulate via `tl.dot(x, w, acc)`.
   - **Per-output-channel scales applied AFTER the K reduction**, not
     inside the loop. Mathematically equivalent
     (`scales[n] * sum_k x[m,k] * w_int8[n,k]`), and crucially: cheaper
     and stops the int8→bf16 cast from happening on every K iter. The
     inside-the-loop variant tripped a Triton MLIR pass assertion (see
     iteration log below).
   - **K loop is `range(0, K, BLOCK_K)`**, no per-iter K mask. Caller
     contract is K divisible by BLOCK_K; Qwen2.5-0.5B and 7B both have
     K ∈ {896, 4864} which are multiples of 32 and 64.
   - **Two block-size profiles**: 16×64×64 decode (M ≤ 16, num_warps=2),
     64×64×32 prefill (M > 16, num_warps=4). Sized so the
     `(BLOCK_M × BLOCK_N)` fp32 accumulator stays well within Ampere
     SM register budgets at the chosen warp count.
4. **No autotune in V1**. Two hand-picked profiles. Autotune is an
   explicit follow-up.

## Why this design (vs alternatives)

- **`tl.dot(x, w, acc)` (canonical pattern, used here)** vs
  `acc += tl.dot(x, w)`: the explicit-accumulator form maps cleanly to
  the HMMA tensor-core "MMA with accumulator" instructions and is the
  pattern in Triton's official matmul tutorial.
- **Scales applied after K-loop** vs per-iter: the post-loop variant
  works around an MLIR lowering assertion that the per-iter cast +
  multiply variant tripped. It's also a hair cheaper (one multiply per
  output element instead of one per K iter).
- **No K masking** vs per-iter K masks: K-divisibility is a contract
  Qwen satisfies and the simpler loop is more reliable to compile.
- **Two profiles** vs one: M=1 decode and M=80 prefill have very
  different optimal tile shapes; one profile would either waste compute
  on padding (decode at BLOCK_M=64) or waste registers on too-big M
  (prefill at BLOCK_M=16).
- **Hand-picked tiles** vs `triton.autotune`: autotune is the right
  long-term answer but compile-time blow-up + the iteration cost of
  diagnosing crashes pointed to "ship two known-good profiles first,
  add autotune later if benchmarks justify it."

## Iteration log (what didn't work)

This kernel took three Modal iterations to ship. Documenting the dead
ends so the same MLIR assertion doesn't bite next time:

1. **Iteration 1**: 128×128×32 prefill profile, 16×64×64 decode,
   per-iter K mask, scales multiplied in the loop.
   → Triton crashed with `llvm::ArrayRef::operator[]: Index < Length`
   on JIT compile of the first kernel call (prefill warmup).
2. **Iteration 2**: same block sizes, switched to `tl.dot(x, w, acc)`
   form, added a `DTYPE: tl.constexpr` to make types static.
   → Same MLIR assertion.
3. **Iteration 3 (shipped)**: dropped the prefill tile to 64×64×32,
   moved scales out of the K loop, set explicit `num_warps=2/4`, dropped
   K masking (Qwen K is BLOCK_K-divisible).
   → Compiles and runs. The exact triggering combination wasn't
   isolated, but the working configuration is conservatively sized
   relative to per-warp register budgets, which is the most plausible
   root cause.

## Empirical findings — model-size regime check

Same kernel, two model sizes, A10 bf16, identical workload (~80-token
prompt, max_tokens=32, three concurrencies):

| Model        | C=1 fused/naive | C=4 fused/naive | C=8 fused/naive |
|--------------|----------------:|----------------:|----------------:|
| 0.5B         | **0.92x**       | **0.74x**       | **0.99x**       |
| 7B           | **2.74x**       | **1.65x**       | **1.47x**       |

The same code base flips from "loses to cuBLAS" at 0.5B to
"**2.74x at decode**" at 7B. Cause: 7B's bf16 weights total ~14 GB; the
naive path's HBM round-trip on those weights becomes the dominant cost
on every forward, and that's exactly what fused skips. At 0.5B the
weights total ~1 GB and the matmul work is compute-bound, where cuBLAS
bf16 GEMM is hard to beat.

This matches the pattern from ADR-008 (paged-FA varlen): a kernel
technique whose published wins are at production scale doesn't
automatically reproduce at toy scale, but the same implementation
delivers when the workload moves into its target regime.

## Alternatives considered

- **`bitsandbytes.Linear8bitLt`**: drop-in, would deliver the speedup
  for free. Rejected because the implementation is what we're
  demonstrating; importing someone else's kernel defeats the goal.
- **`torch.compile` with a custom op**: in principle could fuse
  dequant + matmul through Inductor. In practice doesn't fuse INT8
  weight paths reliably without a custom op definition; lower-effort,
  lower-payoff.
- **CUTLASS C++ kernel**: a serious production engine would use
  CUTLASS templates. Out of scope for the project's pedagogical
  framing; Triton is the right vehicle for showing kernel-engineering
  thought.
- **Inside-the-loop scale multiply**: tripped the MLIR assertion in
  iteration 1; post-loop is functionally equivalent and compiles.

## Consequences

- **Positive**:
  - On 7B at decode (the production-relevant regime): up to **2.74x**
    throughput vs the naive int8 path, and the same kernel scales
    smoothly to higher concurrencies (1.47–1.65x at C=4–8).
  - Memory advantage from ADR-010 (~30% smaller weights) is preserved;
    nothing about INT8 storage changes.
  - The kernel + dispatch is local: ~200 LOC of new code, fully behind
    the existing `Int8Linear` API. CPU/MPS continues to use the naive
    forward unchanged.
- **Negative**:
  - At 0.5B (toy-scale models), fused regresses 0.74–0.92x vs naive.
    Honest finding; documented in the bench report. Fix: autotune or
    fall through to naive when M small AND model small.
  - Hand-picked tile sizes leave performance on the table at any
    specific shape; autotune is a real follow-up.
  - One kernel, no fp8 / W4A16 variants. The dispatch boundary is
    clean enough to add them later (same `Int8Linear` shape, different
    storage / kernel path).
- **Reversibility**: the fused path is gated by `supports_fused_kernel`
  + `_FUSED_DISABLED_FOR_BENCH`. The naive path stays as the fallback;
  reverting is a one-line change in the dispatch.

## Validation

- **M1 (CPU, fp32)**: 8 unit tests for the kernel module
  (`@requires_cuda`-marked, skipped on M1 via `tests/conftest.py`).
  Existing 143-test unit suite stays green; the dispatch change in
  `Int8Linear.forward` doesn't affect CPU path. New CPU sanity test
  asserts that `forward` does NOT call the fused kernel when `x.is_cuda`
  is False.
- **CUDA (A10, bf16)**:
  - Qwen2.5-0.5B: fused 0.92/0.74/0.99x naive at C=1/4/8 (worse — the
    workload isn't HBM-bound at 0.5B).
  - Qwen2.5-7B: fused **2.74/1.65/1.47x** naive at C=1/4/8.
  - Numbers in `docs/benchmarks/2026-04-29-fused-int8-kernel.md`.
- **Modal-side token-level correctness verified**. The
  `quant_kernel` bench now runs a greedy parity check (same prompt,
  naive vs fused dispatch on the same int8 model) before the
  throughput sweeps. Confirmed match on:
  - Qwen2.5-0.5B / A10 / bf16: `[12095, 13, 1084, 374, 279, 7772, 3283, 304]`
    = " Paris. It is the largest city in" — naive == fused, exact.
  - Qwen2.5-7B / A10 / bf16: `[12095, 13, 15920, 315, 279, 2701, 12239, 374]`
    = " Paris. Which of the following statements is" — naive == fused, exact.

## Pointers

- Implementation: `src/mini_infer/quant/int8_kernel.py`.
- Dispatch: `src/mini_infer/quant/int8.py::Int8Linear.forward`.
- Unit tests: `tests/unit/test_int8_kernel.py` (CUDA-only, parity vs
  naive across decode / prefill / GQA k_proj / MLP down_proj shapes).
- CPU dispatch sanity: `tests/unit/test_int8_quant.py::test_int8_linear_cpu_does_not_attempt_fused_kernel`.
- Bench: `scripts/modal_packed_bench.py --config quant_kernel`.
- ADR-010 (the W8A16 baseline this builds on): `docs/decisions/ADR-010-int8-weight-quant.md`.

## Follow-ups

- **Triton autotune attempted, reverted (regression)**. Wrapping the
  kernel in `@triton.autotune` over an 8-config space keyed on
  `(M, N, K)` regressed throughput badly at C=4/C=8 on both 0.5B
  (0.19x/0.29x of naive) and 7B (0.46x/0.51x of naive). Diagnosis:
  continuous batching produces many distinct M values per run (M=1
  decode, M=80 single-prefill, M=320 4-way packed prefill, M=4 packed
  decode, ...), each firing a fresh 8-config sweep, and the sweep cost
  dominated the timed window. Hand-picked profiles ship instead, with
  a clearer fix as a real follow-up below.
- **M-bucketed autotune**: split into two `@triton.jit` kernels —
  `_w8a16_decode_kernel` (autotune over small-M configs only,
  `key=['N', 'K']`) and `_w8a16_prefill_kernel` (autotune over large-M
  configs only, `key=['N', 'K']`). Wrapper dispatches by M. Each
  kernel's autotune sweep fires once per (N, K), regardless of M
  variation within the bucket. Should close the 0.5B regression and
  give ~10% at 7B.
- **W4A16** (4-bit weights, GPTQ/AWQ dequant). Same kernel skeleton
  with a 4-bit unpacking step in the K-loop body; biggest memory win.
- **W8A8** (full INT8). Real Tensor-Core INT8 GEMM throughput on
  Ampere; needs activation calibration + outlier handling
  (SmoothQuant-style).
- **Integration with paged attention**: today's kernel is for the
  matmul layers (q/k/v/o/gate/up/down). Attention is a separate kernel
  path (FA varlen). They're orthogonal.
