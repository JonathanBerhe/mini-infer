# ADR-012: Fused W8A16 Triton kernel for `Int8Linear`

Date: 2026-04-29
Status: Accepted

## Context

ADR-010 added weight-only INT8 (W8A16: int8 weights, bf16 activations). Its
`Int8Linear.forward` wrote the dequantized bf16 weight to HBM and read it
back for `F.linear`. The weights got about 30% smaller on
Qwen2.5-0.5B, but throughput stayed about the same. ADR-010 named a fused
dequant-matmul kernel as the follow-up.

## Decision

1. A Triton W8A16 GEMM kernel in `src/mini_infer/quant/int8_kernel.py`:
   `_w8a16_gemm_kernel` and the launch wrapper `fused_w8a16_linear`. It runs
   on CUDA only.
2. On CUDA with Triton, `Int8Linear.forward` calls the fused kernel. On CPU
   or MPS, it uses the naive path. The `_FUSED_DISABLED_FOR_BENCH` toggle
   lets a benchmark compare both paths on one model load.
3. The kernel design:
   - The standard Triton matmul loop accumulates with `tl.dot(x, w, acc)`,
     which maps to the tensor-core MMA with an accumulator.
   - The per-output-channel scales apply after the K reduction:
     `scales[n] * sum_k x[m,k] * w_int8[n,k]`. This is one multiply per
     output instead of one per K step.
   - The K loop is `range(0, K, BLOCK_K)` with no K mask, so K must be a
     multiple of BLOCK_K. The K values of Qwen2.5-0.5B and 7B are multiples
     of 32 and 64.
   - Two tile profiles: 16×64×64 for decode (M ≤ 16, num_warps=2) and
     64×64×32 for prefill (M > 16, num_warps=4). One profile wastes compute
     on padding at decode or registers at prefill. The fp32 accumulator
     stays inside the register budget of an Ampere SM.
4. No autotune. The two profiles are hand-picked.

## Alternatives considered

- Scales inside the K loop, a K mask and a 128×128×32 prefill tile. This
  first version crashed at its first JIT compile with
  `llvm::ArrayRef::operator[]: Index < Length`. The shipped version uses
  the smaller tile, scales after the loop, explicit `num_warps=2/4` and no K
  mask. The exact trigger is not known. Register pressure is the likely
  cause.
- `@triton.autotune` over an 8-config space with the key `(M, N, K)`. This
  was tried and reverted. Continuous batching gives many M values per run
  (M=1, 80, 320, 4, ...). Each new M started a new 8-config sweep, and the
  sweeps took most of the timed window. At C=4/C=8 the fused path fell to
  0.19x/0.29x of naive on 0.5B and to 0.46x/0.51x on 7B.
- `bitsandbytes.Linear8bitLt`. It is a drop-in, but the kernel itself is
  the work of this ADR.
- `torch.compile` with a custom op. Inductor does not fuse INT8 weight paths
  reliably without a custom op definition.
- A CUTLASS C++ kernel. The kernels in this repository are in Triton.

## Consequences

Fused throughput over naive, A10, bf16, a prompt of about 80 tokens,
max_tokens=32 (from `docs/benchmarks/2026-04-29-fused-int8-kernel.md`):

| Model | C=1 | C=4 | C=8 |
|-------|----:|----:|----:|
| 0.5B  | 0.92x | 0.74x | 0.99x |
| 7B    | 2.74x | 1.65x | 1.47x |

- The bf16 weights of 7B are about 14 GB. The HBM round trip of the naive
  path is the main cost there, and the fused path skips it.
- On 0.5B the weights are about 1 GB and the matmul is compute-bound.
  cuBLAS bf16 GEMM is hard to beat there. ADR-008 (paged FA varlen) showed
  the same pattern. A fix is autotune, or the naive path when M and the
  model are small.
- The 30% weight saving of ADR-010 does not change.
- Hand-picked tiles leave speed unused at specific shapes.
- `supports_fused_kernel` and `_FUSED_DISABLED_FOR_BENCH` gate the fused
  path, so a revert is a one-line change.
- On M1, the 143 existing unit tests stay green.
  `tests/unit/quant/test_int8_kernel.py` has 8 CUDA-only tests that compare fused
  with naive at decode and prefill shapes.
  `tests/unit/quant/test_int8_quant.py::test_int8_linear_cpu_does_not_attempt_fused_kernel`
  asserts that the CPU path does not call the fused kernel.
- This closes the "Modal-side correctness not verified" gap.
  `scripts/bench/modal_packed_bench.py --config quant_kernel` decodes one prompt
  greedily with naive and fused dispatch before the throughput sweeps. The
  token IDs are equal on Qwen2.5-0.5B and Qwen2.5-7B.

## Follow-ups

- M-bucketed autotune. Split the kernel into `_w8a16_decode_kernel`
  (small-M configs) and `_w8a16_prefill_kernel` (large-M configs), each
  autotuned with `key=['N', 'K']`. The wrapper picks one by M, so each sweep
  runs once per (N, K). This can close the 0.5B regression and give about
  10% on 7B.
- W4A16 (GPTQ or AWQ): the same kernel with a 4-bit unpack in the K loop,
  for the largest memory saving.
- W8A8 (full INT8): INT8 tensor-core GEMM on Ampere. It needs activation
  calibration and outlier handling (SmoothQuant-style).
