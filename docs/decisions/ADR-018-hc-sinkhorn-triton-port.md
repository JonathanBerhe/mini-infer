# ADR-018: Triton port of `hc_split_sinkhorn`

Date: 2026-05-17 (accepted 2026-06-02 after live validation)
Status: Accepted

## Context

The Hyper-Connections (HC) sublayer of DeepSeek-V4 (paper §2.5) mixes
several residual streams around every attention and FFN call. Its core is
`hc_split_sinkhorn`. For each row of `mixes` it computes `pre` and `post`
(sigmoids) and `comb`, an `(hc, hc)` matrix. Sinkhorn normalization divides
`comb` by its row sums, then by its column sums, many times. At the end,
every row and column sums to about 1 (doubly stochastic).

The V4 reference (`third_party/deepseek_v4_reference/kernel.py:371`) is a
tilelang kernel. Our earlier PyTorch transcription gave bit-equivalent
output, but each step was a separate op. V4-Flash has 61 layers with 2 HC
sublayers each, so one forward pass made about `122 * 50 = 6,100` launches
for this arithmetic. The compute per call is tiny (16 elements for
`hc = 4`), so the launches are the cost.

The port gives back the fusion of the reference kernel. It also cuts the
launches per HC sublayer from 50 to 1 on the critical path of each decoded
token.

## Decision

Add the Triton kernel `_hc_split_sinkhorn_kernel` with the entry point
`hc_split_sinkhorn_triton` and the predicate
`supports_hc_kernel(device, hc_mult)`. The old PyTorch body becomes
`_hc_split_sinkhorn_torch` without changes. It is the numerical oracle (the
reference result) and the CPU and MPS fallback. The public
`hc_split_sinkhorn` picks one of the two by the predicate, so no call site
changes.

- One Triton program per row of `(B*T, mix_hc)`. The `(hc, hc)` matrix stays
  in registers for all iterations.
- A scale-vector Sinkhorn loop. The loop carries two 1D vectors, `row_scale`
  and `col_scale`. Each iteration computes
  `comb * row_scale[:, None] * col_scale[None, :]` and sums that value. To
  fold `1 / (row_sum[j] + eps)` into `row_scale[j]` equals a division of
  row j, so the sums match the reference. The FP difference is 3.6e-7
  max-abs in FP32 at `hc=4, iters=20`.
- `hc_mult` and `sinkhorn_iters` are constexpr, so each variant compiles
  once. `num_warps` keeps its default, because `num_warps=1` gave no gain.
- `hc_mult` must be a power of 2, so `tl.arange(0, HC)` needs no mask. V4
  uses `hc=4`. Other values (for example `hc=3`) go to the PyTorch path.

The direct transcription of the reference loop crashes the Triton 3.1.0
compiler with SIGSEGV at JIT time (torch 2.5.1, CUDA 12.4, L40S). A bisect
(`scripts/modal_hc_kernel_probe.py`) ran each variant in its own subprocess:

| Variant | Reduction operand | Result |
|---|---|---|
| direct loop, axis-0 or trans + axis-1 sums | loop-carried tile | SIGSEGV |
| `tl.static_range(3)` unroll | plain value | OK, diff ~1e-7 |
| `tl.static_range(19)` unroll | plain value | compile hang (5+ min) |
| scale-vector, axis-0 sums | body-local value | OK, diff 3.6e-7 |

The crash needs a reduction whose operand comes from a 2D tile that the loop
carries. FlashAttention-style kernels never hit it, because their loops
reduce values that the body computes. The scale-vector loop has that shape,
with 1D statistics like the `m` and `l` of FlashAttention.

The parity contract:

- FP32 everywhere, as in the reference, and forward only.
- Cosine similarity > 0.999 against the PyTorch oracle, the bar of the
  codebase. Also `torch.testing.assert_close` on `pre` and `post` at
  rtol=1e-4, atol=1e-5, because cosine alone is too loose for small sigmoid
  outputs. `comb` gets rtol=1e-3, atol=1e-4, because 20 normalizations
  compound any difference in the order of the sums.
- After 20 iterations, every row and column of `comb` sums to about 1, apart
  from `eps`, as the oracle asserts.

## Alternatives Considered

1. Keep only the PyTorch path. Rejected: about 6,100 launches per V4
   forward, and roadmap item #11 names this kernel as the first
   paper-faithful Triton port.
2. Fuse the upstream `linear + rsqrt` of `hc_pre` into the kernel. It saves
   an HBM round-trip on `mixes`, but changes the grid and the memory
   traffic. Rejected for now: the Sinkhorn kernel alone removes the
   launches.
3. Port to CUDA C++. Rejected: every other kernel here is Triton, and an
   nvcc build path is too much for one small kernel.
4. A BF16 or FP16 version. Rejected: the inputs and the reference are FP32,
   and the 20-iteration loop needs its own parity bar.

## Consequences

- About 50 PyTorch ops per call become one Triton launch. The first call
  compiles the kernel in a few hundred ms, mostly during the first decoded
  token.
- `supports_hc_kernel` answers "can this kernel run" in one place, so no
  caller tests `device.type == "cuda"`.
- If the reference changes the Sinkhorn iteration,
  `_hc_split_sinkhorn_torch` and `_hc_split_sinkhorn_kernel` must change
  together. The parity test guards this.
- A non-power-of-2 `hc_mult` takes the slower PyTorch path without a
  warning, as the docstring of `supports_hc_kernel` says.
- The PyTorch oracle stays canonical on M1, CI and any non-CUDA path.
  `tests/unit/_v4_reference_helpers.py` still patches
  `kernel.hc_split_sinkhorn = <our public dispatcher>`, so on CPU the V4
  reference parity test reaches the oracle as before.

`scripts/bench_hc_sinkhorn.py` on an L40S, 2026-06-02 (torch 2.5.1+cu124,
triton 3.1.0, FP32, `hc_mult=4`, `batch=2`, `sinkhorn_iters=20`), median
per-call latency over 100 iterations after 10 warm-up iterations:

| seqlen | PyTorch (us) | Triton (us) | Speedup |
|---:|---:|---:|---:|
| 1 | 1006.6 | 70.7 | 14.25x |
| 64 | 1007.8 | 72.5 | 13.90x |
| 256 | 1092.7 | 73.6 | 14.84x |
| 1024 | 1020.9 | 73.6 | 13.87x |

- Both columns are flat. The PyTorch cost is the dispatch of about 50 ops,
  not compute. The Triton kernel runs all rows in parallel.
- V4-Flash decode makes 122 HC calls per token. At these isolated-call
  rates, that is about 123 ms against 8.8 ms of serial HC overhead per
  token. In-model rates differ.
- All 18 parity configurations of the same run passed the bars above. They
  cover hc ∈ {2,4,8} x T ∈ {1,16,256} x iters ∈ {1,20}, with raw randn
  inputs. The doubly-stochastic test on tempered inputs passed at the 5e-3
  tolerance of the CPU test.

## Pointers

- Kernel and predicate: [src/mini_infer/models/blocks/hc_sinkhorn_kernel.py](../../src/mini_infer/models/blocks/hc_sinkhorn_kernel.py).
- Dispatcher and oracle: [src/mini_infer/models/blocks/hyper_connections.py](../../src/mini_infer/models/blocks/hyper_connections.py).
- Bench: [scripts/bench_hc_sinkhorn.py](../../scripts/bench_hc_sinkhorn.py).
- Parity tests: [tests/unit/test_hc_sinkhorn_kernel.py](../../tests/unit/test_hc_sinkhorn_kernel.py).
- Plan: [docs/plans/hc-sinkhorn-triton.md](../plans/hc-sinkhorn-triton.md).
