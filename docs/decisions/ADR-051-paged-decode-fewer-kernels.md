# ADR-051: Fewer kernels in the paged decode step

Date: 2026-11-02
Status: Accepted

## Context

[ADR-050](ADR-050-paged-decode-cuda-graph.md) replays the paged decode step
from a CUDA graph. After it, the step of Qwen2.5-7B on one H100 at
concurrency 8 took about 9.8 ms. About 7.8 ms of that was GPU work and about
2 ms was a host gap. The launch of the graph costs host time for every
node. Each kernel in the step therefore costs GPU time and launch time.
Most kernels were small. An RMS norm was 8 kernels. The rotation of q and k and the K/V write
were about 12 kernels per layer. The q, k and v projections were three GEMMs
(matrix multiplies) per layer. For k and v, cuBLAS added a split-K reduce (a
kernel that sums partial results) and a bias copy. The MLP GEMMs took about
4 ms near the memory roofline, so they were not a lever.

## Decision

1. The fused RMS norm kernel of [ADR-045](ADR-045-fused-rms-norm.md) is on
   by default for Llama, Qwen2, Qwen3 and Mixtral (`fused_rms_norm`). The
   per-head q and k norms use it too.
2. The FlashInfer plan of a replayed step no longer waits for the GPU. It
   reads pinned host copies of the sequence boundaries and keeps the page
   indices on the device.
3. One Triton kernel per layer (`models/blocks/rope_write_kernel.py`)
   rotates q and k and writes the rotated k and v into the pool pages.
   `PagedKVCache.append_kv_with` does the bookkeeping and hands the pool
   positions to the kernel. The kernel keeps the rounding of the PyTorch
   path, so q and the cached k are bit-equal on CUDA.
4. One `QKVParallelLinear` per attention block holds the q, k and v rows.
   The loader concatenates the HF tensors at load time (`merge_qkv_weights`).
   On CUDA the block runs one GEMM and reads q, k and v as views. On the CPU
   and on MPS the layer runs one GEMM per row slice, so the bit-parity tests
   against HF keep their arithmetic.

## Alternatives Considered

- Prepare step t+1 on the host while replay t runs. It hides about 0.4 ms
  of the 2 ms gap and needs double-buffered step state. Not worth it yet.
- Keep the kernels and make the launch cheaper. The launch cost grows with
  the node count, so a shorter graph is the only lever.
- Run the merged GEMM on the CPU too. The CPU BLAS picks another kernel for
  the wider shape, and the bit-parity CI against the pinned HF checkpoints
  fails.
- Fuse the residual add into the norm. The adds and casts that are left
  cost about 0.2 ms per step. Deferred.

## Consequences

Measured on one H100, Qwen2.5-7B bf16, FlashInfer backend, graph on, 32 new
tokens, concurrency 1 and 8. Each change passed its own A/B run in one
container before it merged, so the baselines differ a little between rows.
The two variants decoded identical tokens in every run.

| Change | Measured per decode step | tok/s at concurrency 8 |
|---|---|---:|
| Fused RMS norm | GPU 7.75 to 7.20 ms, kernels 1363 to 1048 | 771 to 818 |
| Plan without host waits | host syncs 6 to 3, host gap at concurrency 1 2.6 to 2.0 ms | not measured apart |
| Fused RoPE and K/V write | GPU 7.31 to 6.47 ms, kernels 964 to 628 | 840 to 926 |
| Merged q, k and v GEMM | GPU 6.16 to 5.79 ms, kernels 544 to 460 | 930 to 979 |

- From the eager path of ADR-050 to this ADR, concurrency 8 went from 414 to
  979 tok/s on this workload. The tokens stayed identical at every step.
- The GPU work of a step is now about 5.8 ms. The MLP GEMMs take about 4.0
  ms of it near the memory roofline. The attention projections take about
  0.7 ms and the `lm_head` 0.35 ms.
- The host gap of about 2 ms per step is now the largest lever. The launch
  of the graph alone takes about 0.76 ms of host time per step. The full fix
  is async sampling: the argmax runs inside the graph, and the host reads
  the tokens one step late.
- Qwen3 and Gemma pay two small copies per layer, because the fused norm
  kernel wants contiguous q and k rows.
- The state dict of the Llama-shaped models names `qkv_proj` in place of
  `q_proj`, `k_proj` and `v_proj`. HF checkpoints load as before through
  `load_weights`.
- The rotation kernel runs on CUDA with Triton and an uncompressed pool.
  Every other case takes the PyTorch path.
- cuBLAS can pick another kernel for the wider GEMM, so the CUDA logits can
  differ in the last bit.
