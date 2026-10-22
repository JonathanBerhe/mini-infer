# ADR-036: Grouped expert matmuls for MoE layers

Date: 2026-10-21
Status: Accepted

## Context

The MoE blocks dispatched their routed experts one expert at a time.
`MoEFFN` (Mixtral, DeepSeek-V2) and `GlmMoeFFN` (GLM-MoE-DSA, MiniMax-M3)
looped over the local experts. Each step of the loop found the expert's
tokens with `torch.where`, ran the expert MLP, and added the weighted output
back with `index_add_`. `torch.where` syncs with the host, so a layer with E
experts made about E host syncs and 10 to 15 small launches per expert.
A CUDA graph cannot hold such a forward.

Nemotron-H stores its experts stacked and had two paths. Up to 64 (token,
expert) pairs it gathered each pair's expert weights, which is capturable
but copies the weights: about 0.5 GB per MoE layer at batch 4
([ADR-035](ADR-035-nemotron-h-port.md)). Above 64 pairs it looped like the
other blocks.

The target GPUs are the H100 and the B200. PyTorch ships
`torch.nn.functional.grouped_mm` (a wrapper of `torch._grouped_mm`), which
runs a CUTLASS grouped GEMM on Hopper and Blackwell: one launch computes a
different matmul for each group of rows.

## Decision

1. **Two grouped matmuls per layer** (`models/blocks/grouped_experts.py`).
   `grouped_expert_mlp` flattens the (token, expert) pairs and sorts them by
   expert on the GPU. `searchsorted` gives each expert's end row. Then the
   gate-up and the down projections each run as one `functional.grouped_mm`
   over all experts. The weighted outputs go back to pair order with
   `index_copy_` and are summed over each token's k experts. Nothing reads
   back to the host, so the forward runs inside a CUDA graph.
2. **On by default on Hopper and newer.** `supports_grouped_experts(device)`
   gates the path on compute capability 9.0 or higher (the new
   `is_hopper_or_newer_device` in `device.py`). On older GPUs, the CPU and
   MPS, `functional.grouped_mm` falls back to a loop inside PyTorch, so those
   devices keep the old paths. The old paths stay the reference.
3. **Stacked weights without a new checkpoint layout.** Nemotron-H passes
   its stacked tensors directly. For the bf16 `MixtralExpert`s of `MoEFFN`
   and `GlmMoeFFN`, `PackedSwigluExperts` stacks the weights on the first
   grouped forward. It copies `w1` and `w3` into one `(E, 2I, in)` tensor
   and `w2` into one `(E, out, I)` tensor, then points each expert's
   parameters at slices of the stacks. The memory is not doubled, the
   experts still work alone, and `state_dict` keys and loaders do not
   change. When a reload or `module.to` gives a parameter new storage, a
   pointer test sees it and the block packs again.
4. **Same accumulation as each block's old path.** Nemotron-H sums the
   weighted expert outputs in fp32, as its reference does. `MoEFFN` and
   `GlmMoeFFN` sum in the model dtype.
5. **Expert parallelism unchanged.** Pairs that route to another rank's
   experts sort after the local groups and add zero. The all-reduce and the
   shared experts work as before.
6. **One process measures before and after.** The private toggle
   `grouped_experts._DISABLED_FOR_BENCH` switches the blocks back to the old
   paths. `scripts/modal_moe_grouped_bench.py` uses it for the A/B, the
   profile and the real-model run.

## Alternatives Considered

- **FlashInfer's fused MoE kernels.** They also fuse the routing and the
  activation, and they take FP8 and FP4 weights. But they compile at run
  time, and the FlashInfer JIT was slow on Modal before. `grouped_mm`
  is in PyTorch, needs no compile step, and covers the bf16 experts.
- **A Triton grouped GEMM.** It would be more code to own for a matmul
  that PyTorch already ships. It is the path for a weight format that
  `grouped_mm` does not take: see
  [ADR-037](ADR-037-block-fp8-grouped-kernel.md) for block-FP8 experts.
- **Gluon or CuTe DSL kernels.** These make sense only when profiling
  shows that the grouped GEMM itself is the limit. The profile shows that
  most of the old time was host syncs and small launches.
- **Keep the gather for Nemotron-H at one token.** The B200 is 3.6 percent
  faster with the gather at batch 1, the H100 is 8 percent slower. A
  threshold for each GPU is not worth it for a case that serving reaches
  only at `max_batch_size=1`.

## Consequences

- Measured on one H100 and one B200
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)): prefill of
  the real Nemotron 3.5 Lightning 30B-A3B is 8.0 to 8.5 times faster at 512
  tokens and 4.9 to 6.7 times faster at 2048 tokens. Decode with the CUDA
  graph is 1.21 to 1.52 times faster at batch 4. At batch 16 the decode
  step now runs in a CUDA graph: 19.7 ms (H100) and 14.2 ms (B200) per
  step, against 214 and 203 ms eager before.
- One Nemotron MoE layer at 2048 tokens went from 1959 GPU operations and
  130 host syncs to 66 operations and none.
- Before, a Nemotron-H decode graph above 10 rows (64 pairs at top-6) hit
  the looping path, which syncs with the host, so it was not capturable.
  Now the graph captures at 16 rows on both GPUs.
- The grouped path is slower in three cases: batch-1 graph decode on the
  B200 (7.74 against 7.47 ms), eager decode at small batches (host-bound,
  10 to 12 more launches per layer), and Mixtral at 16384 tokens on the
  B200 (0.88x, where cuBLAS runs the 8 large expert matmuls faster).
- The two paths differ only in rounding. On the real model, the 512-token
  prefill logits have a cosine of 0.9994 to 0.9995 between the paths. This
  is the same kind of gap as the port against transformers, from bf16
  rounding at top-6 routing near-ties.
- `functional.grouped_mm` is new in PyTorch. The path tests for it with
  `hasattr`, and `tests/unit/test_grouped_experts.py` runs it on the CPU
  against a token-by-token reference.
- Block-FP8 experts (`Fp8Expert`) got a Triton grouped kernel
  ([ADR-037](ADR-037-block-fp8-grouped-kernel.md)). FP4-resident experts
  (`FP4Expert`), Inkling, Kimi Linear and the DeepSeek-V4 hash-routed block
  keep their loops.
