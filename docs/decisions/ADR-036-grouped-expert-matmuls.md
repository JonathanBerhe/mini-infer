# ADR-036: Grouped expert matmuls for MoE layers

Date: 2026-10-21
Status: Accepted

## Context

`MoEFFN` (Mixtral, DeepSeek-V2) and `GlmMoeFFN` (GLM-MoE-DSA, MiniMax-M3)
ran their routed experts one at a time, with `torch.where` to find the
tokens of each expert. `torch.where` syncs with the host, so a layer with E
experts made about E host syncs and 10 to 15 small launches per expert. A
CUDA graph cannot hold such a forward. Nemotron-H copied about 0.5 GB of
expert weights per MoE layer at batch 4
([ADR-035](ADR-035-nemotron-h-port.md)), and looped above 64 (token,
expert) pairs.

On the H100 and the B200, `torch.nn.functional.grouped_mm` runs a CUTLASS
grouped GEMM: one launch computes a different matmul for each group of rows.

## Decision

1. `grouped_expert_mlp` (`models/blocks/grouped_experts.py`) sorts the
   (token, expert) pairs by expert on the GPU. The gate-up and the down
   projections each run as one `functional.grouped_mm`. Nothing reads back
   to the host, so the forward runs in a CUDA graph.
2. `supports_grouped_experts(device)` turns the path on for compute
   capability 9.0 or higher. Older GPUs, the CPU and MPS keep the old
   paths, which stay the reference.
3. `PackedSwigluExperts` stacks the bf16 expert weights on the first
   grouped forward and points each expert at slices of the stacks. So
   memory does not double, and `state_dict` keys do not change.
4. Each block keeps its old accumulation: fp32 for Nemotron-H, the model
   dtype for `MoEFFN` and `GlmMoeFFN`.
5. Under expert parallelism, pairs routed to another rank add zero.

## Alternatives Considered

- FlashInfer's fused MoE kernels. Their JIT compile was slow on Modal
  before.
- A Triton grouped GEMM. It is more code for a matmul that PyTorch ships.
- Gluon or CuTe DSL kernels. The profile showed that host syncs and small
  launches, not the GEMM, took most of the old time.

## Consequences

- On one H100 and one B200
  ([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)), prefill of
  the real Nemotron 3.5 Lightning 30B-A3B is 8.0 to 8.5 times faster at 512
  tokens. At 2048 tokens it is 4.9 to 6.7 times faster. Graph decode at
  batch 4 is 1.21 to 1.52 times faster. At batch 16 the decode step now
  runs in a CUDA graph: 19.7 ms (H100) and 14.2 ms (B200), against 214 and
  203 ms eager.
- One Nemotron MoE layer at 2048 tokens went from 1959 GPU operations and
  130 host syncs to 66 operations and none.
- The grouped path is slower for batch-1 graph decode on the B200 (7.74
  against 7.47 ms). It is also slower for eager decode at small batches and
  Mixtral at 16384 tokens on the B200 (0.88x).
- The paths differ only in bf16 rounding: a prefill logit cosine of 0.9994
  to 0.9995 on the real model.
- `tests/unit/test_grouped_experts.py` runs the path on the CPU against a
  token-by-token reference.
- Block-FP8 and FP4 experts got Triton grouped kernels
  ([ADR-037](ADR-037-block-fp8-grouped-kernel.md),
  [ADR-040](ADR-040-fp4-grouped-kernel.md)). Inkling and Kimi Linear keep
  their loops.
