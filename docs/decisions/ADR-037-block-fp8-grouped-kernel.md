# ADR-037: Triton grouped kernel for block-FP8 MoE experts

Date: 2026-10-21
Status: Accepted

## Context

The GLM-5 FP8 and MiniMax-M3 checkpoints keep their routed experts in e4m3,
with one fp32 scale per 128x128 block, as `Fp8Expert`s
([ADR-021](ADR-021-glm-moe-dsa-port.md)). After
[ADR-036](ADR-036-grouped-expert-matmuls.md), bf16 experts run as grouped
matmuls, but `Fp8Expert`s kept a per-expert loop. One GLM-5.3 MoE layer
with FP8 experts took 9 to 190 ms per forward on an H100 and a B200.
`torch.nn.functional.scaled_grouped_mm` does not take 128x128 weight
blocks.

## Decision

1. A W8A16 grouped kernel in Triton (`models/blocks/moe/fp8_grouped_kernel.py`).
   It multiplies each e4m3 weight by its block scale in fp32 and rounds it
   to bf16, as `Fp8Expert.forward` does. The bf16 tensor cores then
   multiply into fp32. With no host sync, it runs in a CUDA graph.
2. Two splits. Up to 256 (token, expert) rows, a program starts at each
   expert segment of `BLOCK_M` rows, so a decode batch loads all experts
   in parallel. Above 256 rows, a program walks the experts of a row tile.
3. Weights as the left operand (`out.T = W @ rows.T`). Hopper's `wgmma`
   reads that operand from registers, where the dequantized tile already is.
4. `grouped_expert_mlp` takes a `BlockFp8Weight` in place of a bf16 tensor.
5. `PackedFp8Experts` stacks the experts as `PackedSwigluExperts` does. `I`
   must be a multiple of 128, as in GLM-5.3 (2048), or the loop runs.
6. `GlmMoeFFN` takes the path on Hopper or newer, with Triton and bf16 or
   fp16 activations. Otherwise it keeps the loop, which stays the reference.
7. Tiles for each GPU family. Three tiers by row count (up to 256, up to
   8192, above), with one table for Hopper and one for Blackwell.
8. `tests/unit/models/blocks/moe/test_fp8_grouped_kernel.py` runs on CUDA or under Triton's
   interpreter (the `mini-infer-triton-interp` Docker image). There the
   kernel multiplies in fp32, because the interpreter gets a bf16 `tl.dot`
   wrong.

## Alternatives Considered

- `scaled_grouped_mm` with row-wise or MXFP8 scales. It also quantizes the
  activations, so the results do not match the loop.
- Split-K for decode. The per-segment split already fills the GPU.
- W8A8 with per-token, per-128 activation scales, as DeepSeek, vLLM and
  transformers run these checkpoints. Tried and removed. At 16384 tokens it
  took 55 ms, against 43 ms (H100) and 28 ms (B200) for W8A16. Its error
  against fp32 math was 8 times larger (4.4e-2 against 5.4e-3).
- Dequantize the hit experts into a bf16 stack. Finding them needs a host
  sync.
- Gluon or CuTe DSL. Two kernels to own, and no interpreter.
  [ADR-038](ADR-038-gluon-fp8-grouped-hopper.md) later added a Gluon kernel
  for the H100.

## Consequences

The GLM-5.3 layer on an H100 and a B200
([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)):

- The first version made the layer 13 to 30 times faster than the loop up
  to 1024 tokens and 2.7 to 3.2 times at 16384.
- The second version (decisions 2, 3 and 7) is another 1.1 to 2.4 times
  faster in a CUDA graph from 1 to 64 tokens. From 256 to 16384 tokens it
  is 1.05 to 1.39 times faster.
- At 2048 tokens, about 8400 GPU operations and 256 host syncs became 63 to
  65 operations and none.
- FP8 experts use half the memory of bf16 experts. Against bf16 experts on
  `grouped_mm`, the FP8 layer is 1.05 to 2.0 times faster up to 256 tokens
  and about equal at 1024. From 4096 tokens it is 1.26 to 1.46 times
  slower. ADR-038 closes this gap on the H100.
- The outputs differ from the loop by a relative 3.8e-3, because the loop
  adds the 8 expert outputs in bf16 one at a time.
- No real FP8 checkpoint ran end to end, because they need 4 to 8 GPUs.

## Update 2026-10-22: tiles of one expert each

Above 256 rows, the default split now gives each program one tile of one
expert. With row tiles, a program ran the whole K loop once per expert of
its rows. The test of decision 8 covers all three splits. Per GPU
([benchmark](../benchmarks/2026-10-21-moe-grouped-gemm.md)):

- The outputs of the two splits are bit-identical.
- On the B200, expert tiles are 1.08 to 1.42 times faster than row tiles
  from 64 to 16384 tokens. On the H100 they are 1.06 to 1.31 times faster.
- On the B200 the FP8 layer then beats bf16 at 1024 tokens (4.02 against
  5.12 ms). It is still 1.20 and 1.40 times slower at 4096 and 16384
  tokens.
- If the Triton version and the operands fit (ADR-038, decision 8), a
  Gluon kernel runs above 256 rows. It always runs on the H100 (sm90), and
  on the B200 (sm100) only above 64 mean rows per expert
  ([ADR-039](ADR-039-gluon-fp8-grouped-blackwell.md)).
