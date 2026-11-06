# ADR-058: Run the Kimi MoE experts as grouped matmuls

Date: 2026-11-06
Status: Accepted

## Context

Kimi Linear 48B has 26 MoE layers of 256 routed experts, top-8. Kimi K3 has
896 experts, top-16. `_KimiMoe.forward` looped over every expert and ran
`torch.where` on the routing mask of each. On a GPU, `torch.where` with one
argument runs `nonzero`, which waits for the GPU to learn the size of its
result. That is a host sync (the CPU waits for the GPU). One decode step of
the 48B had about 6,700 of them (26 layers of 258).

The README shows 2.1 tok/s decode for the 48B on one B200, eager. The other
MoE families (Mixtral, DeepSeek-V2, GLM, MiniMax-M3, DeepSeek-V4, Nemotron-H)
already run their experts through `grouped_expert_mlp` on Hopper and newer
GPUs, with no host sync.

## Decision

1. On Hopper and newer GPUs with bf16 activations
   (`supports_grouped_experts(device, dtype)`), `_KimiMoe` packs its experts
   with `PackedSwigluExperts` on the first forward. Each step then runs two
   grouped matmuls over all (token, expert) pairs and sums each token's
   experts in fp32, as DeepSeek-V4 does. Nothing reads back to the host.
2. The pack points the parameters of the experts at slices of its stacks, so
   the checkpoint names (`experts.N.w1/w2/w3`) and the loader do not change.
   The pack takes the activation of the experts, so the SiTU of K3 works.
   In the latent MoE of K3, the pack runs at the latent width, between
   `routed_expert_down_proj` and `routed_expert_norm`.
3. The router `_KimiMoeGate` does not change. Both of its conventions (the
   biased weights of Kimi Linear, the unbiased weights of K3) feed both paths.
4. Other devices (the CPU, MPS, older GPUs, fp32 models) run
   `_loop_experts`. It sorts the pairs by expert with a stable sort in
   slot-major order, reads the pair count of each expert once, and runs only
   the experts with pairs. Each expert gets its tokens in the order of the
   old loop and adds into the fp32 sum in expert order. So the result has
   the same bits as the old loop, with one host read per layer instead of
   one per expert.
5. `test_kimi_moe_dispatch.py` keeps the old loop as the reference. The
   fallback must equal it bit for bit. The grouped path must match it within
   the order of the fp32 sum, and on a CUDA GPU it must run with no sync.

## Alternatives Considered

- Gather the weights of each token's experts and run batched matmuls, as
  the small-batch path of Nemotron-H does. It has no sync, but it copies 8
  experts per token: 113 MB per token per layer for the 48B in bf16.
- Keep the old loop on devices without the grouped path, as `MoEFFN` does.
  That keeps 256 syncs per layer on an A100 or L4 and a slow loop on the
  CPU. The sorted loop is about 20 lines.

## Consequences

- On the CPU (the fallback path), `profile_kimi_decode.py split` of PR #93
  counts the ops that read a value back to the host. At the 48B widths with
  256 experts, one MoE layer went from 258 such ops to 0, and from 636 ATen
  ops to 133. The CPU count does not see the one `tolist` per layer.
- On one H100 in bf16, an eager decode step of a 4-layer slice of the 48B
  shape (3 MoE layers, 256 experts, random weights, 1,024 history tokens):

  | Batch | Step, before | after | GPU work, before | after | Host syncs, before | after |
  |---|---:|---:|---:|---:|---:|---:|
  | 1 | 30.5 ms | 6.0 ms | 7.9 ms | 1.6 ms | 770 | 2 |
  | 8 | 54.9 ms | 7.4 ms | 14.4 ms | 3.8 ms | 770 | 2 |

  At batch 1 the kernels per step went from 3,789 to 519, and the MoE GPU
  time from 6.8 ms to 0.5 ms. The 2 syncs left come from the MLA layer.
- The first forward copies the experts of each MoE layer into the stacks.
  For a 48B layer that is 3.6 GB more for a moment. The old tensors are
  freed when the experts point at the stacks.
- The step is still host-bound: 6.0 ms for 1.6 ms of GPU work. The MoE no
  longer blocks a CUDA graph, but the MLA layers still read `row_starts` on
  the host, so a graph for the Kimi decode is a separate change.
- Measured on an H100 only, without the real checkpoint. The grouped path
  is the one that the other families already run on H200, B200 and B300.
